"""Conditional forward models of the cognitive load index – the twin's missing simulator.

The Digital Twin records what happened; it cannot answer "if I schedule deep work at 3pm, what state
will I be in?". The only day-scale simulator in the repo (`paper/run_daily_experiment.py`) draws each
window IID from its activity's fixed Gaussian, so **permuting the schedule changes nothing** and an
optimiser over task order sees a flat landscape. Everything here exists to fix that one defect.

Design choices and why:

**Discrete levels, not continuous load.** Load is quantised into `n_levels` quantile bins. With ~36
labelled windows per activity a continuous density is not estimable, a categorical distribution over
16 bins is, and a categorical distribution is exactly what a Born machine models natively – so the
classical and quantum arms share one interface (`predict_proba` over levels).

**Absolute time, segmented.** Steps are only formed between windows that are genuinely adjacent
(`trajectory_contract.segment_bounds`). Treating a 40-minute break as one 5 s step would teach the
model that load collapses instantly, destroying the recovery dynamics the planner needs.

**Backoff, not flat smoothing.** `P(next | prev, activity, context)` has more cells than this dataset
has windows. A Jelinek-Mercer style interpolation backs off through
`(prev, activity, bucket) -> (prev, activity) -> (prev) -> marginal`, weighting each level by how much
evidence it actually has, so sparse cells degrade gracefully instead of becoming noise.

Every model implements the same three methods, so `trajectory_benchmark.py` and the quantum arm are
interchangeable:

    fit(steps)                      -> self
    predict_proba(steps)            -> (n, n_levels)
    sample_next(steps, rng)         -> (n,) sampled levels

    python trajectory_model.py --selftest
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_contract import (  # noqa: E402
    UNTAGGED,
    WINDOW_SECONDS,
    broadcast_field,
    segment_bounds,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

EPS = 1e-12
DEFAULT_LEVELS = 16


# -------------------------- quantisation ------------------------------------
class LevelQuantiser:
    """Quantile binning of the load index, fitted on training folds only."""

    def __init__(self, n_levels: int = DEFAULT_LEVELS):
        if n_levels < 2:
            raise ValueError("n_levels must be at least two")
        self.n_levels = int(n_levels)
        self.edges: Optional[np.ndarray] = None
        self.centres: Optional[np.ndarray] = None

    def fit(self, load: np.ndarray) -> "LevelQuantiser":
        values = np.asarray(load, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < self.n_levels:
            raise ValueError(f"need at least {self.n_levels} values to fit {self.n_levels} levels")
        quantiles = np.linspace(0.0, 1.0, self.n_levels + 1)[1:-1]
        self.edges = np.unique(np.quantile(values, quantiles))
        levels = np.searchsorted(self.edges, values, side="right")
        self.centres = np.array(
            [
                float(np.median(values[levels == k])) if np.any(levels == k) else float(np.nan)
                for k in range(self.n_levels)
            ]
        )
        # Empty bins happen when load is heavily tied; fill them by interpolation so `inverse` is
        # always usable rather than returning NaN into a downstream sum.
        if np.isnan(self.centres).any():
            index = np.arange(self.n_levels)
            good = ~np.isnan(self.centres)
            self.centres[~good] = np.interp(index[~good], index[good], self.centres[good])
        return self

    def transform(self, load: np.ndarray) -> np.ndarray:
        if self.edges is None:
            raise RuntimeError("quantiser is not fitted")
        return np.searchsorted(self.edges, np.asarray(load, dtype=float), side="right").astype(int)

    def inverse(self, levels: np.ndarray) -> np.ndarray:
        if self.centres is None:
            raise RuntimeError("quantiser is not fitted")
        return self.centres[np.clip(np.asarray(levels, dtype=int), 0, self.n_levels - 1)]


# -------------------------- step extraction ---------------------------------
@dataclass
class Steps:
    """One autoregressive transition per row, plus everything needed to condition it."""

    prev_level: np.ndarray
    next_level: np.ndarray
    activity: np.ndarray            # string slug
    day_id: np.ndarray
    session_id: np.ndarray
    hour: np.ndarray                # 0..23, fractional
    time_on_task: np.ndarray        # seconds since the current activity began
    time_since_break: np.ndarray    # seconds since an explicitly recorded break; NaN means unknown
    elapsed: np.ndarray             # seconds since the start of the day
    complexity: np.ndarray
    n_levels: int = DEFAULT_LEVELS
    activity_names: Sequence[str] = field(default_factory=tuple)
    segment_id: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.prev_level)

    def subset(self, mask: np.ndarray) -> "Steps":
        mask = np.asarray(mask)
        return Steps(
            prev_level=self.prev_level[mask], next_level=self.next_level[mask],
            activity=self.activity[mask], day_id=self.day_id[mask],
            session_id=self.session_id[mask], hour=self.hour[mask],
            time_on_task=self.time_on_task[mask], time_since_break=self.time_since_break[mask],
            elapsed=self.elapsed[mask], complexity=self.complexity[mask],
            n_levels=self.n_levels, activity_names=self.activity_names,
            segment_id=self.segment_id[mask] if self.segment_id is not None else None,
        )

    def matrix(self) -> np.ndarray:
        """Continuous conditioning matrix for the regression and neural arms.

        Hour is encoded as sin/cos so 23:00 and 00:00 are adjacent; the two durations are log1p'd
        because their effect is steeply diminishing – the difference between 2 and 20 minutes on task
        matters far more than between 100 and 118.
        """
        angle = 2.0 * np.pi * self.hour / 24.0
        return np.column_stack(
            [
                self.prev_level.astype(float) / max(1, self.n_levels - 1),
                np.sin(angle), np.cos(angle),
                np.log1p(np.maximum(self.time_on_task, 0.0)),
                np.log1p(np.maximum(np.nan_to_num(self.time_since_break, nan=0.0), 0.0)),
                np.isfinite(self.time_since_break).astype(float),
                np.log1p(np.maximum(self.elapsed, 0.0)),
                self.complexity,
            ]
        )


def build_steps(
    sessions: Sequence[Mapping[str, np.ndarray]],
    quantiser: LevelQuantiser,
    max_gap: float = 60.0,
) -> Steps:
    """Turn exported sessions into autoregressive steps, never crossing a recording gap."""
    cols: dict[str, list] = {k: [] for k in (
        "prev", "next", "activity", "day", "session", "hour",
        "on_task", "since_break", "elapsed", "complexity", "segment",
    )}
    names: set[str] = set()

    for data in sessions:
        t = np.asarray(data["t"], dtype=float)
        levels = quantiser.transform(np.asarray(data["load"], dtype=float))
        activity = broadcast_field(data, "activity")
        day = broadcast_field(data, "day_id")
        session = broadcast_field(data, "session_id")
        complexity = (
            np.asarray(data["complexity"], dtype=float) if "complexity" in data
            else np.ones(len(t), dtype=float)
        )
        valid = (
            np.asarray(data["valid"]).astype(bool) if "valid" in data
            else np.ones(len(t), dtype=bool)
        )
        # A disconnected sensor is not a recovery break. Versioned exporters may supply the
        # timestamp of the last explicit break; legacy recordings have unknown break context.
        last_break = np.asarray(data.get("last_break_at", np.full(len(t), np.nan)), dtype=float)
        if last_break.shape != t.shape or np.isinf(last_break).any() or np.any(np.isfinite(last_break) & (last_break > t)):
            raise ValueError("last_break_at must align with timestamps, use NaN for unknown and cannot refer to the future")
        names.update(str(a) for a in activity)

        day_start = t[0]
        for lo, hi in segment_bounds(t, max_gap=max_gap):
            seg = slice(lo, hi)
            t_seg, lv_seg = t[seg], levels[seg]
            act_seg, ok_seg = activity[seg], valid[seg]
            # Time on task restarts whenever the activity label changes inside a segment.
            on_task = np.zeros(len(t_seg))
            for i in range(1, len(t_seg)):
                on_task[i] = 0.0 if act_seg[i] != act_seg[i - 1] else on_task[i - 1] + (t_seg[i] - t_seg[i - 1])
            since_break = t_seg - last_break[seg]

            run = 0
            for i in range(len(t_seg) - 1):
                if not (ok_seg[i] and ok_seg[i + 1]):
                    run += 1
                    continue
                if day[seg][i] != day[seg][i + 1] or session[seg][i] != session[seg][i + 1]:
                    run += 1
                    continue
                cols["segment"].append(f"{session[seg][i]}:{day[seg][i]}:{lo}:{run}")
                # Only context already observed at the forecast origin is available.
                # A task switch/break at the target window must not leak into features.
                hour = (t_seg[i] % 86400.0) / 3600.0
                cols["prev"].append(int(lv_seg[i]))
                cols["next"].append(int(lv_seg[i + 1]))
                cols["activity"].append(str(act_seg[i]))
                cols["day"].append(str(day[seg][i + 1]))
                cols["session"].append(str(session[seg][i + 1]))
                cols["hour"].append(hour)
                cols["on_task"].append(float(on_task[i]))
                cols["since_break"].append(float(since_break[i]))
                cols["elapsed"].append(float(t_seg[i] - day_start))
                cols["complexity"].append(float(complexity[seg][i]))

    return Steps(
        prev_level=np.asarray(cols["prev"], dtype=int),
        next_level=np.asarray(cols["next"], dtype=int),
        activity=np.asarray(cols["activity"], dtype=object),
        day_id=np.asarray(cols["day"], dtype=object),
        session_id=np.asarray(cols["session"], dtype=object),
        hour=np.asarray(cols["hour"], dtype=float),
        time_on_task=np.asarray(cols["on_task"], dtype=float),
        time_since_break=np.asarray(cols["since_break"], dtype=float),
        elapsed=np.asarray(cols["elapsed"], dtype=float),
        complexity=np.asarray(cols["complexity"], dtype=float),
        n_levels=quantiser.n_levels,
        activity_names=tuple(sorted(names)),
        segment_id=np.asarray(cols["segment"], dtype=object),
    )


# -------------------------- models ------------------------------------------
class TransitionModel:
    """Interface every arm implements, classical and quantum alike.

    `context_length` is how many trailing steps the model needs to see to predict the next one.
    One-step models leave it at 1; a recurrent model sets it to its sequence length. `rollout` reads
    it to decide how much history to hand over – without it, a sequence model would be fed a single
    left-padded row at every step of a simulated day and would behave as if it had no memory, which
    silently turns the strongest baseline into the weakest.
    """

    name = "base"
    context_length = 1

    def fit(self, steps: Steps) -> "TransitionModel":
        raise NotImplementedError

    def predict_proba(self, steps: Steps) -> np.ndarray:
        raise NotImplementedError

    def sample_next(self, steps: Steps, rng: np.random.Generator) -> np.ndarray:
        probability = self.predict_proba(steps)
        cumulative = np.cumsum(probability, axis=1)
        draws = rng.random(len(probability))[:, None]
        return np.argmax(cumulative > draws, axis=1)

    def log_prob(self, steps: Steps) -> np.ndarray:
        probability = self.predict_proba(steps)
        return np.log(np.clip(probability[np.arange(len(steps)), steps.next_level], EPS, None))


class MarginalModel(TransitionModel):
    """P(next | activity) – ignores the previous level and all timing.

    This is the model the current day simulator implements, and it is here to be beaten: because it
    conditions on nothing that changes with ordering, it must score exactly zero on the
    order-sensitivity test. If it ever scores above zero, the test is broken.
    """

    name = "marginal"

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.table: dict[str, np.ndarray] = {}
        self.fallback: Optional[np.ndarray] = None
        self.n_levels = DEFAULT_LEVELS

    def fit(self, steps: Steps) -> "MarginalModel":
        self.n_levels = steps.n_levels
        self.fallback = _counts(steps.next_level, self.n_levels, self.alpha)
        for activity in np.unique(steps.activity):
            mask = steps.activity == activity
            self.table[str(activity)] = _counts(steps.next_level[mask], self.n_levels, self.alpha)
        return self

    def predict_proba(self, steps: Steps) -> np.ndarray:
        if self.fallback is None:
            raise RuntimeError("model is not fitted")
        return np.array([self.table.get(str(a), self.fallback) for a in steps.activity])


class MarkovModel(TransitionModel):
    """P(next | prev, activity, timing bucket) with interpolated backoff.

    Backoff order, most specific first:

        (prev, activity, bucket) -> (prev, activity) -> (prev) -> marginal

    Each level is mixed in proportion to the evidence behind it (`n / (n + tau)`), so a cell seen
    three times contributes little and a cell seen three hundred times dominates. Flat Laplace
    smoothing over the full product space would instead drown every real cell in uniform noise, which
    at ~36 windows per activity is the difference between a usable model and none.
    """

    name = "markov"

    def __init__(self, tau: float = 8.0, n_buckets: int = 3, alpha: float = 0.5):
        self.tau = float(tau)
        self.n_buckets = int(n_buckets)
        self.alpha = float(alpha)
        self.n_levels = DEFAULT_LEVELS
        self._edges: Optional[np.ndarray] = None
        self._full: dict[tuple, np.ndarray] = {}
        self._prev_act: dict[tuple, np.ndarray] = {}
        self._prev: dict[int, np.ndarray] = {}
        self._marginal: Optional[np.ndarray] = None

    def _bucket(self, steps: Steps) -> np.ndarray:
        if self._edges is None:
            raise RuntimeError("model is not fitted")
        return np.searchsorted(self._edges, np.log1p(np.maximum(steps.time_on_task, 0.0)), "right")

    def fit(self, steps: Steps) -> "MarkovModel":
        self.n_levels = steps.n_levels
        on_task = np.log1p(np.maximum(steps.time_on_task, 0.0))
        quantiles = np.linspace(0.0, 1.0, self.n_buckets + 1)[1:-1]
        self._edges = np.unique(np.quantile(on_task, quantiles)) if len(on_task) else np.array([0.0])
        buckets = self._bucket(steps)

        self._marginal = _counts(steps.next_level, self.n_levels, self.alpha)
        for level in np.unique(steps.prev_level):
            mask = steps.prev_level == level
            self._prev[int(level)] = _counts(steps.next_level[mask], self.n_levels, self.alpha)
            for activity in np.unique(steps.activity[mask]):
                sub = mask & (steps.activity == activity)
                self._prev_act[(int(level), str(activity))] = _raw_counts(
                    steps.next_level[sub], self.n_levels
                )
                for bucket in np.unique(buckets[sub]):
                    cell = sub & (buckets == bucket)
                    self._full[(int(level), str(activity), int(bucket))] = _raw_counts(
                        steps.next_level[cell], self.n_levels
                    )
        return self

    def predict_proba(self, steps: Steps) -> np.ndarray:
        if self._marginal is None:
            raise RuntimeError("model is not fitted")
        buckets = self._bucket(steps)
        out = np.empty((len(steps), self.n_levels), dtype=float)
        for i in range(len(steps)):
            level, activity, bucket = int(steps.prev_level[i]), str(steps.activity[i]), int(buckets[i])
            distribution = self._prev.get(level, self._marginal)
            for counts in (
                self._prev_act.get((level, activity)),
                self._full.get((level, activity, bucket)),
            ):
                if counts is None:
                    continue
                total = float(counts.sum())
                if total <= 0:
                    continue
                weight = total / (total + self.tau)
                distribution = weight * (counts / total) + (1.0 - weight) * distribution
            out[i] = distribution / max(distribution.sum(), EPS)
        return out


class GaussianARModel(TransitionModel):
    """Ridge regression of the next level on the conditioning matrix, read as a discrete density.

    The point estimate is Gaussian around the prediction with the residual standard deviation, then
    integrated over each level's cell. It is the natural "just fit a line" comparator: if a linear
    autoregression on the same inputs matches the categorical models, nothing more expressive is
    warranted, and that is worth knowing before reaching for a quantum circuit.
    """

    name = "gaussian_ar"

    def __init__(self, ridge: float = 1.0):
        self.ridge = float(ridge)
        self.n_levels = DEFAULT_LEVELS
        self._coef: Optional[np.ndarray] = None
        self._sigma = 1.0
        self._activities: tuple = ()

    def _design(self, steps: Steps) -> np.ndarray:
        onehot = np.zeros((len(steps), len(self._activities)))
        index = {a: i for i, a in enumerate(self._activities)}
        for i, activity in enumerate(steps.activity):
            j = index.get(str(activity))
            if j is not None:
                onehot[i, j] = 1.0
        return np.column_stack([np.ones(len(steps)), steps.matrix(), onehot])

    def fit(self, steps: Steps) -> "GaussianARModel":
        self.n_levels = steps.n_levels
        self._activities = tuple(sorted({str(a) for a in steps.activity}))
        X = self._design(steps)
        y = steps.next_level.astype(float)
        gram = X.T @ X + self.ridge * np.eye(X.shape[1])
        self._coef = np.linalg.solve(gram, X.T @ y)
        residual = y - X @ self._coef
        self._sigma = float(max(residual.std(), 0.5))
        return self

    def predict_proba(self, steps: Steps) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("model is not fitted")
        mean = self._design(steps) @ self._coef
        levels = np.arange(self.n_levels)[None, :]
        # Integrate the Gaussian over each unit-width level cell rather than evaluating its density,
        # so the result is a genuine probability mass function.
        upper = _phi((levels + 0.5 - mean[:, None]) / self._sigma)
        lower = _phi((levels - 0.5 - mean[:, None]) / self._sigma)
        probability = np.clip(upper - lower, EPS, None)
        return probability / probability.sum(axis=1, keepdims=True)


# -------------------------- helpers -----------------------------------------
def _raw_counts(values: np.ndarray, n_levels: int) -> np.ndarray:
    return np.bincount(np.asarray(values, dtype=int), minlength=n_levels)[:n_levels].astype(float)


def _counts(values: np.ndarray, n_levels: int, alpha: float) -> np.ndarray:
    counts = _raw_counts(values, n_levels) + alpha
    return counts / counts.sum()


def _phi(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF via the error function, without pulling in scipy."""
    from math import sqrt
    return 0.5 * (1.0 + np.vectorize(_erf)(z / sqrt(2.0)))


def _erf(x: float) -> float:
    import math
    return math.erf(x)


def make_model(name: str, **kwargs) -> TransitionModel:
    """Build a model by name. The quantum and torch arms are imported lazily.

    `trajectory_born` and `trajectory_gru` both import `TransitionModel` from this module, so
    importing them at the top would be circular; and the torch arm should not make this module
    unimportable on a machine without torch.
    """
    table = {
        "marginal": MarginalModel,
        "markov": MarkovModel,
        "gaussian_ar": GaussianARModel,
    }
    if name in table:
        return table[name](**kwargs)
    if name == "born":
        from trajectory_born import ConditionalBornMachine  # noqa: PLC0415

        return ConditionalBornMachine(**kwargs)
    if name == "gru":
        from trajectory_gru import GRUTransitionModel  # noqa: PLC0415

        return GRUTransitionModel(**kwargs)
    if name in ("persistence", "softmax", "mlp"):
        from trajectory_classical import PersistenceModel, SklearnTransitionModel
        return PersistenceModel(**kwargs) if name == "persistence" else SklearnTransitionModel(kind=name, **kwargs)
    known = sorted([*table, "born", "gru", "persistence", "softmax", "mlp"])
    raise ValueError(f"unknown model {name!r}; known: {', '.join(known)}")


CLASSICAL_MODELS = ("marginal", "markov", "gaussian_ar")
ALL_MODELS = ("persistence", "marginal", "markov", "gaussian_ar", "softmax", "mlp", "gru", "born")


# -------------------------- self-test ---------------------------------------
def synthetic_sessions(
    n_days: int,
    order_dependent: bool,
    seed: int = 0,
    per_activity: int = 240,
    fatigue_per_hour: float = 260.0,
    break_seconds: float = 900.0,
) -> list[dict[str, np.ndarray]]:
    """Days built from three activities, with or without carry-over between them.

    When `order_dependent` is True a task inherits elevated load from whatever preceded it and
    fatigue accumulates with time-of-day, so the ORDER of the same three tasks changes the
    trajectory. When False each task draws from its own fixed distribution regardless of position –
    the behaviour of `paper/run_daily_experiment.py`, and the null the order test must reject.

    `per_activity` defaults to 240 windows = 20 minutes per task, giving a ~1.5 hour day with two
    15-minute breaks. An earlier default of 40 produced 21-minute "days", over which fatigue barely
    accumulated and the order effect was too small for any model to learn – a testbed that flattered
    every arm equally and measured nothing. `fatigue_per_hour` is expressed per hour so the total
    drift stays realistic (~25-30% of baseline over a day) as the day length changes.
    """
    rng = np.random.default_rng(seed)
    activities = ["email", "coding", "reading"]
    base = {"email": 600.0, "coding": 1700.0, "reading": 700.0}
    sessions = []
    for day in range(n_days):
        order = list(rng.permutation(activities))
        t0 = 1_700_000_000.0 + day * 86400.0 + 9 * 3600.0
        t, load, activity = [], [], []
        carry, clock = 0.0, 0.0
        for position, name in enumerate(order):
            for step in range(per_activity):
                level = base[name]
                if order_dependent:
                    level += carry + fatigue_per_hour * clock / 3600.0
                    carry = 0.985 * carry + 0.004 * base[name]   # decays over ~minutes, not windows
                t.append(t0 + clock)
                load.append(max(1.0, level + rng.normal(0, 60)))
                activity.append(name)
                clock += WINDOW_SECONDS
            clock += break_seconds if position < len(order) - 1 else 0.0
        sessions.append(
            {
                "t": np.asarray(t), "load": np.asarray(load),
                "activity": np.asarray(activity, dtype=object),
                "session_id": np.asarray([f"s{day:02d}"]),
                "day_id": np.asarray([f"d{day:02d}"]),
            }
        )
    return sessions


def _selftest() -> int:
    q = LevelQuantiser(8)
    sessions = synthetic_sessions(6, order_dependent=True, seed=1)
    all_load = np.concatenate([s["load"] for s in sessions])
    q.fit(all_load)

    levels = q.transform(all_load)
    assert levels.min() >= 0 and levels.max() < 8, "levels out of range"
    assert len(np.unique(levels)) >= 6, "quantiser collapsed the distribution"
    assert np.isfinite(q.inverse(np.arange(8))).all(), "inverse produced non-finite centres"

    steps = build_steps(sessions, q)
    assert len(steps) > 0, "no steps built"
    assert steps.prev_level.shape == steps.next_level.shape
    # A 300 s inter-task gap exceeds the 60 s threshold, so no step may straddle one.
    assert np.isnan(steps.time_since_break).all(), "legacy gaps must not become recorded breaks"
    assert np.isfinite(steps.matrix()).all(), "unknown break context needs a separate availability flag"
    print(f"  built {len(steps)} steps over {len(set(steps.day_id.tolist()))} days, "
          f"{len(steps.activity_names)} activities")

    for name in CLASSICAL_MODELS:
        model = make_model(name).fit(steps)
        probability = model.predict_proba(steps)
        assert probability.shape == (len(steps), 8), f"{name} bad shape {probability.shape}"
        assert np.allclose(probability.sum(axis=1), 1.0, atol=1e-9), f"{name} rows must sum to 1"
        assert (probability >= 0).all(), f"{name} produced a negative probability"
        mean_ll = float(model.log_prob(steps).mean())
        sampled = model.sample_next(steps, np.random.default_rng(0))
        assert sampled.min() >= 0 and sampled.max() < 8, f"{name} sampled out of range"
        print(f"  {name:<12} in-sample mean log-likelihood {mean_ll:7.4f}")

    # The conditional models must beat the order-blind one on data that genuinely has carry-over.
    marginal = make_model("marginal").fit(steps).log_prob(steps).mean()
    markov = make_model("markov").fit(steps).log_prob(steps).mean()
    assert markov > marginal, f"markov ({markov:.4f}) failed to beat marginal ({marginal:.4f})"
    print("trajectory_model self-test OK")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Conditional forward models of the load index.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    print("import LevelQuantiser/build_steps/make_model, or run --selftest.")
