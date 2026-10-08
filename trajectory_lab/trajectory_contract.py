"""Canonical contract for exported per-window session trajectories.

The forward model in `trajectory_model.py` is trained on the user's own recorded days, which live in
Azure SQL. Rather than let every script issue its own query, the export is frozen into one auditable
NPZ shape – the same discipline `flow-classifier/dataset_contract.py` applies to affect data, and for
the same reason: every model must see the same examples, and a query that quietly changed would
silently change results.

One file per session. A *day* may contain several sessions, so `day_id` is what folds are grouped by:
holding out a session while another session from the same day is in training would leak the day's
fatigue state.

Required, all aligned to the same length:

    t              unix seconds, strictly increasing – the sampling grid is NOT assumed uniform
    load           cognitive load index for the window (Vn)
    activity       activity slug per window, "untagged" where no ActionSpike covers it
    session_id     one distinct value per file
    day_id         local calendar day

Optional, used when present:

    complexity     estimated task complexity C
    quality        signal-quality indicator in 0..1
    hrv            RMSSD
    valid          bool, False for windows failing the artifact gate

Time is kept as absolute unix seconds rather than a step index. A recording gap is missing data,
not evidence of a recovery break. Only explicit recorded break events may describe recovery context.
Legacy `n_breaks` below counts recording gaps only; use the unambiguous `n_recording_gaps` alias.

    python trajectory_contract.py <session.npz>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

REQUIRED = ("t", "load", "activity", "session_id", "day_id")
OPTIONAL_ALIGNED = ("complexity", "quality", "hrv", "valid")
UNTAGGED = "untagged"

# A window is 5 s. Anything beyond this between consecutive windows is a break in recording, not a
# sampling jitter, and the model must treat it as elapsed time rather than as one step.
WINDOW_SECONDS = 5.0
MAX_CONTIGUOUS_GAP = 60.0


class TrajectoryContractError(ValueError):
    """Raised when an exported trajectory cannot support the forward model."""


def _text(values: np.ndarray, field: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise TrajectoryContractError(f"{field} must be one-dimensional")
    out = np.asarray([str(v).strip() for v in values], dtype=object)
    if any(not v for v in out):
        raise TrajectoryContractError(f"{field} contains an empty value")
    return out


def validate_trajectory(data: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Validate one exported session and return an auditable summary."""
    missing = [f for f in REQUIRED if f not in data]
    if missing:
        raise TrajectoryContractError(f"missing required fields: {', '.join(missing)}")

    t = np.asarray(data["t"], dtype=float)
    if t.ndim != 1 or len(t) == 0:
        raise TrajectoryContractError("t must be a non-empty one-dimensional vector")
    if not np.isfinite(t).all():
        raise TrajectoryContractError("t contains NaN or infinity")
    if np.any(np.diff(t) <= 0):
        raise TrajectoryContractError("t must be strictly increasing")

    n = len(t)
    load = np.asarray(data["load"], dtype=float)
    if len(load) != n:
        raise TrajectoryContractError(f"load must contain exactly {n} rows")
    if not np.isfinite(load).all():
        raise TrajectoryContractError("load contains NaN or infinity")
    if (load < 0).any():
        raise TrajectoryContractError("load cannot be negative")

    for field in ("activity", "session_id", "day_id"):
        if len(np.asarray(data[field]).reshape(-1)) not in (1, n):
            raise TrajectoryContractError(f"{field} must contain 1 or {n} values")

    activity = _text(np.broadcast_to(np.asarray(data["activity"]).reshape(-1), (n,)), "activity")
    session = _text(np.broadcast_to(np.asarray(data["session_id"]).reshape(-1), (n,)), "session_id")
    day = _text(np.broadcast_to(np.asarray(data["day_id"]).reshape(-1), (n,)), "day_id")
    if len(np.unique(session)) != 1:
        raise TrajectoryContractError("one file must hold exactly one session")

    for field in OPTIONAL_ALIGNED:
        if field not in data:
            continue
        values = np.asarray(data[field])
        if len(values) != n:
            raise TrajectoryContractError(f"{field} must contain exactly {n} rows")
        if field == "valid":
            continue
        numeric = values.astype(float)
        if not np.isfinite(numeric).all():
            raise TrajectoryContractError(f"{field} contains NaN or infinity")
        if field == "quality" and ((numeric < 0) | (numeric > 1)).any():
            raise TrajectoryContractError("quality must stay in 0..1")

    gaps = np.diff(t)
    return {
        "windows": int(n),
        "session_id": str(session[0]),
        "days": sorted({str(d) for d in day}),
        "activities": sorted({str(a) for a in activity}),
        "tagged_fraction": float(np.mean(activity != UNTAGGED)),
        "duration_seconds": float(t[-1] - t[0]),
        "median_gap_seconds": float(np.median(gaps)) if len(gaps) else 0.0,
        "n_breaks": int(np.sum(gaps > MAX_CONTIGUOUS_GAP)),  # historical alias, NOT recovery events
        "n_recording_gaps": int(np.sum(gaps > MAX_CONTIGUOUS_GAP)),
        "load_median": float(np.median(load)),
        "has_complexity": "complexity" in data,
        "has_quality": "quality" in data,
        "has_hrv": "hrv" in data,
    }


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        data = {name: np.asarray(archive[name]).copy() for name in archive.files}
    validate_trajectory(data)
    return data


def load_many(paths) -> list[dict[str, np.ndarray]]:
    """Load several sessions, refusing duplicates – a repeated session would double-count a day."""
    out, seen = [], set()
    for path in paths:
        data = load_trajectory(Path(path))
        session = str(np.asarray(data["session_id"]).reshape(-1)[0])
        if session in seen:
            raise TrajectoryContractError(f"session {session} appears more than once")
        seen.add(session)
        out.append(data)
    return out


def broadcast_field(data: Mapping[str, np.ndarray], field: str) -> np.ndarray:
    """Return `field` at full length, whether it was stored per-window or once per session."""
    n = len(np.asarray(data["t"], dtype=float))
    values = np.asarray(data[field]).reshape(-1)
    return np.broadcast_to(values, (n,)).copy()


def segment_bounds(t: np.ndarray, max_gap: float = MAX_CONTIGUOUS_GAP) -> list[tuple[int, int]]:
    """Split a session into contiguously-recorded segments at gaps larger than `max_gap`.

    An autoregressive step is only meaningful between windows that are actually adjacent in time.
    Crossing a 40-minute break as if it were one 5 s step would teach the model that load drops
    instantly, which is exactly the recovery dynamic it is supposed to learn properly.
    """
    t = np.asarray(t, dtype=float)
    if len(t) == 0:
        return []
    breaks = np.flatnonzero(np.diff(t) > float(max_gap)) + 1
    edges = [0, *breaks.tolist(), len(t)]
    return [(int(a), int(b)) for a, b in zip(edges, edges[1:]) if b > a]


def _selftest() -> int:
    print("trajectory_contract self-test")

    def base(n: int = 20) -> dict[str, np.ndarray]:
        return {
            "t": 1_700_000_000.0 + np.arange(n) * WINDOW_SECONDS,
            "load": np.linspace(500.0, 900.0, n),
            "activity": np.asarray(["coding"] * n, dtype=object),
            "session_id": np.asarray(["s1"]),
            "day_id": np.asarray(["2026-09-17"]),
        }

    summary = validate_trajectory(base())
    assert summary["windows"] == 20 and summary["session_id"] == "s1"
    assert summary["tagged_fraction"] == 1.0 and summary["n_breaks"] == 0
    print(f"  valid session accepted: {summary['windows']} windows, "
          f"{summary['duration_seconds']:.0f} s")

    # Each of these must be REFUSED. A contract that accepts broken input is worse than none: the
    # failure would surface later as a silently wrong model rather than an error here.
    cases = {
        "missing field": lambda d: d.pop("load"),
        "non-increasing time": lambda d: d.__setitem__("t", np.r_[d["t"][:5], d["t"][4], d["t"][6:]]),
        "NaN in load": lambda d: d["load"].__setitem__(3, np.nan),
        "negative load": lambda d: d["load"].__setitem__(3, -1.0),
        "length mismatch": lambda d: d.__setitem__("load", d["load"][:-1]),
        "two sessions in one file": lambda d: d.__setitem__(
            "session_id", np.asarray(["s1"] * 10 + ["s2"] * 10)
        ),
        "quality out of range": lambda d: d.__setitem__("quality", np.full(20, 1.5)),
        "empty activity": lambda d: d["activity"].__setitem__(2, "  "),
    }
    for label, damage in cases.items():
        data = base()
        damage(data)
        try:
            validate_trajectory(data)
        except TrajectoryContractError:
            continue
        raise AssertionError(f"contract accepted broken input: {label}")
    print(f"  all {len(cases)} malformed inputs refused")

    # Gaps must split into segments; contiguous data must not.
    gapped = base()
    gapped["t"] = np.r_[gapped["t"][:10], gapped["t"][10:] + 3600.0]
    assert len(segment_bounds(gapped["t"])) == 2, "a one-hour gap must start a new segment"
    assert len(segment_bounds(base()["t"])) == 1, "contiguous data must be one segment"
    assert validate_trajectory(gapped)["n_breaks"] == 1
    print("  segment splitting: 1 contiguous, 2 across an hour-long gap")

    # A session-level scalar must broadcast to window length.
    assert len(broadcast_field(base(), "day_id")) == 20
    print("trajectory_contract self-test OK")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an exported session trajectory NPZ")
    parser.add_argument("session", type=Path, nargs="*")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    if not args.session:
        parser.error("pass one or more NPZ files, or --selftest")
    for path in args.session:
        with np.load(path, allow_pickle=True) as archive:
            data = {name: np.asarray(archive[name]).copy() for name in archive.files}
        print(json.dumps(validate_trajectory(data), indent=2))


if __name__ == "__main__":
    main()
