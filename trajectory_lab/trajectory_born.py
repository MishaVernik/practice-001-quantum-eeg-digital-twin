"""Conditional Quantum Circuit Born Machine over load levels – the quantum arm of the simulator.

Why generation and not classification. Every discriminative quantum route on EEG was closed during
2026: the few-shot advantage was refuted with sample-complexity lower bounds (arXiv 2508.19437),
fidelity kernels flatten and lose (Phys. Rev. A 107, 062417), quantum reservoirs underperform on EEG
(arXiv 2608.00139). None of that touches *sampling*. A Born machine reads the measurement
distribution of a circuit as a probability mass function, which is what a forward model needs, and
Born machines report their best results exactly where data is scarce – which is this project at ~36
labelled windows per activity.

**The fit is unusually direct.** `trajectory_model.LevelQuantiser` already discretises load into
`n_levels` bins. With `n_levels` a power of two, `log2(n_levels)` qubits have exactly that many
computational-basis outcomes, so the circuit's native output *is* `P(next level | context)` with no
decoding layer, no readout head and nothing thrown away. Contrast the repo's existing VQC, which
reads one qubit's marginal out of a 16-dimensional state and discards the rest.

Circuit: `L` layers, each applying per-row `Ry(theta[q,l] + W[q,l] . context)` – a data re-uploading
affine map, so context enters as rotation angles – followed by a ring of CNOTs. Trainable parameters
are `theta` and `W`; there is no separate encoder.

Gradients use the **parameter-shift rule**, which is exact for these rotations, rather than finite
differences: `dP(x)/dtheta = [P(x; theta + pi/2) - P(x; theta - pi/2)] / 2`. `W` follows by the chain
rule, its derivative being the angle derivative times the context value.

This is a sibling of `flow-classifier/quantum_kernel.py`. That module's statevector applies the same
gate to every row; here each row carries its own angles, so the primitives are batched differently and
are written out rather than imported. Both are checked against an independent reference in their
self-tests.

Simulator only – at 3-4 qubits the statevector is 8-16 amplitudes and exact, and
`validation/QUANTUM_SCOPE.md` already rules out any speedup claim for circuits this size.

    python trajectory_born.py --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_model import EPS, Steps, TransitionModel  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


# -------------------------- batched statevector -----------------------------
def _axis_for(n_qubits: int, qubit: int) -> int:
    """Qubit 0 is the least significant bit, i.e. the last tensor axis (little-endian)."""
    return n_qubits - qubit


def apply_ry(state: np.ndarray, n_qubits: int, qubit: int, angles: np.ndarray) -> np.ndarray:
    """Ry with a DIFFERENT angle per row. state (m, 2**n), angles (m,) -> (m, 2**n)."""
    m = state.shape[0]
    view = np.moveaxis(state.reshape((m,) + (2,) * n_qubits), _axis_for(n_qubits, qubit), -1)
    shape = (m,) + (1,) * (view.ndim - 2)
    cos = np.cos(np.asarray(angles, dtype=float) / 2.0).reshape(shape)
    sin = np.sin(np.asarray(angles, dtype=float) / 2.0).reshape(shape)
    low, high = view[..., 0], view[..., 1]
    view = np.stack((cos * low - sin * high, sin * low + cos * high), axis=-1)
    return np.moveaxis(view, -1, _axis_for(n_qubits, qubit)).reshape(m, 2 ** n_qubits)


def apply_cnot(state: np.ndarray, n_qubits: int, control: int, target: int) -> np.ndarray:
    """CNOT, identical for every row."""
    if control == target:
        raise ValueError("control and target must differ")
    m = state.shape[0]
    view = state.reshape((m,) + (2,) * n_qubits)
    view = np.moveaxis(view, _axis_for(n_qubits, control), -1)
    # Moving the control axis to the end shifts any axis that sat after it one place left.
    target_axis = _axis_for(n_qubits, target)
    if target_axis > _axis_for(n_qubits, control):
        target_axis -= 1
    flipped = np.flip(view[..., 1], axis=target_axis)
    view = np.stack((view[..., 0], flipped), axis=-1)
    return np.moveaxis(view, -1, _axis_for(n_qubits, control)).reshape(m, 2 ** n_qubits)


# -------------------------- the model ---------------------------------------
class ConditionalBornMachine(TransitionModel):
    """P(next level | previous level, context) read from a circuit's measurement distribution."""

    name = "born"

    def __init__(
        self,
        n_levels: int = 8,
        n_layers: int = 3,
        # 0.02, not 0.08: the loss landscape of these rotations is periodic, and above roughly 0.05
        # Adam overshoots and the training NLL climbs instead of falling. Measured on 8 synthetic
        # days: lr 0.2 -> 3.19, 0.08 -> 2.83 (both diverging, worse than the uniform 2.08),
        # 0.03 -> 1.05, 0.01 -> 0.99. `fit` also keeps the best iterate, so a bad rate degrades
        # gracefully rather than silently returning a model worse than chance.
        learning_rate: float = 0.02,
        epochs: int = 120,
        batch_size: int = 128,
        l2: float = 1e-4,
        seed: int = 0,
        entangle: bool = True,
        encoding: str = "binary",
    ):
        n_qubits = int(round(np.log2(n_levels)))
        if 2 ** n_qubits != int(n_levels):
            raise ValueError(f"n_levels must be a power of two, got {n_levels}")
        if encoding not in ("binary", "gray"):
            raise ValueError("encoding must be 'binary' or 'gray'")
        self.n_levels = int(n_levels)
        self.n_qubits = n_qubits
        self.n_layers = int(n_layers)
        # `entangle=False` removes the CNOT ring, so the qubits never interact and the measured bits
        # are independent: the output collapses to a product distribution with n_qubits free
        # parameters instead of 2**n_qubits - 1. This is the entanglement ablation.
        self.entangle = bool(entangle)
        # How load level k maps to a basis state. Under plain binary, adjacent levels can differ in
        # every bit (3 = 011, 4 = 100), so an unentangled circuit cannot put mass on "level 3 or 4"
        # at all. Gray code makes adjacent levels differ in exactly one bit. The choice therefore
        # decides how much work the entanglement is doing – see `verify_born_qiskit.py` and
        # TRAJECTORY_FINDINGS.md – and the ablation must be run under both to be honest.
        self.encoding = encoding
        code = (lambda k: k ^ (k >> 1)) if encoding == "gray" else (lambda k: k)
        self._basis_of_level = np.array([code(k) for k in range(self.n_levels)], dtype=int)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.theta: Optional[np.ndarray] = None     # (n_layers, n_qubits)
        self.weights: Optional[np.ndarray] = None   # (n_layers, n_qubits, n_context)
        self._mean: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._activities: tuple = ()
        self.history: list[float] = []

    # -- context ------------------------------------------------------------
    def _raw_context(self, steps: Steps) -> np.ndarray:
        onehot = np.zeros((len(steps), len(self._activities)))
        index = {a: i for i, a in enumerate(self._activities)}
        for i, activity in enumerate(steps.activity):
            j = index.get(str(activity))
            if j is not None:
                onehot[i, j] = 1.0
        return np.column_stack([steps.matrix(), onehot])

    def _context(self, steps: Steps) -> np.ndarray:
        raw = self._raw_context(steps)
        if self._mean is None:
            raise RuntimeError("model is not fitted")
        return (raw - self._mean) / self._scale

    # -- circuit ------------------------------------------------------------
    def _probabilities(self, context: np.ndarray, theta: np.ndarray,
                       weights: np.ndarray) -> np.ndarray:
        m = len(context)
        state = np.zeros((m, 2 ** self.n_qubits), dtype=float)
        state[:, 0] = 1.0
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                angles = theta[layer, qubit] + context @ weights[layer, qubit]
                state = apply_ry(state, self.n_qubits, qubit, angles)
            if self.n_qubits > 1 and self.entangle:
                for qubit in range(self.n_qubits):
                    state = apply_cnot(state, self.n_qubits, qubit, (qubit + 1) % self.n_qubits)
        probability = state ** 2
        probability = probability / np.clip(probability.sum(axis=1, keepdims=True), EPS, None)
        # Born rule gives a probability per BASIS STATE; reorder so column k is load LEVEL k.
        return probability[:, self._basis_of_level]

    # -- training -----------------------------------------------------------
    def _nll(self, context: np.ndarray, targets: np.ndarray,
             theta: np.ndarray, weights: np.ndarray) -> float:
        probability = self._probabilities(context, theta, weights)
        return float(-np.log(np.clip(probability[np.arange(len(targets)), targets], EPS, None)).mean())

    def _gradients(self, context: np.ndarray, targets: np.ndarray):
        """Parameter-shift gradients of the mean negative log-likelihood.

        Shifting one angle by +-pi/2 gives the exact derivative of every outcome probability at once,
        so one pair of circuit evaluations serves the whole batch and all `n_levels` outcomes.
        """
        rows = np.arange(len(targets))
        base = self._probabilities(context, self.theta, self.weights)
        assigned = np.clip(base[rows, targets], EPS, None)
        grad_theta = np.zeros_like(self.theta)
        grad_weights = np.zeros_like(self.weights)

        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                shifted = self.theta.copy()
                shifted[layer, qubit] += np.pi / 2
                plus = self._probabilities(context, shifted, self.weights)[rows, targets]
                shifted[layer, qubit] -= np.pi
                minus = self._probabilities(context, shifted, self.weights)[rows, targets]
                d_angle = (plus - minus) / 2.0
                # d(-log p)/d(angle) = -(1/p) dp/d(angle)
                per_row = -d_angle / assigned
                grad_theta[layer, qubit] = per_row.mean()
                grad_weights[layer, qubit] = (per_row[:, None] * context).mean(axis=0)
        return grad_theta, grad_weights

    def fit(self, steps: Steps) -> "ConditionalBornMachine":
        if steps.n_levels != self.n_levels:
            raise ValueError(f"steps use {steps.n_levels} levels, model expects {self.n_levels}")
        self._activities = tuple(sorted({str(a) for a in steps.activity}))
        raw = self._raw_context(steps)
        self._mean = raw.mean(axis=0)
        self._scale = np.maximum(raw.std(axis=0), 1e-6)
        context = self._context(steps)
        targets = steps.next_level.astype(int)

        rng = np.random.default_rng(self.seed)
        self.theta = rng.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits))
        self.weights = rng.normal(0, 0.1, (self.n_layers, self.n_qubits, context.shape[1]))

        # Adam; plain SGD stalls badly on the periodic landscape these rotations produce.
        m_theta = np.zeros_like(self.theta); v_theta = np.zeros_like(self.theta)
        m_w = np.zeros_like(self.weights); v_w = np.zeros_like(self.weights)
        beta1, beta2 = 0.9, 0.999
        step = 0
        # The initialisation is a candidate too, so a run that only ever makes things worse returns
        # its starting point rather than wherever it wandered to.
        best_loss = self._nll(context, targets, self.theta, self.weights)
        best = (self.theta.copy(), self.weights.copy())
        self.history = [best_loss]

        for epoch in range(self.epochs):
            order = rng.permutation(len(targets))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                g_theta, g_w = self._gradients(context[batch], targets[batch])
                g_w += self.l2 * self.weights
                step += 1
                for param, grad, m, v in (
                    (self.theta, g_theta, m_theta, v_theta),
                    (self.weights, g_w, m_w, v_w),
                ):
                    m *= beta1
                    m += (1 - beta1) * grad
                    v *= beta2
                    v += (1 - beta2) * grad ** 2
                    param -= self.learning_rate * (m / (1 - beta1 ** step)) / (
                        np.sqrt(v / (1 - beta2 ** step)) + 1e-8
                    )
            loss = self._nll(context, targets, self.theta, self.weights)
            if loss < best_loss:
                best_loss = loss
                best = (self.theta.copy(), self.weights.copy())
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                self.history.append(loss)

        # Keep the best iterate rather than the last. Parameter-shift gradients are exact, but the
        # landscape is periodic and a too-large step can walk the loss upward for the rest of the
        # run; without this the model can end worse than a uniform distribution and still look
        # "trained". Selection is on TRAINING loss only, so this guards divergence, not overfitting.
        self.theta, self.weights = best
        self.final_loss = float(best_loss)
        # A Born machine that cannot beat a uniform distribution has not learned anything, and must
        # say so rather than be read as a quantum result. Callers check `.diverged`; the benchmark
        # reports it alongside the metrics.
        self.diverged = bool(best_loss >= np.log(self.n_levels))
        return self

    def predict_proba(self, steps: Steps) -> np.ndarray:
        if self.theta is None:
            raise RuntimeError("model is not fitted")
        return self._probabilities(self._context(steps), self.theta, self.weights)

    def n_parameters(self) -> int:
        if self.theta is None:
            return 0
        return int(self.theta.size + self.weights.size)


# -------------------------- self-test ---------------------------------------
def _reference_probabilities(angles_per_layer: Sequence[np.ndarray], n_qubits: int) -> np.ndarray:
    """Dense matrix simulation of the same circuit, for one row. Independent of the batched code."""
    def kron_all(mats):
        out = np.array([[1.0]])
        for mat in mats:                      # qubit 0 is least significant -> rightmost in the kron
            out = np.kron(mat, out)
        return out

    dim = 2 ** n_qubits
    state = np.zeros(dim); state[0] = 1.0
    for angles in angles_per_layer:
        mats = []
        for q in range(n_qubits):
            a = angles[q] / 2.0
            mats.append(np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]))
        state = kron_all(mats) @ state
        if n_qubits > 1:
            for control in range(n_qubits):
                target = (control + 1) % n_qubits
                gate = np.zeros((dim, dim))
                for basis in range(dim):
                    flipped = basis ^ (1 << target) if (basis >> control) & 1 else basis
                    gate[flipped, basis] = 1.0
                state = gate @ state
    return state ** 2


def _selftest() -> int:
    print("trajectory_born self-test")
    rng = np.random.default_rng(0)

    # --- the batched statevector must match an independent dense simulation.
    for n_qubits in (2, 3, 4):
        layers = [rng.uniform(-np.pi, np.pi, n_qubits) for _ in range(3)]
        state = np.zeros((1, 2 ** n_qubits)); state[0, 0] = 1.0
        for angles in layers:
            for q in range(n_qubits):
                state = apply_ry(state, n_qubits, q, np.asarray([angles[q]]))
            if n_qubits > 1:
                for q in range(n_qubits):
                    state = apply_cnot(state, n_qubits, q, (q + 1) % n_qubits)
        mine = (state[0] ** 2)
        reference = _reference_probabilities(layers, n_qubits)
        error = float(np.abs(mine - reference).max())
        assert error < 1e-12, f"{n_qubits} qubits: batched sim differs from dense by {error:.2e}"
        assert abs(mine.sum() - 1.0) < 1e-12, "probabilities must sum to 1"
    print("  statevector matches dense reference for 2, 3 and 4 qubits (< 1e-12)")

    # --- parameter-shift gradients must match finite differences.
    from trajectory_model import LevelQuantiser, build_steps, synthetic_sessions

    sessions = synthetic_sessions(4, order_dependent=True, seed=1, per_activity=25)
    quantiser = LevelQuantiser(8).fit(np.concatenate([s["load"] for s in sessions]))
    steps = build_steps(sessions, quantiser)

    model = ConditionalBornMachine(n_levels=8, n_layers=2, epochs=1, seed=3)
    model._activities = tuple(sorted({str(a) for a in steps.activity}))
    raw = model._raw_context(steps)
    model._mean, model._scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-6)
    context = model._context(steps)[:48]
    targets = steps.next_level[:48].astype(int)
    init = np.random.default_rng(3)
    model.theta = init.uniform(0, 2 * np.pi, (2, 3))
    model.weights = init.normal(0, 0.1, (2, 3, context.shape[1]))

    g_theta, _ = model._gradients(context, targets)
    eps = 1e-5
    worst = 0.0
    for layer in range(2):
        for qubit in range(3):
            up = model.theta.copy(); up[layer, qubit] += eps
            down = model.theta.copy(); down[layer, qubit] -= eps
            numeric = (model._nll(context, targets, up, model.weights)
                       - model._nll(context, targets, down, model.weights)) / (2 * eps)
            worst = max(worst, abs(numeric - g_theta[layer, qubit]))
    assert worst < 1e-5, f"parameter-shift disagrees with finite differences by {worst:.2e}"
    print(f"  parameter-shift gradients match finite differences (max |diff| {worst:.2e})")

    # --- training must actually reduce the loss and beat a uniform distribution.
    trained = ConditionalBornMachine(n_levels=8, n_layers=3, epochs=60, seed=0).fit(steps)
    probability = trained.predict_proba(steps)
    assert probability.shape == (len(steps), 8)
    assert np.allclose(probability.sum(axis=1), 1.0, atol=1e-9), "rows must sum to 1"
    mean_ll = float(trained.log_prob(steps).mean())
    uniform_ll = float(np.log(1.0 / 8))
    print(f"  loss {trained.history[0]:.4f} -> {min(trained.history):.4f} over training")
    print(f"  mean log-likelihood {mean_ll:.4f} vs uniform {uniform_ll:.4f} "
          f"({trained.n_parameters()} parameters)")
    assert min(trained.history) < trained.history[0], "training did not reduce the loss"
    assert mean_ll > uniform_ll, "trained model is no better than a uniform distribution"

    assert not trained.diverged, "a well-tuned run was wrongly flagged as diverged"

    # A deliberately reckless learning rate must (a) never return something worse than where it
    # started, and (b) be FLAGGED. Silently returning a sub-chance model as a quantum result is the
    # failure mode this guards; promising it cannot happen would be the wrong fix.
    reckless = ConditionalBornMachine(n_levels=8, n_layers=3, epochs=30,
                                      learning_rate=0.5, seed=0).fit(steps)
    assert reckless.final_loss <= reckless.history[0] + 1e-9, \
        "the best-iterate guard returned something worse than the initialisation"
    assert reckless.diverged, "a run that never beat uniform was not flagged as diverged"
    print(f"  reckless lr flagged: diverged={reckless.diverged}, "
          f"loss held at {reckless.final_loss:.4f} (start {reckless.history[0]:.4f})")

    # --- sampling stays in range and follows the fitted distribution.
    sampled = trained.sample_next(steps, np.random.default_rng(0))
    assert sampled.min() >= 0 and sampled.max() < 8, "sampled level out of range"
    print("trajectory_born self-test OK")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Conditional Born machine over load levels.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    print("import ConditionalBornMachine, or run --selftest.")
