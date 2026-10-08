"""Classical predictor baselines for the state-trajectory experiment, not emotion recognition."""
import numpy as np
from trajectory_model import Steps, TransitionModel


class PersistenceModel(TransitionModel):
    name = "persistence"

    def __init__(self, smoothing=0.01):
        self.smoothing = smoothing

    def fit(self, steps):
        self.n_levels = steps.n_levels
        return self

    def predict_proba(self, steps):
        p = np.full((len(steps), self.n_levels), self.smoothing / self.n_levels)
        p[np.arange(len(steps)), steps.prev_level] += 1 - self.smoothing
        return p

    def n_parameters(self):
        return 0  # genuinely no fitted coefficients


class SklearnTransitionModel(TransitionModel):
    """Same contemporaneous context as ConditionalBornMachine, including unknownness flags.

    The sequence GRU is also retained as a stronger, history-aware comparator. This class
    does not pretend the feedforward/QCBM arms have the GRU's longer memory.
    """
    def __init__(self, kind="softmax", C=0.1, hidden=8, seed=0):
        self.kind, self.C, self.hidden, self.seed = kind, C, hidden, seed
        self.name = kind

    def features(self, steps):
        names = {a: i for i, a in enumerate(self.activities)}
        onehot = np.zeros((len(steps), len(names)))
        for i, a in enumerate(steps.activity):
            if a in names:
                onehot[i, names[a]] = 1
        return np.column_stack([steps.matrix(), onehot])

    def fit(self, steps):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        self.n_levels = steps.n_levels
        self.activities = tuple(sorted(set(steps.activity)))
        self.scaler = StandardScaler().fit(self.features(steps))
        X = self.scaler.transform(self.features(steps))
        self.model = (LogisticRegression(C=self.C, max_iter=1000, random_state=self.seed)
                      if self.kind == "softmax" else MLPClassifier(hidden_layer_sizes=(self.hidden,),
                          activation="tanh", solver="lbfgs", alpha=1 / self.C, max_iter=500, random_state=self.seed))
        self.model.fit(X, steps.next_level)
        return self

    def predict_proba(self, steps):
        raw = self.model.predict_proba(self.scaler.transform(self.features(steps)))
        p = np.full((len(steps), self.n_levels), 1e-12)
        p[:, self.model.classes_.astype(int)] = raw
        return p / p.sum(axis=1, keepdims=True)

    def n_parameters(self):
        if self.kind == "softmax":
            return self.model.coef_.size + self.model.intercept_.size
        return sum(x.size for x in self.model.coefs_ + self.model.intercepts_)


def stored_numeric_scalars(model):
    """Storage proxy, not trainable-parameter count; includes fitted tables/scalers/configs.

    Count arrays once, not their referenced aliases. Byte size is reported separately by
    actual serialized artifact size. Closed-form/tabular models are not reported as free.
    """
    seen = set()
    def count(x):
        if isinstance(x, (float, int, np.number)):
            return 1
        if id(x) in seen:
            return 0
        seen.add(id(x))
        if isinstance(x, np.ndarray):
            return int(x.size) if np.issubdtype(x.dtype, np.number) else 0
        if isinstance(x, dict):
            return sum(count(v) for v in x.values())
        if isinstance(x, (tuple, list)):
            return sum(count(v) for v in x)
        if hasattr(x, "__dict__"):
            return count(vars(x))
        return 0
    return count(model)
