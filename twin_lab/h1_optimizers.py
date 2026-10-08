"""Optimisers for Н1 (PREREG_H1.md §5), all over genes in [0, 2π] and all MINIMISING.

* `NSGA2` – Deb et al. 2002: fast non-dominated sort, crowding distance, binary tournament,
  SBX (η = 15, p = 0.9) and polynomial mutation (η = 20, p = 1/n). The Н1 method.
* `GA`, `PSO`, `SA`, `ACOR` – line-for-line ports of
  `CmeSim.Api/Services/Optimizers/PopulationOptimizers.cs`, same constants. The C# code MAXIMISES
  fitness; here each receives −J, so the logic is unchanged.
* `hypervolume_2d`, `nondominated` – the Е3 endpoint.

Every optimiser speaks ask/tell: `ask()` returns the next population, `tell(pop, objectives)` reports it.

    python h1_optimizers.py --selftest     # ZDT1 for NSGA-II, a sphere for the four ports, HV checks
"""
from __future__ import annotations

import sys
from typing import List, Optional

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

LO, HI = 0.0, 2.0 * np.pi


# ── Pareto utilities ───────────────────────────────────────────────────────

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated(objs: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows of an (N, M) objective matrix (minimisation)."""
    objs = np.asarray(objs, dtype=float)
    keep = []
    for i in range(len(objs)):
        if not any(dominates(objs[j], objs[i]) for j in range(len(objs)) if j != i):
            keep.append(i)
    # identical points: keep the first copy only
    seen, out = set(), []
    for i in keep:
        key = tuple(np.round(objs[i], 12))
        if key not in seen:
            seen.add(key); out.append(i)
    return np.array(out, dtype=int)


def hypervolume_2d(objs: np.ndarray, ref=(1.0, 1.0)) -> float:
    """Area dominated by a 2-objective set and bounded by `ref` (minimisation)."""
    pts = np.asarray([p for p in np.asarray(objs, dtype=float) if p[0] < ref[0] and p[1] < ref[1]])
    if len(pts) == 0:
        return 0.0
    pts = pts[nondominated(pts)]
    pts = pts[np.argsort(pts[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        hv += (ref[0] - x) * (prev_y - y)
        prev_y = y
    return float(hv)


def fast_nondominated_sort(objs: np.ndarray) -> List[List[int]]:
    n = len(objs)
    dominated_by = [[] for _ in range(n)]
    count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                dominated_by[p].append(q)
            elif dominates(objs[q], objs[p]):
                count[p] += 1
        if count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                count[q] -= 1
                if count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(objs: np.ndarray) -> np.ndarray:
    n, m = objs.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        order = np.argsort(objs[:, k])
        span = objs[order[-1], k] - objs[order[0], k]
        d[order[0]] = d[order[-1]] = np.inf
        if span <= 0:
            continue
        d[order[1:-1]] += (objs[order[2:], k] - objs[order[:-2], k]) / span
    return d


# ── NSGA-II ─────────────────────────────────────────────────────────────────

class NSGA2:
    name = "nsga2"

    def __init__(self, dim: int, pop_size: int, rng: np.random.Generator,
                 eta_c: float = 15.0, p_c: float = 0.9, eta_m: float = 20.0,
                 lo: float = LO, hi: float = HI):
        self.dim, self.n, self.rng = dim, pop_size, rng
        self.eta_c, self.p_c, self.eta_m, self.p_m = eta_c, p_c, eta_m, 1.0 / dim
        self.lo, self.hi = lo, hi
        self.pop: Optional[np.ndarray] = None
        self.objs: Optional[np.ndarray] = None
        self.rank = self.crowd = None

    def ask(self) -> np.ndarray:
        if self.pop is None:
            return self.rng.uniform(self.lo, self.hi, (self.n, self.dim))
        return self._offspring()

    def tell(self, pop: np.ndarray, objs: np.ndarray) -> None:
        if self.pop is not None:
            pop = np.vstack([self.pop, pop]); objs = np.vstack([self.objs, objs])
        fronts = fast_nondominated_sort(objs)
        chosen, rank, crowd = [], [], []
        for r, front in enumerate(fronts):
            cd = crowding(objs[front])
            if len(chosen) + len(front) <= self.n:
                chosen += front; rank += [r] * len(front); crowd += list(cd)
            else:
                order = np.argsort(-cd)[: self.n - len(chosen)]
                chosen += [front[i] for i in order]; rank += [r] * len(order); crowd += list(cd[order])
                break
        self.pop, self.objs = pop[chosen], objs[chosen]
        self.rank, self.crowd = np.array(rank), np.array(crowd)

    def _tournament(self) -> int:
        a, b = self.rng.integers(0, self.n, 2)
        if self.rank[a] != self.rank[b]:
            return a if self.rank[a] < self.rank[b] else b
        return a if self.crowd[a] >= self.crowd[b] else b

    def _sbx(self, x1: np.ndarray, x2: np.ndarray):
        c1, c2 = x1.copy(), x2.copy()
        if self.rng.random() > self.p_c:
            return c1, c2
        for i in range(self.dim):
            if self.rng.random() > 0.5 or abs(x1[i] - x2[i]) < 1e-14:
                continue
            y1, y2 = min(x1[i], x2[i]), max(x1[i], x2[i])
            u = self.rng.random()
            for sign, bound in ((-1, self.lo), (1, self.hi)):
                beta = 1.0 + 2.0 * ((y1 - self.lo) if sign < 0 else (self.hi - y2)) / (y2 - y1)
                alpha = 2.0 - beta ** -(self.eta_c + 1.0)
                bq = ((u * alpha) ** (1.0 / (self.eta_c + 1.0)) if u <= 1.0 / alpha
                      else (1.0 / (2.0 - u * alpha)) ** (1.0 / (self.eta_c + 1.0)))
                child = 0.5 * ((y1 + y2) + sign * bq * (y2 - y1))
                if sign < 0:
                    lo_child = child
                else:
                    hi_child = child
            lo_child, hi_child = np.clip([lo_child, hi_child], self.lo, self.hi)
            if self.rng.random() <= 0.5:
                lo_child, hi_child = hi_child, lo_child
            c1[i], c2[i] = lo_child, hi_child
        return c1, c2

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        span = self.hi - self.lo
        for i in range(self.dim):
            if self.rng.random() > self.p_m:
                continue
            d1, d2 = (x[i] - self.lo) / span, (self.hi - x[i]) / span
            u, p = self.rng.random(), 1.0 / (self.eta_m + 1.0)
            if u < 0.5:
                dq = (2 * u + (1 - 2 * u) * (1 - d1) ** (self.eta_m + 1)) ** p - 1
            else:
                dq = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (self.eta_m + 1)) ** p
            x[i] = np.clip(x[i] + dq * span, self.lo, self.hi)
        return x

    def _offspring(self) -> np.ndarray:
        kids = []
        while len(kids) < self.n:
            a, b = self.pop[self._tournament()], self.pop[self._tournament()]
            for c in self._sbx(a, b):
                kids.append(self._mutate(c))
        return np.array(kids[: self.n])


# ── Ports of the C# single-objective optimisers (fitness MAXIMISED = −J) ─────

class _Port:
    """ask/tell wrapper: the C# interface is NextGeneration(population, fitness)."""

    def __init__(self, dim: int, size: int, rng: np.random.Generator):
        self.dim, self.size, self.rng = dim, size, rng
        self._next: Optional[np.ndarray] = None

    def ask(self) -> np.ndarray:
        if self._next is None:
            self._next = self.rng.uniform(LO, HI, (self.size, self.dim))
        return self._next

    def tell(self, pop: np.ndarray, objs: np.ndarray) -> None:
        fitness = -np.asarray(objs, dtype=float).reshape(len(pop), -1)[:, 0]
        self._next = np.clip(self.next_generation(np.asarray(pop), fitness), LO, HI)

    def next_generation(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GA(_Port):
    """Elitist GA: top half survives, uniform crossover, per-gene Gaussian mutation."""
    name = "genetic"
    MUTATION_RATE, MUTATION_SIGMA = 0.10, 0.25

    def next_generation(self, pop, fitness):
        ranked = pop[np.argsort(-fitness, kind="stable")]
        survivors = ranked[: max(1, self.size // 2)]
        nxt = [s.copy() for s in survivors]
        while len(nxt) < self.size:
            a = survivors[self.rng.integers(len(survivors))]
            b = survivors[self.rng.integers(len(survivors))]
            child = np.where(self.rng.random(self.dim) < 0.5, a, b)
            mut = self.rng.random(self.dim) < self.MUTATION_RATE
            child = np.where(mut, np.clip(child + self.rng.normal(0, self.MUTATION_SIGMA, self.dim), LO, HI), child)
            nxt.append(child)
        return np.array(nxt)


class PSO(_Port):
    """Particle swarm, Clerc constriction pair (0.729, 1.494, 1.494), |v| ≤ 0.2 · range."""
    name = "pso"
    INERTIA, COGNITIVE, SOCIAL, VMAX_FRAC = 0.729, 1.494, 1.494, 0.2

    def __init__(self, dim, size, rng):
        super().__init__(dim, size, rng)
        self.vmax = self.VMAX_FRAC * (HI - LO)
        self.vel = self.pbest = self.pbest_f = self.gbest = None
        self.gbest_f = -np.inf

    def next_generation(self, pop, fitness):
        if self.vel is None:
            self.vel = self.rng.uniform(-1, 1, (self.size, self.dim)) * self.vmax
            self.pbest = pop.copy()
            self.pbest_f = np.full(self.size, -np.inf)
        for i in range(self.size):
            if fitness[i] > self.pbest_f[i]:
                self.pbest_f[i], self.pbest[i] = fitness[i], pop[i].copy()
            if fitness[i] > self.gbest_f:
                self.gbest_f, self.gbest = fitness[i], pop[i].copy()
        r1 = self.rng.random((self.size, self.dim)); r2 = self.rng.random((self.size, self.dim))
        v = (self.INERTIA * self.vel + self.COGNITIVE * r1 * (self.pbest - pop)
             + self.SOCIAL * r2 * (self.gbest - pop))
        self.vel = np.clip(v, -self.vmax, self.vmax)
        return np.clip(pop + self.vel, LO, HI)


class SA(_Port):
    """Simulated annealing, one chain per slot; T0 = 0.5, cooling 0.9, step σ = 0.6 · T/T0."""
    name = "simulated_annealing"
    T0, COOLING, T_MIN, STEP_SIGMA = 0.5, 0.9, 1e-4, 0.6

    def __init__(self, dim, size, rng):
        super().__init__(dim, size, rng)
        self.cur = self.cur_f = None
        self.T = self.T0

    def next_generation(self, pop, fitness):
        if self.cur is None:
            self.cur, self.cur_f = pop.copy(), fitness.copy()
        else:
            for i in range(self.size):
                delta = fitness[i] - self.cur_f[i]
                if delta >= 0 or self.rng.random() < np.exp(delta / max(self.T, self.T_MIN)):
                    self.cur[i], self.cur_f[i] = pop[i].copy(), fitness[i]
        self.T = max(self.T_MIN, self.T * self.COOLING)
        step = self.STEP_SIGMA * self.T / self.T0
        return np.clip(self.cur + self.rng.normal(0, step, self.cur.shape), LO, HI)


class ACOR(_Port):
    """ACO_R (Socha & Dorigo 2008): archive of the best, Gaussian kernel around a rank-weighted guide."""
    name = "aco"
    LOCALITY_Q, EVAPORATION_XI = 0.1, 0.85

    def __init__(self, dim, size, rng):
        super().__init__(dim, size, rng)
        self.archive_size = max(2, size)
        self.arch = np.empty((0, dim)); self.arch_f = np.empty(0)

    def next_generation(self, pop, fitness):
        arch = np.vstack([self.arch, pop]); f = np.concatenate([self.arch_f, fitness])
        order = np.argsort(-f, kind="stable")[: self.archive_size]
        self.arch, self.arch_f = arch[order], f[order]
        k = len(self.arch)
        ranks = np.arange(k)
        w = np.exp(-ranks ** 2 / (2 * (self.LOCALITY_Q * k) ** 2)) / (self.LOCALITY_Q * k * np.sqrt(2 * np.pi))
        w = w / w.sum()
        nxt = np.empty((self.size, self.dim))
        for ant in range(self.size):
            g = self.rng.choice(k, p=w)
            spread = np.abs(self.arch - self.arch[g]).sum(axis=0)
            sigma = self.EVAPORATION_XI * spread / max(1, k - 1)
            sigma = np.where(sigma <= 1e-9, 1e-3, sigma)
            nxt[ant] = self.arch[g] + self.rng.normal(0, 1, self.dim) * sigma
        return np.clip(nxt, LO, HI)


PORTS = {"genetic": GA, "pso": PSO, "simulated_annealing": SA, "aco": ACOR}


# ── self-test ───────────────────────────────────────────────────────────────

def _zdt1(x: np.ndarray) -> np.ndarray:
    u = x / HI                                              # map genes to [0, 1]
    f1 = u[:, 0]
    g = 1 + 9 * u[:, 1:].mean(axis=1)
    return np.column_stack([f1, g * (1 - np.sqrt(f1 / g))])


def _selftest() -> int:
    assert abs(hypervolume_2d(np.array([[0.5, 0.5]])) - 0.25) < 1e-12
    assert abs(hypervolume_2d(np.array([[0.2, 0.6], [0.6, 0.2], [0.7, 0.7]])) - (0.8 * 0.4 + 0.4 * 0.4)) < 1e-12
    assert list(nondominated(np.array([[1, 2], [2, 1], [2, 2], [1, 2]]))) == [0, 1]

    rng = np.random.default_rng(1)
    opt = NSGA2(dim=10, pop_size=50, rng=rng)
    for _ in range(100):
        pop = opt.ask(); opt.tell(pop, _zdt1(pop))
    hv = hypervolume_2d(opt.objs, ref=(1.1, 1.1))
    hv_true = 1.1 * 1.1 - 1.0 / 3.0 - 0.1 * 0.0      # area above f2 = 1 − √f1 on [0,1], bounded at 1.1
    print(f"NSGA-II ZDT1 hypervolume {hv:.3f} (true front {hv_true:.3f})")
    assert hv > 0.95 * hv_true, "NSGA-II failed to approach the ZDT1 front"

    def sphere(pop):
        return (((pop - np.pi) / np.pi) ** 2).sum(axis=1, keepdims=True)

    for name, cls in PORTS.items():
        o = cls(dim=8, size=20, rng=np.random.default_rng(2))
        start = best = None
        for _ in range(60):
            pop = o.ask(); f = sphere(pop)
            start = f.min() if start is None else start
            best = f.min() if best is None else min(best, f.min())
            o.tell(pop, f)
        print(f"{name:>20}: sphere {start:.3f} → {best:.4f}")
        assert best < 0.5 * start, f"{name} did not improve"
    print("h1_optimizers selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
