import numpy as np

def nll(probabilities,truth):
    p=np.asarray(probabilities);y=np.asarray(truth,dtype=int)
    return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())

def choose_config(rows,budget):
    candidates=[r for r in rows if r['gate_shots']<=budget]
    if not candidates:raise ValueError('No configuration fits the budget')
    return min(candidates,key=lambda r:(r['dev_nll'],r['gate_shots']))

def route_probability(born,classical,quantum_available):
    value=born if quantum_available and born is not None else classical
    if value is None:raise ValueError('No available model')
    p=np.asarray(value,dtype=float)
    if p.shape!=(8,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):raise ValueError('Invalid probabilities')
    return p
