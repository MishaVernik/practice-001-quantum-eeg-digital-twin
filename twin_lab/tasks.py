"""Three student functions for the thesis-oriented lab."""
import numpy as np

def nll(probabilities,truth):
    """Mean negative natural log of the actual next-level probability; floor 1e-12."""
    raise NotImplementedError('TODO 1')

def choose_config(rows,budget):
    """Return min dev_nll config with gate_shots <= budget; tie: lower cost.
    Raise ValueError when the budget admits none. Never use a test metric.
    """
    raise NotImplementedError('TODO 2')

def route_probability(born,classical,quantum_available):
    """Choose Born when available, otherwise classical; validate eight probabilities.
    Raise ValueError for invalid/missing distributions. Domain check belongs to adapter.
    """
    raise NotImplementedError('TODO 3')
