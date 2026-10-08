"""Implement these three functions, then run check_tasks.py.
The reference implementation is in solutions.py for review after completion.
"""
import numpy as np

def band_integral(frequency, psd, low, high):
    """Integral by rectangular sum, low <= frequency < high; output microvolt²."""
    raise NotImplementedError('TODO 1: select bins, multiply sum by frequency step')

def resting_cli(theta, alpha, rest_theta, rest_alpha):
    """Positive robust z score of log(theta/alpha), using only resting baseline."""
    raise NotImplementedError('TODO 2: median, 1.4826 MAD, max(z, 0)')

def paired_mae(reference, estimate):
    """Mean absolute difference between paired, already accepted windows."""
    raise NotImplementedError('TODO 3: mean(abs(estimate-reference))')
