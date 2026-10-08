import numpy as np
def negative_log_likelihood(probabilities,truth):
    return float(-np.log(np.maximum(probabilities[np.arange(len(truth)),truth],1e-12)).mean())
def sample_levels(probabilities,rng):
    return np.array([rng.choice(probabilities.shape[1],p=p) for p in probabilities])
def high_level_probability(probabilities,first_high=6):
    return probabilities[:,first_high:].sum(axis=1)
