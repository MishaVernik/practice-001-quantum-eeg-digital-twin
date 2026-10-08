"""Checks computations against independent analytic examples and held-out data."""
import argparse,json,importlib
from pathlib import Path
import numpy as np
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--solution',action='store_true');args=ap.parse_args()
    m=importlib.import_module('solutions' if args.solution else 'tasks')
    p=np.array([[.75,.25],[.1,.9]])
    assert abs(m.negative_log_likelihood(p,np.array([0,1]))+np.log(.675)/2)<1e-12
    assert abs(m.negative_log_likelihood(np.full((3,8),.125),np.array([0,4,7]))-np.log(8))<1e-12
    deterministic=np.eye(8)[[7,0,3]]
    assert np.array_equal(m.sample_levels(deterministic,np.random.default_rng(1)),[7,0,3])
    rng=np.random.default_rng(82);s=m.sample_levels(np.tile([.2,.8],(10000,1)),rng)
    assert abs(np.mean(s==1)-.8)<.02
    np.testing.assert_allclose(m.high_level_probability(np.eye(8)),[0,0,0,0,0,0,1,1])
    root=Path(__file__).resolve().parent
    a=json.loads((root/'evidence/predictions.json').read_text());y=np.array(a['truth'])
    born=m.negative_log_likelihood(np.array(a['probabilities']['born_gray_entangled']),y)
    assert abs(born-1.3408)<.0001
    print('PASS: NLL, conditional sampling, probability of high levels, frozen evidence.')
if __name__=='__main__':main()
