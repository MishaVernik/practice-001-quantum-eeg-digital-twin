from pathlib import Path
import argparse,importlib,csv,json
import numpy as np
from twin_adapter import TwinEstimator

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--solution',action='store_true');ap.add_argument('--run',type=Path,default=Path(__file__).resolve().parent/'reference_run');a=ap.parse_args()
    s=importlib.import_module('solutions' if a.solution else 'tasks')
    p=np.load(a.run/'selected_predictions.npz');r=json.loads((a.run/'report.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(s.nll(p['finite_shots'],p['truth']),r['test_nll']['selected_born_shots'],atol=1e-10)
    np.testing.assert_allclose(s.nll(np.full((2,8),1/8),[0,7]),np.log(8))
    rows=list(csv.DictReader((a.run/'configurations.csv').open(encoding='utf-8')))
    for row in rows:row['gate_shots']=int(row['gate_shots']);row['dev_nll']=float(row['dev_nll'])
    chosen=s.choose_config([row for row in rows if row['visited']=='True'],4608)
    assert chosen['key']==r['search']['selected']['key'] and int(chosen['shots'])==r['search']['selected']['shots']
    try:s.choose_config(rows,0)
    except ValueError:pass
    else:raise AssertionError('An impossible budget must be rejected')
    b,c=p['finite_shots'][0],p['classical'][0]
    np.testing.assert_allclose(s.route_probability(b,c,True),b)
    np.testing.assert_allclose(s.route_probability(b,c,False),c)
    try:s.route_probability(np.ones(8),c,True)
    except ValueError:pass
    else:raise AssertionError('Unnormalised output must be rejected')
    adapter=TwinEstimator()
    try:adapter.estimate(domain='neuroergo_task',source_id='P01',observation={},born=b,classical=c,quantum_available=True)
    except ValueError:pass
    else:raise AssertionError('Synthetic model must not be attached to real EEG')
    offline=adapter.estimate(domain='synthetic_trajectory',source_id='d10',observation={},born=b,classical=c,quantum_available=False)
    assert offline['mode']=='classical_fallback' and offline['forecast'] and not offline['planning_allowed']
    print('PASS: test NLL; development-only budget choice; fallback; probability validation; domain boundary')

if __name__=='__main__':main()
