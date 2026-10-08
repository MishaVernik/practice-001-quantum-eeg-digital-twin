"""Reproduce the frozen PhD next-state experiment and generate measured figures."""
from pathlib import Path
import argparse, csv, hashlib, json, platform
import numpy as np
from trajectory_model import LevelQuantiser, build_steps, make_model, synthetic_sessions

ROOT = Path(__file__).resolve().parent
SEED = 20261005

def dataset():
    sessions = synthetic_sessions(15, order_dependent=True, seed=SEED, per_activity=60)
    q = LevelQuantiser(8).fit(np.concatenate([s['load'] for s in sessions[:8]]))
    return q, [build_steps(part, q) for part in (sessions[:8], sessions[8:10], sessions[10:])]

def nll(p, y):
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())

def score(p, y):
    return {'nll': nll(p,y), 'accuracy': float((p.argmax(1)==y).mean()),
            'brier': float(((p-np.eye(8)[y])**2).sum(1).mean())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['replay','train'], default='replay')
    ap.add_argument('--out', type=Path, default=ROOT/'run')
    args=ap.parse_args()
    if args.out.exists():
        raise SystemExit('Choose a new --out directory; existing results are preserved.')
    args.out.mkdir(parents=True)
    q,(train,dev,test)=dataset()
    archived=json.loads((ROOT/'evidence/predictions.json').read_text(encoding='utf-8'))
    assert archived['truth']==test.next_level.tolist()
    assert archived['day']==test.day_id.tolist()
    names=['marginal','markov','gaussian_ar','softmax','born_gray_product','born_gray_entangled']
    predictions={'uniform': np.full((len(test),8),1/8)}
    history={}; models={}; reproduction={}
    for name in names:
        if args.mode=='train':
            if name.startswith('born'):
                model=make_model('born',n_levels=8,epochs=8,seed=SEED,encoding='gray',entangle=name.endswith('entangled')).fit(train)
                history[name]=model.history
            elif name=='softmax':
                candidates=[make_model(name,C=c,seed=SEED).fit(train) for c in [.01,.1,1]]
                model=min(candidates,key=lambda m:nll(m.predict_proba(dev),dev.next_level))
            else: model=make_model(name).fit(train)
            p=model.predict_proba(test); models[name]=model
            reproduction[name]=float(np.max(np.abs(p-np.asarray(archived['probabilities'][name]))))
        else: p=np.asarray(archived['probabilities'][name])
        assert np.isfinite(p).all() and (p>=0).all()
        np.testing.assert_allclose(p.sum(1),1,atol=1e-10)
        predictions[name]=p
        print(name, f'NLL={nll(p,test.next_level):.6f}',flush=True)
    result={name:score(p,test.next_level) for name,p in predictions.items()}
    # Every reported point comes from held-out probabilities, never a cosmetic trend.
    with (args.out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['model','nll','accuracy','brier']);w.writeheader()
        for name,s in result.items():w.writerow({'model':name,**s})
    (args.out/'report.json').write_text(json.dumps({'mode':args.mode,'seed':SEED,'synthetic':True,
        'split':{'train':len(train),'selection':len(dev),'test':len(test)},'metrics':result,
        'training_history':history,'max_abs_difference_from_archive':reproduction,
        'python':platform.python_version(),'numpy':np.__version__,
        'evidence_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'evidence').glob('*.json')},
        'scope':'One-step prediction on generated days; no human EEG validation or quantum advantage.'},indent=2),encoding='utf-8')
    if args.mode=='train':
        # Export non-executable numeric state and a matching context for integration.
        model=models['born_gray_entangled']
        np.savez_compressed(args.out/'born_state.npz',theta=model.theta,weights=model.weights,
            context=model._context(test),previous=test.prev_level,truth=test.next_level,
            probabilities=predictions['born_gray_entangled'],quantiser_edges=q.edges)
    from figures import build_figures
    build_figures(args.out,predictions,test,result)
    assert result['born_gray_entangled']['nll'] < result['marginal']['nll'] < np.log(8)
    assert result['softmax']['nll'] < result['born_gray_entangled']['nll']
    if args.mode=='train':
        assert max(reproduction.values())<1e-6, reproduction
    print('VERIFIED: learned Born distribution beats uniform and task-only; softmax is stronger.')

if __name__=='__main__':main()
