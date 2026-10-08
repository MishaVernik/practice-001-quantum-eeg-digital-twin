"""Thesis teaching integration: conditional Born, NSGA-II and a typed twin adapter.
Real EEG observation and synthetic transition model remain separate domains.
"""
from pathlib import Path
import sys,json,csv,argparse,time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'trajectory_lab'))
from run_lab import dataset,nll,SEED
from trajectory_model import make_model
from h1_optimizers import NSGA2,nondominated
from twin_adapter import TwinEstimator

SHOTS=[64,128,256,512,1024]
BUDGET=4608

def counts_probability(p,shots,seed):
    rng=np.random.default_rng(seed)
    counts=np.array([rng.multinomial(shots,row/row.sum()) for row in p])
    return (counts+.5)/(shots+4.)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise SystemExit('Choose a new --out directory')
    a.out.mkdir(parents=True);out=a.out
    q,(train,dev,test)=dataset();models={};dev_p={};history={};fit_seconds={}
    for layers in [1,2,3]:
        for entangle in [False,True]:
            key=f'L{layers}_CNOT{int(entangle)}';start=time.perf_counter()
            m=make_model('born',n_levels=8,n_layers=layers,encoding='gray',entangle=entangle,epochs=8,seed=SEED).fit(train)
            models[key]=m;dev_p[key]=m.predict_proba(dev)
            history[key]={'epochs':[0]+[e+1 for e in range(m.epochs) if e%10==0 or e==m.epochs-1], 'nll':m.history}
            fit_seconds[key]=time.perf_counter()-start
            print('TRAINED',key,flush=True)
    # All objective values come from trained distributions and DEVELOPMENT targets.
    rows=[]
    for layers in [1,2,3]:
        for entangle in [False,True]:
            key=f'L{layers}_CNOT{int(entangle)}'
            for shots in SHOTS:
                losses=[nll(counts_probability(dev_p[key],shots,20261007+r),dev.next_level) for r in range(8)]
                rows.append({'key':key,'layers':layers,'cnot':int(entangle),'shots':shots,
                    'dev_nll':float(np.mean(losses)),'dev_mc_sd':float(np.std(losses,ddof=1)),
                    'gate_shots':shots*3*layers*(1+int(entangle))})
    def decode(x):
        return min(int(x[0]*3),2)*10+min(int(x[1]*2),1)*5+min(int(x[2]*5),4)
    opt=NSGA2(dim=3,pop_size=12,rng=np.random.default_rng(20261007),lo=0.,hi=1.)
    seen=set();search=[]
    for generation in range(10):
        pop=opt.ask();ids=[decode(x) for x in pop];seen.update(ids)
        objectives=np.array([[rows[i]['dev_nll'],rows[i]['gate_shots']/18432] for i in ids])
        opt.tell(pop,objectives)
        feasible=[rows[i] for i in seen if rows[i]['gate_shots']<=BUDGET]
        search.append({'generation':generation+1,'distinct_configs':len(seen),'best_dev_nll_under_budget':min(r['dev_nll'] for r in feasible) if feasible else None})
    selected=min((i for i in seen if rows[i]['gate_shots']<=BUDGET),key=lambda i:(rows[i]['dev_nll'],rows[i]['gate_shots']))
    exhaustive=min((i for i,r in enumerate(rows) if r['gate_shots']<=BUDGET),key=lambda i:(rows[i]['dev_nll'],rows[i]['gate_shots']))
    # Selection is complete before any test probabilities are calculated.
    choice=rows[selected];m=models[choice['key']];exact=m.predict_proba(test)
    shot_p=counts_probability(exact,choice['shots'],20261008)
    candidates=[make_model('softmax',C=c,seed=SEED).fit(train) for c in [.01,.1,1]]
    softmax=min(candidates,key=lambda k:nll(k.predict_proba(dev),dev.next_level));classic=softmax.predict_proba(test)
    marginal=make_model('marginal').fit(train).predict_proba(test)
    metrics={'uniform':float(np.log(8)),'task_only':nll(marginal,test.next_level),'selected_born_exact':nll(exact,test.next_level),'selected_born_shots':nll(shot_p,test.next_level),'softmax':nll(classic,test.next_level)}
    frontier=nondominated(np.array([[r['dev_nll'],r['gate_shots']] for r in rows])).tolist()
    with (out/'configurations.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])+['visited','pareto']);w.writeheader()
        for i,r in enumerate(rows):w.writerow({**r,'visited':i in seen,'pareto':i in frontier})
    eeg=list(csv.DictReader((ROOT.parent/'data_lab/reference_run/eeg_predictions.csv').open(encoding='utf-8')))[0]
    adapter=TwinEstimator()
    scenarios={
        'real_eeg_observation':adapter.estimate(domain='neuroergo_task',source_id=eeg['subject'],observation={'p_difficult':float(eeg['probability']),'target':'MATB-II difficulty'},born=None,classical=None,quantum_available=True),
        'synthetic_born':adapter.estimate(domain='synthetic_trajectory',source_id=str(test.day_id[0]),observation={'previous_level':int(test.prev_level[0])},born=shot_p[0],classical=classic[0],quantum_available=True),
        'synthetic_fallback':adapter.estimate(domain='synthetic_trajectory',source_id=str(test.day_id[0]),observation={'previous_level':int(test.prev_level[0])},born=shot_p[0],classical=classic[0],quantum_available=False)}
    result={'date':'2026-10-07','source':'New teaching experiment using PhD conditional Born and NSGA2 implementations',
        'split':{'train':len(train),'dev':len(dev),'test':len(test)},'training_seed':SEED,'epochs':8,
        'search':{'method':'NSGA-II','population':12,'generations':10,'proposal_count':120,'distinct_configs':len(seen),'full_grid':30,
                  'selection_budget_gate_shots':BUDGET,'selected':choice,'exhaustive_selected':rows[exhaustive],
                  'gap_to_exhaustive_dev_nll':choice['dev_nll']-rows[exhaustive]['dev_nll'],'history':search},
        'test_nll':metrics,'training_history':history,'training_seconds_local_cpu':fit_seconds,'adapter_examples':scenarios,
        'scope':'Synthetic transition fixture. Real EEG is an observation-only domain. Cost is logical gate count times simulated shots; no hardware latency or planning benefit claim.'}
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/'selected_predictions.npz',exact=exact,finite_shots=shot_p,classical=classic,truth=test.next_level,day=test.day_id)
    from twin_figures import build
    build(out,rows,frontier,selected,metrics,history)
    print(json.dumps({'selected':choice,'gap':result['search']['gap_to_exhaustive_dev_nll'],'test_nll':metrics},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
