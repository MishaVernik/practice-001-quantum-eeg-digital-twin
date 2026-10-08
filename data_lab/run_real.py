"""Real-data teaching experiment: EEG/CLI fitting and HRV/emotion reanalysis."""
from pathlib import Path
import argparse,csv,json
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve
from signal_tools import filtered_signal
ROOT=Path(__file__).resolve().parent

def scale(train,test):
    lo,hi=np.percentile(train,[1,99],axis=0);span=np.where(hi-lo>1e-9,hi-lo,1)
    return [np.clip(2*(x-lo)/span-1,-1,1) for x in [train,test]]

def cli(theta,alpha,rest_theta,rest_alpha):
    rest=np.log(rest_theta/rest_alpha);mu=np.median(rest);sd=1.4826*np.median(abs(rest-mu))
    if sd<1e-6:raise ValueError('Degenerate resting baseline')
    return np.maximum((np.log(theta/alpha)-mu)/sd,0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise SystemExit('Use a new --out directory.')
    args.out.mkdir(parents=True);out=args.out
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
    d=np.load(ROOT/'data/neuroergo_features.npz',allow_pickle=False)
    mask=d['level']!=1;x=d['x'][mask];y=(d['level'][mask]==2).astype(int);sub=d['subject'][mask]
    people=np.unique(sub);pred=np.zeros(len(y));scores=[]
    for s in people:
        te=sub==s;a,b=scale(x[~te],x[te]);m=LogisticRegression(C=1,max_iter=2000).fit(a,y[~te])
        pred[te]=m.predict_proba(b)[:,1];scores.append(roc_auc_score(y[te],pred[te]))
    rng=np.random.default_rng(20261007);samples=rng.integers(0,len(scores),(10000,len(scores)))
    ci=np.quantile(np.array(scores)[samples].mean(1),[.025,.975])
    with (out/'eeg_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['subject','target','probability']);w.writerows(zip(sub,y,pred))
    rows=list(csv.DictReader((ROOT/'data/cli_bands.csv').open(encoding='utf-8')))
    cli_rows=[];cli_auc=[];medians={c:[] for c in ['lv0','lv1','lv2']}
    for s in people:
        sy=[];sp=[]
        for se in ['S1','S2']:
            base=[r for r in rows if r['subject']==s and r['session']==se and r['condition']=='base']
            bt=np.array([float(r['theta_frontal_uv2']) for r in base]);ba=np.array([float(r['alpha_posterior_uv2']) for r in base])
            for c in ['lv0','lv1','lv2']:
                rr=[r for r in rows if r['subject']==s and r['session']==se and r['condition']==c]
                t=np.array([float(r['theta_frontal_uv2']) for r in rr]);a=np.array([float(r['alpha_posterior_uv2']) for r in rr])
                values=cli(t,a,bt,ba);medians[c].append(float(np.median(values)))
                cli_rows.extend([[s,se,c,i,float(t[i]),float(a[i]),float(v)] for i,v in enumerate(values)])
                if c!='lv1':sy.extend([int(c=='lv2')]*len(values));sp.extend(values)
        cli_auc.append(roc_auc_score(sy,sp))
    with (out/'cli_values.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['subject','session','condition','epoch','theta_uv2','alpha_uv2','cli']);w.writerows(cli_rows)
    with (out/'participant_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['subject','eeg_lr_auc','cli_auc']);w.writerows(zip(people,scores,cli_auc))
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained');pos=np.arange(len(people))
    ax.plot(pos,scores,'o-',label=f'8 EEG-ознак + LR: середнє {np.mean(scores):.3f}',color='#176d92')
    ax.plot(pos,cli_auc,'s--',label=f'Індекс CLI: середнє {np.mean(cli_auc):.3f}',color='#bb7635')
    ax.axhline(.5,color='gray',ls=':',label='Випадкове ранжування: 0,5')
    ax.set(xticks=pos,xticklabels=people,ylabel='AUROC',ylim=(0,1),xlabel='Тестовий учасник',title='Реальні EEG: легкий і складний MATB-II')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,fontsize=9);save(fig,'eeg_result')
    fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained')
    vals=np.array([medians[c] for c in ['lv0','lv1','lv2']]).T
    for v in vals:ax.plot([0,1,2],v,color='#aab8bd',alpha=.55,lw=.7)
    ax.plot([0,1,2],np.median(vals,axis=0),'o-',color='#176d92',lw=2,label='Медіана 30 сесій')
    ax.set(xticks=[0,1,2],xticklabels=['Легке','Середнє','Складне'],ylabel='CLI, одиниці відхилення від спокою',title='CLI: медіана епох у кожній реальній сесії');ax.legend();save(fig,'cli_result')
    raw=np.load(ROOT/'data/raw_eeg_probe.npz');fs=float(raw['sfreq']);names=raw['channel_names'].tolist();ch=names.index('FZ')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    displayed_psd=[]
    for task,label,col in [('RS','Спокій','#777777'),('MATBeasy','Легке','#176d92'),('MATBdiff','Складне','#bb7635')]:
        sig=filtered_signal(raw[task][ch],fs);t=np.arange(len(sig))/fs;f,psd=welch(sig,fs=fs,nperseg=len(sig),detrend='constant')
        axes[0].plot(t,sig,label=label,color=col,lw=.8);axes[1].semilogy(f,psd,label=label,color=col)
        displayed_psd.extend(psd[(f>=1)&(f<=35)])
    axes[0].set(xlabel='Час усередині епохи, с',ylabel='EEG Fz після фільтрації, мкВ',title='P01 / S1 / епоха 0')
    axes[1].set(xlim=(1,35),ylim=(min(displayed_psd)*.7,max(displayed_psd)*1.4),xlabel='Частота, Гц',ylabel='PSD, мкВ²/Гц',title='Періодограма тієї самої епохи')
    axes[1].axvspan(4,8,alpha=.1,color='orange');axes[1].axvspan(8,13,alpha=.1,color='blue');axes[1].legend(fontsize=9);save(fig,'raw_eeg')
    cardiac=json.loads((ROOT/'data/cardiac_windows.json').read_text());paired=[r for r in cardiac if r['ecg']['valid'] and r['prv_native']['valid']]
    hr={}
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for ax,field,label in zip(axes,['rate_bpm','rmssd_ms'],['Частота, уд/хв','RMSSD, мс']):
        a=np.array([r['ecg'][field] for r in paired]);b=np.array([r['prv_native'][field] for r in paired]);delta=b-a
        hr[field]={'mae':float(abs(delta).mean()),'bias':float(delta.mean()),'loa':[float(delta.mean()-1.96*delta.std(ddof=1)),float(delta.mean()+1.96*delta.std(ddof=1))]}
        if field=='rate_bpm':
            ax.scatter(a,b,s=18,color='#176d92');lo=min(a.min(),b.min());hi=max(a.max(),b.max());ax.plot([lo,hi],[lo,hi],'k--',lw=.7)
            ax.set(xlabel='ECG, уд/хв',ylabel='PPG, уд/хв',title=f"Частота: MAE {hr[field]['mae']:.2f} уд/хв")
        else:
            ax.scatter((a+b)/2,delta,s=18,color='#176d92');ax.axhline(delta.mean(),color='#bb7635')
            for limit in hr[field]['loa']:ax.axhline(limit,color='gray',ls='--')
            ax.set(xlabel='Середнє RMSSD ECG і PPG, мс',ylabel='PPG − ECG, мс',title=f"RMSSD: зміщення {delta.mean():+.2f} мс")
    save(fig,'cardiac_result')
    probe=np.load(ROOT/'data/raw_cardiac_probe.npz',allow_pickle=False)
    fig,axes=plt.subplots(2,1,figsize=(9,3.6),layout='constrained',sharex=True)
    keep=(probe['time']>=4)&(probe['time']<=10)
    for ax,key in zip(axes,['ecg','ppg']):
        ax.plot(probe['time'][keep],probe[key][keep],color='#176d92',lw=.8)
        ax.set(ylabel=f'{key.upper()}, од. файлу')
    axes[0].set_title(f"Реальний запис {probe['participant'].item()}: одночасні ECG і PPG")
    axes[1].set_xlabel('Час від початку запису, с');save(fig,'raw_cardiac')
    emotion={}
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for ax,part,title in zip(axes,['participant','stimulus'],['Нові учасники','Нові відео']):
        rr=json.loads((ROOT/f'data/faced_{part}_predictions.json').read_text());yy=np.array([r['y'] for r in rr]);em={}
        for name,key in [('LSTM–GRU','p'),('Класична EEG','classical_eeg'),('Лише відео','context')]:
            pp=np.array([r[key] if key=='p' else r['controls'][key] for r in rr]);auc=roc_auc_score(yy,pp);em[name]={'auroc':float(auc),'average_precision':float(average_precision_score(yy,pp))}
            fpr,tpr,_=roc_curve(yy,pp);ax.plot(fpr,tpr,label=f'{name}: {auc:.3f}')
        ax.plot([0,1],[0,1],ls=':',color='gray');ax.set(xlabel='Частка хибних позитивних',ylabel='Частка виявлених позитивних',title=f'{title}: {len(rr)} проб, {sum(yy)} позитивних');ax.legend(fontsize=8)
        emotion[part]={'n':len(rr),'positives':int(sum(yy)),'metrics':em,'named_output_threshold':rr[0]['threshold']}
    save(fig,'emotion_result')
    result={'eeg':{'participants':len(people),'sessions':30,'binary_windows':len(y),'all_windows':len(d['level']),'lr_mean_subject_auroc':float(np.mean(scores)),'descriptive_subject_bootstrap_ci':ci.tolist(),'subjects_above_chance':int(sum(np.array(scores)>.5))},
        'cli':{'mean_subject_auroc':float(np.mean(cli_auc)),'median_session_cli_by_difficulty':np.median(vals,axis=0).tolist(),'scope':'Mobile formula adapted to native research-cap regions, not a Muse validation'},
        'cardiac':{'candidate_windows':len(cardiac),'paired_windows':len(paired),'participants':len({r['participant'] for r in paired}),'metrics':hr},'emotion':emotion}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
