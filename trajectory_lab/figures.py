"""Matplotlib figures derived only from supplied measured probabilities."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'uniform':'Рівномірний розподіл','marginal':'Лише тип завдання','markov':'Контекстна модель Маркова',
 'gaussian_ar':'Гауссівська авторегресія','softmax':'Класична softmax',
 'born_gray_product':'Born Gray без CNOT','born_gray_entangled':'Born Gray з CNOT'}

def build_figures(out,predictions,test,metrics):
    out=Path(out);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(out/(name+'.'+ext),dpi=200,bbox_inches='tight')
        plt.close(fig)
    names=list(predictions)
    fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained')
    vals=[metrics[n]['nll'] for n in names]
    colors=['#adb5bd' if n=='uniform' else '#b57936' if 'product' in n else '#176d92' if 'entangled' in n else '#466950' for n in names]
    bars=ax.barh([LABELS[n] for n in names],vals,color=colors)
    ax.bar_label(bars,fmt='%.4f',padding=4);ax.set_xlim(0,2.4);ax.invert_yaxis()
    ax.set_xlabel('NLL на тесті, нат / перехід – менше краще')
    ax.set_title('885 переходів • 5 нових синтетичних днів')
    ax.grid(axis='x',alpha=.15);save(fig,'comparison')
    days=np.unique(test.day_id)
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
    for name in ['uniform','marginal','born_gray_entangled','softmax']:
        p=predictions[name];loss=-np.log(np.maximum(p[np.arange(len(test)),test.next_level],1e-12))
        ax.plot(days,[loss[test.day_id==d].mean() for d in days],marker='o',label=LABELS[name])
    ax.set_ylabel('NLL, нат / перехід');ax.set_title('Результат окремо для кожного тестового дня')
    ax.legend(fontsize=9,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.10),frameon=False)
    ax.grid(alpha=.15);save(fig,'per_day')
    # Fixed first test day: show every segment, never choose the prettiest example.
    idx=np.flatnonzero(test.day_id==days[0]);y=test.next_level[idx]
    fig,axes=plt.subplots(2,1,figsize=(9,5),layout='constrained',sharex=True)
    for ax,name in zip(axes,['born_gray_entangled','softmax']):
        p=predictions[name][idx]
        im=ax.imshow(p.T,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='Blues',interpolation='nearest')
        ax.plot(np.arange(len(y)),y,'.',color='#be3c29',markersize=3,label='Спостережений рівень')
        for k in range(1,len(idx)):
            if test.segment_id[idx[k]]!=test.segment_id[idx[k-1]]:ax.axvline(k-.5,color='black',ls='--',lw=.7)
        ax.set_ylabel('Рівень');ax.set_yticks([0,2,4,6,7]);ax.set_title(LABELS[name]+' • день d10',fontsize=11)
        ax.legend(loc='upper right',fontsize=8)
    axes[-1].set_xlabel('Порядковий номер допустимого переходу; пунктир – межа сегмента')
    fig.colorbar(im,ax=axes,label='Умовна ймовірність',shrink=.85);save(fig,'next_state')
    # Sampling is a probabilistic output of a trained distribution, not invented data.
    p=predictions['born_gray_entangled'][0];rng=np.random.default_rng(20261007)
    samples=rng.choice(8,size=4096,p=p);freq=np.bincount(samples,minlength=8)/len(samples)
    fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained');x=np.arange(8)
    ax.bar(x-.18,p,.36,label='Точна модель Born',color='#176d92')
    ax.bar(x+.18,freq,.36,label='4096 згенерованих результатів',color='#75a6b7')
    ax.axhline(1/8,color='#7b7b7b',ls='--',label='Рівномірне вгадування')
    ax.set(xticks=x,xlabel='Наступний рівень',ylabel='Ймовірність / частота',title='Перший тестовий контекст d10 – без вибору за результатом')
    ax.legend(fontsize=9);save(fig,'sampling')
    np.savetxt(out/'sampling.csv',np.column_stack([x,p,freq]),delimiter=',',header='level,probability,frequency_4096',comments='')
