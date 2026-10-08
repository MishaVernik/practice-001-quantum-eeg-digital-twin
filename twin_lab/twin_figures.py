import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build(out,rows,frontier,selected,metrics,history):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    for c,label,marker in [(0,'Без CNOT','o'),(1,'Із CNOT','s')]:
        rr=[r for r in rows if r['cnot']==c];ax.scatter([r['gate_shots'] for r in rr],[r['dev_nll'] for r in rr],label=label,marker=marker,alpha=.7)
    ff=sorted([rows[i] for i in frontier],key=lambda r:r['gate_shots'])
    ax.plot([r['gate_shots'] for r in ff],[r['dev_nll'] for r in ff],color='#176d92',ls='--',label='Точний фронт 30 конфігурацій')
    r=rows[selected];ax.scatter([r['gate_shots']],[r['dev_nll']],s=150,marker='*',color='#a32b36',label='Обрано NSGA-II за бюджетом')
    ax.axvline(4608,color='gray',ls=':',label='Бюджет 4608');ax.set(xscale='log',xlabel='Логічні gate-shots на прогноз',ylabel='Валідаційний NLL (менше – краще)',title='Виміряні оцінки конфігурацій на 354 переходах')
    ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2);save(fig,'pareto')
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
    labels=['Рівномірний','Лише діяльність','Обрана Born\nточний розподіл','Обрана Born\nскінченні shots','Softmax']
    bars=ax.barh(labels,list(metrics.values()),color=['#aaa','#999','#176d92','#398dab','#466950'])
    ax.bar_label(bars,fmt='%.4f',padding=4);ax.invert_yaxis();ax.set(xlim=(0,max(metrics.values())*1.18),xlabel='NLL (менше – краще)',title='Окремий тест: 885 синтетичних переходів');save(fig,'twin_result')
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
    for k,h in history.items():ax.plot(h['epochs'],h['nll'],label=k,marker='o',markersize=3)
    ax.set_xticks([0,1,8])
    ax.set(xlabel='Епоха навчання; 0 – ініціалізація',ylabel='Навчальний NLL (менше – краще)',title='Шість фактично навчених схем Борна');ax.legend(ncol=3,fontsize=8);save(fig,'learning')
    fig,ax=plt.subplots(figsize=(10,3.8),layout='constrained');ax.axis('off')
    boxes=[(.16,.78,'Реальні EEG\nспостереження'),(.16,.28,'Синтетичні дні\nконтекст прогнозу'),(.48,.78,'NSGA-II\nякість і бюджет'),(.48,.28,'Прогноз\nBorn / softmax'),(.83,.53,'Відповідь двійника\nджерело, режим,\nоцінка або прогноз')]
    for x,y,label in boxes:ax.text(x,y,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#e9f1f5',ec='#176d92'))
    ax.plot([.16,.16,.83,.83],[.89,.98,.98,.74],color='#176d92',lw=1)
    for start,end in [((.83,.74),(.83,.69)),((.28,.28),(.37,.28)),((.48,.63),(.48,.43)),((.59,.28),(.72,.43))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#176d92'))
    ax.text(.5,.01,'Два незалежні домени даних; спільний програмний контракт',ha='center',fontsize=11)
    ax.set(xlim=(0,1),ylim=(-.07,1));save(fig,'architecture')
