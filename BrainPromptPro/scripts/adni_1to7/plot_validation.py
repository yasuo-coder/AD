"""Publication-style figures from anonymous verified validation summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--ablation',type=Path,required=True);p.add_argument('--mechanisms',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    summary=json.loads(a.summary.read_text());ablation=json.loads(a.ablation.read_text());mechanisms=json.loads(a.mechanisms.read_text())
    if summary['verification']!='PASSED' or summary.get('protocol')!='sample':raise ValueError('Expected verified sample validation')
    methods=[f'A{i}' for i in range(8)];groups={x['experiment']:x for x in summary['groups']}
    if any(groups[x]['status']!='COMPLETE' for x in methods):raise ValueError('Eight complete three-seed groups required')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    a.output.mkdir(parents=True,exist_ok=False)
    def save(fig,name):
        fig.savefig(a.output/(name+'.png'),dpi=200,bbox_inches='tight');fig.savefig(a.output/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
        svg=a.output/(name+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n');plt.close(fig)
    metrics=[('accuracy','Accuracy'),('f1_macro','Macro-F1'),('precision_macro','Macro precision'),('recall_macro','Balanced accuracy / macro recall'),('roc_auc_macro','Macro OvR AUC'),('f1_weighted','Weighted-F1')]
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    colors=['#0072B2']+['#E69F00']*7
    for ax,(metric,title) in zip(axes.flat,metrics):
        means=[100*groups[x]['metrics'][metric]['mean'] for x in methods];sd=[100*groups[x]['metrics'][metric]['sample_sd'] for x in methods]
        ax.bar(methods,means,yerr=sd,color=colors,alpha=.8,capsize=3)
        for i,method in enumerate(methods):
            vals=[100*r['metrics'][metric] for r in sorted(summary['runs'],key=lambda r:r['seed']) if r['experiment']==method]
            ax.scatter(np.array([i]*3)+np.array([-.09,0,.09]),vals,s=14,c='black',zorder=5)
        ax.axhline(means[0],color='#0072B2',ls='--',lw=1);ax.set_title(title);ax.set_ylabel('%');ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2)
    fig.suptitle('ADNI 1–7: frozen sample-level validation (three training seeds)',fontsize=14)
    fig.text(.5,.015,'A0: BrainPrompt | A1: P2 | A2: P3 | A3: P5 | A4: P2+P3 | A5: P2+P5 | A6: P3+P5 | A7: all\nDots: training seeds; error bars: sample SD, not patient confidence intervals. TEST not used.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95));save(fig,'validation_matrix')
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,(metric,title) in zip(axes,[('f1_macro','Macro-F1'),('balanced_accuracy','Balanced accuracy'),('roc_auc_macro','Macro OvR AUC')]):
        for i,name in enumerate(['P2','P3','P5']):
            item=ablation['modules'][name];v=item['mean_marginal'][metric]
            ax.errorbar(i,100*v['mean'],yerr=100*v['sample_sd'],fmt='o',ms=8,capsize=5,color='#0072B2')
            backgrounds=[100*x['metrics'][metric]['mean'] for x in item['backgrounds']]
            ax.scatter(np.full(4,i)+np.linspace(-.15,.15,4),backgrounds,s=22,color='#D55E00',marker='x')
        ax.axhline(0,color='black',lw=1);ax.set_xticks([0,1,2],['P2','P3','P5']);ax.set_title(title);ax.set_ylabel('Paired change (percentage points)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Module effects across four combination backgrounds',fontsize=14)
    fig.text(.5,.015,'Blue: within-seed average over backgrounds, then mean ± SD across 3 seeds. Orange: each background mean.\nThe 12 correlated contrasts are not 12 independent subjects. Validation only.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.94));save(fig,'paired_module_effects')
    fig,ax=plt.subplots(figsize=(7.5,5.5))
    for name,color in [('A0','#0072B2'),('A1','#D55E00'),('P2_joint_rec_only','#009E73')]:
        row=next(x for x in mechanisms['runs'] if x['experiment']==name);values=[row['clean_f1']]+[x['f1'] for x in row['robustness']]
        ax.errorbar([0,5,10,15],[100*x['mean'] for x in values],yerr=[100*x['sample_sd'] for x in values],label=name,color=color,marker='o',capsize=4)
    ax.set(xlabel='Symmetric FC masking (%)',ylabel='Validation macro-F1 (%)',title='P2 masking robustness across three training seeds',xticks=[0,5,10,15]);ax.grid(alpha=.2);ax.legend()
    fig.text(.5,.01,'Within each trained seed, average 3 fixed perturbation masks. Error bars: training-seed SD.\nDifferent visits and prior development exposure limit generalization claims.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));save(fig,'p2_robustness')
    (a.output/'README.md').write_text('Figures use anonymous verified validation metrics only. Error bars are training-seed sample SD, not patient confidence intervals. The source protocol is sample-level; TEST is not included.\n')
    print(json.dumps({'figures':3,'formats':['png','svg'],'test_accessed':False,'output':str(a.output)}))


if __name__=='__main__':main()
