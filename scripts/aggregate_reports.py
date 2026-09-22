"""Aggregate the deterministic 5 x 4 x 3 benchmark result layout."""
import os,sys,itertools,math,argparse,tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'scolar_mpl'))
from common import *
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEY=['joint_correct_known','novelty_ap','full_prednovel_ami']
ENDPOINTS=KEY+['known_retention','retained_known_accuracy','novelty_auroc','threshold_precision','threshold_recall','threshold_f1','full_prednovel_ari','full_prednovel_nmi','lcc_yield','assigned_cell_coverage','lineage_U']
CONTRASTS=[('REAL_CL','DEPTH_SHUFFLED_CL'),('REAL_CL','GENERIC_STAR'),('DEPTH_SHUFFLED_CL','GENERIC_STAR')]
LABELS={'REAL_CL':'Real CL','DEPTH_SHUFFLED_CL':'Depth shuffled','GENERIC_STAR':'Generic star'}

def status():
    completion=[];values=[]
    for d in DATASETS:
        for seed in SEEDS:
            for arm in ARMS:
                ev=ROOT/'metrics'/d/f'seed{seed}'/arm
                result_path=ev/'result.json';marker_path=ev/'EVALUATION_DONE.json'
                state='PENDING';reason='result.json not found'
                if result_path.is_file():
                    state='COMPLETE';reason=''
                    if marker_path.is_file():
                        marker=json.loads(marker_path.read_text())
                        if not all((ev/f).is_file() and sha(ev/f)==h for f,h in marker.get('files',{}).items()):
                            state='FAILED_INTEGRITY';reason='EVALUATION_DONE.json hash verification failed'
                    if state=='COMPLETE':
                        result=json.loads(result_path.read_text())
                        expected=(d,int(seed),arm)
                        observed=(result.get('dataset'),int(result.get('seed',-1)),result.get('arm'))
                        if observed!=expected:
                            state='FAILED_INTEGRITY';reason=f'result identity {observed!r} != {expected!r}'
                        else:values.append(result)
                completion.append(dict(dataset=d,seed=seed,arm=arm,run_id=f'P0_{d}_s{seed}_{arm}',status=state,
                    result_path=str(result_path.relative_to(ROOT)),diagnostics_complete=(ev/'DIAGNOSTICS_DONE.json').exists(),reason=reason))
    return completion,values

def uncertainty(groups):
    arrays=[np.asarray(v,dtype=float) for v in groups.values() if len(v)];n=len(arrays)
    if not n:return dict(effect=None,ci_low=None,ci_high=None,p_exact=None,n_datasets=0)
    means=np.asarray([v.mean() for v in arrays]);effect=float(means.mean())
    rng=np.random.default_rng(9012026);choice=rng.integers(0,n,size=(10000,n));boot=np.zeros_like(choice,dtype=float)
    for d,a in enumerate(arrays):
        mask=choice==d;count=int(mask.sum());boot[mask]=a[rng.integers(0,len(a),size=(count,len(a)))].mean(1)
    b=boot.mean(1);lo,hi=np.quantile(b,[.025,.975])
    flips=np.asarray(list(itertools.product([-1.,1.],repeat=n)));stats=np.abs((flips*means).mean(1))
    p=float(np.mean(stats>=abs(effect)-1e-14))
    return dict(effect=effect,ci_low=float(lo),ci_high=float(hi),p_exact=p,n_datasets=n,n_positive_datasets=int((means>0).sum()),n_negative_datasets=int((means<0).sum()),minimum_two_sided_p=float(2/2**n) if n else None)

def holm(rows,indices):
    order=sorted(indices,key=lambda i:rows[i]['p_exact'] if rows[i]['p_exact'] is not None else 1.)
    running=0.;m=len(order)
    for rank,i in enumerate(order):
        p=rows[i]['p_exact'];running=max(running,min(1.,(m-rank)*(p if p is not None else 1.)))
        rows[i]['p_holm']=running if p is not None else None

def aggregate(completion,values):
    frame=pd.DataFrame(values);raw=[];dataset_effects=[];summary=[]
    lookup={(r['dataset'],int(r['seed']),r['arm']):r for r in values}
    for a,b in CONTRASTS:
        contrast=f'{a}-{b}'
        for metric in ENDPOINTS:
            groups={};controls=[]
            for d in DATASETS:
                paired=[]
                for seed in SEEDS:
                    ra=lookup.get((d,seed,a));rb=lookup.get((d,seed,b))
                    if ra is None or rb is None:continue
                    va=ra.get(metric);vb=rb.get(metric)
                    if va is None or vb is None:continue
                    assert ra['split_sha256']==rb['split_sha256'];delta=float(va-vb);paired.append(delta)
                    raw.append(dict(dataset=d,seed=seed,contrast=contrast,endpoint=metric,real_or_first=va,control_or_second=vb,effect=delta,
                       three_arm_block_complete=all((d,seed,arm) in lookup for arm in ARMS)))
                if paired:
                    groups[d]=paired
                    controls.append(np.mean([lookup[(d,s,b)][metric] for s in SEEDS if (d,s,a) in lookup and (d,s,b) in lookup and lookup[(d,s,a)].get(metric) is not None and lookup[(d,s,b)].get(metric) is not None]))
                dataset_effects.append(dict(dataset=d,contrast=contrast,endpoint=metric,n_paired_seeds=len(paired),effect=float(np.mean(paired)) if paired else None,
                   seed_sd=float(np.std(paired,ddof=1)) if len(paired)>1 else None,status='FOUR_SEEDS_COMPLETE' if len(paired)==4 else 'PARTIAL_OR_NOT_EVALUABLE'))
            item=dict(contrast=contrast,endpoint=metric,planned=(a=='REAL_CL'),**uncertainty(groups),n_pairs=sum(map(len,groups.values())),
               complete_5x4=(len(groups)==5 and all(len(v)==4 for v in groups.values())),p_holm=None)
            cm=float(np.mean(controls)) if controls else None
            item['relative_effect']=item['effect']/cm if cm is not None and cm>0 and item['effect'] is not None else None
            summary.append(item)
    holm(summary,[i for i,r in enumerate(summary) if r['planned'] and r['endpoint'] in KEY])
    holm(summary,[i for i,r in enumerate(summary) if r['planned'] and r['endpoint']=='lineage_U'])
    writecsv(ROOT/'tables/completion_matrix.csv',completion)
    writecsv(ROOT/'metrics/paired_run_level.csv',raw,['dataset','seed','contrast','endpoint','real_or_first','control_or_second','effect','three_arm_block_complete'])
    writecsv(ROOT/'metrics/paired_effects_by_dataset.csv',dataset_effects)
    writecsv(ROOT/'metrics/paired_effects_and_CI.csv',summary)
    if values:
        try:
            frame.to_parquet(ROOT/'metrics/results_run_level.parquet',index=False)
        except ImportError as exc:
            (ROOT/'metrics/PARQUET_PENDING.md').write_text('Parquet export dependency pending; complete CSV outputs remain authoritative.\n'+str(exc)+'\n')
        frame.to_csv(ROOT/'metrics/results_run_level.csv',index=False)
        frame.melt(id_vars=['dataset','seed','arm'],value_vars=[m for m in ENDPOINTS if m in frame],var_name='endpoint',value_name='value').to_csv(ROOT/'metrics/results_long.csv',index=False)
        means=frame.groupby(['dataset','arm'],sort=False)[ENDPOINTS].agg(['mean','std','count']);means.columns=['_'.join(c) for c in means.columns];means.to_csv(ROOT/'metrics/aggregate_by_dataset.csv')
    else:
        writecsv(ROOT/'metrics/results_run_level.csv',[],['dataset','seed','arm','status']+ENDPOINTS)
        writecsv(ROOT/'metrics/results_long.csv',[],['dataset','seed','arm','endpoint','value'])
        writecsv(ROOT/'metrics/aggregate_by_dataset.csv',[],['dataset','arm'])
    return frame,raw,dataset_effects,summary

def savefig(fig,name,csvframe):
    (ROOT/'figures').mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    for fmt in ['svg','pdf','png']:fig.savefig(ROOT/'figures'/f'{name}.{fmt}',dpi=300,bbox_inches='tight')
    plt.close(fig);csvframe.to_csv(ROOT/'figures'/f'{name}.csv',index=False)

def figures(frame,paired):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    if frame.empty:return
    pr=pd.DataFrame(paired)
    if not pr.empty:
        fig,axes=plt.subplots(3,2,figsize=(10,9),squeeze=False)
        for i,m in enumerate(KEY):
            for j,(a,b) in enumerate(CONTRASTS[:2]):
                ax=axes[i,j];sub=pr[(pr.endpoint==m)&(pr.contrast==f'{a}-{b}')]
                ax.axvline(0,color='.6',lw=.8)
                for k,d in enumerate(DATASETS):
                    ds=sub[sub.dataset==d]
                    ax.scatter(ds.effect,np.full(len(ds),k)+np.linspace(-.12,.12,len(ds)),s=18,alpha=.65,color='#247a8a')
                    if len(ds):ax.scatter([ds.effect.mean()],[k],marker='D',color='#c35b35',s=26)
                ax.set_yticks(range(5),DATASETS);ax.set_title(m.replace('_',' ')+'\nReal − '+LABELS[b]);ax.set_xlabel('Paired effect (fraction units)')
        savefig(fig,'paired_ontology_effects',pr)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    routing=frame.groupby(['dataset','arm'])[['n_known','tn','joint_correct_known_n','n_novel','tp']].mean().reset_index()
    for ax,arm in zip(axes,ARMS):
        sub=routing[routing.arm==arm]
        for _,r in sub.iterrows():ax.plot([0,1,2],[r.n_known,r.tn,r.joint_correct_known_n],marker='o',label=r.dataset)
        ax.set_xticks([0,1,2],['True known','Retained','Correct known'],rotation=20);ax.set_title(LABELS[arm]);ax.set_ylabel('Cells (mean across available seeds)')
    axes[-1].legend(fontsize=7);savefig(fig,'routing_waterfall',routing)
    curveframes=[]
    for _,r in frame.iterrows():
        p=ROOT/'metrics'/r.dataset/f'seed{int(r.seed)}'/r.arm/'calibration_risk_curve.csv'
        c=pd.read_csv(p);c=c.iloc[::max(1,len(c)//200)].copy();c['dataset']=r.dataset;c['seed']=r.seed;c['arm']=r.arm;curveframes.append(c)
    if curveframes:
        allc=pd.concat(curveframes,ignore_index=True);fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        colors=dict(zip(ARMS,['#247a8a','#c35b35','#777777']))
        for (_,_,arm),g in allc.groupby(['dataset','seed','arm']):
            axes[0].plot(g.recall,g.precision,color=colors[arm],alpha=.25,lw=.8)
            axes[1].plot(g.recall,g.known_retention,color=colors[arm],alpha=.25,lw=.8)
            axes[2].plot(g.known_retention,g.selective_known_risk,color=colors[arm],alpha=.25,lw=.8)
        for arm,color in colors.items():axes[0].plot([],[],color=color,label=LABELS[arm])
        axes[0].set(xlabel='Novel recall',ylabel='Novel precision');axes[1].set(xlabel='Novel recall',ylabel='Known retention');axes[2].set(xlabel='Known retention',ylabel='Retained-known risk');axes[0].legend(fontsize=7)
        savefig(fig,'calibration_risk_curves',allc)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,m in zip(axes,['full_prednovel_ami','full_prednovel_ari','full_prednovel_nmi']):
        for j,arm in enumerate(ARMS):
            f=frame[frame.arm==arm];ax.scatter([DATASETS.index(d)+j*.15-.15 for d in f.dataset],f[m],label=LABELS[arm],s=16,alpha=.65)
        ax.set_xticks(range(5),DATASETS,rotation=30,ha='right');ax.set_ylabel(m)
    axes[0].legend(fontsize=7);savefig(fig,'full_prednovel_grouping',frame[['dataset','seed','arm','pred_novel_n','known_contamination','full_prednovel_ami','full_prednovel_ari','full_prednovel_nmi']])
    for filename,name,x,y in [('lcc_sensitivity.csv','lcc_coverage_correctness','assigned_cell_coverage','lineage_U'),
         ('ontology_distance.csv','ontology_distance_mechanism','nearest_source_cl_distance','recall'),
         ('oracle_routing_K_decomposition.csv','oracle_routing_K_decomposition',None,'ami')]:
        fs=[]
        for _,r in frame.iterrows():
            p=ROOT/'metrics'/r.dataset/f'seed{int(r.seed)}'/r.arm/filename
            if p.exists():
                f=pd.read_csv(p);f['dataset']=r.dataset;f['seed']=r.seed;f['arm']=r.arm;fs.append(f)
        if not fs:continue
        data=pd.concat(fs,ignore_index=True);fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for ax,arm in zip(axes,ARMS):
            sub=data[data.arm==arm]
            if x:
                for d,g in sub.groupby('dataset'):ax.scatter(g[x],g[y],s=13,alpha=.6,label=d)
                ax.set(xlabel=x.replace('_',' '),ylabel=y.replace('_',' '))
            else:
                sub=sub[sub.grouping!='frozen_OverallJ'];cats=['full_PredNovel/no_K_Leiden','GT_novel_oracle_routing/no_K_Leiden','full_PredNovel/oracle_K_KMeans','GT_novel_oracle_routing/oracle_K_KMeans']
                cat=sub.population+'/'+sub.grouping
                for j,c in enumerate(cats):
                    vals=sub.loc[cat==c,y];ax.scatter(np.full(len(vals),j),vals,s=12,alpha=.6)
                ax.set_xticks(range(4),['Deployment','Routing oracle','K oracle','Both oracle'],rotation=30,ha='right');ax.set_ylabel('AMI')
            ax.set_title(LABELS[arm])
        if x:axes[-1].legend(fontsize=6)
        savefig(fig,name,data)

def report(completion,frame,summary):
    n=sum(c['status']=='COMPLETE' for c in completion);failed=[c for c in completion if c['status'].startswith('FAILED')]
    pending=[c for c in completion if c['status']=='PENDING'];full=n==60;diag=sum(bool(c['diagnostics_complete']) for c in completion)
    lines=['# Aggregated benchmark results',
      f'Generated from deterministic discovery of `metrics/<dataset>/seed<seed>/<arm>/result.json`. Complete: {n}/60; integrity failures: {len(failed)}; pending: {len(pending)}; optional diagnostics: {diag}/60.',
      'The matrix is 5 datasets × 4 seeds × 3 frozen arms. Dataset is the inferential unit and seeds are technical repeats.',
      '| Contrast | Endpoint | Equal-dataset paired effect | 95% hierarchical bootstrap CI | Complete pairs | Exact p / Holm p |',
      '|---|---|---:|---|---:|---|']
    fmt=lambda x:'NA' if x is None else f'{x:.4f}'
    for r in summary:
        if r['planned'] and r['endpoint'] in KEY+['lineage_U']:
            lines.append(f"| {r['contrast']} | {r['endpoint']} | {fmt(r['effect'])} | [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {r['n_pairs']}/20 | {fmt(r['p_exact'])} / {fmt(r['p_holm'])} |")
    lines+=['With five datasets the minimum two-sided exact sign-flip p is 0.0625. Incomplete pairs or datasets are descriptive only.',
      '## Completion matrix','| Dataset / seed | REAL_CL | DEPTH_SHUFFLED_CL | GENERIC_STAR |','|---|---|---|---|']
    for d in DATASETS:
        for seed in SEEDS:
            cells=[next(c for c in completion if c['dataset']==d and int(c['seed'])==seed and c['arm']==a)['status'] for a in ARMS]
            lines.append(f'| {d} / {seed} | '+' | '.join(cells)+' |')
    lines+=['## Generated artifacts']+[f'- `{p}`' for p in ['tables/completion_matrix.csv','metrics/results_run_level.csv','metrics/results_long.csv','metrics/aggregate_by_dataset.csv','metrics/paired_run_level.csv','metrics/paired_effects_by_dataset.csv','metrics/paired_effects_and_CI.csv','figures/']]
    for c in completion:
        if c['status']!='COMPLETE':lines.append(f"- `{c['run_id']}`: {c['status']}; {c['reason']}")
    report='\n'.join(line if line.startswith('|') else '\n'+line+'\n' for line in lines)+'\n'
    (ROOT/'reports').mkdir(parents=True,exist_ok=True);(ROOT/'reports/RESULTS_SUMMARY.md').write_text(report)
    writej(ROOT/'statistics/aggregation_status.json',dict(utc=now(),complete=n,failed_integrity=len(failed),pending=len(pending),diagnostics_complete=diag,complete_matrix=full))
    (ROOT/'audit').mkdir(parents=True,exist_ok=True)
    pd.DataFrame(failed,columns=list(completion[0]) if completion else None).to_csv(ROOT/'audit/FAILED_RUNS.tsv',sep='\t',index=False)

def main():
    completion,values=status();frame,raw,ds,summary=aggregate(completion,values);report(completion,frame,summary);figures(frame,raw)
    print('AGGREGATION',sum(c['status']=='COMPLETE' for c in completion),'/60',now(),flush=True)

if __name__=='__main__':main()
