"""Physical evaluator process: verifies frozen outputs BEFORE accessing sealed truth."""
import os,sys,argparse,math,tempfile
from pathlib import Path
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(tempfile.gettempdir())/'scolar_numba'));os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'scolar_mpl'))
from common import *
sys.path.insert(0,str(ROOT/'scolar/frozen'))
import numpy as np,torch,igraph as ig
from sklearn.metrics import average_precision_score,roc_auc_score,homogeneity_score,completeness_score
import frozen_common_evaluator as frozen
from blind_inference import lcc_partition
from ontology_adapter import view

def ratio(a,b):return float(a/b) if b else None
def scores(y,c):
    s=frozen.safe_scores(y,c)
    s['na_reason']='population<2 or truth<2 classes or partition<2 clusters' if s['ami'] is None else ''
    return s
def routing(y,p,correct):
    known=~y;tp=int((y&p).sum());fp=int((known&p).sum());tn=int((known&~p).sum());fn=int((y&~p).sum());joint=int((known&~p&correct).sum())
    return dict(n_target=len(y),n_known=int(known.sum()),n_novel=int(y.sum()),tp=tp,fp=fp,tn=tn,fn=fn,joint_correct_known_n=joint,
      known_retention=ratio(tn,tn+fp),retained_known_accuracy=ratio(joint,tn),joint_correct_known=ratio(joint,tn+fp),false_novel_rate=ratio(fp,tn+fp),
      threshold_precision=ratio(tp,tp+fp),threshold_recall=ratio(tp,tp+fn),threshold_f1=ratio(2*tp,2*tp+fp+fn),pred_novel_n=tp+fp,pred_novel_fraction=ratio(tp+fp,len(y)),
      known_contamination=ratio(fp,tp+fp),selective_known_risk=ratio(tn-joint,tn))

class Lineage:
    def __init__(self):
        ck=torch.load(PROJECT/'data/cl.pt',map_location='cpu',weights_only=False)
        self.idx=ck['mappings']['id2idx'];self.terms=ck['structure']['all_terms'];self.a=ck['structure']['ancestor_matrix_incl_self'].numpy();self.d=ck['structure']['depths'].numpy()
        self.graph=ig.Graph(n=len(self.d),edges=np.argwhere(ck['structure']['parent_matrix'].numpy()).tolist(),directed=True);self.cache={}
    def distance(self,a,b):
        key=(int(a),int(b))
        if key not in self.cache:self.cache[key]=self.graph.distances(source=int(a),target=int(b),mode='all')[0][0]
        v=self.cache[key];return float(v) if np.isfinite(v) else None
    def evaluate(self,records,part,truth,arm):
        ti=np.asarray([self.idx.get(str(t),-1) for t in truth]);valid=(ti>=0)
        valid[valid]=self.d[ti[valid]]>0
        rows=[];eligible_evaluable=0;assigned_evaluable=0;utility=0.;compatible_total=0;distance_sum=0.;distance_n=0;exact=[];q=[];over=under=0
        for rec in records:
            r=dict(rec);mask=(part==int(rec['cluster_id']));tv=ti[mask&valid];n=len(tv)
            r['n_evaluable']=n;r['n_unmapped_or_root_truth']=int(mask.sum())-n
            r['lineage_status']='not_applicable_synthetic' if arm=='GENERIC_STAR' else ('EVALUABLE' if n else 'NOT_EVALUABLE')
            r.update(compatible_n=0,utility_numerator=0.,q=None,exact_lca=None,mean_shortest_path=None,over_specific_n=0,under_specific_n=0)
            if rec['eligible']:
                eligible_evaluable+=n
                pred=self.idx.get(rec.get('lcc_node',''),-1)
                if rec['assigned'] and n and pred>=0 and self.d[pred]>0:
                    compatible=self.a[tv,pred];spec=np.minimum(1.,self.d[pred]/self.d[tv]);u=float((compatible*spec).sum())
                    r.update(compatible_n=int(compatible.sum()),utility_numerator=u,q=float(compatible.mean()),
                       over_specific_n=int((self.a[pred,tv]&(tv!=pred)).sum()),under_specific_n=int((compatible&(tv!=pred)).sum()))
                    common=np.flatnonzero(self.a[np.unique(tv)].all(0));deepest=common[self.d[common]==self.d[common].max()] if len(common) else []
                    r['exact_lca']=int(pred in deepest);exact.append(r['exact_lca']);q.append(r['q'])
                    distances=[self.distance(pred,t) for t in tv];distances=[d for d in distances if d is not None]
                    r['mean_shortest_path']=float(np.mean(distances)) if distances else None
                    compatible_total+=int(compatible.sum());utility+=u;assigned_evaluable+=n;distance_sum+=sum(distances);distance_n+=len(distances)
                    over+=r['over_specific_n'];under+=r['under_specific_n']
            rows.append(r)
        applicable=arm!='GENERIC_STAR' and eligible_evaluable>0
        return dict(lineage_U=ratio(utility,eligible_evaluable) if applicable else None,lineage_utility_numerator=utility if applicable else None,
          lineage_eligible_evaluable_n=eligible_evaluable,lineage_assigned_evaluable_n=assigned_evaluable,lineage_compatible_n=compatible_total,
          lineage_status='EVALUABLE' if applicable else ('not_applicable_synthetic' if arm=='GENERIC_STAR' else 'NOT_EVALUABLE'),
          gt_mapping_coverage=ratio(int(valid.sum()),len(valid)),lineage_q_cluster_macro=float(np.mean(q)) if q and applicable else None,
          lineage_q_cell_weighted=ratio(compatible_total,assigned_evaluable) if applicable else None,
          exact_lca_accuracy=float(np.mean(exact)) if exact and applicable else None,lineage_mean_shortest_path=ratio(distance_sum,distance_n) if applicable else None,
          over_specific_n=over,under_specific_n=under),rows

def verify(out):
    sentinel=json.loads((out/'PREDICTION_DONE.json').read_text());assert sentinel['manifest_sha256']==sha(out/'PREDICTIONS_FROZEN.json')
    manifest=json.loads((out/'PREDICTIONS_FROZEN.json').read_text());assert not manifest['target_truth_access']
    assert all(sha(out/f)==h for f,h in manifest['files'].items()),'frozen output drift'
    meta=json.loads((out/'run_metadata.json').read_text());assert not meta['smoke'] and meta['checkpoint_epoch']==100
    assert meta['protocol_sha256']==sha(PROTOCOL)
    return meta

def evaluate(out,diagnostics=True):
    out=Path(out);meta=verify(out);dataset=meta['dataset'];seed=meta['seed'];arm=meta['arm']
    dest=ROOT/'metrics'/dataset/f'seed{seed}'/arm;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'EVALUATION_DONE.json').exists():
        mark=json.loads((dest/'EVALUATION_DONE.json').read_text());assert all(sha(dest/f)==h for f,h in mark['files'].items());return
    pred=np.load(out/'predictions.npz',allow_pickle=False)
    folder=ROOT/'splits'/dataset/f'seed{seed}';splitmeta=json.loads((folder/'metadata.json').read_text())
    assert sha(folder/'truth.npz')==splitmeta['truth_sha256']
    ledger(dataset,seed,'separate_postfreeze_evaluator',folder/'truth.npz','all sealed truth fields','all prediction/threshold/partition/native LCC manifests verified; no feedback to training')
    truth=np.load(folder/'truth.npz',allow_pickle=False)
    assert np.array_equal(pred['cell_ids'],truth['cell_ids']);assert objsha(pred['cell_ids'].tolist())==meta['target_cell_order_sha256']
    y=truth['true_novel'].astype(bool);p=pred['pred_novel'].astype(bool);part=pred['cluster_id'];labels=truth['true_label'];correct=pred['known_prediction']==truth['true_canonical_row']
    assert np.array_equal(p,pred['mls']<float(pred['alpha']));assert np.all(part[~p]==-1);assert np.all(part[p]>=0)
    assert arrsha(p)==meta['membership_sha256'] and arrsha(part)==meta['partition_sha256']
    r=dict(dataset=dataset,seed=seed,arm=arm,run_id=f'P0_{dataset}_s{seed}_{arm}',status='COMPLETE',protocol_sha256=meta['protocol_sha256'],split_sha256=meta['split_sha256'],
           alpha=float(pred['alpha']),novelty_threshold=float(pred['threshold']),**routing(y,p,correct))
    r['novelty_ap']=float(average_precision_score(y,pred['novelty_score'])) if len(np.unique(y))==2 else None
    r['novelty_auroc']=float(roc_auc_score(y,pred['novelty_score'])) if len(np.unique(y))==2 else None
    r['ranking_na_reason']='' if r['novelty_ap'] is not None else 'single truth class'
    gs=scores(labels[p],part[p]);r.update({'full_prednovel_'+k:v for k,v in gs.items()})
    r['predicted_cluster_count']=len(np.unique(part[p]));r['homogeneity']=float(homogeneity_score(labels[p],part[p])) if p.any() else None
    r['completeness']=float(completeness_score(labels[p],part[p])) if p.any() else None
    r['fragmentation_mean_clusters_per_type']=float(np.mean([len(np.unique(part[p&(labels==l)])) for l in np.unique(labels[p])])) if p.any() else None
    r['merging_mean_types_per_cluster']=float(np.mean([len(np.unique(labels[part==c])) for c in np.unique(part[p])])) if p.any() else None
    lcc=json.loads((out/'lcc.json').read_text())['records'];eligible=[c for c in lcc if c['eligible']];assigned=[c for c in eligible if c['assigned']]
    r.update(lcc_eligible_clusters=len(eligible),lcc_assigned_clusters=len(assigned),lcc_eligible_cells=sum(c['n_cells'] for c in eligible),lcc_assigned_cells=sum(c['n_cells'] for c in assigned),
      lcc_yield=ratio(len(assigned),len(eligible)),assigned_cell_coverage=ratio(sum(c['n_cells'] for c in assigned),sum(c['n_cells'] for c in eligible)))
    lin=Lineage();lm,clusters=lin.evaluate(lcc,part,truth['true_cl_term'],arm);r.update(lm)
    for c in clusters:c.update(dataset=dataset,seed=seed,arm=arm)
    writej(dest/'cluster_level_lcc.json',clusters)
    writecsv(dest/'cluster_level_lcc.csv',[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in c.items()} for c in clusters])
    # Full exact threshold sequence, tied scores routed together (novelty > threshold).
    score=pred['novelty_score'];order=np.argsort(-score,kind='stable');ys=y[order].astype(int);ks=(~y)[order].astype(int);cs=((~y)&correct)[order].astype(int)
    tp=np.cumsum(ys);fp=np.cumsum(ks);lost=np.cumsum(cs);ends=np.flatnonzero(np.r_[score[order][:-1]!=score[order][1:],True]);curve=[]
    nknown=int((~y).sum());nnovel=int(y.sum());nc=int(((~y)&correct).sum())
    for j in np.r_[-1,ends]:
        a=int(tp[j]) if j>=0 else 0;b=int(fp[j]) if j>=0 else 0;joint=nc-(int(lost[j]) if j>=0 else 0)
        threshold=float(np.nextafter(float(score[order[j]]),-np.inf)) if j>=0 else float(np.max(score))
        curve.append(dict(threshold=threshold,precision=ratio(a,a+b),recall=ratio(a,nnovel),known_retention=ratio(nknown-b,nknown),joint_correct_known=ratio(joint,nknown),
             selective_known_risk=ratio(nknown-b-joint,nknown-b),tp=a,fp=b,tn=nknown-b,fn=nnovel-a,is_frozen_operating_point=False))
    curve.append(dict(threshold=float(pred['threshold']),precision=r['threshold_precision'],recall=r['threshold_recall'],known_retention=r['known_retention'],joint_correct_known=r['joint_correct_known'],
      selective_known_risk=r['selective_known_risk'],tp=r['tp'],fp=r['fp'],tn=r['tn'],fn=r['fn'],is_frozen_operating_point=True))
    writecsv(dest/'calibration_risk_curve.csv',curve)
    # Threshold sensitivities retain every eligible cluster, including unassigned.
    sensitivity=[]
    for c in [.25,.30,.4,.5,.6,.7]:
        rec=[]
        for native in lcc:
            d=dict(native);d['assigned']=bool(d['eligible'] and d.get('candidate_node') and d['top3_coverage']>=c)
            d['lcc_node']=d.get('candidate_node','') if d['assigned'] else '';d['lcc_depth']=d.get('candidate_depth') if d['assigned'] else None;rec.append(d)
        lineage,_=lin.evaluate(rec,part,truth['true_cl_term'],arm);assigned_c=[x for x in rec if x['assigned']]
        sensitivity.append(dict(coverage_threshold=c,n_eligible=len(eligible),n_assigned=len(assigned_c),assignment_yield=ratio(len(assigned_c),len(eligible)),
            assigned_cell_coverage=ratio(sum(x['n_cells'] for x in assigned_c),r['lcc_eligible_cells']),**lineage))
    writecsv(dest/'lcc_sensitivity.csv',sensitivity)
    # Per-type ontology distance, fixed strata chosen before outcome access.
    distrows=[]
    for label in np.unique(labels):
        mask=labels==label;terms=np.unique(truth['true_cl_term'][mask]);term=str(terms[0]) if len(terms)==1 else '';ti=lin.idx.get(term,-1)
        distances=[lin.distance(ti,int(s)) for s in pred['source_rows']] if ti>=0 else [];distances=[v for v in distances if v is not None];d=min(distances) if distances else None
        stratum='NOT_EVALUABLE' if d is None else ('near_sibling' if d<=2 else 'intermediate' if d<=4 else 'distant')
        distrows.append(dict(dataset=dataset,seed=seed,arm=arm,true_type=label,n_cells=int(mask.sum()),true_novel=bool(y[mask][0]),nearest_source_cl_distance=d,distance_stratum=stratum,
          true_cl_term=term,ontology_depth=int(lin.d[ti]) if ti>=0 else None,mean_novelty_score=float(score[mask].mean()),target_prevalence=float(mask.mean()),
          recall=ratio(int((mask&p).sum()),int(mask.sum())) if y[mask][0] else None,**{f'grouping_{k}':v for k,v in scores(labels[mask&p],part[mask&p]).items()}))
    writecsv(dest/'ontology_distance.csv',distrows)
    # Explicit secondaries run only after native artifacts were saved and sealed.
    posthoc=[]
    for lcc_arm in (['REAL_CL','DEPTH_SHUFFLED_CL'] if arm!='GENERIC_STAR' else ['REAL_CL']):
        oo=view(dataset,lcc_arm,pred['source_rows'],'cpu')
        records,_=lcc_partition(pred['source_logits'],part,p,oo,pred['source_rows'])
        vals,_=lin.evaluate(records,part,truth['true_cl_term'],lcc_arm)
        er=[x for x in records if x['eligible']]
        posthoc.append(dict(training_arm=arm,lcc_arm=lcc_arm,secondary_only=True,lcc_yield=ratio(sum(x['assigned'] for x in er),len(er)),**vals))
    writecsv(dest/'training_interpretation_decomposition.csv',posthoc)
    # Write cell-level evaluated output with explicit evaluation-only truth provenance.
    cells=[]
    for i,cid in enumerate(pred['cell_ids']):
        cells.append(dict(cell_id=cid,dataset=dataset,seed=seed,arm=arm,mapping_seed=meta['ontology_mapping_seed'],novelty_score=float(score[i]),threshold=float(pred['threshold']),
          pred_novel=bool(p[i]),known_prediction=int(pred['known_prediction'][i]),cluster_id=int(part[i]),lcc_node=pred['lcc_node'][i],
          evaluation_only_true_type=labels[i],evaluation_only_true_novel=bool(y[i]),evaluation_only_true_cl=truth['true_cl_term'][i],truth_source=str(folder/'truth.npz')))
    writecsv(dest/'cell_level_outputs.csv',cells)
    writej(dest/'result.json',r)
    files=[f.name for f in dest.iterdir() if f.is_file() and f.suffix in ['.json','.csv'] and f.name!='EVALUATION_DONE.json']
    writej(dest/'EVALUATION_DONE.json',dict(utc=now(),prediction_manifest_sha256=sha(out/'PREDICTIONS_FROZEN.json'),files={f:sha(dest/f) for f in files}))
    if diagnostics:oracle(out,dest,pred,truth,r)
    print('EVALUATED',dataset,seed,arm,flush=True)

def oracle(out,dest,pred,truth,result):
    rows=[];z=pred['embedding'];labels=truth['true_label'];novel=truth['true_novel'].astype(bool);p=pred['pred_novel'].astype(bool)
    for population,mask in [('full_PredNovel',p),('GT_novel_oracle_routing',novel)]:
        y=frozen.dense_labels(labels[mask]);k=len(np.unique(y))
        if len(y)<2 or k<2:
            for method in ['no_K_Leiden','oracle_K_KMeans']:rows.append(dict(population=population,grouping=method,n_cells=len(y),true_K=k,ami=None,ari=None,nmi=None,na_reason='insufficient population/classes',oracle=True))
            continue
        leiden=pred['cluster_id'][mask] if population=='full_PredNovel' else frozen.frozen_leiden(z[mask])
        s=scores(y,leiden);rows.append(dict(population=population,grouping='no_K_Leiden',n_cells=len(y),true_K=k,**s,oracle=population!='full_PredNovel'))
        km=frozen.frozen_kmeans(z[mask],y,k);rows.append(dict(population=population,grouping='oracle_K_KMeans',n_cells=len(y),true_K=k,**{m:km[m] for m in ['ami','ari','nmi','weighted','macro']},oracle=True))
        if population=='GT_novel_oracle_routing':
            source_rows=pred['source_rows'];lookup={int(v):i for i,v in enumerate(source_rows)}
            kp=np.asarray([lookup[int(v)] for v in pred['known_prediction'][~novel]])
            kt=np.asarray([lookup.get(int(v),-1) for v in truth['true_canonical_row'][~novel]])
            if np.all(kt>=0):
                overall=frozen.cluster_acc(np.r_[kp,km['pred']+len(source_rows)],np.r_[kt,y+len(source_rows)])
                rows.append(dict(population='all_target_oracle_routing',grouping='frozen_OverallJ',OverallJ=overall,oracle=True,n_cells=len(z)))
    writecsv(dest/'oracle_routing_K_decomposition.csv',rows)
    writej(dest/'DIAGNOSTICS_DONE.json',dict(utc=now(),files={'oracle_routing_K_decomposition.csv':sha(dest/'oracle_routing_K_decomposition.csv')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--skip-oracle',action='store_true');a=p.parse_args();torch.set_num_threads(4);evaluate(a.run_dir,not a.skip_oracle)
