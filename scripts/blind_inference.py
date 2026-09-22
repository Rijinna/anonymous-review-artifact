"""Prediction/partition/LCC only. No truth access or evaluator imports."""
import sys
from common import *
sys.path.insert(0,str(ROOT/'scolar/frozen'))
import numpy as np, torch
from deployment_partition import common_leiden

def lcc_partition(source_logits,partition,prednovel,onto,source_rows,thresholds=(.25,.30,.40,.50,.60,.70)):
    records=[];node=np.full(len(partition),'',dtype='<U64')
    depths=onto.depths.detach().cpu();compute_device=onto.ancestor_matrix.device
    anc=onto.ancestor_matrix.float()[torch.as_tensor(source_rows,device=compute_device)]
    # Match production: source softmax and projection on model device, then CPU means.
    probs=torch.softmax(torch.as_tensor(source_logits,dtype=torch.float32,device=compute_device)/2.,dim=1)
    projected=(probs@anc).cpu()
    for cid in np.unique(partition[prednovel]):
        mask=(partition==cid)&prednovel;n=int(mask.sum());eligible=n>=5 and int(prednovel.sum())>=10
        base=dict(cluster_id=int(cid),n_cells=n,eligible=eligible,assigned=False,lcc_node='',lcc_depth=None,top3_coverage=0.,top1_score=0.,reason='cluster<5 or total PredNovel<10')
        if eligible:
            # Production computes per-cell projection then mean; retain operation order.
            avg=projected[mask].mean(0);filtered=avg*(depths>=2).float();nv=int((filtered>0).sum())
            if nv:
                scores,idx=torch.topk(filtered,min(3,nv));coverage=float(scores.sum()/(filtered.sum()+1e-8))
                deepest=int(idx[int(np.argmax(depths[idx].numpy()))])
                base.update(top3_coverage=coverage,top1_score=float(scores[0]),candidate_node=onto.all_terms[deepest],candidate_depth=int(depths[deepest]),
                            top3_nodes=[onto.all_terms[int(i)] for i in idx],reason='coverage below .40')
                if coverage>=.4:base.update(assigned=True,lcc_node=onto.all_terms[deepest],lcc_depth=int(depths[deepest]),reason='assigned');node[mask]=base['lcc_node']
            else:base['reason']='no positive probability at depth >=2'
        base['sensitivity']={str(c):bool(eligible and base.get('candidate_node') and base['top3_coverage']>=c) for c in thresholds}
        records.append(base)
    return records,node

def save_outputs(model,onto,alpha,source_rows,blind,out,metadata):
    model.eval();zs=[];mls=[];sl=[]
    device=next(model.parameters()).device;rows=source_rows.detach().cpu().numpy()
    with torch.no_grad():
        for x in torch.split(blind['target_tensors'][0],1024):
            r=model(x.to(device));logits=r['logits_fine'].cpu();zs.append(r['z'].cpu().numpy());mls.append(logits.max(1).values.numpy());sl.append(logits[:,rows].numpy())
    z=np.concatenate(zs);maximum=np.concatenate(mls);source_logits=np.concatenate(sl)
    assert np.isfinite(z).all() and np.isfinite(maximum).all() and np.isfinite(alpha)
    prednovel=maximum<float(alpha);partition=np.full(len(z),-1,dtype=np.int64)
    if prednovel.any():partition[prednovel]=common_leiden(z[prednovel],resolution=1.,n_neighbors=15,random_state=0)
    part_hash=arrsha(partition);membership_hash=arrsha(prednovel)
    records,lccnodes=lcc_partition(source_logits,partition,prednovel,onto,rows)
    assert arrsha(partition)==part_hash and arrsha(prednovel)==membership_hash
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',cell_ids=np.asarray(blind['target_cell_ids'],dtype=str),embedding=z,mls=maximum,
        novelty_score=-maximum,threshold=np.array(-float(alpha)),alpha=np.array(float(alpha)),pred_novel=prednovel,
        known_prediction=rows[source_logits.argmax(1)],cluster_id=partition,lcc_node=lccnodes,source_logits=source_logits,source_rows=rows)
    writej(out/'lcc.json',dict(condition=onto.condition,records=records,thresholds=[.25,.30,.4,.5,.6,.7],temperature=2.,min_depth=2,min_cluster_size=5,min_total_prednovel=10))
    writej(out/'run_metadata.json',dict(metadata,split_sha256=blind['split_sha256'],preprocessing_sha256=blind['preprocessing_sha256'],
        membership_sha256=membership_hash,partition_sha256=part_hash,source_cell_order_sha256=objsha(blind['source_cell_ids']),target_cell_order_sha256=objsha(blind['target_cell_ids']),
        ancestor_view_sha256=arrsha(onto.ancestor_matrix.cpu().numpy()),fine_to_coarse_sha256=arrsha(onto.fine_to_coarse_mask.cpu().numpy()),depth_sha256=arrsha(onto.depths.cpu().numpy()),
        ontology_mapping_seed=onto.mapping_seed,canonical_to_view=onto.canonical_to_view,target_truth_access=False))
    paths=['predictions.npz','lcc.json','run_metadata.json','final_model.pth','initial_model.json','final_model.calibration_draws.json','materialized_config.json']
    writej(out/'PREDICTIONS_FROZEN.json',dict(utc=now(),files={p:sha(out/p) for p in paths},target_truth_access=False))
    writej(out/'PREDICTION_DONE.json',dict(utc=now(),manifest_sha256=sha(out/'PREDICTIONS_FROZEN.json')))
