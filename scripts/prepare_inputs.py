"""Label-access authority. Materialize the original mutually-exclusive splits once.
Only this process and the later evaluator may open truth; trainer sees blind.pt.
"""
import os, sys, argparse, gc, re, collections, tempfile
from pathlib import Path
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(tempfile.gettempdir())/'scolar_numba'))
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'scolar_mpl'))
os.environ.setdefault('OMP_NUM_THREADS','4')
from common import *
sys.path.insert(0,str(ROOT/'scolar/frozen'))
import numpy as np, torch, scanpy as sc
import preparation_authority as orig
import train_core
from utils import OntologyManager, norm

def indices(labels,dataset,seed):
    shared,sp,novel=orig.split_open_set_protocol(labels,dataset_name=dataset,seed=seed)
    rng=np.random.RandomState(seed); s=[];t=[]
    for i,c in enumerate(labels):
        if c in shared: (s if rng.rand()<.5 else t).append(i)
        elif c in sp:s.append(i)
        elif c in novel:t.append(i)
    return np.array(s,dtype=np.int64),np.array(t,dtype=np.int64),shared,sp,novel

def trusted_mapping(obs,onto,dataset):
    import obonet
    graph=obonet.read_obo(PROJECT/'data/cl.obo')
    candidates=collections.defaultdict(set)
    for term,d in graph.nodes(data=True):
        if term not in onto.id2idx:continue
        if d.get('name'):candidates[norm(d['name'])].add(term)
        for synonym in d.get('synonym',[]):
            m=re.match(r'"(.*?)"\s+EXACT\b',synonym)
            if m:candidates[norm(m.group(1))].add(term)
    aliases={'early embryonic cell':'CL:0000007','embryonic cell':'CL:0002321'} if dataset=='Wagner' else {}
    result={}
    for label,g in obs.groupby('cell_ontology_class',observed=True):
        label=str(label);reason='';term='';ids=[]
        if 'cell_ontology_id' in g:
            ids=sorted({str(v) for v in g.cell_ontology_id.dropna()})
        if len(ids)==1 and ids[0] in onto.id2idx:
            term=ids[0];reason='preexisting unique valid cell_ontology_id; no outcome-informed edits'
        elif norm(label) in candidates and len(candidates[norm(label)])==1:
            term=next(iter(candidates[norm(label)]));reason='unique preexisting CL canonical name or EXACT synonym'
        elif label in aliases:
            term=aliases[label];reason='preexisting production dataset-scoped curated alias'
        else:reason='NOT_EVALUABLE: absent/ambiguous ID and no unique exact-name/synonym rule'
        result[label]=dict(term=term,reason=reason,preexisting_ids=ids,n_cells=len(g))
    return result,graph.graph.get('data-version','unknown')

def prepare(dataset):
    marker=ROOT/'splits'/dataset/'PREPARED.json'
    if marker.exists():
        info=json.loads(marker.read_text())
        assert all(sha(p)==h for p,h in info['files'].items())
        print(dataset,'verified resume');return
    data=PROJECT/'data'/f'{dataset}.h5ad'
    ledger(dataset,'all','split_and_mapping_authority',data,'obs names/class/ontology_id/donor/sample/batch','freeze registry-exclusive split and sealed evaluation mapping; no model outputs')
    ad=sc.read_h5ad(data,backed='r');obs=ad.obs.copy();ad.file.close();del ad
    labels=obs['cell_ontology_class'].astype(str).to_numpy();cell_ids=np.asarray(obs.index.astype(str))
    if len(set(cell_ids))!=len(cell_ids):raise RuntimeError('duplicate stable cell IDs')
    onto=OntologyManager(PROJECT/'data/cl.pt',device='cpu',coarse_depth_threshold=3)
    gtmap,obo_release=trusted_mapping(obs,onto,dataset)
    writej(ROOT/'ontology'/f'sealed_gt_{dataset}.json',dict(dataset=dataset,obo_release=obo_release,mapping=gtmap,ontology_sha256=sha(PROJECT/'data/cl.pt')))
    args=train_core.make_parser().parse_args(['--dataset',dataset,'--seed','101','--ontology_path',str(PROJECT/'data/cl.pt'),'--no-posthoc_oracle_diagnostic'])
    captured=[];reader=orig._read_h5ad_fail_closed
    def tracked(path):
        ledger(dataset,101,'original_preprocessing_authority',path,'expression and obs class','exact frozen preprocessing and original split used once; no predictions exist')
        a=reader(path);captured.append(a);return a
    orig._read_h5ad_fail_closed=tracked
    out=orig.prepare_datasets_robust(args,onto)
    orig._read_h5ad_fail_closed=reader
    s0,t0,shared,sp,novel=indices(labels,dataset,101)
    tensors_s=out[0].dataset.tensors;tensors_t=out[2].dataset.tensors
    assert len(s0)==len(tensors_s[0]) and len(t0)==len(tensors_t[0])
    X=[]
    for j in range(3):
        a=torch.empty((len(labels),)+tensors_s[j].shape[1:],dtype=tensors_s[j].dtype)
        a[s0]=tensors_s[j];a[t0]=tensors_t[j];X.append(a)
    used=set(s0.tolist()+t0.tolist())
    canonical,valid=onto.map_labels(labels)
    if dataset=='Wagner':
        for label,term in {'early embryonic cell':'CL:0000007','embryonic cell':'CL:0002321'}.items():
            canonical[labels==label]=onto.id2idx[term];valid[labels==label]=True
    canonical[np.isin(labels,list(novel))]=-1
    assert torch.equal(canonical[s0],tensors_s[3])
    assert torch.equal(canonical[t0],tensors_t[3])
    source_label_by_row={}
    for row,label in zip(canonical[s0].tolist(),labels[s0]):
        if row in source_label_by_row and source_label_by_row[row]!=label:raise RuntimeError('multiple source classes share classifier row; adapter must be revised before freeze')
        source_label_by_row[row]=str(label)
    pre=dict(dataset=dataset,h5ad_sha256=sha(data),hvg_genes=[str(v) for v in captured[0].var_names[captured[0].var.highly_variable]],
             expression_fit='original transductive all-input expression normalize_total1e4/log1p/HVG2000/zscore/clip10; original size factors',
             tensors_sha256=[arrsha(a.numpy()[sorted(used)]) for a in X])
    pre_sha=objsha(pre);writej(ROOT/'splits'/dataset/'preprocessing.json',pre)
    files={}
    for seed in SEEDS:
        s,t,shared,sp,novel=indices(labels,dataset,seed)
        assert not np.intersect1d(s,t).size and set(labels[s]).isdisjoint(novel)
        assert set(s.tolist()+t.tolist())==used
        valid_s=valid[s];assert bool(valid_s.all())
        src=tuple(a[s] for a in X)+(canonical[s],torch.ones(len(s),dtype=torch.bool))
        tgt=tuple(a[t] for a in X)
        split_sha=objsha(dict(source=cell_ids[s].tolist(),target=cell_ids[t].tolist()))
        dst=ROOT/'splits'/dataset/f'seed{seed}';dst.mkdir(parents=True,exist_ok=True)
        blind=dict(source_tensors=src,target_tensors=tgt,source_cell_ids=cell_ids[s].tolist(),target_cell_ids=cell_ids[t].tolist(),
                   unique_src_ids=torch.unique(canonical[s]),source_label_by_row=source_label_by_row,split_sha256=split_sha,preprocessing_sha256=pre_sha)
        torch.save(blind,dst/'blind.pt')
        np.savez_compressed(dst/'truth.npz',cell_ids=cell_ids[t],true_label=labels[t],true_label_id=out[7].transform(labels[t]),
             true_novel=np.isin(labels[t],list(novel)),true_canonical_row=canonical[t].numpy(),
             true_cl_term=np.asarray([gtmap[l]['term'] for l in labels[t]],dtype=str))
        overlaps={}
        for c in obs.columns:
            if any(k in c.lower() for k in ['donor','sample','batch','individual']):
                ss=set(obs.iloc[s][c].dropna().astype(str));tt=set(obs.iloc[t][c].dropna().astype(str))
                overlaps[c]=dict(n_source=len(ss),n_target=len(tt),n_overlap=len(ss&tt),note='cell-exclusive benchmark; group-exclusive split not claimed')
        meta=dict(dataset=dataset,seed=seed,n_source=len(s),n_target=len(t),split_sha256=split_sha,preprocessing_sha256=pre_sha,
                  source_class_rows=sorted(source_label_by_row),source_label_by_row=source_label_by_row,
                  source_cell_order_sha256=objsha(cell_ids[s].tolist()),target_cell_order_sha256=objsha(cell_ids[t].tolist()),
                  source_target_cell_overlap=0,target_private_in_source=0,group_overlap=overlaps,
                  group_metadata_missing=not bool(overlaps),blind_sha256=sha(dst/'blind.pt'),truth_sha256=sha(dst/'truth.npz'))
        writej(dst/'metadata.json',meta)
        for fn in ['blind.pt','truth.npz','metadata.json']:files[str(dst/fn)]=sha(dst/fn)
    # Exact 202 split cross-check against the original path, one small dataset only.
    if dataset=='Cao':
        args.seed=202
        ledger(dataset,202,'preprocessing_equivalence_test',data,'expression/obs class','test reconstructed frozen tensors against direct original function')
        direct=orig.prepare_datasets_robust(args,onto)
        saved=torch.load(ROOT/'splits'/dataset/'seed202/blind.pt',weights_only=False)
        assert all(torch.equal(a,b) for a,b in zip(direct[0].dataset.tensors,saved['source_tensors']))
        assert all(torch.equal(a,b) for a,b in zip(direct[3].dataset.tensors,saved['target_tensors']))
        writej(ROOT/'audit/PREPROCESSING_EQUIVALENCE.json',dict(status='PASS',dataset='Cao',seed=202,bitwise_equal=True))
    writej(marker,dict(status='PREPARED',utc=now(),files=files,preprocessing_sha256=pre_sha))
    print(dataset,'ALL FOUR SPLITS PREPARED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS);a=p.parse_args()
    torch.set_num_threads(4)
    for dataset in ([a.dataset] if a.dataset else DATASETS):prepare(dataset);gc.collect()
