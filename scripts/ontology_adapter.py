"""Canonical classifier rows are immutable; this object is a row-indexed ontology VIEW.
Its columns retain ontology node identity. It is not a rewritten CL graph.
"""
import sys, copy, collections
from common import *
sys.path.insert(0,str(ROOT/'scolar/frozen'))
import numpy as np, torch
from utils import OntologyManager

def build_mapping(dataset, source_label_by_row):
    ck=torch.load(PROJECT/'data/cl.pt',map_location='cpu',weights_only=False)
    depths=ck['structure']['depths'].numpy();terms=ck['structure']['all_terms']
    root=ck['mappings']['id2idx']['CL:0000000'];rows=sorted(map(int,source_label_by_row))
    groups=collections.defaultdict(list)
    for row in rows:groups[int(depths[row])].append(row)
    rng=np.random.default_rng(27101+DATASETS.index(dataset));mapping={};used=set();unshufflable=[]
    for depth,group in sorted(groups.items()):
        movable=[r for r in group if r!=root]
        if root in group:mapping[root]=root;used.add(root);unshufflable.append(root)
        if len(movable)>1:
            order=np.asarray(movable)[rng.permutation(len(movable))].tolist()
            for a,b in zip(order,order[1:]+order[:1]):mapping[a]=b;used.add(b)
        elif movable:
            row=movable[0]
            pool=[int(r) for r in np.flatnonzero(depths==depth) if r not in rows and r not in used and r!=root]
            if pool:other=int(rng.choice(pool));mapping[row]=other;used.add(other)
            else:mapping[row]=row;used.add(row);unshufflable.append(row)
    assert len(set(mapping.values()))==len(rows)
    assert sorted(depths[rows])==sorted(depths[list(mapping.values())])
    import igraph as ig
    edges=np.argwhere(ck['structure']['parent_matrix'].numpy())
    graph=ig.Graph(n=len(terms),edges=edges.tolist(),directed=True)
    observed=np.asarray([int(v[0]) if np.isfinite(v[0]) else -1 for v in graph.distances(target=root,mode='out')])
    assert np.array_equal(observed,depths),'compiled depth disagrees with shortest directed path to CL root'
    anc=ck['structure']['ancestor_matrix_incl_self']
    records=[]
    for row in rows:
        dest=mapping[row]
        records.append(dict(source_label=str(source_label_by_row.get(row,source_label_by_row.get(str(row)))),canonical_row=row,
            real_cl=terms[row],ontology_row=dest,shuffled_cl=terms[dest],depth=int(depths[row]),fixed_point=row==dest,
            unshufflable=row in unshufflable,real_ancestors=int(anc[row].sum()),shuffled_ancestors=int(anc[dest].sum()),
            real_parent_degree=graph.degree(row,mode='out'),shuffled_parent_degree=graph.degree(dest,mode='out')))
    dists=[]
    for name,rr in [('REAL_CL',rows),('DEPTH_SHUFFLED_CL',[mapping[r] for r in rows])]:
        ds=np.asarray(graph.distances(source=rr,target=rr,mode='all'))
        dists.extend(dict(arm=name,row_a=rows[i],row_b=rows[j],distance=float(ds[i,j])) for i in range(len(rows)) for j in range(i+1,len(rows)) if np.isfinite(ds[i,j]))
    return dict(dataset=dataset,mapping_seed=27101+DATASETS.index(dataset),root='CL:0000000',depth_definition='shortest directed root-to-node path (compiled child-to-parent graph traversed upward)',
                graph_sha256=sha(PROJECT/'data/cl.pt'),cl_release=ck['meta']['version'],canonical_to_view={str(k):v for k,v in mapping.items()},
                label_fixed_point_fraction=sum(r['fixed_point'] for r in records)/len(rows),records=records,pair_distances=dists)

def prepare_mappings(datasets=DATASETS):
    for dataset in datasets:
        meta=json.loads((ROOT/'splits'/dataset/'seed101/metadata.json').read_text())
        mapping=build_mapping(dataset,meta['source_label_by_row'])
        again=build_mapping(dataset,meta['source_label_by_row']);assert mapping==again
        writej(ROOT/'ontology'/f'{dataset}_primary_mapping.json',mapping)
        writecsv(ROOT/'ontology'/f'{dataset}_mapping_qc.csv',mapping['records'])
        writecsv(ROOT/'ontology'/f'{dataset}_path_distances.csv',mapping['pair_distances'])

def view(dataset,arm,source_rows,device='cpu'):
    o=OntologyManager(PROJECT/'data/cl.pt',device=device,coarse_depth_threshold=3)
    o.condition=arm;o.canonical_to_view={int(r):int(r) for r in source_rows};o.mapping_seed=None
    if arm=='DEPTH_SHUFFLED_CL':
        m=json.loads((ROOT/'ontology'/f'{dataset}_primary_mapping.json').read_text())
        o.canonical_to_view={int(k):int(v) for k,v in m['canonical_to_view'].items()};o.mapping_seed=m['mapping_seed']
        rows=torch.tensor(sorted(o.canonical_to_view),device=device)
        target=torch.tensor([o.canonical_to_view[int(r)] for r in rows.tolist()],device=device)
        original=o.ancestor_matrix.clone();o.ancestor_matrix[rows]=original[target]
        coarse=o.fine_to_coarse_mask.clone();o.fine_to_coarse_mask[rows]=coarse[target]
    elif arm=='GENERIC_STAR':
        root=o.id2idx['CL:0000000'];rows=list(map(int,source_rows));assert root not in rows
        o.ancestor_matrix=torch.zeros_like(o.ancestor_matrix)
        o.ancestor_matrix[root,root]=True
        o.depths=torch.full_like(o.depths,-1);o.depths[root]=0
        o.all_terms=[f'PADDING:{i}' for i in range(o.num_classes)]
        o.all_terms[root]='SYNTHETIC_ROOT'
        for row in rows:
            o.ancestor_matrix[row,root]=True;o.ancestor_matrix[row,row]=True;o.depths[row]=1;o.all_terms[row]=f'SYNTHETIC_LEAF:{row}'
        # Original coarse excludes root and self. A star has no valid coarse parent;
        # retain the original 773-column interface with all entries masked to zero.
        # Thus DBR uses zero coarse similarity; no artificial lineage is inserted.
        o.fine_to_coarse_mask=torch.zeros_like(o.fine_to_coarse_mask)
        o.id2name={i:name for i,name in enumerate(o.all_terms)}
    elif arm!='REAL_CL':raise ValueError(arm)
    return o

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=DATASETS)
    args=parser.parse_args()
    prepare_mappings([args.dataset] if args.dataset else DATASETS)
