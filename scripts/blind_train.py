"""100-epoch frozen production training with a blind-input and ontology-view adapter."""
import os,sys,argparse,time,tempfile
from pathlib import Path
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(tempfile.gettempdir())/'scolar_numba'));os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'scolar_mpl'))
from common import *
sys.path.insert(0,str(ROOT/'scolar/frozen'))
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset,RandomSampler
import train_core as core
from ontology_adapter import view
from blind_inference import save_outputs

class IsolatedLegacySampler(RandomSampler):
    stream=None
    def __iter__(self):
        if self.stream is None:
            yield from super().__iter__()
        else:
            seed=int(torch.empty((),dtype=torch.int64).random_(generator=self.stream).item())
            generator=torch.Generator();generator.manual_seed(seed)
            yield from iter(RandomSampler(self.data_source,replacement=self.replacement,num_samples=self.num_samples,generator=generator))

def loaders(blind):
    source=TensorDataset(*blind['source_tensors']);target=TensorDataset(*blind['target_tensors'])
    ls=[DataLoader(source,batch_size=1024,sampler=IsolatedLegacySampler(source),num_workers=0),DataLoader(source,batch_size=1024,shuffle=False,num_workers=0),
        DataLoader(target,batch_size=1024,shuffle=False,num_workers=0),DataLoader(target,batch_size=1024,sampler=IsolatedLegacySampler(target),num_workers=0)]
    return (*ls,source.tensors[0].shape[1],blind['unique_src_ids'],None,None)

def configure_rng(*loaders):
    # Clone the exact post-initialization CPU RNG state used by production loaders.
    # No subsequent training CPU random draws exist; CUDA dropout/HCL is separate.
    g=torch.Generator(device='cpu');g.set_state(torch.get_rng_state())
    for loader in loaders:
        loader.generator=g
        if isinstance(loader.sampler,IsolatedLegacySampler):loader.sampler.stream=g

def run(args):
    out=ROOT/('smoke' if args.smoke else 'runs')/args.dataset/f'seed{args.seed}'/args.arm
    out.mkdir(parents=True,exist_ok=True)
    if (out/'PREDICTION_DONE.json').exists():
        frozen=json.loads((out/'PREDICTIONS_FROZEN.json').read_text());assert all(sha(out/p)==h for p,h in frozen['files'].items());return
    if (out/'final_model.pth').exists():raise RuntimeError('incomplete existing attempt: preserve and use new attempt directory')
    split=ROOT/'splits'/args.dataset/f'seed{args.seed}'
    meta=json.loads((split/'metadata.json').read_text());assert sha(split/'blind.pt')==meta['blind_sha256']
    blind=torch.load(split/'blind.pt',map_location='cpu',weights_only=False)
    assert len(blind['target_tensors'])==3 and len(blind['source_tensors'])==5
    assert torch.cuda.is_available(),'GPU required; refusing silent CPU training'
    torch.set_num_threads(4)
    ontology=view(args.dataset,args.arm,blind['unique_src_ids'].tolist(),'cuda')
    core.OntologyManager=lambda *a,**k:ontology
    core.prepare_datasets_robust=lambda *a,**k:loaders(blind)
    core.configure_loader_rng=configure_rng
    def record(model,cfg):
        params={k:arrsha(p.detach().cpu().numpy()) for k,p in model.named_parameters()}
        writej(out/'initial_model.json',dict(parameter_hashes=params,parameter_count=sum(p.numel() for p in model.parameters()),
                 cpu_rng_sha256=arrsha(torch.get_rng_state().numpy()),cuda_rng_sha256=arrsha(torch.cuda.get_rng_state().cpu().numpy())))
    core.record_initial_model=record
    config=core.make_parser().parse_args(['--dataset',args.dataset,'--seed',str(args.seed),'--output_dir',str(out),
         '--ontology_path',str(PROJECT/'data/cl.pt'),'--epochs','100','--lr_schedule_epochs','100','--no-posthoc_oracle_diagnostic'])
    # A full 100-epoch one-block smoke exercises every schedule transition; kept separate.
    writej(out/'materialized_config.json',dict(vars(config),ontology_condition=args.arm,mapping_seed=ontology.mapping_seed))
    started=time.monotonic()
    model,ontology,alpha,rows,plan,train_seconds=core.train(config)
    metadata=dict(dataset=args.dataset,seed=args.seed,arm=args.arm,smoke=args.smoke,checkpoint_epoch=100,training_seconds=train_seconds,
            protocol_sha256=sha(PROTOCOL),runtime_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),start_utc=now())
    save_outputs(model,ontology,alpha,rows,blind,out,metadata)
    writej(out/'runtime.json',dict(total_seconds=time.monotonic()-started,training_seconds=train_seconds,end_utc=now()))
    print('BLIND RUN COMPLETE',args.dataset,args.seed,args.arm,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--smoke',action='store_true');run(p.parse_args())
