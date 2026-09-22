from pathlib import Path
import hashlib, json, os, csv
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT
PROTOCOL=ROOT/'configs'/'protocol_locked.json'

try:
    import fcntl
except ImportError:  # Windows reviewer environments do not provide fcntl.
    fcntl=None
DATASETS=['Cao','Quake_10x','Quake_Smart-seq2','Wagner','Zeisel_2018']
SEEDS=[101,202,303,404]
ARMS=['REAL_CL','DEPTH_SHUFFLED_CL','GENERIC_STAR']
def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def objsha(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def arrsha(arr):
    import numpy as np
    a=np.asarray(arr)
    if a.dtype.kind in 'OUS': return objsha(a.tolist())
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()
def writej(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,p)
def append_json(p,v):
    with open(p,'a') as f:
        if fcntl is not None:fcntl.flock(f,fcntl.LOCK_EX)
        f.write(json.dumps(v,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        if fcntl is not None:fcntl.flock(f,fcntl.LOCK_UN)
def ledger(dataset,seed,stage,path,fields,reason):
    p=ROOT/'TARGET_LABEL_ACCESS_LEDGER.tsv'
    with p.open('a') as f:
        if fcntl is not None:fcntl.flock(f,fcntl.LOCK_EX)
        if f.tell()==0:f.write('utc\tpid\tdataset\tseed\tstage\tpath\tfields\treason\n')
        f.write('\t'.join(map(str,[now(),os.getpid(),dataset,seed,stage,path,fields,reason]))+'\n');f.flush()
        if fcntl is not None:fcntl.flock(f,fcntl.LOCK_UN)
def writecsv(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if fields is None: fields=list(dict.fromkeys(k for row in rows for k in row))
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,p)
