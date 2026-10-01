"""Pre-import guard for a separately approved Wikispeedia public-graph GPU probe."""
import argparse,base64,hashlib,json,os,runpy,sys
from pathlib import Path
from nacl.signing import VerifyKey

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def admit(document,authority,root):
    if document.get('signer')!=authority:raise ValueError('resource authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),base64.b64decode(document['signature'],validate=True))
    p=document['payload'];root=Path(root).resolve()
    if p.get('revision')!='original-wikispeedia-window-model-resources-v1' or p.get('remote')!=str(root) or p.get('model_execution')is not True or p.get('optimizer_ran')is not False or p.get('chain_transactions')is not False or p.get('provider_namespace_controlled')is not True or p.get('full_transitive_closure_claimed')is not False:raise ValueError('bounded controlled model resource scope')
    expected=p.get('files');actual={}
    if not isinstance(expected,dict)or not expected:raise ValueError('resource file membership')
    for name in ['source','dependencies','resources','operator','worker.py']:
        q=root/name
        if not q.exists()or q.is_symlink():raise ValueError('resource root')
        for path in ([q]if q.is_file()else q.rglob('*')):
            if path.is_symlink():raise ValueError('resource symlink')
            if path.is_file():actual[str(path.relative_to(root))]=path
    if set(actual)!=set(expected):raise ValueError('exact controlled resource membership')
    for name,path in actual.items():
        record=expected[name]
        if path.stat().st_size!=record['size']or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('approved resource bytes')
    if any(name in sys.modules for name in ['wikispeedia_v1','affine_wikispeedia_v1','verifiers','subnet','torch']):raise ValueError('provider or model imported before admission')
    required={'WIKISPEEDIA_CACHE_DIR':str(root/'resources/wikispeedia'),'XDG_CACHE_HOME':str(root/'private-cache'),'XDG_DATA_HOME':str(root/'private-data'),'MPLCONFIGDIR':str(root/'private-cache/matplotlib'),'PYTHONPATH':[str(root/'dependencies'),str(root/'source')]}
    if p.get('resource_environment')!=required:raise ValueError('qualified private resource environment')
    return p

def main():
    if not sys.flags.isolated or not sys.dont_write_bytecode:raise ValueError("fresh isolated no-bytecode worker required (-I -B)")
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True,type=Path);p.add_argument('--authority',required=True);p.add_argument('--root',required=True,type=Path);p.add_argument('--verify',action='store_true');a=p.parse_args();profile=admit(json.loads(a.profile.read_bytes()),a.authority,a.root);root=a.root.resolve();env=profile['resource_environment']
    for k in ['WIKISPEEDIA_CACHE_DIR','XDG_CACHE_HOME','XDG_DATA_HOME','MPLCONFIGDIR']:os.environ[k]=env[k]
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    # The unchanged approved GPU runtime applies its signed numerical/thread profile.
    sys.dont_write_bytecode=True;sys.path[:0]=env['PYTHONPATH'];os.chdir(root/'source');sys.argv=[str(root/'worker.py')]+(['--verify']if a.verify else[]);runpy.run_path(str(root/'worker.py'),run_name='__main__')
if __name__=='__main__':main()
