"""Operator-owned, JSON-only isolated role transport. No tensor unpickling."""
import base64,copy,hashlib,io,json,os,struct,subprocess,threading
from types import SimpleNamespace
VERSION='native-role-process-json-v1'
MAX_FRAME_BYTES=550_000_000

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read_frame(stream,limit=MAX_FRAME_BYTES):
    header=stream.read(4)
    if not header:raise EOFError('role worker closed')
    if len(header)!=4:raise ValueError('short frame header')
    size=struct.unpack('!I',header)[0]
    if not 0<size<=limit:raise ValueError('frame budget')
    chunks=[];remaining=size
    while remaining:
        block=stream.read(min(remaining,1048576))
        if not block:raise ValueError('short frame body')
        chunks.append(block);remaining-=len(block)
    return json.loads(b''.join(chunks),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def write_frame(stream,value,limit=MAX_FRAME_BYTES):
    raw=canonical(value)
    if not 0<len(raw)<=limit:raise ValueError('frame budget')
    stream.write(struct.pack('!I',len(raw)));stream.write(raw);stream.flush()
def tokens(value,descriptor,maximum=None):
    if not isinstance(value,list) or len(value)>(maximum or descriptor['max_context']) or any(type(t) is not int or not 0<=t<descriptor['vocab_size'] for t in value):raise ValueError('token geometry')
    return value

def unpack_array(value,rows,vocab):
    import numpy as np
    if not isinstance(value,dict) or set(value)!={'npy_base64','sha256','shape','dtype'} or value['shape']!=[rows,vocab] or value['dtype']!='float32':raise ValueError('array metadata')
    encoded=value['npy_base64'];maximum=rows*vocab*4+10000
    if not isinstance(encoded,str) or len(encoded)>4*((maximum+2)//3):raise ValueError('array encoded budget')
    raw=base64.b64decode(encoded,validate=True)
    if len(raw)>maximum or hashlib.sha256(raw).hexdigest()!=value['sha256']:raise ValueError('array byte binding')
    # Inspect shape/dtype BEFORE numpy allocates the declared array.
    stream=io.BytesIO(raw);version=np.lib.format.read_magic(stream)
    if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=10000)
    elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream,max_header_size=10000)
    else:raise ValueError('NPY version')
    if shape!=(rows,vocab) or fortran or dtype!=np.dtype('float32') or len(raw)-stream.tell()!=rows*vocab*4:raise ValueError('NPY preheader geometry')
    array=np.load(io.BytesIO(raw),allow_pickle=False)
    if not np.isfinite(array).all():raise ValueError('nonfinite probabilities')
    return array

class TokenizerProxy:
    def __init__(self,owner):self.owner=owner
    def encode(self,text,add_special_tokens=False):return self.owner._rpc('tokenizer',method='encode',text=text,options={'add_special_tokens':add_special_tokens})
    def decode(self,ids,skip_special_tokens=True):return self.owner._rpc('tokenizer',method='decode',tokens=ids,options={'skip_special_tokens':skip_special_tokens})
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=True):
        return self.owner._rpc('tokenizer',method='apply_chat_template',messages=messages,options={'tokenize':tokenize,'add_generation_prompt':add_generation_prompt})

class ProcessRoleRuntime:
    """argv/env are trusted operator configuration, never miner-supplied fields.

    stderr goes to a caller-owned private log; default DEVNULL prevents accidental
    context disclosure. Worker startup independently approves descriptor/source.
    """
    def __init__(self,argv,descriptor,env=None,stderr=None):
        if not isinstance(argv,(list,tuple)) or not argv or any(not isinstance(x,str) or not x for x in argv):raise ValueError('operator argv sequence')
        self.descriptor=copy.deepcopy(descriptor);self.lock=threading.Lock();self.counter=0;self.handle=None
        self.process=subprocess.Popen(list(argv),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr if stderr is not None else subprocess.DEVNULL,env={**os.environ,**(env or {})})
        try:
            ready=read_frame(self.process.stdout)
            if canonical(ready)!=canonical({'version':VERSION,'ready':True,'descriptor':self.descriptor}):raise ValueError('worker approved readiness')
        except BaseException:self.close();raise
        self.tokenizer=TokenizerProxy(self)
    def approved_descriptor(self):return copy.deepcopy(self.descriptor)
    def _rpc(self,command,**fields):
        with self.lock:
            self.counter+=1;request={'id':self.counter,'command':command,**fields}
            write_frame(self.process.stdin,request);response=read_frame(self.process.stdout)
            if not isinstance(response,dict) or response.get('id')!=self.counter or set(response)-{'id','result','error'} or ('result' in response)==('error' in response):raise ValueError('worker response framing')
            if 'error' in response:raise ValueError('role worker rejected: '+str(response['error']))
            return response['result']
    def compute(self,prompt,output):
        tokens(prompt,self.descriptor);tokens(output,self.descriptor,self.descriptor['max_output_tokens'])
        result=self._rpc('compute',prompt=prompt,output=output)
        array=unpack_array(result['probabilities'],len(output),self.descriptor['vocab_size'])
        self.handle=result['handle'];return self.handle,array
    def _current(self,handle):
        if not isinstance(handle,str) or handle!=self.handle:raise ValueError('stale activation handle')
    def build_proofs(self,handle,decode_batching_size=16,topk=128):
        self._current(handle);return self._rpc('build_proofs',handle=handle,decode_batching_size=decode_batching_size,topk=topk)
    def verify_proofs(self,handle,proofs,decode_batching_size=16,topk=128):
        self._current(handle)
        return [SimpleNamespace(**r) for r in self._rpc('verify_proofs',handle=handle,proofs=proofs,decode_batching_size=decode_batching_size,topk=topk)]
    def sample(self,prompt,seed,temperature,top_p,max_tokens):
        return self._rpc('sample',prompt=prompt,seed=seed,temperature=temperature,top_p=top_p,max_tokens=max_tokens)
    def close(self):
        process=getattr(self,'process',None)
        if process is not None:
            if process.poll() is None:process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            for stream in (process.stdin,process.stdout):
                if stream:stream.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
