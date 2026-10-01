"""Fixed-registry role worker; binary stdout is exclusively framed JSON."""
import base64,contextlib,hashlib,io,json,pathlib,secrets,sys
from .native_role_process import VERSION,canonical,read_frame,write_frame,tokens

class Worker:
    def __init__(self,runtime,descriptor):
        if canonical(runtime.approved_descriptor())!=canonical(descriptor):raise ValueError('approved worker runtime descriptor')
        self.runtime=runtime;self.descriptor=descriptor;self.handle=None;self.activations=None
    def dispatch(self,r):
        if not isinstance(r,dict) or type(r.get('id')) is not int or r['id']<1:raise ValueError('request identity')
        command=r.get('command')
        fields={'compute':{'prompt','output'},'sample':{'prompt','seed','temperature','top_p','max_tokens'},'build_proofs':{'handle','decode_batching_size','topk'},'verify_proofs':{'handle','decode_batching_size','topk','proofs'},'tokenizer':None}
        if command not in fields:raise ValueError('command allowlist')
        if canonical(self.runtime.approved_descriptor())!=canonical(self.descriptor):raise ValueError('runtime descriptor drift')
        if command=='tokenizer':return self.tokenizer(r)
        if set(r)!={'id','command'}|fields[command]:raise ValueError('exact command fields')
        if command in ('compute','sample'):
            prompt=tokens(r['prompt'],self.descriptor)
            if not prompt:raise ValueError('empty prompt')
        if command=='compute':
            import numpy as np
            output=tokens(r['output'],self.descriptor,self.descriptor['max_output_tokens'])
            if not output or len(prompt)+len(output)>self.descriptor['max_context']:raise ValueError('complete context budget')
            self.handle=None;self.activations=None
            acts,array=self.runtime.compute(prompt,output)
            if not isinstance(array,np.ndarray) or array.dtype!=np.float32 or array.shape!=(len(output),self.descriptor['vocab_size']) or not np.isfinite(array).all():raise ValueError('runtime full float32 probabilities')
            stream=io.BytesIO();np.save(stream,array,allow_pickle=False);raw=stream.getvalue()
            self.activations=acts;self.handle=secrets.token_hex(24)
            return {'handle':self.handle,'probabilities':{'npy_base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),'shape':list(array.shape),'dtype':'float32'}}
        if command=='sample':
            import math
            if type(r['seed']) is not int or not 0<=r['seed']<2**63 or type(r['max_tokens']) is not int or not 1<=r['max_tokens']<=self.descriptor['max_output_tokens'] or len(prompt)+r['max_tokens']>self.descriptor['max_context']:raise ValueError('sampling seed/budget')
            if any(type(r[k]) not in (int,float) or not math.isfinite(r[k]) for k in ('temperature','top_p')) or r['temperature']<=0 or not 0<r['top_p']<=1:raise ValueError('sampling policy')
            return tokens(self.runtime.sample(prompt,r['seed'],r['temperature'],r['top_p'],r['max_tokens']),self.descriptor,r['max_tokens'])
        if r['handle']!=self.handle or self.handle is None:raise ValueError('stale activation handle')
        if type(r['decode_batching_size']) is not int or r['decode_batching_size']!=16 or type(r['topk']) is not int or r['topk']!=128:raise ValueError('proof policy')
        if command=='build_proofs':return self.runtime.build_proofs(self.activations,decode_batching_size=16,topk=128)
        proofs=r['proofs']
        if not isinstance(proofs,list) or len(proofs)>33 or any(not isinstance(p,str) or not 0<len(p)<=16384 for p in proofs):raise ValueError('proof framing budget')
        checks=self.runtime.verify_proofs(self.activations,proofs,decode_batching_size=16,topk=128)
        import math
        normalized=[]
        for row in checks:
            count=getattr(row,'exp_mismatches');mean=float(row.mant_err_mean);median=float(row.mant_err_median)
            if isinstance(count,bool) or int(count)!=count or int(count)<0 or not math.isfinite(mean) or not math.isfinite(median):raise ValueError('proof result scalar framing')
            normalized.append({'exp_mismatches':int(count),'mant_err_mean':mean,'mant_err_median':median})
        return normalized
    def tokenizer(self,r):
        method=r.get('method');options=r.get('options')
        allowed={'encode':({'text'},{'add_special_tokens'}),'decode':({'tokens'},{'skip_special_tokens'}),'apply_chat_template':({'messages'},{'tokenize','add_generation_prompt'})}
        if method not in allowed:raise ValueError('tokenizer method allowlist')
        args,opts=allowed[method]
        if set(r)!={'id','command','method','options'}|args or not isinstance(options,dict) or set(options)!=opts or any(type(v) is not bool for v in options.values()):raise ValueError('tokenizer exact options')
        tokenizer=self.runtime.tokenizer
        if method=='encode':
            if not isinstance(r['text'],str) or len(r['text'].encode())>4_000_000:raise ValueError('text budget')
            return tokens(list(tokenizer.encode(r['text'],**options)),self.descriptor)
        if method=='decode':return tokenizer.decode(tokens(r['tokens'],self.descriptor),**options)
        messages=r['messages']
        if options!={'tokenize':True,'add_generation_prompt':True} or not isinstance(messages,list) or not messages or len(canonical(messages))>4_000_000:raise ValueError('complete chat template budget')
        result=tokenizer.apply_chat_template(messages,**options)
        if hasattr(result,'keys'):result=result['input_ids']
        return tokens(list(result),self.descriptor)

def serve(runtime,descriptor,input_stream,output_stream):
    """Trusted injected fixture entry for conformance; CLI has no factory option."""
    worker=Worker(runtime,descriptor)
    write_frame(output_stream,{'version':VERSION,'ready':True,'descriptor':descriptor})
    while True:
        try:request=read_frame(input_stream)
        except EOFError:return
        try:
            with contextlib.redirect_stdout(sys.stderr):result=worker.dispatch(request)
            response={'id':request['id'],'result':result}
        except Exception as error:
            # No model inputs, stack trace or credential-bearing paths in protocol.
            response={'id':request.get('id') if isinstance(request,dict) else None,'error':type(error).__name__}
        write_frame(output_stream,response)

def main():
    import argparse,importlib.metadata
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config_path=pathlib.Path(args.config)
    if config_path.is_symlink() or not config_path.is_file() or config_path.stat().st_mode&0o077 or config_path.stat().st_size>1_000_000:raise ValueError('private bounded operator config')
    config=json.loads(config_path.read_bytes())
    if set(config)!={'version','checkpoint','descriptor'} or config['version']!=VERSION:raise ValueError('operator startup configuration')
    descriptor=config['descriptor'];root=pathlib.Path(__file__).resolve().parent.parent
    for required in ('subnet/native_role_worker.py','subnet/native_role_process.py','subnet/native_tau2_role_runtime.py'):
        if required not in descriptor['source_files']:raise ValueError('worker source closure')
    for name,expected in descriptor['source_files'].items():
        path=root/name
        if pathlib.PurePosixPath(name).is_absolute() or '..' in pathlib.PurePosixPath(name).parts or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('approved source bytes')
    if hashlib.sha256(pathlib.Path(sys.executable).resolve().read_bytes()).hexdigest()!=descriptor['interpreter_sha256']:raise ValueError('approved interpreter')
    if any(importlib.metadata.version(name)!=version for name,version in descriptor['runtime_versions'].items()):raise ValueError('approved packages')
    input_stream=sys.stdin.buffer;output_stream=sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        from .native_tau2_role_runtime import load_role
        runtime=load_role(config['checkpoint'],descriptor)
    serve(runtime,descriptor,input_stream,output_stream)
if __name__=='__main__':main()
