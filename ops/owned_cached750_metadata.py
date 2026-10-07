"""Metadata-only controls. Never load a model, mutate metadata or sign a request."""
import hashlib,json
from pathlib import Path
ALLOWED=frozenset(('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','merges.txt','vocab.json'))
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def fetch_metadata(bucket,checkpoint,directory):
 """Caller authenticates ROOT descriptor; every GET is guarded by allowlist."""
 directory=Path(directory)
 if directory.is_symlink():raise ValueError('private metadata namespace')
 directory.mkdir(mode=0o700,parents=True,exist_ok=True);receipts={}
 for name,expected in checkpoint['files'].items():
  if name not in ALLOWED:continue
  key='public/checkpoints/'+checkpoint['id']+'/'+name
  raw=bucket.get(key)
  if len(raw)>32*1024**2 or hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('full metadata SHA/size bound')
  path=directory/name
  if path.exists():
   if path.is_symlink()or path.read_bytes()!=raw:raise ValueError('immutable original metadata differs')
  else:path.write_bytes(raw);path.chmod(0o600)
  receipts[name]=dict(key=key,sha256=expected,full_read_bytes=len(raw))
 return receipts

def compare_all750(base,current,base_config,current_config,session,groups,render):
 """Use authentic frozen native reset/render supplied by the checked CPU caller."""
 indices=[i for group in groups for i in group['indices']]
 if len(groups)!=24 or [len(g['indices'])for g in groups]!=[32]*23+[14]:raise ValueError('complete24-group750 population')
 if len(indices)!=750 or len(set(indices))!=750:raise ValueError('complete unique750 population')
 rows=[]
 for group in groups:
  if group['seeds']!=[20261002+i*1000 for i in group['indices']]:raise ValueError('exact predeclared seeds')
  for index,seed in zip(group['indices'],group['seeds']):
   initial=session.reset(index,0)
   b=render(base,initial['messages'],initial.get('tools',[]),group['harness']);a=render(current,initial['messages'],initial.get('tools',[]),group['harness'])
   rows.append(dict(index=index,seed=seed,task_hash=initial['task_hash'],base_prompt_sha256=hashlib.sha256(canonical(b)).hexdigest(),current_prompt_sha256=hashlib.sha256(canonical(a)).hexdigest(),prompt_tokens=len(b),equal=b==a))
 ignored={'_name_or_path','transformers_version','torch_dtype','dtype'}
 differences={k:dict(base=base_config.get(k),current=current_config.get(k))for k in set(base_config)|set(current_config)if base_config.get(k)!=current_config.get(k)}
 semantic={k:v for k,v in differences.items()if k not in ignored}
 result=dict(tasks=750,all750_prompt_tokens_equal=all(r['equal']for r in rows),vocabulary_id_maps_equal=base.get_vocab()==current.get_vocab(),special_tokens_map_equal=base.special_tokens_map==current.special_tokens_map,eos_equal=base.eos_token_id==current.eos_token_id,chat_template_equal=base.chat_template==current.chat_template,inference_config_differences=semantic,serialization_config_differences={k:v for k,v in differences.items()if k in ignored},rows=rows)
 result['passed']=all(result[k]for k in ('all750_prompt_tokens_equal','vocabulary_id_maps_equal','special_tokens_map_equal','eos_equal'))and not semantic
 return result
