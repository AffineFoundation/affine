"""Bounded capability-only hydration of explicitly authorized corpus shard assets.

Call only after signed job/manifest/source admission. This function authenticates
asset bytes against that admitted descriptor; it does not authenticate a signature.
"""
import gzip,hashlib,io,json,os,re,tempfile
from pathlib import Path
VERSION='original-math-corpus-shard-v1'
MAX_COMPRESSED=16*1024**2
MAX_RAW=32*1024**2

def validate_asset(binding):
 for field,limit in [('size',MAX_RAW),('compressed_size',MAX_COMPRESSED)]:
  if type(binding.get(field)) is not int or not 0<binding[field]<=limit:raise ValueError('task asset budget')
 for field in ('sha256','compressed_sha256'):
  if not isinstance(binding.get(field),str) or re.fullmatch('[0-9a-f]{64}',binding[field]) is None:raise ValueError('task asset digest')
 if binding.get('path')!='assets/math-corpora/'+binding['sha256']+'.tasks.json':raise ValueError('task asset path')

def admit_bytes(body,binding):
 validate_asset(binding)
 if len(body)!=binding['compressed_size'] or hashlib.sha256(body).hexdigest()!=binding['compressed_sha256']:raise ValueError('compressed task asset integrity')
 with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:raw=stream.read(binding['size']+1)
 if len(raw)!=binding['size'] or hashlib.sha256(raw).hexdigest()!=binding['sha256']:raise ValueError('raw task asset integrity')
 rows=json.loads(raw)
 if not isinstance(rows,list) or len(rows)!=binding['rows']:raise ValueError('task asset row count')
 from .math_corpus import SYSTEM
 for row in rows:
  if not isinstance(row,dict) or set(row)!={'task_class','task_config','data'} or row['task_class']!='MathTask':raise ValueError('task asset class')
  data=row['data']
  if data.get('system_prompt')!=SYSTEM or not isinstance(data.get('problem'),str) or data.get('prompt')!=data['problem'] or not isinstance(data.get('answer'),str):raise ValueError('question-only task asset')
 return raw

def hydrate(root,binding,url,fetch=None):
 validate_asset(binding)
 root=Path(root).resolve();path=root/binding['path']
 # Existing aliases or malformed immutable caches fail; never overwrite them.
 cursor=path
 while cursor!=root:
  if cursor.is_symlink():raise ValueError('task asset symlink')
  cursor=cursor.parent
 if path.exists():
  if not path.is_file() or path.stat().st_size!=binding['size'] or hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:raise ValueError('existing task asset mismatch')
  return path
 from .source_bootstrap import download,r2_url
 r2_url(url)
 body=(fetch or download)(url,binding['compressed_size']);raw=admit_bytes(body,binding)
 path.parent.mkdir(parents=True,exist_ok=True)
 fd,name=tempfile.mkstemp(prefix='.hydrate-',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
  os.chmod(name,0o444);os.replace(name,path)
 finally:
  if os.path.exists(name):os.unlink(name)
 return path
