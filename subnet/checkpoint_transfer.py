"""Bounded public checkpoint transport, with full-file hash admission.

Transport functions match the qualified operator hydration helper. This packaged
module imports no operator, GPU, model, sampler or training code.
"""
import hashlib,os,re,time
from pathlib import Path
from urllib.parse import parse_qs,unquote,urlparse
import requests
MAX_FILE=20*1024**3
MAX_TOTAL=64*1024**3

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
 return h.hexdigest()

def read_url(url,cp,name):
 p=urlparse(url);q=parse_qs(p.query)
 if p.scheme!='https' or not (p.hostname or '').endswith('.r2.cloudflarestorage.com') or p.username or p.password or p.fragment or p.port not in (None,443) or q.get('X-Amz-Algorithm')!=['AWS4-HMAC-SHA256'] or len(q.get('X-Amz-Signature',[]))!=1 or not re.fullmatch('[0-9a-f]{64}',q['X-Amz-Signature'][0]):raise ValueError('scoped direct signed R2 read')
 path=unquote(p.path)
 if '..' in path.split('/') or not path.endswith('/public/checkpoints/'+cp+'/'+name):raise ValueError('read URL exact checkpoint/file')
 return url

def download(session,url,partial,expected_bytes,expected_sha,deadline,clock=time.time,*,chunk_bytes=64*1024**2,max_retries=3):
 partial=Path(partial)
 if partial.is_symlink() or (partial.exists() and not partial.is_file()):raise ValueError('regular preserved partial')
 offset=partial.stat().st_size if partial.exists() else 0
 if offset>expected_bytes:raise ValueError('partial exceeds approved object')
 if offset==expected_bytes:
  if sha(partial)!=expected_sha:raise ValueError('preserve mismatching complete partial')
  return {'resumed_bytes':offset,'downloaded_bytes':0}
 if type(chunk_bytes)is not int or not 1<=chunk_bytes<=64*1024**2 or type(max_retries)is not int or not 0<=max_retries<=10:raise ValueError('bounded transfer settings')
 start=offset;failures=0
 while offset<expected_bytes:
  if clock()>=deadline:raise ValueError('read plan expired; preserve partial')
  end=min(expected_bytes-1,offset+chunk_bytes-1)
  # A fresh connection also bounds transport intermediaries that truncate a
  # long-lived connection after cumulative bytes across otherwise valid ranges.
  headers={'Accept-Encoding':'identity','Connection':'close','Range':'bytes='+str(offset)+'-'+str(end)}
  try:
   with session.get(url,headers=headers,stream=True,timeout=(30,180),allow_redirects=False) as response:
    if response.status_code>=500:response.raise_for_status()
    if response.status_code!=206 or response.headers.get('Content-Encoding','identity')!='identity':raise ValueError('exact GET/range status/encoding')
    if response.headers.get('Content-Range')!='bytes '+str(offset)+'-'+str(end)+'/'+str(expected_bytes):raise ValueError('exact resumed object range')
    length=response.headers.get('Content-Length')
    if length is not None and (not length.isdecimal() or int(length)!=end-offset+1):raise ValueError('exact remaining length')
    # Append only to retained same-object partial; each response is bounded.
    with partial.open('ab' if partial.exists() else 'xb') as stream:
     partial.chmod(0o600);written=offset
     for part in response.iter_content(1024**2):
      if clock()>=deadline:raise ValueError('read plan expired during transfer')
      if not part:continue
      if written+len(part)>end+1:raise ValueError('object range size bound')
      stream.write(part);written+=len(part)
     stream.flush();os.fsync(stream.fileno())
    if written!=end+1:raise requests.exceptions.ChunkedEncodingError('short bounded checkpoint range')
   offset=partial.stat().st_size
  except requests.RequestException:
   failures+=1
   if failures>max_retries:raise
   offset=partial.stat().st_size if partial.exists()else 0
 if partial.stat().st_size!=expected_bytes or sha(partial)!=expected_sha:raise ValueError('complete downloaded size/SHA mismatch; preserve')
 return {'resumed_bytes':start,'downloaded_bytes':expected_bytes-start,'range_bytes':chunk_bytes,'transfer_retries':failures}
