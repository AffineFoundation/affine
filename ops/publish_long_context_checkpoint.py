#!/usr/bin/env python3
"""Publish remote-only trained weights with six scoped PUTs and operator hashes."""
import argparse,base64,hashlib,json,pathlib,shlex,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from subnet.long_context_runtime import authenticate,AUTHORITY,canonical,digest,file_sha

# Signed delegation is checked before reading checkpoint files or upload URLs.
REMOTE=r'''
import base64,hashlib,json,pathlib,sys,urllib.request
from nacl.signing import VerifyKey
raw=json.load(sys.stdin);p=raw['payload'];authority='d54a3a345d0de3e2c7898f30c0942d78f931f8c4b8036ffdc6adffcd2525062f'
canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
if raw['signer']!=authority:raise ValueError('upload authority')
VerifyKey(bytes.fromhex(authority)).verify(canonical(p),base64.b64decode(raw['signature'],validate=True))
if p['role']!='long-context-exact-checkpoint-upload-v1' or p['payable'] is not False or p['chain_transactions'] is not False:raise ValueError('upload role')
cp=p['checkpoint'];root=pathlib.Path(cp['path']);records={}
if hashlib.sha256(canonical(cp['files'])).hexdigest()!=cp['id']:raise ValueError('checkpoint identity')
if set(p['capabilities'])!=set(cp['files']):raise ValueError('exact upload capabilities')
for name,expected in cp['files'].items():
 if pathlib.Path(name).name!=name:raise ValueError('checkpoint filename')
 path=root/name;h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 if h.hexdigest()!=expected:raise ValueError('remote checkpoint hash')
 cap=p['capabilities'][name]
 if cap is not None:
  with path.open('rb') as f:
   request=urllib.request.Request(cap['url'],data=f,method='PUT',headers={'Content-Type':'application/octet-stream','Content-Length':str(path.stat().st_size)})
   response=urllib.request.urlopen(request,timeout=600);response.read();response.close()
 records[name]={'sha256':expected,'size':path.stat().st_size}
print(json.dumps({'checkpoint_id':cp['id'],'files':records}))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--job',default=str(ROOT/'state/long-context-eog/optimizer-job.json'));a=p.parse_args();out=pathlib.Path(a.out)
 job=authenticate(json.loads(pathlib.Path(a.job).read_text()),AUTHORITY)
 report=json.loads((out/'report.json').read_text())
 if job.get('role')!='long-context-proof-probe' or job.get('experiment') not in ('full-gradient-optimizer-v1','full-sequential-seven-agent-turns-v1'):raise ValueError('authorized long-context training publication role')
 if report.get('job_hash')!=digest(job) or report.get('completed') is not True or report.get('fresh_changed_checkpoint_verified') is not True or report.get('full_model_finetune') is not True or report.get('quality_improvement_claimed') is not False:raise ValueError('complete authorized training control')
 cp=report['checkpoint']
 if digest(cp['files'])!=cp['id'] or len(cp['files'])!=6:raise ValueError('exact six-file checkpoint')
 from subnet.storage import Bucket
 from nacl.signing import SigningKey
 from botocore.exceptions import ClientError
 key=SigningKey(bytes.fromhex((ROOT/'state/service-conformance/authority.seed').read_text()))
 if key.verify_key.encode().hex()!=AUTHORITY:raise ValueError('operator signer')
 def sign(value):return {'payload':value,'signer':AUTHORITY,'signature':base64.b64encode(key.sign(canonical(value)).signature).decode()}
 bucket=Bucket(json.loads((ROOT/'state/r2-direct.json').read_text()))
 def checked(object_key,expected):
  try:response=bucket.client.get_object(Bucket=bucket.name,Key=object_key)
  except ClientError as e:
   if str(e.response['Error']['Code']) in ('404','NoSuchKey','NotFound'):return None
   raise
  body=response['Body'];h=hashlib.sha256();size=0
  try:
   for b in iter(lambda:body.read(1024*1024),b''):h.update(b);size+=len(b)
  finally:body.close()
  if h.hexdigest()!=expected:raise ValueError('immutable published checkpoint collision')
  return {'sha256':expected,'size':size}
 caps={}
 for name,expected in cp['files'].items():
  object_key=f"public/checkpoints/{cp['id']}/{name}"
  caps[name]=None if checked(object_key,expected) else {'url':bucket.presign(object_key,'put_object',3600)}
 payload={'role':'long-context-exact-checkpoint-upload-v1','checkpoint':cp,'capabilities':caps,'payable':False,'chain_transactions':False,'publisher_source_sha256':file_sha(__file__)}
 ssh=['ssh','-p','20059','-o','UserKnownHostsFile='+str(ROOT/'state/registered-pod-retained-known-hosts'),'-o','StrictHostKeyChecking=yes','root@90.95.12.246',shlex.join(['/root/miner-venv/bin/python','-c',REMOTE])]
 remote=json.loads(subprocess.check_output(ssh,input=canonical(sign(payload)),timeout=900))
 verified={name:checked(f"public/checkpoints/{cp['id']}/{name}",expected) for name,expected in cp['files'].items()}
 if remote['checkpoint_id']!=cp['id'] or verified!=remote['files'] or any(v is None for v in verified.values()):raise ValueError('independent stream hash/size')
 descriptor={'id':cp['id'],'files':cp['files']};envelope=sign(descriptor);object_key=f"public/checkpoints/{cp['id']}/authorities/{AUTHORITY}/checkpoint.json"
 try:previous=json.loads(bucket.get(object_key))
 except ClientError as e:
  if str(e.response['Error']['Code']) not in ('404','NoSuchKey','NotFound'):raise
  bucket.json(object_key,envelope)
 else:
  if authenticate(previous,AUTHORITY)!=descriptor:raise ValueError('immutable descriptor collision')
 if authenticate(json.loads(bucket.get(object_key)),AUTHORITY)!=descriptor:raise ValueError('descriptor readback')
 record={'checkpoint':cp,'verified_files':verified,'descriptor_key':object_key,'report_sha256':file_sha(out/'report.json'),'approved_job_hash':digest(job),'publisher_source_sha256':file_sha(__file__),'independent_r2_stream_hash_verified':True,'full_model_finetune':True,'quality_improvement_claimed':False,'payable':False,'chain_transactions':False,'completed_at':time.time()}
 with (out/'checkpoint-publication.json').open('x') as f:f.write(json.dumps(sign(record),indent=2)+'\n')
 print(json.dumps({'checkpoint':cp['id'],'verified_files':len(verified),'independent_r2_stream_hash_verified':True}),flush=True)
if __name__=='__main__':main()
