"""Conditional immutable asset upload and independent streamed R2 byte readback.

The private capability registry is separate from URL-free immutable descriptors.
This operator-only tool never publishes epochs or executes model/runtime roles.
"""
import argparse,concurrent.futures,hashlib,json,os,re,time
from pathlib import Path
from botocore.exceptions import ClientError
from subnet.storage import Bucket
from subnet.math_corpus_assets import admit_bytes

def atomic(path,value):
 temporary=path.with_suffix(path.suffix+'.new');temporary.write_text(json.dumps(value,sort_keys=True));temporary.chmod(0o600);os.replace(temporary,path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--asset-directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(exist_ok=False);out.chmod(0o700);manifestbody=Path(a.manifest).read_bytes();manifest=json.loads(manifestbody);config=json.loads(Path(a.config).read_bytes());bucket=Bucket(config['bucket']);assets=Path(a.asset_directory).resolve();prefix='operator-assets/math-corpora-v2/'+manifest['catalog_sha256']+'/'
 completed={};registry={};started=time.time()
 def upload(entry):
  b=entry['asset'];path=assets/entry['object_file']
  if path.parent!=assets or path.is_symlink() or not path.is_file():raise ValueError('asset local path')
  body=path.read_bytes();admit_bytes(body,b);key=prefix+b['sha256']+'/'+b['compressed_sha256']+'.tasks.json.gz';created=False
  try:head=bucket.client.head_object(Bucket=bucket.name,Key=key)
  except ClientError as exc:
   code=str(exc.response.get('Error',{}).get('Code'))
   if code not in ('404','NoSuchKey','NotFound'):raise
   try:
    bucket.client.put_object(Bucket=bucket.name,Key=key,Body=body,ContentType='application/gzip',IfNoneMatch='*');created=True
   except ClientError as race:
    if str(race.response.get('Error',{}).get('Code')) not in ('PreconditionFailed','412'):raise
  response=bucket.client.get_object(Bucket=bucket.name,Key=key);stream=response['Body'];digest=hashlib.sha256();size=0
  try:
   if response['ContentLength']!=b['compressed_size']:raise ValueError('existing/remote size collision')
   while True:
    block=stream.read(1024*1024)
    if not block:break
    size+=len(block)
    if size>b['compressed_size']:raise ValueError('remote read bound')
    digest.update(block)
  finally:stream.close()
  if size!=b['compressed_size'] or digest.hexdigest()!=b['compressed_sha256']:raise ValueError('existing/remote content collision')
  url=bucket.presign(key,expires=604800)
  return b['sha256'],{'id':entry['id'],'object_key':key,'created_new':created,'size':size,'sha256':digest.hexdigest(),'raw_size':b['size'],'raw_sha256':b['sha256'],'remote_stream_verified':True,'verified_at':time.time()}, {'read_url':url,'compressed_sha256':b['compressed_sha256'],'compressed_size':b['compressed_size']}
 failures=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  pending={pool.submit(upload,e):e['id'] for e in manifest['shards']}
  for future in concurrent.futures.as_completed(pending):
   try:key,receipt,cap=future.result()
   except Exception as exc:
    failures.append({'id':pending[future],'error_type':type(exc).__name__});atomic(out/'failure-status.json',{'failures':failures,'completed':len(completed)});continue
   if key in completed:raise ValueError('duplicate raw asset')
   completed[key]=receipt;registry[key]=cap
   atomic(out/'streamed-readback-progress.json',{'started_at':started,'assets':completed,'failures':failures});atomic(out/'task-assets.private.json',registry)
   print(json.dumps({'completed':len(completed),'total':len(pending),'id':receipt['id'],'streamed_bytes':receipt['size'],'remote_sha_match':True}),flush=True)
 if failures:raise RuntimeError('asset publication failed; preserved partial receipts')
 if len(completed)!=len(manifest['shards']):raise ValueError('publication membership mismatch')
 atomic(out/'publication-complete.json',{'started_at':started,'finished_at':time.time(),'manifest_sha256':hashlib.sha256(manifestbody).hexdigest(),'catalog_sha256':manifest['catalog_sha256'],'assets':completed,'asset_count':len(completed),'total_streamed_bytes':sum(x['size'] for x in completed.values()),'private_registry_sha256':hashlib.sha256((out/'task-assets.private.json').read_bytes()).hexdigest(),'capabilities_expires_after_seconds':604800,'GPU_jobs':0,'epochs_published':0,'chain_transactions':False,'no_existing_objects_overwritten':True})
 (out/'descriptors.json').write_bytes(manifestbody);(out/'descriptors.json').chmod(0o600);print(json.dumps({'complete':True,'asset_count':len(completed),'streamed_bytes':sum(x['size'] for x in completed.values())}),flush=True)
if __name__=='__main__':main()
