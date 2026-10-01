"""Private operator evidence storage for native roles, including unadmitted data.

An exact-byte upload/readback is storage evidence, NEVER model/native admission.
Every array must later be independently recomputed before any training/score.
"""
import hashlib,json,pathlib
from .long_context_runtime import AUTHORITY,authenticate,canonical,digest,file_sha
from .native_tau2_common_service import sign,write
VERSION='native-tau2-private-unadmitted-role-storage-v1'
LIMIT=512*200000*4+10000

def exact_read(bucket,row):
 if set(row)!={'key','sha256','size'} or not row['key'].startswith('private/native-tau2-common-live/') or type(row['size']) is not int or not 0<row['size']<=LIMIT:raise ValueError('exact bounded private role object')
 response=bucket.client.get_object(Bucket=bucket.name,Key=row['key']);body=response['Body']
 try:
  if response['ContentLength']!=row['size']:raise ValueError('role response declared size')
  raw=body.read(row['size']+1)
 finally:body.close()
 if len(raw)!=row['size'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('private role exact byte hash')
 return raw

def offload(bucket,path,key,receipt,keypair,epoch):
 path=pathlib.Path(path);envelope=authenticate(receipt,AUTHORITY);manifest=authenticate(epoch,AUTHORITY)
 if envelope['manifest_sha256']!=digest(manifest) or envelope['probabilities_file']!=path.name or path.is_symlink() or not path.is_file() or path.stat().st_size>LIMIT or file_sha(path)!=envelope['probabilities_sha256']:raise ValueError('signed role array before storage')
 row={'key':key,'sha256':file_sha(path),'size':path.stat().st_size};bucket.upload(key,path);exact_read(bucket,row)
 inventory_path=path.parent.parent/'signed-private-role-storage.json'
 if inventory_path.exists():
  prior=authenticate(json.loads(inventory_path.read_bytes()),AUTHORITY)
  if prior['manifest_sha256']!=digest(manifest) or prior['version']!=VERSION:raise ValueError('immutable role storage namespace')
  rows=prior['files']
 else:rows={}
 if path.name in rows and rows[path.name]!=row:raise ValueError('role object overwrite forbidden')
 rows[path.name]=row
 write(inventory_path,sign({'version':VERSION,'manifest_sha256':digest(manifest),'files':rows,'uploaded_exact_bytes_verified':True,'model_or_native_admission_claimed':False,'payable':False},keypair))
 # The signed exact-byte inventory is durable BEFORE removing only this cache.
 path.unlink();return row

def read_array(bucket,out,name,epoch):
 out=pathlib.Path(out);local=out/'roles'/name
 if local.exists():return local.read_bytes()
 inventory=authenticate(json.loads((out/'signed-private-role-storage.json').read_bytes()),AUTHORITY);manifest=authenticate(epoch,AUTHORITY)
 if inventory['version']!=VERSION or inventory['manifest_sha256']!=digest(manifest) or inventory['model_or_native_admission_claimed'] is not False:raise ValueError('unadmitted storage lineage')
 return exact_read(bucket,inventory['files'][name])
