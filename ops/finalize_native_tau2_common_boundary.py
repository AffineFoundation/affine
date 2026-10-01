"""Operator-only final atomic object selection after a native epoch deadline.

Predeadline admission is not final selection. This helper holds only the exact
owned coordinator identity while selecting a final immutable R2 snapshot. Any
late/changed object leaves that coordinator held for explicit reconciliation.
No generation, chain transactions, uploaded imports or model loading.
"""
import argparse,base64,hashlib,json,os,pathlib,signal,time
from nacl.signing import SigningKey
from subnet.long_context_runtime import AUTHORITY,authenticate,canonical,digest,file_sha
from subnet.native_tau2_common_service import pid_identity,write,sign
from subnet.storage import Bucket
VERSION='native-tau2-final-boundary-selection-v1'

def select(snapshot,pre,deadline,completed_at):
 if snapshot is None or completed_at<deadline:raise ValueError('final selection after deadline')
 if type(snapshot.get('completed_at')) not in (int,float) or snapshot['completed_at']>=deadline:raise ValueError('late object is not an eligible submission')
 if hashlib.sha256(snapshot['data']).hexdigest()!=pre['zip_sha256'] or len(snapshot['data'])!=pre['zip_size'] or snapshot['etag']!=pre['r2_etag']:raise ValueError('final object differs from admitted cumulative bytes')
 return {'version':VERSION,'epoch_manifest_sha256':pre['manifest_sha256'],'registered_uid':pre['registered_uid'],'deadline':deadline,'final_atomic_get_completed_at':completed_at,'object_last_modified_at':snapshot['completed_at'],'zip_sha256':pre['zip_sha256'],'zip_size':pre['zip_size'],'r2_etag':snapshot['etag'],'predeadline_receipt_sha256':digest(pre),'final_selection_after_deadline':True,'late_objects_rejected':True,'payable':False,'chain_transactions':False}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=pathlib.Path,required=True);a=p.parse_args();plan=authenticate(json.loads(a.plan.read_bytes()),AUTHORITY)
 if plan.get('version')!=VERSION or plan.get('payable') is not False or plan.get('chain_transactions') is not False:raise ValueError('operator scoped boundary plan')
 folder=pathlib.Path(plan['folder']);epoch=authenticate(json.loads((folder/'signed-epoch.json').read_bytes()),AUTHORITY);deadline=epoch['submission_window']['deadline'];pre=authenticate(json.loads((folder/'signed-batch-freeze.json').read_bytes()),AUTHORITY)
 if pre['manifest_sha256']!=digest(epoch):raise ValueError('predeadline manifest')
 caps=json.loads((folder/'private-upload-capability.json').read_bytes());expected='private/native-tau2-common-live/'+epoch['epoch']+'/uid131.zip'
 if caps['object_key']!=expected or plan['coordinator_identity'].get('state')=='Z':raise ValueError('owned staging scope')
 config=json.loads(pathlib.Path(plan['config']).read_bytes());key=SigningKey(bytes.fromhex(pathlib.Path(config['authority_seed_file']).read_text().strip()))
 if key.verify_key.encode().hex()!=AUTHORITY:raise ValueError('operator signer')
 while time.time()<deadline:time.sleep(min(5,deadline-time.time()))
 identity=pid_identity(plan['coordinator_identity']['pid'])
 if not identity or identity['start_ticks']!=plan['coordinator_identity']['start_ticks'] or identity['state']=='Z':raise ValueError('exact owned coordinator no longer active')
 os.kill(identity['pid'],signal.SIGSTOP)
 try:
  bucket=Bucket(config['bucket']);snapshot=bucket.snapshot(expected,limit=250000000);report=select(snapshot,pre,deadline,time.time());target='private/native-tau2-common-live/'+epoch['epoch']+'/final-frozen-'+report['zip_sha256']+'.zip';bucket.put(target,snapshot['data']);report['immutable_final_object_key']=target;report['predeadline_signed_receipt_file_sha256']=file_sha(folder/'signed-batch-freeze.json');write(folder/'signed-final-boundary-selection.json',sign(report,key));bucket.json(target+'.receipt.json',sign(report,key))
 except BaseException as error:
  write(folder/'signed-final-boundary-rejection.json',sign({'version':VERSION,'epoch':epoch['epoch'],'error_type':type(error).__name__,'reason':str(error)[:160],'coordinator_held':True,'payable':False,'chain_transactions':False},key));raise
 os.kill(identity['pid'],signal.SIGCONT)
 write(folder/'final-boundary-selector-completed.json',{'completed_at':time.time(),'coordinator_resumed':True,'payable':False})
if __name__=='__main__':main()
