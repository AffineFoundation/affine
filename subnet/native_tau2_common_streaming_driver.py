"""Prospective native driver with private exact-byte per-role R2 offload.

Unadmitted data remains signed/retained in private R2. Fresh verification reads
one full probability array at a time, then replays the original native episode.
This is a new source closure; it does not modify historical mixed-role controls.
"""
import argparse,json,pathlib
from nacl.signing import SigningKey
from . import native_tau2_common_mixed_driver as base
from .native_tau2_common_role_storage import offload,read_array
from .native_tau2_common_service import disk_guard
from .native_tau2_common_search_contract import admit_sample,AUDIT_VERSION,digest
from .native_tau2_common_search_endpoint import verify_receipt
from .native_tau2_common_replay import checked_records,replay
from .storage import Bucket


def generate(epoch,authority,fixed,workers,key,index,attempt,data,public,private,out,bucket):
 original=base.CommonRoleEndpoint
 class StreamingEndpoint(original):
  def response(self,request):
   roles=[r for r in self.manifest['roles'].values() if r['request_model']==request.get('model')]
   if len(roles)!=1:raise ValueError('unique approved model role')
   reserve=roles[0]['max_output_tokens']*roles[0]['vocab_size']*4+128*1024*1024
   disk_guard(out,max(512*1024*1024,reserve+256*1024*1024))
   result=super().response(request);record=result[1]['payload'];path=self.artifact_dir/record['probabilities_file'];object_key='private/native-tau2-common-live/'+self.manifest['epoch']+'/raw/'+pathlib.Path(out).name+'/'+path.name
   offload(bucket,path,object_key,result[1],key,epoch)
   return result
 base.CommonRoleEndpoint=StreamingEndpoint
 try:return base.generate(epoch,authority,fixed,workers,key,index,attempt,data,public,private,out)
 finally:base.CommonRoleEndpoint=original


def verify(epoch,authority,fixed,workers,key,index,attempt,data,public,private,out,bucket):
 out=pathlib.Path(out).resolve();manifest,runtimes,policies=base.dependencies(epoch,authority,fixed,workers)
 if json.loads((out/'signed-epoch.json').read_bytes())!=epoch:raise ValueError('artifact original signed epoch')
 receipts=json.loads((out/'signed-receipts.json').read_bytes());checked_records(epoch,receipts,authority,fixed,index,attempt);checks=[]
 for ordinal,envelope in enumerate(receipts):
  record=envelope['payload']
  if record.get('probabilities_file')!=f'role-{ordinal}.npy':raise ValueError('exact ordinal role array')
  raw=read_array(bucket,out,record['probabilities_file'],epoch)
  checks.append(verify_receipt(epoch,envelope,raw,authority,fixed,runtimes[record['role']],index,trajectory_attempt=attempt,candidate_policy=policies[record['role']]))
  del raw
 native=replay(epoch,receipts,json.loads((out/'signed-native-simulation.json').read_bytes()),authority,fixed,index,data,public,private,out/'independent-native-replay',attempt)
 report={**native,'role_checks':checks,'all_model_roles_verified':True,'derived_responses_verified':True,'source_closure_verified':True,'source_scope':'operator-trusted-exact-declared-source-interpreter-package-versions-v1','full_transitive_binary_dependency_closure_claimed':False,'private_role_array_storage':'exact-byte-R2-one-array-per-fresh-check-v1'}
 audit={'version':AUDIT_VERSION,'manifest_sha256':digest(manifest),'signed_receipts_sha256':digest(receipts),'verification_report_sha256':digest(report),'trajectory_attempt':attempt,'epoch':manifest['epoch'],'environment_id':manifest['environment']['id'],'environment_version':manifest['environment']['version'],'environment_index':index,'task_hash':report['task_hash'],'reward':report['reward'],'originally_sampled':False,'payable':False,'sampler_provenance':manifest['sampler_provenance']}
 for field in ('full_native_trajectory_verified','all_model_roles_verified','derived_responses_verified','source_closure_verified'):audit[field]=report[field]
 envelope=base.sign(audit,key);view=admit_sample(epoch,receipts,envelope,report,authority,fixed)
 for name,value in (('independent-full-verification.json',report),('role-audit.json',envelope),('admitted-sample.json',view)):base.save(out/name,value)
 return report


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['generate','verify'])
 for name in ('epoch','fixed-user','worker-configs','seed-file','data','public-tasks','private-tasks','out','bucket-config'):p.add_argument('--'+name,type=pathlib.Path,required=True)
 p.add_argument('--authority',required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--attempt',type=int,required=True);a=p.parse_args();key=SigningKey(bytes.fromhex(a.seed_file.read_text().strip()))
 if key.verify_key.encode().hex()!=a.authority:raise ValueError('operator signer')
 epoch=json.loads(a.epoch.read_bytes());fixed=json.loads(a.fixed_user.read_bytes())
 base.validate_epoch(epoch,a.authority,fixed)
 bucket=Bucket(json.loads(a.bucket_config.read_bytes()))
 try:(generate if a.command=='generate' else verify)(epoch,a.authority,fixed,json.loads(a.worker_configs.read_bytes()),key,a.index,a.attempt,a.data,a.public_tasks,a.private_tasks,a.out,bucket)
 finally:base.close_workers()
if __name__=='__main__':main()
