"""CPU dispatch only: mining cache hints never authorize checkpoint files.
The unchanged scientific checkpoint() authenticates matching default files and
hydrates missing/mismatched members from its signed R2 map. Other roles retain
all original cache/optimizer authorization behavior.
"""
VERSION='miner-default-checkpoint-materialization-v1'

def install(remote_backend,grant,source_sha256,runtime_files,digest):
 expected={'version':VERSION,'roles':['mine'],'source_sha256':source_sha256,'scientific_runtime_files_sha256':digest(runtime_files)}
 if {k:v for k,v in grant.items()if k not in ('helper_path','helper_file_sha256')}!=expected:raise ValueError('exact CPU mining hydration policy')
 original=remote_backend.RemoteJobs.run
 def run(self,label,role,manifest,cache=None,dispatch_only=False,**fields):
  if role=='mine':
   if manifest.get('source_bundle',{}).get('sha256')!=source_sha256 or self.metadata.get('source_files')!=runtime_files:raise ValueError('mining hydration bound scientific source')
   cache=None
  return original(self,label,role,manifest,cache,dispatch_only=dispatch_only,**fields)
 remote_backend.RemoteJobs.run=run
 return original
