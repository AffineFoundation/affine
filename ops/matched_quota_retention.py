"""Own trial artifacts retire automatically only after genuine full R2 ACK."""
import hashlib,json
from pathlib import Path
from subnet.backend_jobs import signed
from ops.matched_quota_trial import digest

def retire_artifacts(scope_envelope,ack_envelope,authority,*,live_phase,cache_factory=None):
 scope=signed(scope_envelope,authority);ack=signed(ack_envelope,authority);root=Path(scope['workspace'])
 if scope.get('production_state_changes')is not False or scope.get('version')!='matched-quota-research-v1'or root.resolve()!=root or root.is_symlink():raise ValueError('exact research-only namespace')
 if(ack.get('version')!='matched-quota-full-evidence-durable-ack-v1'or ack.get('scope_sha256')!=digest(scope)or ack.get('R2_full_GET_verified')is not True or set(ack.get('completed_phases',[]))!=set(scope['phases'])):raise ValueError('genuine all-required-phase full-byte durability ACK')
 if any(live_phase(phase)for phase in scope['phases']):raise ValueError('original scoped process still live')
 report_path=root/'generation-result.json'
 if report_path.is_symlink()or hashlib.sha256(report_path.read_bytes()).hexdigest()!=ack['generation_result_sha256']:raise ValueError('exact durable original generation metadata')
 report=json.loads(report_path.read_bytes())
 if report['scope_sha256']!=digest(scope):raise ValueError('original generation scope')
 paths=[]
 for rows in report['streams'].values():
  for row in rows:
   if 'artifact_path'not in row:continue
   p=Path(row['artifact_path']);expected=Path('jobs')/digest(scope)
   if p.parent!=expected or p.name!=f'submission-{scope["task_indices"].index(row["index"])*16+row["attempt"]}.zip':raise ValueError('only code-created original artifact paths')
   paths.append(str(p))
 if cache_factory is None:
  from subnet.cache_lifecycle import CacheLifecycle
  cache_factory=CacheLifecycle
 # record_download happened at actual file creation. CacheLifecycle verifies the
 # recorded inode/owner/size/mtime and refuses foreign or swapped files.
 return cache_factory(root).retire_downloads(digest(scope),only=paths)
