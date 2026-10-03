"""Publish current mining availability from a live, pinned controller.

This operator-host observer cannot open an epoch, extend a deadline or submit
weights. It closes discovery when the original controller is absent or the
signed upload window has ended. It never publishes upload capabilities.
"""
import argparse, json, os, time
from pathlib import Path
from ops.live_reward_exporter import atomic
from subnet.live_reward_bridge import signed
from subnet.source_bootstrap import r2_url

def process_live(record, proc=Path('/proc')):
 try:
  fields=(proc/str(record['child_pid'])/'stat').read_text().rsplit(')',1)[1].split()
  return fields[0] not in ('Z','X') and fields[19]==str(record['child_ticks'])
 except (OSError,KeyError,IndexError,ValueError):return False

def project(previous, config, status, discovery, envelope, authority, *, live, now):
 result=dict(previous,accepting_submissions=False,status='between_epochs',status_updated_at=now)
 active=status.get('active')
 if not live:
  result.update(status='controller_unavailable',notice='Mining is closed while the controller is unavailable. Watch this document for the next signed epoch.')
  return result
 if not active or active.get('phase') not in ('mine','collect'):
  result['notice']='Uploads are closed during verification, training and evaluation. The next epoch opens after its checkpoint is published.'
  return result
 manifest=signed(envelope,authority)
 contract=manifest.get('live_reward_contract',{})
 # Read capabilities are refreshed at opening. Model identity is the exact
 # immutable ID/file map, rather than equality of expiring URL metadata.
 same_checkpoint=all(manifest['checkpoint'].get(k)==status['checkpoint'].get(k) for k in ('id','files'))
 if (manifest['epoch']!=active['epoch'] or manifest['source_bundle']['sha256']!=config['source_bundle']['sha256']
     or not same_checkpoint or contract.get('version')!='live-verified-subset-reward-v1'
     or contract.get('epoch')!=manifest['epoch'] or contract.get('payable') is not True
     or discovery['authority']!=authority or discovery['expires_at']<=now):
  raise ValueError('current signed reward discovery binding')
 url=r2_url(discovery['current_url'])
 deadline=manifest['deadline']
 accepting=manifest.get('start',now)<=now<deadline
 result.update(epoch_id=manifest['epoch'],deadline=deadline,current_url=url,expires_at=discovery['expires_at'],
     authority=authority,checkpoint=manifest['checkpoint']['id'],source_bundle_sha256=manifest['source_bundle']['sha256'],
     max_batches=manifest['max_batches'],audit_policy=manifest['audit_policy'],payable=False,prospective_reward_eligible=True,
     accepting_submissions=accepting,status='mining' if accepting else 'epoch_closed',
     notice='Mining is open. Read the signed current manifest and approved source; watch main and /llms.txt for updates.' if accepting else 'The signed upload deadline has passed. Watch for the next epoch.')
 if previous.get('epoch_id')!=manifest['epoch']:
  result.update(verified_task_points=0,verified_contributing_miners=0,latest_controller_completed=False)
 result['latest_training_checkpoint']=status['checkpoint']['id']
 return result

def run_once(config_path, process_path, target, authority):
 target=Path(target);previous=json.loads(target.read_text());now=time.time()
 try:
  config=json.loads(Path(config_path).read_text());state=Path(config['state'])
  status=json.loads((state/'controller.json').read_text())
  live=process_live(json.loads(Path(process_path).read_text()))
  discovery=json.loads((state/'direct-discovery.json').read_text())
  active=status.get('active');envelope=None
  if live and active and active.get('phase') in ('mine','collect'):
   envelope=json.loads((state/(active['epoch']+'-first-signed-manifest.json')).read_text())
  result=project(previous,config,status,discovery,envelope,authority,live=live,now=now)
 except Exception:
  result=dict(previous,accepting_submissions=False,status='discovery_binding_error',status_updated_at=now,
      notice='Mining availability could not be authenticated. Uploads remain closed while this is checked.')
 # Atomic and readable by the static HTTP server; private operator files stay private.
 atomic(target,result);target.chmod(0o644)
 return dict(status=result['status'],accepting_submissions=result['accepting_submissions'])

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--process',required=True);p.add_argument('--target',required=True);p.add_argument('--authority',required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
 while True:
  print(json.dumps(run_once(a.config,a.process,a.target,a.authority)),flush=True)
  if not a.watch:return
  time.sleep(5)
if __name__=='__main__':main()
