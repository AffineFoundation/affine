"""Truthful terminal closure of a signed freeze budget with incomplete metadata."""
import json,time
from .storage import canonical,sha
from .hourly_policy import cutoff

VERSION='commitment-capture-status-v1'
class InfrastructureSkipped(RuntimeError):
 def __init__(self,document):self.document=document;super().__init__('freeze metadata incomplete after signed cutoff')

def terminal(controller,manifest):
 """No fake frozen receipts, audits, scores, training or reward attestations."""
 from .commitment_transport import VERSION as TRANSPORT
 from .remote_backend import save
 from .backend_jobs import signed
 epoch=manifest['epoch'];until=cutoff(manifest,'freeze')
 if manifest.get('submission_transport_policy')!=TRANSPORT or until is None or time.time()<until:raise ValueError('prospective signed expired freeze only')
 state=controller.gateway.epochs[epoch]
 if 'frozen_receipts'in state or state.get('commitment_capture_complete'):raise ValueError('incomplete capture cannot replace complete freeze')
 path=controller.state/(epoch+'-capture-status.json')
 if path.exists():
  document=json.loads(path.read_text());payload=signed(document,controller.authority.id)
  if (payload.get('version')!=VERSION or payload.get('status')!='metadata_incomplete' or payload.get('complete')is not False or payload.get('manifest_sha256')!=sha(canonical(manifest)) or payload.get('freeze_cutoff')!=until):raise ValueError('original incomplete capture status binding')
 else:
  pending=state.get('commitment_pending',{});snapshots=state.get('commitment_snapshots',{});known={m:p['sha256']for m,p in dict(pending,**snapshots).items()}
  discovery=state.get('commitment_discovery');population=discovery['miners']if discovery and discovery.get('complete')is True else sorted(state['miners'])
  unresolved=sorted(set(population)-set(known)-set(state.get('rejections',{})))
  payload=dict(version=VERSION,epoch=epoch,status='metadata_incomplete',complete=False,manifest_sha256=sha(canonical(manifest)),source_sha256=manifest['source_bundle']['sha256'],checkpoint=manifest['checkpoint']['id'],freeze_cutoff=until,observed_at=time.time(),discovery_complete=bool(discovery and discovery.get('complete')is True),known_captured=[dict(miner=m,commitment_sha256=known[m])for m in sorted(known)],unresolved_miners=unresolved,structurally_rejected_miners=sorted(state.get('rejections',{})),metadata_failure=state.get('commitment_metadata_incomplete'),verification_claim=False,audits_started=False,accepted_batches=0,rewards_eligible=False)
  document=controller.signed(payload);save(path,document)
 controller.bucket.json('public/'+epoch+'/capture-status.json',document)
 return document

def close_epoch(controller,manifest,status,statuspath,prefix):
 """Advance only the epoch counter; preserve checkpoint and optimizer parent."""
 from .remote_backend import save
 from .backend_jobs import signed
 epoch=manifest['epoch'];document=terminal(controller,manifest);capture=signed(document,controller.authority.id)
 if not status.get('active')or status['active']['epoch']!=epoch or status['checkpoint']['id']!=capture['checkpoint']:raise ValueError('same original infrastructure skipped epoch')
 record=dict(epoch=epoch,status='infrastructure_skipped_metadata_incomplete',capture_status_sha256=sha(canonical(document)),checkpoint=status['checkpoint']['id'],training_updates=0,verification_claim=False,payable=False,chain_transactions=False,completed_at=time.time())
 closure=controller.state/(epoch+'-infrastructure-skipped.json')
 if closure.exists():record=json.loads(closure.read_text())
 else:save(closure,record)
 controller.bucket.json('public/'+epoch+'/infrastructure-skipped.json',controller.signed(record))
 ledgerpath=controller.state/'infrastructure-skipped-epochs.json';ledger=json.loads(ledgerpath.read_text())if ledgerpath.exists()else []
 if not any(row['epoch']==epoch for row in ledger):ledger.append(record);save(ledgerpath,ledger)
 controller.bucket.json('public/streams/'+prefix+'/infrastructure-skips.json',controller.signed(dict(version=VERSION,epochs=ledger)))
 status['active']=None;status['round']+=1;save(statuspath,status)
 return record
