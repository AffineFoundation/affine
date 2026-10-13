"""Read-only, allowlisted MiMo dashboard projection; never exports submissions."""
import base64
import hashlib
import json
import math
from pathlib import Path
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(document, authority):
    if document.get('signer') != authority:
        raise ValueError('dashboard evidence signer')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),
        base64.b64decode(document['signature'], validate=True))
    return document['payload']


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256*1024**2:
        raise ValueError('bounded ordinary dashboard evidence')
    return json.loads(path.read_bytes())


def finite(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def evidence_availability(snapshot, history, authority, now):
    """Project the signed publisher's availability, never infer it from training."""
    result = dict(snapshot, summary=dict(snapshot['summary']))
    summary = result['summary']
    if history is None:
        return result
    try:
        payload = signed(history, authority)
        entry = payload.get('mimo_finalized_epoch_evidence')
        if entry is None:
            return result
        run = snapshot['training_run']
        finalized = [e for e in snapshot['epochs'] if e['finalized']]
        count = entry['epoch_count']
        if (entry['version'] != 'mimo-finalized-evidence-v1'
                or entry['run_id'] != run['run_id']
                or entry['first_round'] != run['first_round']
                or type(count) is not int or not 0 <= count <= len(finalized)
                or entry['last_finalized_round'] != (run['first_round'] + count - 1 if count else None)
                or entry['catalog']['method'] != 'GET'
                or entry['created_at'] > now):
            raise ValueError('dashboard finalized evidence catalog binding')
        if entry['catalog']['expires_at'] <= now:
            summary['evidence_status'] = 'mimo-finalized-artifact-links-expired'
        elif count:
            summary['evidence_status'] = 'mimo-finalized-artifacts-available'
        summary['published_evidence_epochs'] = count
    except (KeyError, TypeError, ValueError, BadSignatureError):
        # A stale/invalid optional publication must not hide operational data.
        summary['evidence_status'] = 'mimo-finalized-evidence-unavailable'
    return result


class Projection:
    def __init__(self, root, policy, authority, workers, validate_report, *, first_round, started_at, model, audit_projection=None):
        self.audit_projection = audit_projection
        self.root, self.policy, self.authority = Path(root), policy, authority
        self.workers, self.validate_report = workers, validate_report
        self.first_round, self.started_at, self.model = first_round, started_at, model
        if type(first_round) is not int or first_round < 0 or finite(started_at) is None:
            raise ValueError("dashboard explicit run boundary")

    def opening(self, envelope):
        value = signed(envelope, self.authority)
        self.policy.validate_manifest(value)
        if value['run_id'] != self.policy.run_id:
            raise ValueError('dashboard run binding')
        return value

    def worker_report(self, envelope, request, role, opening, now):
        job = signed(envelope, self.authority)
        if job['role'] != role or job['manifest'] != opening:
            raise ValueError('dashboard original job/opening binding')
        worker = request.get('signer')
        if role not in self.workers.get(worker, ()):
            raise ValueError('dashboard admitted worker role')
        body = signed(request, worker)
        if body.get('action') != 'report' or body.get('job_id') != job['job_id']:
            raise ValueError('dashboard original worker request')
        if role == 'train' and job.get('trainer_worker') != worker:
            raise ValueError('dashboard exact trainer')
        report = body['report']
        self.validate_report(report, job, self.opening(opening), self.authority, digest(job), now)
        return job, report

    def project(self, previous, now):
        epochs, openings = [], {}
        for directory in sorted((self.root/'epochs').iterdir(), key=lambda p:int(p.name)):
            path = directory/'coordinator/OPENING.json'
            if not directory.is_dir() or not path.exists():
                continue
            opening = read(path); m = self.opening(opening); eid = m['epoch']
            index = int(directory.name)
            # Global round is an operator-authenticated release setting, not a UI guess.
            first_round = self.first_round
            display_epoch = len(epochs)+1
            row = dict(id=eid, finalized=False, mode='live', reward_eligible=m['payable'],
                payable=m['payable'], start=m['start'], deadline=m['deadline'], phase='collecting' if now<m['deadline'] else 'closed',
                checkpoint=m['checkpoint']['id'], indices=sum(len(e['indices']) for e in m['environments']),
                k=m['group_contract']['min_success'], l=m['group_contract']['min_completed_failure'],
                participants=len(m.get('capabilities',{})), submissions=0, accepted=0,rejected=0,unchecked=0,
                points=0,batches=0,batches_available=False,grid=[0]*256,grid_outcomes=None,unassigned_batches=0,
                miners=[],training=None,audit_policy='continuous',source='live-reward-math',mechanism='mimo-fixed-group',learner_eligible=None,
                learner_excluded=None,learner_grid=None,learner_input_assurance='unaudited',audit_breakdown_available=False,
                learner_eligible_identities=None,batch_count_source=None,learner_provenance_sha256=None,
                finalization_source=None,completion_sha256=None,display_epoch=display_epoch,training_run_id=m['run_id'])
            # Never read per-miner capture files or open-epoch submissions.
            population_path=directory/'coordinator/POPULATION.json'
            dispatch_path=directory/'coordinator/DISPATCH.json'
            dispatch=None
            if now>=m['deadline'] and population_path.exists() and dispatch_path.exists():
                population=signed(read(population_path),self.authority)
                dispatch_envelope=read(dispatch_path);dispatch=signed(dispatch_envelope,self.authority)
                if (population['epoch']!=eid or population['manifest_sha256']!=digest(m) or population['sealed_at']<m['deadline'] or dispatch['epoch']!=eid or dispatch['checkpoint']!=m['checkpoint']['id']
                        or dispatch['manifest_sha256']!=digest(m) or dispatch['population_sha256']!=digest(population)):
                    raise ValueError('dashboard frozen population binding')
                captures=[signed(c,self.authority) for c in population['captures']]
                if any(c['epoch']!=eid or c['manifest_sha256']!=digest(m) or c['miner']not in m['capabilities'] or type(c['total_groups']) is not int or not 0<=c['total_groups']<=m['max_batches'] for c in captures) or len({c['miner']for c in captures})!=len(captures):
                    raise ValueError('dashboard frozen group count')
                total=sum(c['total_groups'] for c in captures)
                row.update(batches=total,batches_available=True,unchecked=total,unassigned_batches=total,
                    submissions=len(captures),phase='training',batch_count_source='signed-mimo-frozen-population',
                    learner_provenance_sha256=digest(dispatch_envelope),learner_eligible=dispatch['selected_task_count'])
            final_path=directory/'FINALIZED.json'
            if final_path.exists():
                final_envelope=read(final_path);final=signed(final_envelope,self.authority)
                if (dispatch is None or final['policy_sha256']!=self.policy.policy_sha256 or final['epoch']!=eid
                        or final['opening_sha256']!=digest(opening) or final['dispatch_sha256']!=digest(dispatch_envelope)
                        or final['input_checkpoint']!=m['checkpoint']['id'] or final['round']!=first_round+display_epoch-1):
                    raise ValueError('dashboard finalized epoch binding')
                result=final['result'];row.update(finalized=True,phase='no-update',finalization_source='signed-mimo-finalized',completion_sha256=digest(final_envelope))
                if result['status']=='updated':
                    job,report=self.worker_report(read(directory/'TRAINING-JOB.json'),read(directory/'TRAINING-RESULT.json'),'train',opening,now)
                    t=report['training'];receipt=report['publication_receipt'];published=signed(receipt,job['training']['publisher_id'])
                    if (digest(receipt)!=result['publication_receipt_sha256'] or published['descriptor']!=result['checkpoint']
                            or t['output_checkpoint']!=result['checkpoint']['id'] or t['optimizer_step']!=result['optimizer_step']):
                        raise ValueError('dashboard published training result')
                    row.update(phase='trained',training=dict(steps=t['optimizer_step']-t['optimizer_step_before'],
                        losses=[finite(t.get('objective_loss'))] if finite(t.get('objective_loss')) is not None else None,
                        weights_changed=True,checkpoint=t['output_checkpoint'],objective='grpo',input_pairs=t['group_count']))
                elif result['status']!='no-update' or result['checkpoint']['id']!=m['checkpoint']['id']:
                    raise ValueError('dashboard unknown final state or changed no-update checkpoint')
            epochs.append(row);openings[eid]=opening
        evaluations=[]
        for path in sorted((self.root/'evaluator').glob('*.receipt.json')):
            receipt=signed(read(path),self.authority)
            if receipt['policy_sha256']!=self.policy.policy_sha256:
                raise ValueError('dashboard evaluation release')
            envelope=receipt['job'];opening=signed(envelope,self.authority)['manifest'];m=self.opening(opening)
            if m['epoch'] not in openings or openings[m['epoch']]!=opening:
                raise ValueError('dashboard evaluation original opening')
            job,report=self.worker_report(envelope,receipt['worker_report'],'evaluate',opening,now)
            e=report['evaluation'];suite=e['suite'];h=e['harness']
            evaluations.append(dict(run_id=job['job_id'],epoch_id=m['epoch'],checkpoint=m['checkpoint']['id'],
                model=self.model,env_id=suite['env_id'],dataset_id=e['dataset_id'],seed=suite['seed'],
                count=e['count'],successes=e['successes'],mean_reward=finite(e['mean_reward']),timestamp=report['completed_at'],
                status=e['status'],requested_count=e['requested_count'],completed_count=e['count'],
                taskset_hash=e['dataset_id'],harness_config={k:h[k] for k in ('max_output_tokens','temperature','top_p','top_k')},
                output_token_budget=h['max_output_tokens'],original_report_sha256=digest(report),native_graded=True,
                proof_verification_performed=False,training_run_id=m['run_id']))
        boundary=dict(version='dashboard-training-run-boundary-v1',run_id=self.policy.run_id,started_at=min((e['start']for e in epochs),default=self.started_at),
            first_round=self.first_round,base_checkpoint=self.policy.scope['initial_checkpoint']['id'],model=self.model,
            source_sha256=self.policy.scope['immutable_manifest']['source_bundle']['sha256'])
        result=dict(summary=dict(updated_at=now,network='Finney',netuid=120,current_source='mimo-mainnet',epochs=len(epochs),
            evidence_status='mimo-finalized-artifacts-not-yet-published',
            evaluation_status='complete' if any(e['status']=='complete'for e in evaluations)else'pending',
            accepted=0,rejected=0,unchecked=sum(e['unchecked']for e in epochs),training_steps=sum((e['training']or{}).get('steps',0)for e in epochs)),
            training_run=boundary,epochs=sorted(epochs,key=lambda e:e['start'],reverse=True),evaluations=sorted(evaluations,key=lambda e:e['timestamp']),training_evaluations=[])
        for key in ('incentive','incentive_history','incentive_prices'):result[key]=previous.get(key)
        return self.audit_projection(result, openings, now) if self.audit_projection is not None else result
