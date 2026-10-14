"""Complete-group evidence for asynchronous audits and the sole hourly writer.

This adapter authenticates the new wire format directly. It never invents a
legacy pair receipt, claims unaudited inputs are verified, or writes weights.
The caller owns the ROOT-qualified release gate and immutable population store.
"""
from collections import Counter
from copy import deepcopy
import json

from .backend_jobs import signed
from .storage import canonical, sha
from .complete_group_admission import validate_commitment
from .complete_group_native_receipts import _validate as validate_native, COMPACT
from .complete_group_contract import meets_minima
from .continuous_audit_policy import population, digest, finite, integer
from .mimo_worker import submission_reference, validate_submissions, validate_report, VERSION as WORKER

VERSION = 'mimo-continuous-audit-population-v1'


def need(value, message):
    if not value:
        raise ValueError(message)


def cohort(manifest):
    """Identity excludes expiring URLs; includes every scientific contract."""
    fields = ('run_id','source_bundle','model_runtime_revision','mimo_runtime','backend_profile',
              'sampling_contract','sampling_source_hash','harness_source_hash','environments',
              'group_contract','group_transport_policy','mimo_answer_projection')
    # The policy's actual field name is deliberately imported, not assumed.
    from .mimo_answer_projection import POLICY_KEY
    fields = tuple(POLICY_KEY if k == 'mimo_answer_projection' else k for k in fields)
    need(all(k in manifest for k in fields), 'complete MiMo verifier cohort')
    value = {k: deepcopy(manifest[k]) for k in fields}
    value['source_bundle'] = {'sha256': manifest['source_bundle']['sha256']}
    return digest(value)


def build_population(opening, captures, commitments, dispatch, registrations, *, round,
                     committed_at, authority, manifest_guard):
    """Reconstruct one signed native population from exact immutable originals.

    Native eligibility covers selected tasks AND tasks excluded only by the
    learner's compute budget. Cross-miner collisions earn no task credit. Audits
    may nevertheless inspect every committed group, including rejected claims.
    """
    need(callable(manifest_guard), 'qualified release manifest gate required')
    m = signed(opening, authority); manifest_guard(m)
    integer(round, 0, 2**31-1, 'monotonic global epoch round')
    finite(committed_at, m['deadline'], 2**53, 'closed original commitment time')
    d = signed(dispatch, authority)
    need(d.get('version') == 'mimo-complete-group-coordinator-v1' and
         d.get('epoch') == m['epoch'] and d.get('run_id') == m['run_id'] and
         d.get('manifest_sha256') == digest(m) and d.get('checkpoint') == m['checkpoint']['id'] and
         d.get('training_requires_sampling_audit') is False, 'original native dispatch identity')
    regs = signed(registrations, authority)
    need(regs.get('epoch') == m['epoch'] and type(regs.get('snapshot_block')) is int and
         regs['snapshot_block'] >= 0 and type(regs.get('registrations')) is dict,
         'original signed registration snapshot')
    by_public = {}; uids = set()
    for hotkey, row in regs['registrations'].items():
        need(type(hotkey) is str and type(row) is dict and type(row.get('uid')) is int and
             row['uid'] >= 0 and row['uid'] not in uids and row.get('snapshot_block') == regs['snapshot_block'],
             'unique fixed-block network registration')
        key = row.get('public_key')
        need(type(key) is str and len(key) == 64 and key not in by_public,
             'unique cryptographic miner registration')
        by_public[key] = hotkey; uids.add(row['uid'])
    need(set(by_public) == set(m['capabilities']), 'capabilities match the registered network snapshot')
    need(type(captures) is dict and type(commitments) is dict and set(captures) == set(commitments)
         and set(captures) <= set(by_public), 'exact captured registered miner set')
    need(d.get('population_sha256') == digest(dict(version='mimo-complete-group-coordinator-v1',
        epoch=m['epoch'],manifest_sha256=digest(m),captures=[captures[k] for k in sorted(captures)],
        sealed_at=committed_at)), 'dispatch binds original sealed capture population')
    rows = []; lookup = {}; documents = {}
    for miner in sorted(captures):
        c = signed(captures[miner], authority)
        need(c.get('version') == 'mimo-complete-group-coordinator-v1' and
             c.get('epoch') == m['epoch'] and c.get('miner') == miner and
             c.get('manifest_sha256') == digest(m) and c.get('fully_audited') is False and
             c.get('assurance') == 'structurally-eligible-unaudited', 'original structural capture')
        parent = validate_commitment(canonical(commitments[miner]), m, miner, registered_miners=set(by_public))
        claims = parent['payload']['groups']; parent_sha = digest(parent)
        prefix = f'private/{m["epoch"]}/complete-group-snapshots/{miner}/{parent_sha}'
        need(c.get('commitment_sha256') == parent_sha and c.get('snapshot') == prefix and
             c.get('total_groups') == len(claims) and len(c.get('documents', [])) == len(claims),
             'exact immutable captured commitment')
        for claim, doc in zip(claims, c['documents'], strict=True):
            need(type(doc) is dict and set(doc) == {'slot','sha256','key'} and
                 doc['slot'] == claim['slot'] and doc['sha256'] == claim['document_sha256'] and
                 doc['key'] == prefix+f'/training/{claim["slot"]}.json', 'original compact document slot')
            row = dict(epoch=m['epoch'], round=round, checkpoint=m['checkpoint']['id'], miner=miner,
                env_id=claim['env_id'], index=claim['index'], batch_sha256=claim['batch_sha256'],
                proof_sha256=claim['artifact_sha256'], commitment_sha256=parent_sha,
                verifier_contract_sha256=cohort(m), committed_at=committed_at)
            identity = (miner, claim['batch_sha256']); need(identity not in lookup, 'one original group')
            rows.append(row); lookup[identity] = row; documents[identity] = doc
    native_rows = d.get('selected', []) + d.get('excluded', [])
    need(type(d.get('selected')) is list and type(d.get('excluded')) is list and
         len(native_rows) == len(rows), 'complete native disposition population')
    seen = set(); eligible = []; mixed = []
    for item in native_rows:
        key = (item['miner'], item['batch_sha256']); row = lookup.get(key)
        need(row is not None and key not in seen and item['document'] == documents[key] and
             all(item[k] == row[k] for k in ('env_id','index')), 'native disposition original committed group')
        seen.add(key)
        reason = item.get('reason')
        need(reason in (None,'training-task-budget','cross-miner-task-collision',
                        'native-outcome-minima-not-met','native-claim-invalid'), 'known native disposition')
        if reason == 'native-claim-invalid':
            need(item.get('native_grades') is None, 'invalid native claim cannot supply native grades')
            continue
        grade = item['native_grades']; validate_native(grade, COMPACT)
        need(grade['miner'] == item['miner'] and grade['batch_sha256'] == item['batch_sha256'] and
             grade['manifest_sha256'] == digest(m) and
             len(grade['results']) == m['group_contract']['group_size'], 'complete original native grade binding')
        counts = Counter(g['classification'] for g in grade['results'])
        passes = meets_minima(counts, m['group_contract'])
        need(passes == (reason != 'native-outcome-minima-not-met'), 'native outcome minima disposition')
        if passes: mixed.append(row)
        if reason in (None, 'training-task-budget'): eligible.append(digest(row))
    task_counts = Counter((r['env_id'],r['index']) for r in mixed)
    eligible = [digest(r) for r in rows if digest(r) in eligible and task_counts[(r['env_id'],r['index'])] == 1]
    return dict(version=VERSION, round=round, committed_at=committed_at, manifest_document=deepcopy(opening),
        captures=deepcopy(captures), commitments=deepcopy(commitments), dispatch_document=deepcopy(dispatch),
        registrations_document=deepcopy(registrations), records=population(rows, ordered=True),
        eligible_evidence_ids=sorted(eligible), training_requires_sampling_audit=False)


def authenticate_population(document, authority, *, manifest_guard):
    p = signed(document, authority)
    need(p.get('version') == VERSION, 'explicit new complete-group audit population')
    expected = build_population(p['manifest_document'],p['captures'],p['commitments'],
        p['dispatch_document'],p['registrations_document'],round=p['round'],committed_at=p['committed_at'],
        authority=authority,manifest_guard=manifest_guard)
    need(canonical(p) == canonical(expected), 'canonical reconstructed MiMo population')
    return p


def build_audit_job(p, rows, *, authority, manifest_guard, job_id, created_at, expires_at,
                    source_files, runtime_versions, read_url, checkpoint_read_urls=None):
    """Unsigned new-version job; caller persists random selection BEFORE signing."""
    m = signed(p['manifest_document'], authority); manifest_guard(m)
    need(p.get('version') == VERSION and type(rows) is list and 1 <= len(rows) <= 256,
         'bounded complete-group audit selection')
    identities = {digest(r): r for r in p['records']}; seen=set(); objects=[]; parents={}
    for row in rows:
        key=digest(row); need(key in identities and row == identities[key] and key not in seen,
                              'exact selected original audit row'); seen.add(key)
        miner=row['miner']; parent=p['commitments'][miner]
        claims=parent['payload']['groups']; matches=[c for c in claims if c['batch_sha256']==row['batch_sha256']]
        need(len(matches)==1, 'one committed selected group'); slot=matches[0]['slot']
        from .complete_group_transport import paths
        objects.append(submission_reference(m,miner,parent,slot,read_url(paths(m,miner,slot)[0])))
        parents[miner]=parent
    value=dict(schema=1,worker_policy=WORKER,role='verify',job_id=job_id,created_at=created_at,
        expires_at=expires_at,manifest=deepcopy(p['manifest_document']),source_files=deepcopy(source_files),
        runtime_versions=deepcopy(runtime_versions),submissions=objects,group_commitments=deepcopy(parents))
    if checkpoint_read_urls is not None:value['checkpoint_read_urls']=deepcopy(checkpoint_read_urls)
    return value


def admit_queue_reports(queue_rows, populations, *, authority, verifiers, approved_execution,
                       manifest_guard, cutoff):
    """Original signed terminal queue results, without any legacy reinterpretation.

    Invalid/missing worker evidence is refused, never itself proof of miner fraud.
    Infrastructure failures do not acquire a synthetic scientific observation.
    """
    parsed=lambda v:json.loads(v) if type(v) is str else v
    records=[]; openings={}
    for document in populations:
        p=authenticate_population(document,authority,manifest_guard=manifest_guard)
        m=signed(p['manifest_document'],authority)
        need(m['epoch'] not in openings,'one population per epoch');openings[m['epoch']]=p['manifest_document']
        records.extend(p['records'])
    lookup={(r['epoch'],r['miner'],r['batch_sha256']):r for r in population(records)}
    admissions={};refused=[]
    from nacl.exceptions import BadSignatureError
    for q in queue_rows:
        try:
            need(q.get('status')=='complete' and q.get('role')=='verify' and q.get('worker') in verifiers,
                 'completed authorized verifier queue row')
            worker=q['worker'];job=signed(parsed(q['envelope']),authority);m=signed(job['manifest'],authority)
            manifest_guard(m)
            need(job['manifest']==openings.get(m['epoch']) and job.get('worker_policy')==WORKER and
                 job.get('role')=='verify' and job.get('job_id')==q['id'] and digest(job)==q['digest'],
                 'exact original signed audit opening and job')
            execution=approved_execution.get(m['source_bundle']['sha256'])
            need(type(execution)is dict and set(execution)=={'source_files','runtime_versions'} and
                 job.get('source_files')==execution['source_files'] and
                 job.get('runtime_versions')==execution['runtime_versions'],'approved actual verifier execution source')
            validate_submissions(m,job['submissions'],job['group_commitments'])
            report=parsed(q['report']);request=signed(parsed(q['report_request']),worker)
            need(digest(report)==q['report_digest'] and request.get('action')=='report' and
                 request.get('job_id')==job['job_id'] and request.get('token')==q['token'] and
                 request.get('report')==report,'original worker-signed terminal queue report')
            validate_report(report,job,m,authority,digest(job),cutoff)
            need(report['completed_at']<=cutoff,'postcutoff audit evidence')
            observed=[]
            for obj,audit in zip(job['submissions'],report['audits'],strict=True):
                ref=obj['commitment_ref'];row=lookup.get((m['epoch'],ref['miner'],ref['batch_sha256']))
                need(row is not None and row['commitment_sha256']==ref['commitment_sha256'] and
                     row['proof_sha256']==obj['sha256'] and row['verifier_contract_sha256']==cohort(m) and
                     report['completed_at']>=row['committed_at'],'original closed population audited')
                outcome=audit['outcomes'][0]
                if outcome['valid'] is True:
                    need(outcome['fully_audited'] is True and len(outcome['rollouts'])==m['group_contract']['group_size'] and
                         all(type(r.get('positions'))is int and r['positions']>0 and r.get('all_intervals_verified')is True
                             for r in outcome['rollouts']), 'all nonce token positions verified')
                    summary=audit['accepted'][0];counts=summary['outcome_counts']
                    need(meets_minima(counts, m['group_contract']) and
                         summary['generated_tokens']==sum(r['positions'] for r in outcome['rollouts']),
                         'audited native minima and exact total token coverage')
                    result='verified_valid'
                else:
                    need(outcome['failure_kind']=='invalid_sample','definite invalid sample only')
                    result='confirmed_invalid'
                observed.append(dict(version='continuous-audit-observation-v1',epoch=row['epoch'],
                    checkpoint=row['checkpoint'],miner=row['miner'],batch_sha256=row['batch_sha256'],
                    commitment_sha256=row['commitment_sha256'],verifier_contract_sha256=row['verifier_contract_sha256'],
                    outcome=result,completed_at=report['completed_at'],job_sha256=digest(job)))
            value=dict(verifier=worker,observations=observed,original_report_request_sha256=digest(parsed(q['report_request'])))
            need(digest(job)not in admissions or admissions[digest(job)]==value,'conflicting original audit job')
            admissions[digest(job)]=value
        except (ValueError,KeyError,TypeError,BadSignatureError) as error:
            refused.append(dict(job_sha256=q.get('digest'),reason='MiMo-worker-evidence-rejected',
                                error_type=type(error).__name__,fraud_claim=False,validity_credit=False))
    return admissions,refused
