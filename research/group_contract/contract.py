"""Pure future group admission, distinct from model/proof verification.

The caller authenticates the epoch manifest, miner and environment task, and
authenticates verifier results BEFORE using this module. JSON from a miner can
never supply the trusted assessments argument. No function here runs a model,
verifies a cryptographic signature or proves historical physical execution.

Single-turn text math is the initial explicit harness scope. Generic group size
does not imply generic multi-turn terminal semantics.
"""
from dataclasses import dataclass
import hashlib
import json
import math

VERSION = 'complete-canonical-group-v1'
BATCH_VERSION = 'complete-canonical-group-batch-v1'
MAX_GROUP_SIZE = 128


def need(value, message):
    if not value: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(value):
    need(type(value) is str and len(value) == 64 and set(value) <= set('0123456789abcdef'), 'exact lowercase sha256/public identity')
    return value


def fields(value, names, what):
    need(type(value) is dict and set(value) == set(names), 'exact '+what+' fields; no extra salt/offset/selector')


def loads_unique(data, *, max_bytes=16*1024**2):
    need(type(data) is bytes and len(data) <= max_bytes, 'bounded JSON bytes')
    def pairs(rows):
        out = {}
        for key, value in rows:
            need(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def constant(value): raise ValueError('nonfinite JSON constant '+value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def validate_contract(contract):
    fields(contract, ('version','epoch','checkpoint','runtime_sha256','randomness','group_size',
                      'min_success','min_completed_failure','sampler','outcome_policy'), 'contract')
    need(contract['version'] == VERSION, 'prospective version only; historical contracts require original validator')
    need(type(contract['epoch']) is str and bool(contract['epoch']), 'epoch identity')
    for name in ('checkpoint','runtime_sha256','randomness'): digest(contract[name])
    n = contract['group_size']; good = contract['min_success']; bad = contract['min_completed_failure']
    need(type(n) is int and 2 <= n <= MAX_GROUP_SIZE, 'bounded group size')
    need(type(good) is int and type(bad) is int and good >= 1 and bad >= 1 and good+bad <= n,
         'independent outcome minima within total group size')
    need(contract['outcome_policy'] == 'completed-eos-math-unresolved-masked-v1', 'explicit prospective outcome policy')
    s = contract['sampler']
    fields(s, ('version','temperature','top_p','support_size','eos_token_id','max_output_tokens','max_context_tokens'), 'sampler')
    need(s['version'] == 'cached-eager-inverse-cdf-v1', 'declared prescribed sampler')
    need(type(s['temperature']) in (int,float) and math.isfinite(s['temperature']) and s['temperature'] > 0, 'finite positive temperature')
    need(type(s['top_p']) in (int,float) and s['top_p'] == 1., 'initial full-support top_p=1 profile')
    need(type(s['support_size']) is int and 2 <= s['support_size'] <= 200000, 'bounded supported vocabulary')
    need(type(s['eos_token_id']) is int and 0 <= s['eos_token_id'] < s['support_size'], 'supported EOS')
    need(type(s['max_output_tokens']) is int and 1 <= s['max_output_tokens'] <= 32768,
         'bounded response horizon')
    need(type(s['max_context_tokens']) is int and s['max_output_tokens'] < s['max_context_tokens'] <= 65536,
         'bounded context')
    return contract


def validate_task(task):
    fields(task, ('env_id','environment_version','index','task_hash','prompt_sha256'), 'trusted task')
    need(all(type(task[k]) is str and task[k] for k in ('env_id','environment_version')), 'environment identity/version')
    need(type(task['index']) is int and task['index'] >= 0, 'task index')
    digest(task['task_hash']); digest(task['prompt_sha256'])
    return task


def context(contract, miner, task):
    validate_contract(contract); digest(miner); validate_task(task)
    return dict(contract=contract, miner=miner, task=task)


def sampling_receipt(contract, miner, task, nonce):
    binding = context(contract, miner, task)
    need(type(nonce) is int and 0 <= nonce < contract['group_size'], 'canonical nonce range')
    return dict(version=VERSION, binding_sha256=sha(binding), attempt=nonce)


def uniform(contract, miner, task, nonce, position):
    """Deterministic challenge recipe, not an assertion tokens followed it."""
    sampling_receipt(contract, miner, task, nonce)
    need(type(position) is int and 0 <= position < contract['sampler']['max_output_tokens'], 'assigned token position')
    body = dict(context(contract, miner, task), attempt=nonce, turn=0, position=position)
    return (int.from_bytes(hashlib.sha256(canonical(body)).digest()[:8], 'big') >> 11) / 2**53


def credit_id(contract, miner, task):
    """One canonical group identity, independent of tokens and upload slot/name."""
    return sha(dict(domain='one-group-credit-v1', **context(contract, miner, task)))


@dataclass(frozen=True)
class Admission:
    batch_sha256: str
    contract_sha256: str
    context_sha256: str
    credit_id: str
    rollout_sha256: tuple[str, ...]
    claimed_counts: tuple[int, int, int]  # positive, negative, neutral
    claimed_outcomes: tuple[str, ...]
    claimed_qualifies: bool
    sampling_verified: bool = False


def validate_batch(batch, contract, *, authenticated_miner, trusted_task):
    """Cheap structural admission only. Claimed outcomes remain untrusted."""
    binding = context(contract, authenticated_miner, trusted_task)
    fields(batch, ('schema','epoch','checkpoint','miner','env_id','environment_version','index','sample_index',
                   'task_hash','group_contract_sha256','rollouts'), 'batch')
    need(batch['schema'] == BATCH_VERSION, 'explicit prospective batch schema')
    need(batch['epoch'] == contract['epoch'] and batch['checkpoint'] == contract['checkpoint']
         and batch['miner'] == authenticated_miner and batch['group_contract_sha256'] == sha(contract), 'epoch/checkpoint/miner/contract binding')
    need(all(batch[k] == trusted_task[k] for k in ('env_id','environment_version','index','task_hash'))
         and type(batch['index']) is int and type(batch['sample_index']) is int and batch['sample_index'] == batch['index'], 'same trusted task')
    rows = batch['rollouts']; n = contract['group_size']; s = contract['sampler']
    need(type(rows) is list and len(rows) == n, 'all N assigned attempts required')
    counts = {'positive':0,'negative':0,'neutral':0}; hashes = []
    for nonce, row in enumerate(rows):
        fields(row, ('index','sample_index','env_id','environment_version','task_hash','seed','sampling',
                     'classification','reward','turns'), 'rollout')
        need(type(row['seed']) is int and row['seed'] == nonce, 'canonical ordered nonce set 0..N-1 exactly once')
        need(all(row[k] == trusted_task[k] for k in ('index','env_id','environment_version','task_hash'))
             and type(row['index']) is int and type(row['sample_index']) is int and row['sample_index'] == trusted_task['index'], 'rollout task binding')
        fields(row['sampling'], ('version','binding_sha256','attempt'), 'sampling receipt')
        need(type(row['sampling']['attempt']) is int and
             row['sampling'] == sampling_receipt(contract, authenticated_miner, trusted_task, nonce), 'exact assigned sampling receipt')
        turns = row['turns']; need(type(turns) is list and len(turns) == 1, 'initial single-turn math scope')
        turn = turns[0]; fields(turn, ('prompt','output','stop_reason'), 'turn')
        prompt = turn['prompt']; output = turn['output']
        for tokens in (prompt, output):
            need(type(tokens) is list and tokens and all(type(t) is int and 0 <= t < s['support_size'] for t in tokens), 'supported token sequences')
        need(sha(prompt) == trusted_task['prompt_sha256'], 'exact trusted prompt, no miner prefix/salt')
        need(len(output) <= s['max_output_tokens'] and len(prompt)+s['max_output_tokens'] <= s['max_context_tokens'], 'full untruncated horizon')
        eos = s['eos_token_id']; need(eos not in output[:-1], 'no generated tokens after first EOS')
        stop = turn['stop_reason']; category = row['classification']
        need(type(category) is str and category in counts, 'explicit outcome class')
        reward = {'positive':1.,'negative':0.,'neutral':None}[category]
        need(row['reward'] == reward and (row['reward'] is None or type(row['reward']) in (int,float)), 'class/reward consistency')
        if stop == 'eos': need(output[-1] == eos, 'claimed EOS requires actual EOS token')
        elif stop == 'budget':
            need(len(output) == s['max_output_tokens'] and output[-1] != eos and category == 'neutral',
                 'budget caps are unresolved and cannot satisfy negative quota')
        else: raise ValueError('unsupported terminal condition; no replacement attempt')
        counts[category] += 1; hashes.append(sha(row))
    # No content-deduplication filter: independent canonical attempts can agree.
    qualifies = counts['positive'] >= contract['min_success'] and counts['negative'] >= contract['min_completed_failure']
    return Admission(sha(batch), sha(contract), sha(binding), credit_id(contract, authenticated_miner, trusted_task),
                     tuple(hashes), tuple(counts[k] for k in ('positive','negative','neutral')),
                     tuple(r['classification'] for r in rows), qualifies)


@dataclass(frozen=True)
class VerifierAssessment:
    """Caller-owned authenticated result; NOT a type accepted in miner JSON.

    Construction alone does not authenticate evidence. The adapter must verify
    the registered verifier's signature, job/source/profile/checkpoint binding
    and frozen input digest before passing these records.
    """
    nonce: int
    rollout_sha256: str
    context_sha256: str
    sampling: str  # valid / invalid / indeterminate / unchecked
    terminal: str  # valid / invalid / indeterminate / unchecked
    outcome: str  # positive / negative / neutral / indeterminate / unchecked


def audited_verdict(admission, assessments, contract):
    validate_contract(contract)
    need(isinstance(admission, Admission) and type(assessments) in (list,tuple)
         and len(assessments) == contract['group_size'], 'one trusted assessment per original attempt')
    need(admission.contract_sha256 == sha(contract), 'audit applies original admission contract')
    invalid = []; uncertain = []; mismatched = []; observed = {'positive':0,'negative':0,'neutral':0}
    for nonce, a in enumerate(assessments):
        need(type(a) is VerifierAssessment and type(a.nonce) is int and a.nonce == nonce, 'ordered caller-owned verifier assessments')
        need(a.rollout_sha256 == admission.rollout_sha256[nonce] and a.context_sha256 == admission.context_sha256,
             'verifier result binds actual submitted bytes and challenge')
        need(type(a.sampling) is str and type(a.terminal) is str and type(a.outcome) is str
             and a.sampling in ('valid','invalid','indeterminate','unchecked') and a.terminal in ('valid','invalid','indeterminate','unchecked')
             and a.outcome in ('positive','negative','neutral','indeterminate','unchecked'), 'explicit known verifier statuses')
        if a.sampling == 'invalid' or a.terminal == 'invalid': invalid.append(nonce)
        elif a.sampling != 'valid' or a.terminal != 'valid' or a.outcome not in observed: uncertain.append(nonce)
        if a.outcome in observed: observed[a.outcome] += 1
        if a.outcome in observed and a.outcome != admission.claimed_outcomes[nonce]: mismatched.append(nonce)
    counts = tuple(observed[k] for k in ('positive','negative','neutral'))
    # An authenticated outcome mismatch is a rejection, not an automatic fraud
    # or wallet-slashing determination. Escalation/economics belong elsewhere.
    mismatch = bool(mismatched)
    if invalid or mismatch: status = 'invalid'
    elif uncertain: status = 'indeterminate'
    elif observed['positive'] >= contract['min_success'] and observed['negative'] >= contract['min_completed_failure']: status = 'valid_qualifying'
    else: status = 'valid_nonqualifying'
    return dict(status=status, counts=observed, invalid_nonces=invalid, uncertain_nonces=uncertain,
                claimed_outcomes_mismatch=mismatch, sampling_checked=not invalid and not uncertain,
                mismatched_outcome_nonces=mismatched, historical_execution_proven=False, credit_id=admission.credit_id)


def register_eligible_once(existing, admission):
    """Register a structurally eligible claimed group WITHOUT waiting for audits.

    The returned delta is a provisional unique-group count, not a payment or
    proof of correctness. The caller authenticates submission/manifest identity
    and persists the transition atomically under a unique group ID. Training
    eligibility and rewards continue through their existing separate policies.
    """
    need(type(existing) is dict and isinstance(admission, Admission), 'caller-owned eligibility ledger')
    need(admission.claimed_qualifies and admission.sampling_verified is False,
         'structurally eligible claimed group; no audit or proof assertion')
    record = dict(batch_sha256=admission.batch_sha256, contract_sha256=admission.contract_sha256,
                  basis='structurally_eligible_claimed_group', assurance='unaudited',
                  sampling_verified=False, reward_assigned=False)
    previous = existing.get(admission.credit_id)
    if previous is not None:
        need(previous == record, 'canonical eligible group already registered with different bytes')
        return dict(existing), 0
    return dict(existing, **{admission.credit_id:record}), 1


def record_audit_once(existing_evidence, admission, verdict, *, authenticated_evidence_sha256):
    """Keep audit evidence independently of provisional contribution counting.

    The caller has already authenticated the verifier report. Valid, invalid and
    uncertain results are all retained; this function neither assigns weights
    nor rewrites original eligible counts. Multiple distinct reports remain
    evidence for the existing probabilistic trust/penalty policy, not automatic
    last-report-wins adjudication. Persist atomically by unique evidence digest.
    """
    need(type(existing_evidence) is dict and isinstance(admission, Admission), 'caller-owned audit ledger')
    digest(authenticated_evidence_sha256)
    need(type(verdict) is dict and verdict.get('credit_id') == admission.credit_id
         and verdict.get('status') in ('valid_qualifying','valid_nonqualifying','invalid','indeterminate'),
         'separate matching verifier evidence of known status')
    record = dict(group_id=admission.credit_id, batch_sha256=admission.batch_sha256,
                  verdict=json.loads(canonical(verdict)), authenticated_evidence_sha256=authenticated_evidence_sha256)
    previous = existing_evidence.get(authenticated_evidence_sha256)
    if previous is not None:
        need(previous == record, 'authenticated evidence digest already records different result')
        return dict(existing_evidence), 0
    return dict(existing_evidence, **{authenticated_evidence_sha256:record}), 1


def audited_credit_once(existing, admission, verdict):
    """Audit-only utility, NEVER the admission/counting path for the network.

    This does not replace existing cross-miner task collision or hourly trust
    weighting rules. It tracks only the confirmed subset, for diagnostics or
    audit-only comparisons. Use register_eligible_once for unaudited contribution
    counts; keep audit evidence separately with record_audit_once. The production
    mechanism must not wait for this helper before counting eligible groups or
    admitting structurally eligible unaudited inputs to its training policy.
    """
    need(type(existing) is dict and isinstance(admission, Admission), 'caller-owned credit ledger')
    need(verdict.get('credit_id') == admission.credit_id and verdict.get('status') == 'valid_qualifying', 'qualified matching verifier result')
    previous = existing.get(admission.credit_id)
    if previous is not None:
        need(previous == admission.batch_sha256, 'canonical group already credited with different bytes')
        return dict(existing), 0
    return dict(existing, **{admission.credit_id:admission.batch_sha256}), 1
