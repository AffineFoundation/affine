"""Single hourly SN120 writer, independent of training and opening completion."""
import argparse
import hashlib
import json
import math
import signal
import time
from pathlib import Path
from nacl.signing import SigningKey
from subnet.chain import ChainAdapter, OWNER
from subnet.current_assessment import calculate, fallback, recipients, HALF_LIFE_HOURS
from subnet.live_reward_bridge import signed, sha, writer_gate
from ops.live_reward_writer import (authenticate_cutover, global_lock, guard_files,
                                    observe_units, process_identity, read, file_hash)
from ops.live_reward_exporter import atomic, sign

VERSION = 'hourly-current-assessment-writer-v1'
NUMERICAL_RESOLUTION_VERSION = 'hourly-current-assessment-writer-reviewed-numerical-v2'


def validate_policy(document, authority, cutover_document, anchor_document):
    p = signed(document, authority)
    fields = {'version','half_life_hours','first_window','netuid','owner_hotkey',
              'audit_config','source_admission_sha256','verifiers','module_hashes',
              'cutover_sha256','anchor_sha256','execute_enabled','zero_total_policy',
              'registration_change_policy'}
    expected_fields = fields | ({'numerical_resolution_policy_sha256'} if p['version'] == NUMERICAL_RESOLUTION_VERSION else set())
    if set(p) != expected_fields or p['version'] not in (VERSION, NUMERICAL_RESOLUTION_VERSION) or p['half_life_hours'] != HALF_LIFE_HOURS:
        raise ValueError('exact hourly assessment policy')
    if p['version'] == NUMERICAL_RESOLUTION_VERSION:
        import importlib.util
        from subnet.continuous_audit_policy import valid_digest
        if not valid_digest(p['numerical_resolution_policy_sha256']): raise ValueError('explicit ROOT numerical resolution digest')
        required = {str(Path(__file__).resolve())} | {str(Path(importlib.util.find_spec(name).origin).resolve()) for name in ('subnet.numerical_resolution', 'subnet.continuous_audit_policy', 'ops.current_assessment_evidence')}
        if not required <= {str(Path(path).resolve()) for path in p['module_hashes']}:
            raise ValueError('present ROOT pins for numerical resolution execution modules')
    if p['netuid'] != 120 or p['owner_hotkey'] != OWNER or type(p['execute_enabled']) is not bool:
        raise ValueError('owner/netuid/execution policy')
    if p['zero_total_policy'] != 'owner-sink-v1':
        raise ValueError('explicit zero-total policy')
    if p['registration_change_policy'] != 'current-hotkey-snapshot-v1':
        raise ValueError('explicit current-registration policy')
    if p['cutover_sha256'] != sha(cutover_document) or p['anchor_sha256'] != sha(anchor_document):
        raise ValueError('single writer cutover binding')
    if type(p['first_window']) is not int or p['first_window'] % 3600:
        raise ValueError('first assessment UTC hour')
    if not p['verifiers'] or len(p['verifiers']) != len(set(p['verifiers'])):
        raise ValueError('exact admitted verifier identities')
    for file, digest in p['module_hashes'].items():
        if file_hash(file) != digest:
            raise ValueError('reviewed assessment writer bytes')
    return p


def run_once(policy_document, cutover_document, anchor_document, authority, *, execute=False,
             adapter_factory=ChainAdapter, evidence_loader=None):
    p = validate_policy(policy_document, authority, cutover_document, anchor_document)
    c, _ = authenticate_cutover(cutover_document, anchor_document, authority)
    if execute and not p['execute_enabled']:
        raise ValueError('assessment writer execution not admitted')
    if evidence_loader is None:
        from ops.current_assessment_evidence import load_evidence
        evidence_loader = load_evidence
    with global_lock(c['global_lock_path']):
        guard_files(c); observe_units()
        now = time.time(); cutoff = int(now) // 3600 * 3600
        if cutoff < p['first_window']:
            return dict(status='waiting_for_hour', chain_executed=False)
        directory = Path(c['reward_state']) / 'current-assessment-v1'
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = SigningKey(bytes.fromhex(Path(c['authority_seed_file']).read_text().strip()))
        if key.verify_key.encode().hex() != authority:
            raise ValueError('assessment signing authority')
        chain_status = read(Path(c['chain_state'])/'weights.json') if (Path(c['chain_state'])/'weights.json').exists() else {}
        if chain_status.get('last_submitted_window', -1) >= cutoff:
            return dict(status='already_submitted', window_end=cutoff, chain_executed=False)
        cursor_path = directory/'submission.json'
        cursor = read(cursor_path) if cursor_path.exists() else {}
        if cursor.get('status') == 'submitting':
            raise RuntimeError('uncertain chain outcome requires actual-chain reconciliation')
        target = directory/('assessment-'+str(cutoff)+'.json')
        if target.exists():
            document = read(target); assessment = signed(document, authority)
            if assessment['cutoff'] != cutoff or assessment['writer_policy_sha256'] != sha(policy_document):
                raise ValueError('immutable hourly assessment binding')
        else:
            try:
                numerical = dict(expected_numerical_resolution_policy_sha256=p['numerical_resolution_policy_sha256']) if p['version'] == NUMERICAL_RESOLUTION_VERSION else {}
                evidence = evidence_loader(p['audit_config'], authority=authority, cutoff=cutoff,
                                           verifiers=p['verifiers'], expected_source_admission_sha256=p['source_admission_sha256'], **numerical)
                assessment = calculate(evidence['snapshots'], evidence['committed_at_by_epoch'], cutoff)
                assessment.update(evidence_cutoff=cutoff, assessment_stale=False,
                                  evidence_hashes=evidence['evidence_hashes'],
                                  evidence_refusals=evidence.get('refused', []),
                                  evidence_exclusions=evidence.get('excluded', []),
                                  evidence_deferrals=evidence.get('deferred', []))
            except (TimeoutError, ConnectionError) as error:
                last_path = directory/'last-valid-assessment.json'
                if not last_path.exists():
                    atomic(directory/'health.json', dict(status='no_valid_assessment', error_type=type(error).__name__, at=now))
                    raise
                assessment = fallback(signed(read(last_path), authority), cutoff, type(error).__name__)
            assessment['writer_policy_sha256'] = sha(policy_document)
            document = sign(assessment, key); atomic(target, document)
            if not assessment['assessment_stale']:
                atomic(directory/'last-valid-assessment.json', document)
        adapter = adapter_factory(c['chain_state'], netuid=120, expected_owner=OWNER)
        try:
            registrations = adapter.registrations()
            points, selected, excluded = recipients(assessment, registrations)
            identity = process_identity(); now = time.time()
            guard_files(c); old_units = observe_units()
            receipt = sign(dict(version='single-live-reward-writer-v1', netuid=120, observed_at=now,
                                global_writer_lock_held=True, legacy_validator_guard_verified=True,
                                old_writers=old_units, **identity), key)
            if execute:
                writer_gate(receipt, authority, now=now, boot_id=identity['boot_id'],
                            writer_pid=identity['writer_pid'], writer_ticks=identity['writer_start_ticks'])
                atomic(cursor_path, dict(status='submitting', window_end=cutoff, assessment_sha256=sha(document)))
            result = adapter.submit_hour(points, selected, cutoff, execute=execute,
                                         zero_total_policy=p['zero_total_policy'],
                                         registration_change_policy=p['registration_change_policy'])
            # Rate-limited/planned responses are safe to retry the SAME assessment.
            atomic(directory/'last-run.json', dict(at=now, result=result, assessment_sha256=sha(document),
                   positive_identities=len(points), excluded_unregistered=excluded,
                   assessment_stale=assessment['assessment_stale'], chain_executed=result['status']=='submitted'))
            if execute:
                atomic(cursor_path, dict(status=result['status'], window_end=cutoff, assessment_sha256=sha(document)))
            return result
        finally:
            chain = getattr(adapter, 'chain', None)
            if chain is not None and hasattr(chain, 'close'):
                chain.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('policy','cutover','anchor','authority'):p.add_argument('--'+field, required=True)
    p.add_argument('--execute', action='store_true'); args = p.parse_args()
    def timeout(*_):raise TimeoutError('bounded current assessment writer')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(720)
    try:
        result = run_once(read(args.policy), read(args.cutover), read(args.anchor), args.authority, execute=args.execute)
        print(json.dumps({k: result[k] for k in ('status','window_end','remaining_blocks') if k in result}))
    finally:signal.alarm(0)


if __name__ == '__main__':main()
