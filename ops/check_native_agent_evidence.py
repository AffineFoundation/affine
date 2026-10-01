"""Check signed controlled Agent evidence; no model, Docker or chain execution.

Operator reports are authenticated evidence, not a cryptographic proof of GPU
execution. This inspector checks byte binding, not upstream orchestrator coverage.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from subnet.backend_jobs import canonical, signed, file_map
from subnet.native_agent_isolation import validate_descriptor


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect(state, authority):
    require(len(bytes.fromhex(authority)) == 32, 'operator public authority')
    audit = signed(json.loads((state/'signed-proof-pair-audit.json').read_text()), authority)
    plan_path = state/'model-plan.json'
    plan = json.loads(plan_path.read_text())
    descriptor = validate_descriptor(plan['descriptor'])
    require(audit['kind'] == 'controlled-original-agent-proof-pair-v1', 'controlled audit kind')
    require(audit['approved_plan_sha256'] == hashlib.sha256(canonical(plan)).hexdigest(), 'canonical approved plan')
    require(audit['descriptor_sha256'] == hashlib.sha256(canonical(descriptor)).hexdigest(), 'approved native images')
    require(audit['checkpoint'] == plan['checkpoint_id'] == file_map(plan['checkpoint_files']), 'approved weights identity')
    require(audit['full_model_recompute_and_native_tools_grader_replay'] is True and
            audit['payable'] is False and audit['chain_transactions'] is False and
            audit['production_admitted'] is False and audit['full_verifiers_orchestrator'] is False and
            type(audit['optimizer_updates']) is int and audit['optimizer_updates'] == 0,
            'controlled scope and authenticated verification')
    root = Path(__file__).resolve().parents[1]
    require(plan['probe_sha256'] == hashlib.sha256((root/'ops/probe_native_agent.py').read_bytes()).hexdigest(), 'approved probe source')
    for name, digest in plan['source_files'].items():
        require(name.startswith('subnet/') and '..' not in Path(name).parts, 'source inventory path')
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, 'approved implementation source')
    required_files = {'positive.json', 'negative.json', 'generate-summary.json', 'verify-summary.json',
                      'positive-0.npy', 'positive-1.npy', 'positive-2.npy', 'positive-3.npy', 'negative-0.npy'}
    require(set(audit['files']) == required_files, 'complete controlled artifact inventory')
    out = state/'genuine-pair-v2'
    for name, entry in audit['files'].items():
        body = (out/name).read_bytes()
        require(len(body) == entry['size'] and hashlib.sha256(body).hexdigest() == entry['sha256'], 'signed artifact bytes')
    verify_body = (out/'verify-summary.json').read_bytes()
    require(hashlib.sha256(verify_body).hexdigest() == audit['verification_sha256'], 'verification report bytes')
    report = json.loads(verify_body)
    require(report['approved_plan_sha256'] == audit['approved_plan_sha256'] and report['mode'] == 'verify', 'fresh verifier report binding')
    require(report['numerical_tolerances'] == dict(TOPLOC_errors=0, logprobs_atol=1e-5, logprobs_rtol=0), 'strict proof policy')
    require(report['records'] == [dict(full_proof_verified=True, kind='negative', reward=0.0, tool_calls=0, turns=1),
                                 dict(full_proof_verified=True, kind='positive', reward=1.0, tool_calls=3, turns=4)],
            'controlled positive and negative evidence')
    mutations = audit['mutations']
    require(mutations == json.loads((state/'mutations-v2-results.json').read_text()), 'mutation evidence binding')
    # Mutation runner records the original file bytes; inference records canonical JSON.
    require(mutations['approved_plan_sha256'] == hashlib.sha256(plan_path.read_bytes()).hexdigest(), 'raw mutation plan bytes')
    require(mutations['mutation_runner_sha256'] == hashlib.sha256((root/'ops/probe_native_agent_mutations.py').read_bytes()).hexdigest(), 'mutation runner source')
    expected = {'probabilities': 'native full logprobs', 'proof': 'native strict TOPLOC',
                'tool_observation': 'original native tool replay', 'reward': 'native full artifact equality'}
    require(len(mutations['controls']) == 4 and {r['kind'] for r in mutations['controls']} == set(expected), 'all mutation cases')
    for row in mutations['controls']:
        require(row['rejected'] is True and type(row['returncode']) is int and row['returncode'] != 0 and
                row['expected_check'] == expected[row['kind']], 'expected mutation rejection')
    return dict(success=True, checked_at=time.time(), checkpoint=audit['checkpoint'],
        task_name=descriptor['task_name'], files_hashed=len(audit['files']),
        artifact_bytes=sum(e['size'] for e in audit['files'].values()), positive_reward=1, negative_reward=0,
        mutations_rejected=4, reports_are_operator_authenticated=True,
        fresh_model_execution_in_this_check=False, native_replay_execution_in_this_check=False,
        full_verifiers_orchestrator=False, training_performed=False, production_admitted=False, chain_transactions=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = inspect(args.state, args.authority)
    if args.output:
        args.output.write_bytes(canonical(result))
        args.output.chmod(0o600)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
