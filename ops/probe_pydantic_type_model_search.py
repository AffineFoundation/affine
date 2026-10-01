"""Versioned public-AST candidate qualification; no optimizer, credit or chain writes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, base64, hashlib, json, time
from pathlib import Path
from nacl.signing import VerifyKey
from subnet.storage import canonical
from ops.probe_pydantic_model_search import source_membership, gpu_wait

REVISION = 'public-pydantic-same-key-wrong-type-model-v2'


def approved(document, authority):
    if document.get('signer') != authority:
        raise ValueError('approval authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']), base64.b64decode(document['signature'], validate=True))
    plan = document['payload']
    if plan.get('revision') != REVISION or plan.get('payable') is not False or plan.get('chain_transactions') is not False:
        raise ValueError('versioned nonpayable qualification')
    if type(plan.get('search_budget')) is not int or not 1 <= plan['search_budget'] <= 32:
        raise ValueError('bounded search')
    indices = plan.get('indices')
    if not isinstance(indices, list) or not indices or len(indices) > 8 or any(type(i) is not int or not 0 <= i < 16 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('unique original training indices only')
    if plan.get('environment', {}).get('id') != 'affine_pydantic' or plan['environment'].get('adapter') != 'prime_v1':
        raise ValueError('original Pydantic source')
    if plan.get('harness') != {'version': 'text-tools-v1', 'policy': 'candidates', 'temperature': 4.0, 'top_p': 1.0, 'max_output_tokens': 512}:
        raise ValueError('exact per-task candidate policy')
    from subnet.backend_jobs import BACKEND_PROFILE, NUMERICAL_POLICY
    if plan.get('backend_profile') != BACKEND_PROFILE or plan.get('numerical_policy') != NUMERICAL_POLICY:
        raise ValueError('strict numerical profile')
    source_membership(plan.get('source_files'))
    for name, digest in plan['source_files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('source pin')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan.get('probe_sha256'):
        raise ValueError('probe source pin')
    from subnet import public_pydantic_type_mutation as public_pydantic
    if hashlib.sha256(Path(public_pydantic.__file__).read_bytes()).hexdigest() != plan.get('generator_sha256'):
        raise ValueError('public generator pin')
    return plan


def task_harness(spec, base, index):
    from subnet.environments import create_session
    from subnet.public_pydantic_type_mutation import proposals
    session = create_session(spec)
    try:
        initial = session.reset(index, int(spec.config.get('seed', 0)))
        candidates = proposals(initial['messages'])
        return {**base, 'candidates': candidates}, initial
    finally:
        session.close()


def calibration(runtime, initial, harness):
    import torch
    prompt = runtime.prompt(initial['messages'], initial.get('tools', []))
    scores, lengths = [], []
    for candidate in harness['candidates']:
        output = runtime.tokenizer.encode(candidate, add_special_tokens=False)
        if not output or len(output) > harness['max_output_tokens']:
            raise ValueError('candidate output budget')
        if len(prompt) + len(output) > min(8192, runtime.model.config.max_position_embeddings):
            raise ValueError('candidate context budget')
        lengths.append(len(output))
        with torch.inference_mode():
            logits = runtime.model(torch.tensor([prompt + output], device='cuda'), use_cache=False).logits[0, len(prompt)-1:len(prompt)+len(output)-1]
            lp = torch.log_softmax(logits.float(), -1)
            scores.append(float(lp.gather(1, torch.tensor(output, device='cuda')[:, None]).sum()))
    probabilities = torch.softmax(torch.tensor(scores, dtype=torch.float64) / harness['temperature'], -1).tolist()
    return {'prompt_tokens': len(prompt), 'candidate_token_lengths': lengths, 'equal_token_lengths': len(set(lengths)) == 1,
            'sum_logprobs': scores, 'sampling_probabilities': probabilities, 'policy': 'candidate-sum-logprob-temperature-v1'}


def execute(plan, out, verify=False):
    from subnet.gpu_runtime import GPURuntime
    from subnet.environments import EnvironmentSpec
    from subnet.batches import pack, unpack
    gpu_wait()
    out.mkdir(parents=True, exist_ok=True)
    spec = EnvironmentSpec.from_dict(plan['environment'])
    first_harness, _ = task_harness(spec, plan['harness'], plan['indices'][0])
    runtime = GPURuntime(plan['checkpoint_path'], plan['checkpoint']['files'], plan['environment'], first_harness)
    if verify:
        records = json.loads((out / 'search.json').read_text())
        results = []
        for row in records['rows']:
            harness, initial = task_harness(spec, plan['harness'], row['index'])
            if harness != row['harness'] or initial['task_hash'] != row['task_hash']:
                raise ValueError('fresh public task/candidate reconstruction')
            runtime.configure(plan['environment'], harness)
            data = (out / row['artifact']).read_bytes()
            if hashlib.sha256(data).hexdigest() != row['artifact_sha256']:
                raise ValueError('frozen artifact bytes')
            batches = unpack(data)
            if len(batches) != 1 or batches[0][0]['index'] != row['index'] or batches[0][0]['checkpoint'] != plan['checkpoint']['id']:
                raise ValueError('task/checkpoint artifact binding')
            for rollout, arrays in zip(batches[0][0]['rollouts'], batches[0][1]):
                if not runtime.verify(rollout, arrays):
                    raise ValueError('fresh model/native verification')
            results.append({'index': row['index'], 'rollouts_verified': len(batches[0][0]['rollouts']), 'artifact_sha256': row['artifact_sha256'], 'qualifying_K1L1': row['qualifying_K1L1']})
        report = {'revision': REVISION, 'rows': results, 'checkpoint': plan['checkpoint']['id'], 'independent_model_reload': True, 'full_logits_verified': True, 'toploc_verified': True, 'original_native_replay': True, 'optimizer_ran': False, 'payable': False, 'chain_transactions': False, 'completed_at': time.time()}
        (out / 'fresh-verification.json').write_bytes(canonical(report))
        print(json.dumps(report))
        return
    rows = []
    for index in plan['indices']:
        harness, initial = task_harness(spec, plan['harness'], index)
        runtime.configure(plan['environment'], harness)
        diagnostic = calibration(runtime, initial, harness)
        found, attempts = {}, []
        for attempt in range(plan['search_budget']):
            seed = 100 + index * 1000 + attempt
            rollout, arrays = runtime.rollout(index, seed)
            label = rollout['classification']
            attempts.append({'seed': seed, 'reward': rollout['reward'], 'classification': label, 'rollout_sha256': hashlib.sha256(canonical(rollout)).hexdigest()})
            if label in ('positive', 'negative') and label not in found:
                found[label] = (rollout, arrays)
            if len(found) == 2:
                break
        chosen = [found[label] for label in ('positive', 'negative') if label in found]
        batch = {'env_id': spec.id, 'index': index, 'checkpoint': plan['checkpoint']['id'], 'rollouts': [v[0] for v in chosen]}
        data = pack([(batch, [v[1] for v in chosen])])
        name = f'index-{index}.zip'
        (out / name).write_bytes(data)
        rows.append({'index': index, 'task_hash': initial['task_hash'], 'harness': harness, 'calibration': diagnostic, 'attempts': attempts, 'positive': int('positive' in found), 'negative': int('negative' in found), 'qualifying_K1L1': len(found) == 2, 'artifact': name, 'artifact_sha256': hashlib.sha256(data).hexdigest(), 'artifact_size': len(data)})
        (out / 'search.json').write_bytes(canonical({'revision': REVISION, 'checkpoint': plan['checkpoint']['id'], 'rows': rows, 'policy_kind': 'public-AST-per-task-candidates', 'generator_reads_gold_fields': False, 'search_budget': plan['search_budget'], 'training': False, 'payable': False, 'chain_transactions': False, 'completed_at': time.time()}))
    print(json.dumps([{'index': r['index'], 'positive': r['positive'], 'negative': r['negative'], 'attempts': len(r['attempts'])} for r in rows]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--authority', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    execute(approved(json.loads(a.plan.read_bytes()), a.authority), a.out, a.verify)


if __name__ == '__main__':
    main()
