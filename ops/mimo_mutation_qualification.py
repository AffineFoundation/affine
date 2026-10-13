#!/usr/bin/env python3
"""Private adversarial check: alter tokens and recompute their honest TOPLOC.

This exercises sampling verification, not merely a stale proof/hash rejection.
The original artifact is immutable. Results are diagnostic and never payouts.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subnet.storage import canonical, sha
from subnet.complete_group_codec import decode, limits
from subnet.complete_group_runtime import MimoGroupRuntime
from subnet.complete_group_verification import native_projection, verify_rollout
from subnet.audit_policy import InvalidSample


def run(args):
    import numpy as np
    import torch
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    raw = Path(args.manifest).read_bytes()
    if sha(raw) != args.manifest_sha256:
        raise ValueError('operator-authenticated original manifest required')
    manifest = json.loads(raw)
    if not manifest['epoch'].startswith('private-'):
        raise ValueError('private diagnostic only')
    artifact = Path(args.attempt).read_bytes()
    metadata, tensors = decode(artifact, limits(manifest))[0]
    if set(metadata) != {'original_rollout'} or len(tensors) != 1:
        raise ValueError('original single-attempt diagnostic required')
    original = metadata['original_rollout']
    definition = manifest['environments'][0]
    report = dict(version='mimo-recomputed-proof-mutation-diagnostic-v1',
        qualification=False, manifest_sha256=args.manifest_sha256,
        original_artifact_sha256=sha(artifact), miner=args.miner,
        checkpoint=manifest['checkpoint']['id'], mutations=[],
        scope='direct-rollout-sampling-after-fresh-native-toploc-and-selected-logprobs',
        tests_whole_group_structural_admission=False,
        behavior_witness_rebound=False,
        gpu=torch.cuda.get_device_name())
    runtime = None
    started = time.monotonic()
    try:
        runtime = MimoGroupRuntime(args.checkpoint, manifest, definition['spec'],
                                   definition['harness'], miner=args.miner)
        report['original'] = verify_rollout(runtime, original, tensors[0])
        tokens = original['turns'][0]['output']
        closing = runtime.tokenizer.convert_tokens_to_ids('</think>')
        end = tokens.index(closing) if closing in tokens else len(tokens) - 10
        if end < 16:
            raise ValueError('diagnostic requires an actual reasoning span')
        positions = sorted({5, end // 2, end - 5})
        replacement = runtime.tokenizer.encode(' banana', add_special_tokens=False)[0]
        for position in positions:
            if tokens[position] == replacement:
                raise ValueError('requested mutation must change the original token')
            began = time.monotonic()
            changed = deepcopy(original)
            turn = changed['turns'][0]
            turn['output'][position] = replacement
            turn['text'] = runtime.tokenizer.decode(turn['output'], skip_special_tokens=True)
            # Recompute native outcome, probabilities and proofs on the forged
            # path. Keeping stale evidence would test only the easy failure.
            native = native_projection(runtime, changed)
            for key in ('classification', 'reward', 'stop_reason', 'done', 'observations'):
                turn[key] = native[key]
            changed['classification'] = native['classification']
            changed['reward'] = native['reward']
            acts, full = runtime.compute(turn['prompt'], turn['output'])
            turn['proofs'] = runtime.build_proofs(acts, decode_batching_size=16, topk=128)
            values = np.asarray(full)
            selected = values[np.arange(len(turn['output'])), turn['output']].reshape(-1, 1)
            # The original sampling witness stays visible as original evidence;
            # verify_rollout checks the recomputed token path independently.
            row = dict(position=position, original_token=tokens[position],
                replacement_token=replacement, fresh_toploc=True,
                fresh_selected_probabilities=True,
                native_verdict_preserved=native['classification'] == original['classification'])
            try:
                row['verification'] = verify_rollout(runtime, changed, [selected])
                row['status'] = 'unexpected-acceptance'
            except InvalidSample as error:
                row.update(status='rejected', reason=str(error),
                    rejected_by_sampling=str(error) == 'original cached group sampler mismatch')
            row['seconds'] = time.monotonic() - began
            report['mutations'].append(row)
            print(json.dumps(row), flush=True)
        report['status'] = ('all-recomputed-mutations-rejected-by-sampling' if all(
            row.get('rejected_by_sampling') and row['native_verdict_preserved']
            for row in report['mutations']) else 'requires-investigation')
    except BaseException as error:
        report.update(status='error', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic() - started
        report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(canonical(report))
        if runtime is not None:
            runtime.backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'manifest', 'manifest-sha256', 'miner', 'attempt', 'output'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
