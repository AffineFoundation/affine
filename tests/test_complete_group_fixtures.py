"""Real Miner/protocol/sampler/ZIP integration with explicit local-only transport.

Fixture generation is deterministic test data, not claimed model provenance.
The final test separately runs actual CPU model sampling through the collector.
"""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
import numpy as np
from verifiers.v1.errors import TaskError
from subnet.miner import Miner, EpochClosed
from subnet.complete_group_mining import GroupFiltered, GroupIndeterminate, digest
from subnet.complete_group_state import _write
from subnet.batch_contract import REQUIRED_CLIENT_FEATURES
from subnet import forced_sampling as forced, harness
from subnet.batches import pack, unpack
from subnet.storage import canonical


def fixture(n=8):
    m = json.loads((ROOT/'research/group_contract_manifest/tests/fixtures/legacy_epoch171.json').read_text())
    m.pop('K'); m.pop('L')
    m['group_contract'] = dict(version='complete-canonical-group-v1', group_size=n,
        min_success=1, min_completed_failure=1, outcome_policy='completed-eos-math-unresolved-masked-v1')
    m['required_client_features'] = sorted(REQUIRED_CLIENT_FEATURES)
    m['sampling_contract']['version'] = forced.COMPLETE_GROUP_VERSION
    m['sampling_contract'].pop('max_attempts')
    m['sampling_source_hash'] = forced.source_hash()
    m['harness_source_hash'] = harness.source_hash()
    m['model_runtime_revision'] = 'cpu-fixture-cached-fp32'
    m['environment_revision'] = 'trusted-native-fixture-v1'
    m['backend_profile'] = dict(device='cpu', dtype='float32', attention='eager')
    m['runtime_profile'] = {}
    m['source_bundle'] = {'sha256':'f'*64}
    m.pop('artifact_policy', None)
    m.pop('sample_harness_registry', None)
    m['submission_transport_policy'] = 'complete-group-commitment-v1'
    m['transport_policy'] = 'direct-r2-v1'
    m['group_transport_policy']=dict(version='complete-group-bounded-selected-v1',max_context_tokens=64,eos_token_id=15,vocab_size=16)
    m['probability_artifact_policy']={'version':'selected-token-logprobs-v1'}
    m['start']=int(time.time())-10
    m['deadline'] = int(time.time())+120
    m['max_batches'] = 3
    h = harness.normalize(dict(version='text-tools-v1', policy='autoregressive',
                               max_output_tokens=12, temperature=.7, top_p=1.))
    spec = dict(id='math', version='taskset-1', adapter='test-fixture', num_samples=8,
                max_turns=1, max_output_tokens=12, config={'seed':0})
    m['environments'] = [dict(env_id='math', spec=spec, indices=[0,1,2,3], harness=h)]
    c = m['sampling_contract']['calibration']
    c.update(checkpoint=m['checkpoint']['id'], model_runtime_revision=m['model_runtime_revision'],
             backend_profile_sha256=digest(m['backend_profile']), harness_sha256=digest(h), cdf_abs_error=0.)
    return m


class FixtureTokenizer:
    eos_token_id=15
    def __len__(self):return 16


class FixtureRuntime:
    def __init__(self, manifest, miner, outcomes=None, error_at=None):
        definition = manifest['environments'][0]
        self.spec = SimpleNamespace(**definition['spec'])
        self.env_config = definition['spec']
        self.harness = definition['harness']
        self.tokenizer = FixtureTokenizer()
        self.model=SimpleNamespace(config=SimpleNamespace(vocab_size=16,max_position_embeddings=64))
        self.sampling_context = forced.binding(manifest, miner)
        self.outcomes = outcomes or ['positive', 'negative']+['neutral']*(manifest['group_contract']['group_size']-2)
        self.error_at = error_at
        self.calls = []
        self.mutate = None

    def for_environment(self, spec, selected_harness):
        return self

    def rollout(self, index, nonce):
        self.calls.append((index, nonce))
        if nonce == self.error_at:
            self.error_at = None
            raise RuntimeError('fixture interrupted generation')
        category = self.outcomes[nonce]
        if isinstance(category, BaseException):
            raise category
        budget = category == 'budget-positive'
        if budget:
            category = 'positive'
        prompt = [1, index+2]
        # Same text at many nonces is intentional; it must not be deduplicated.
        output = [3]*12 if budget else [{'positive':3,'negative':4,'neutral':5}[category],15]
        reward = 1.0 if category == 'positive' else 0.0
        turn = dict(prompt=prompt, output=output, text='fixture', proofs=[base64.b64encode((40001).to_bytes(2,'big')+bytes(256)).decode()]*2, observations=[],
                    done=True, reward=reward, classification=category)
        row = dict(schema=2, env_id=self.spec.id, environment_version=self.spec.version,
                   index=index, sample_index=index, seed=nonce, env_seed=0,
                   task_hash=digest(dict(index=index, taskset=self.spec.version)),
                   reward=reward, classification=category, turns=[turn],
                   sampling=forced.receipt(self.sampling_context, nonce))
        self.complete_group_sample_evidence = dict(version='sampler-selected-behavior-logprobs-v1',
            attempt=nonce, turn=0, binding_sha256=digest(self.sampling_context),
            prompt_sha256=digest(prompt), output_sha256=digest(output),
            behavior_selected_logprobs=[-1.0]*len(output), stop_reason='budget' if budget else 'eos')
        if self.mutate:
            self.mutate(self, row)
        return row, [np.full((len(output),1), -2.0, dtype=np.float32)]


