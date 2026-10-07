"""Research-only adapter for the current schema-2 batch/rollout wire format.

Caller authenticates the manifest and provides the trusted native reset task hash
and pinned tokenizer. This module performs no sampling, grading or deployment.
"""
import copy
from dataclasses import dataclass

from ops.paired_quota_qualification import ApprovedTask, digest, identities, select_pairs
from subnet import forced_sampling, harness, protocol
from subnet.trajectory_identity import token_trace_sha256


@dataclass(frozen=True)
class CurrentBatchAdapter:
    task: ApprovedTask
    definition: dict
    sampling_context: dict
    decode: object

    @classmethod
    def from_manifest(cls, manifest, env_id, index, *, task_sha256, decode):
        """Manifest payload must already be signature-authenticated by caller.

        task_sha256 is independently derived by the native task reset adapter,
        not copied from a miner rollout. Decode must use the pinned tokenizer.
        """
        if not callable(decode):
            raise ValueError('pinned tokenizer decode required')
        definition = protocol.entry(manifest, env_id)
        if definition.get('evaluation_only', False):
            raise ValueError('evaluation-only task')
        resolved = protocol.harness_for(definition, index)
        context = forced_sampling.binding(manifest)
        if context is None or resolved is None:
            raise ValueError('explicit forced sampling and harness required')
        forced_sampling.validate_harness(resolved)
        heldout = manifest.get('heldout_indices', [])
        if isinstance(heldout, dict):
            heldout = heldout.get(env_id, [])
        if index in heldout:
            raise ValueError('heldout task forbidden')
        task = ApprovedTask(manifest['epoch'], manifest['checkpoint']['id'],
                            digest(definition['spec']), env_id, index, task_sha256,
                            digest(resolved), digest(context),
                            tuple(range(context['contract']['max_attempts'])))
        return cls(task, copy.deepcopy(definition), copy.deepcopy(context), decode)

    def normalize(self, batch):
        """Adapt canonical traces, without claiming the observations are true."""
        task = self.task
        expected = dict(schema=2, epoch=task.epoch, checkpoint=task.checkpoint,
                        env_id=task.env_id, environment_version=self.definition['spec']['version'],
                        index=task.index, sample_index=task.index)
        if not isinstance(batch, dict) or any(type(batch.get(k)) is not type(v) or batch.get(k) != v
                                              for k, v in expected.items()):
            raise ValueError('current batch binding')
        rolls = batch.get('rollouts')
        if not isinstance(rolls, list) or not 1 <= len(rolls) <= len(task.approved_attempts) * 4:
            raise ValueError('bounded current rollout list')
        resolved = protocol.harness_for(self.definition, task.index)
        env_seed = int(self.definition['spec'].get('config', {}).get('seed', 0))
        normalized = []
        for rollout in rolls:
            expected_rollout = dict(schema=2, env_id=task.env_id,
                                    environment_version=expected['environment_version'],
                                    index=task.index, sample_index=task.index,
                                    env_seed=env_seed, task_hash=task.task_sha256)
            if not isinstance(rollout, dict) or any(type(rollout.get(k)) is not type(v) or rollout.get(k) != v
                                                   for k, v in expected_rollout.items()):
                raise ValueError('current rollout task binding')
            # Schema 2 normally scopes epoch/checkpoint on the enclosing batch.
            # If a future wrapper repeats them, inconsistencies are refused.
            for key in ('epoch', 'checkpoint'):
                if key in rollout and rollout[key] != expected[key]:
                    raise ValueError('contradictory rollout wrapper binding')
            attempt = rollout.get('seed')
            if rollout.get('sampling') != forced_sampling.receipt(self.sampling_context, attempt):
                raise ValueError('prescribed draw receipt binding')
            turns = rollout.get('turns')
            max_turns = self.definition['spec'].get('max_turns', 32)
            if not isinstance(turns, list) or not 1 <= len(turns) <= min(max_turns, 32):
                raise ValueError('complete current turns required')
            trace = []
            for ordinal, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    raise ValueError('current turn mapping')
                output = turn.get('output')
                if (not isinstance(output, list) or not output or len(output) > resolved['max_output_tokens'] or
                        any(type(t) is not int or not 0 <= t < 200000 for t in output)):
                    raise ValueError('current output token budget')
                text = self.decode(output)
                if not isinstance(text, str) or turn.get('text') != text:
                    raise ValueError('pinned tokenizer text mismatch')
                observations = turn.get('observations')
                if not isinstance(observations, list):
                    raise ValueError('complete current observations required')
                # Validate the actual harness wire schema, but retain the original
                # structured observations, including order and tool role.
                harness.observations(observations, resolved)
                if type(turn.get('done')) is not bool or turn['done'] != (ordinal == len(turns) - 1):
                    raise ValueError('current terminal turn structure')
                trace.append(dict(prompt=copy.deepcopy(turn.get('prompt')), output=copy.deepcopy(output),
                                  actions=[harness.action(text, resolved)],
                                  observations=[dict(role=o['role'], content=o['content']) for o in observations]))
            row = dict(task.task_binding(), epoch=task.epoch,
                       harness_sha256=task.harness_sha256,
                       sampling_context_sha256=task.sampling_context_sha256,
                       attempt=attempt, classification=rollout.get('classification'), turns=trace)
            identities(task, row)
            if row['classification'] not in ('positive', 'negative'):
                raise ValueError('explicit current classification required')
            normalized.append(row)
        return normalized


class CumulativeTaskSlot:
    """In-memory research revision checker, not a durable optimizer ledger.

    Revisions are cumulative snapshots of ONE task/miner slot. A later revision
    cannot remove or rewrite an approved attempt. Repacking/redelivery is a no-op.
    All updates are checked before mutating in-memory state.
    """
    def __init__(self, adapter, miner):
        self.adapter = adapter
        self.miner = miner
        adapter.task.slot_id(miner)
        self._attempts = {}
        self._rows = []
        self._revision_ids = set()

    def add_revision(self, batch):
        rows = self.adapter.normalize(batch)
        attempts = {}
        labels = {}
        token_traces = {}
        for row in rows:
            execution, content = identities(self.adapter.task, row)
            token_trace = token_trace_sha256(row['turns'])
            if token_trace in token_traces and token_traces[token_trace] != content:
                raise ValueError('conflicting cumulative token trace observations')
            token_traces[token_trace] = content
            value = (content, row['classification'])
            if execution in attempts and attempts[execution] != value:
                raise ValueError('conflicting cumulative attempt')
            attempts[execution] = value
            if content in labels and labels[content] != row['classification']:
                raise ValueError('conflicting cumulative content label')
            labels[content] = row['classification']
        if not self._attempts.keys() <= attempts.keys():
            raise ValueError('cumulative revision removed attempt')
        for execution, value in self._attempts.items():
            if attempts[execution] != value:
                raise ValueError('cumulative revision rewrote attempt')
        revision_id = digest(dict(slot_id=self.adapter.task.slot_id(self.miner),
                                  attempts=sorted(attempts.items())))
        redelivery = revision_id in self._revision_ids
        self._attempts = attempts
        self._rows = copy.deepcopy(rows)
        self._revision_ids.add(revision_id)
        return dict(revision_id=revision_id, redelivery=redelivery,
                    unique_attempts=len(attempts), unique_contents=len(labels),
                    stored_revision_count=len(self._revision_ids))

    def select(self, *, quota=1):
        return select_pairs(self.adapter.task, self.miner, self._rows, quota=quota)
