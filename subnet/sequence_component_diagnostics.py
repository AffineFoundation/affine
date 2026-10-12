"""Bounded detached training diagnostics; never changes the scientific result.

The caller exposes already-computed scalar sums and means during its existing
reference/post-update passes only. This module performs no model calls, writes,
imports of tensor libraries, or objective calculation used for differentiation.
"""
import json
import math

VERSION = 'sequence-component-diagnostics-v1'
MAX_PAIRS = 65536
MAX_DOCUMENT_BYTES = 64 * 1024 * 1024

def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

class SequenceComponentDiagnostics:
    def __init__(self, pairs, tasks, identities, *, max_pairs=MAX_PAIRS,
                 max_document_bytes=MAX_DOCUMENT_BYTES):
        self.phase = None
        self.error = None
        self.current = None
        self.meta = []
        self.values = {'before': {}, 'after': {}}
        self.max_document_bytes = max_document_bytes
        try:
            if (type(max_pairs) is not int or not 1 <= max_pairs <= MAX_PAIRS
                    or type(max_document_bytes) is not int
                    or not 1 <= max_document_bytes <= MAX_DOCUMENT_BYTES
                    or not 0 < len(pairs) <= max_pairs or len(identities) != len(pairs)):
                raise ValueError
            assigned = {}
            for task in tasks:
                if not _hash(task['task_sha256']): raise ValueError
                for index in task['pair_indices']:
                    if type(index) is not int or index in assigned: raise ValueError
                    assigned[index] = task['task_sha256']
            if set(assigned) != set(range(len(pairs))): raise ValueError
            for index, (_, positive, negative) in enumerate(pairs):
                if not _hash(identities[index]): raise ValueError
                row = dict(pair_index=index, pair_sha256=identities[index],
                           task_sha256=assigned[index])
                for role, rollout in (('positive', positive), ('negative', negative)):
                    turns = rollout['turns']
                    outputs = sum(len(turn['output']) for turn in turns)
                    if not turns or not outputs: raise ValueError
                    row[role] = dict(output_tokens=outputs,
                        prompt_tokens=sum(len(turn['prompt']) for turn in turns),
                        turns=len(turns),
                        last_turn_done=turns[-1].get('done') if type(turns[-1].get('done')) is bool else None)
                self.meta.append(row)
        except Exception:
            self._disable('metadata_unavailable')

    def _disable(self, reason):
        # Fixed codes only: exception messages can contain private inputs.
        self.error = reason
        self.phase = None
        self.current = None
        self.values = {'before': {}, 'after': {}}

    def start(self, phase):
        if self.error: return
        if phase not in self.values or self.phase is not None or self.values[phase]:
            self._disable('phase_order_invalid'); return
        self.phase = phase
        self.current = None

    def begin_pair(self, index):
        if self.phase is None: return
        if (type(index) is not int or not 0 <= index < len(self.meta)
                or index in self.values[self.phase]
                or self.current is not None and len(self.values[self.phase][self.current]) != 2):
            self._disable('pair_order_invalid'); return
        self.current = index
        self.values[self.phase][index] = []

    def observe(self, total, mean, tokens):
        if self.phase is None: return
        try:
            row = self.values[self.phase][self.current]
            if len(row) >= 2: raise ValueError
            role = ('positive', 'negative')[len(row)]
            if type(tokens) is not int or tokens != self.meta[self.current][role]['output_tokens']:
                raise ValueError
            # Detach scalars, never retain a tensor or gradient graph.
            total_value, mean_value = float(total.detach()), float(mean.detach())
            if not _finite(total_value) or not _finite(mean_value): raise ValueError
            row.append(dict(total_logprob=total_value, mean_logprob=mean_value))
        except Exception:
            self._disable('component_unavailable')

    def stop(self):
        if self.phase is None: return
        if (set(self.values[self.phase]) != set(range(len(self.meta)))
                or any(len(row) != 2 for row in self.values[self.phase].values())):
            self._disable('incomplete_phase'); return
        self.phase = None
        self.current = None

    def report(self, references, after_margins, updates, *, input_checkpoint,
               global_step_before, global_step_after):
        unavailable = lambda reason: dict(version=VERSION, status='unavailable',
            reason=reason, extra_model_forward_passes=0, objective_changed=False,
            output_checkpoint_binding='containing_original_trainer_report.new_checkpoint.id')
        if self.error: return unavailable(self.error)
        try:
            n = len(self.meta)
            if (self.phase is not None or len(references) != n or len(after_margins) != n
                    or not _hash(input_checkpoint)
                    or any(set(self.values[p]) != set(range(n)) for p in self.values)):
                raise ValueError
            weights = [0.] * n
            uses = [0] * n
            for update in updates:
                for row in update['pairs']:
                    index = row['pair_index']
                    if (type(index) is not int or not 0 <= index < n
                            or row['pair_sha256'] != self.meta[index]['pair_sha256']
                            or row['task_sha256'] != self.meta[index]['task_sha256']
                            or not _finite(row['gradient_weight']) or row['gradient_weight'] <= 0):
                        raise ValueError
                    weights[index] += row['gradient_weight']; uses[index] += 1
            if any(count == 0 for count in uses): raise ValueError
            rows = []
            for index, meta in enumerate(self.meta):
                before = self.values['before'][index]; after = self.values['after'][index]
                if len(before) != 2 or len(after) != 2: raise ValueError
                if not _finite(references[index]) or not _finite(after_margins[index]): raise ValueError
                row = dict(meta, total_gradient_weight=weights[index], gradient_observations=uses[index],
                    before=dict(positive=before[0], negative=before[1], raw_margin=references[index]),
                    after=dict(positive=after[0], negative=after[1], raw_margin=after_margins[index]))
                row['delta'] = dict(raw_margin=after_margins[index]-references[index],
                    positive_mean_logprob=after[0]['mean_logprob']-before[0]['mean_logprob'],
                    negative_mean_logprob=after[1]['mean_logprob']-before[1]['mean_logprob'],
                    positive_total_logprob=after[0]['total_logprob']-before[0]['total_logprob'],
                    negative_total_logprob=after[1]['total_logprob']-before[1]['total_logprob'])
                rows.append(row)
            out = dict(version=VERSION, status='complete', pair_count=n, rows=rows,
                input_checkpoint=input_checkpoint, global_step_before=global_step_before,
                global_step_after=global_step_after, extra_model_forward_passes=0,
                objective_changed=False, log_base='e', sequence_reduction='sum_output_logprobs/output_token_count',
                raw_margin_definition='positive_mean_logprob-minus-negative_mean_logprob',
                raw_margin_beta_scaled=False, prompt_tokens_scored=False,
                output_checkpoint_binding='containing_original_trainer_report.new_checkpoint.id',
                gradient_weight_binding='existing training.updates[].pairs entries',
                reference_scope='immutable-BF16-epoch-input',
                input_assurance='same_as_containing_original_trainer_report',
                task_weighted_delta={})
            counts = {}
            for row in rows: counts[row['task_sha256']] = counts.get(row['task_sha256'], 0) + 1
            out['task_weighted_delta'] = {key: math.fsum(
                row['delta'][key]/(len(counts)*counts[row['task_sha256']]) for row in rows)
                for key in rows[0]['delta']}
            encoded = json.dumps(out, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            if len(encoded) > self.max_document_bytes: return unavailable('document_byte_budget')
            return out
        except Exception:
            return unavailable('report_unavailable')
