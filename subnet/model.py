"""Actual CPU inference, full-distribution verification and preference training."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import harness as policy
from .proofs import validate_framing
from .environments import EnvironmentSpec, create_session, legacy_spec, legacy_harness

NUMERICAL_RUNTIME_REVISION = 'cpu-float32-eager-v2-bounded-toploc'

ENV = dict(num_train_examples=4, num_eval_examples=0, code_length=1, num_symbols=2,
           max_turns=2, use_think=True, seed=42, use_candidate_reduction_reward=False)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def model_files(path):
    files = [p for p in Path(path).iterdir()
             if p.is_file() and p.suffix in ('.json','.safetensors','.txt','.model','.jinja','.bin','.pt','.tiktoken')]
    if not files:
        return {}
    # Read every byte on every invocation. Bound readers rather than trusting
    # file metadata or memoized digests; a failed read fails the whole inventory.
    with ThreadPoolExecutor(max_workers=min(4, len(files))) as readers:
        return dict(zip((p.name for p in files), readers.map(file_hash, files)))


class Runtime:
    def __init__(self, checkpoint, files, threads=4, environment=None, harness=None):
        torch.set_num_threads(threads)
        if not isinstance(files,dict) or not files:
            raise ValueError('empty or invalid checkpoint allowlist')
        actual=model_files(checkpoint)
        if set(actual)!=set(files):
            raise ValueError('unexpected or missing model-relevant checkpoint files')
        for name, expected in files.items():
            if Path(name).name != name or file_hash(Path(checkpoint)/name) != expected:
                raise ValueError('untrusted checkpoint')
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
                                                        dtype=torch.float32,
                                                        attn_implementation='eager',trust_remote_code=False,use_safetensors=True).eval()
        self.configure(environment or ENV, harness)
        from toploc import build_proofs_base64
        from .proofs import verify_mapped_proofs
        from functools import partial
        self.build_proofs, self.verify_proofs = build_proofs_base64, partial(verify_mapped_proofs,num_threads=threads)
        # TOPLOC's default native bit-extraction calls omp_set_num_threads with
        # hardware_concurrency, contaminating later Torch inference. Pin its
        # explicit thread parameter; extraction itself remains bit-identical.
        import toploc.poly as toploc_poly
        from toploc.C.csrc.utils import get_fp_parts
        self.toploc_threads=threads
        toploc_poly.get_fp_parts=lambda tensor:get_fp_parts(tensor,num_threads=threads)

    def configure(self, environment, harness=None):
        self.legacy = 'id' not in environment or environment.get('adapter')=='legacy_mastermind'
        self.env_config = dict(environment)
        self.spec = legacy_spec(environment) if 'id' not in environment else EnvironmentSpec.from_dict(environment)
        self.harness = policy.normalize(harness or (legacy_harness(self.spec.config) if self.legacy else None))
        if self.harness['max_output_tokens'] > self.spec.max_output_tokens:
            raise ValueError('harness exceeds environment output budget')
        return self

    def for_environment(self, environment, harness=None):
        import copy
        chosen = copy.copy(self).configure(environment, harness)
        chosen.native_source_validation=getattr(self,'native_source_validations',{}).get(chosen.spec.id)
        if chosen.native_source_validation is not None:chosen.native_source_validation.validate(chosen.spec)
        if getattr(chosen, 'sampling_context', None) is not None:
            from .forced_sampling import validate_harness
            validate_harness(chosen.harness)
            if getattr(chosen,'fast_sampling_calibration',None)is not None:
                from .fast_prefill_audit import bind
                chosen.fast_sampling_calibration=bind(chosen.fast_sampling_manifest,chosen.harness)
        return chosen

    def sample_output(self, prompt, seed, messages, turn, index, task_hash):
        if getattr(self, 'sampling_context', None) is not None:
            from .forced_sampling import sample
            return sample(self, prompt, seed, turn, index, task_hash)
        if hasattr(self, 'sample'):
            return self.sample(prompt, seed + turn, messages, turn)
        return policy.sample(self.model, self.tokenizer, prompt, seed + turn, self.harness, messages=messages, turn_index=turn)

    def sampling_receipt(self, seed):
        if getattr(self, 'sampling_context', None) is None:
            return {}
        from .forced_sampling import receipt
        return {'sampling': receipt(self.sampling_context, seed)}

    def prompt(self, messages, tools=()):
        return policy.render(self.tokenizer, messages, tools,self.harness)

    def compute(self, prompt, output):
        with torch.inference_mode():
            result = self.model(torch.tensor([prompt+output]), output_hidden_states=True, use_cache=False)
            hidden = result.hidden_states[-1][0].to(torch.bfloat16).contiguous()
            logprobs = torch.log_softmax(result.logits[0, len(prompt)-1:len(prompt)+len(output)-1].float(), -1).numpy()
        acts = [hidden[:len(prompt)]] + [hidden[i:i+1] for i in range(len(prompt), len(prompt)+len(output))]
        return acts, logprobs

    def rollout(self, index, seed):
        token_manifest = getattr(self, 'token_artifact_manifest', None)
        if token_manifest is not None:
            from .token_only_protocol import bind_runtime
            bind_runtime(self, token_manifest)
        env_seed = int(self.spec.config.get('seed', 0))
        validation=getattr(self,'native_source_validation',None)
        session = create_session(self.spec) if validation is None else create_session(self.spec,source_validation=validation)
        try:
            initial = session.reset(index, env_seed)
            messages, tools = initial['messages'], initial.get('tools', [])
            turns, arrays = [], []
            reward, classification, done = 0., 'neutral', False
            for turn_index in range(self.spec.max_turns):
                prompt = self.prompt(messages, tools)
                if len(prompt)+self.harness['max_output_tokens'] > min(getattr(self.model.config, 'max_position_embeddings', 8192), 8192):
                    raise ValueError('model context budget')
                output = self.sample_output(prompt, seed, messages, turn_index, index, initial['task_hash'])
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                token_manifest = getattr(self, 'token_artifact_manifest', None)
                if token_manifest is None:
                    acts, logprobs = self.compute(prompt, output)
                    proofs = self.build_proofs(acts, decode_batching_size=16, topk=128)
                    if not proofs or any(p is None for p in proofs):
                        raise ValueError('proof construction failed')
                else:
                    from .token_only_protocol import bind_runtime
                    bind_runtime(self, token_manifest)
                    proofs = None
                result = session.step(policy.action(text,self.harness))
                done, reward, classification = result['done'], result['reward'], result['classification']
                observations = result['observations']
                turn = dict(prompt=prompt, output=output, text=text,
                            observations=observations, done=done, reward=reward, classification=classification)
                # Keep legacy artifacts readable; new verification authenticates
                # structured observations rather than a guessed single feedback.
                if self.legacy:
                    turn['feedback'] = observations[0]['content'] if observations else ''
                if token_manifest is None:
                    turn['proofs'] = proofs
                    from .probability_artifacts import encode
                    arrays.append(encode(logprobs, output, getattr(self, 'probability_artifact_policy', None)))
                turns.append(turn)
                messages = messages + [dict(role='assistant', content=text)] + policy.observations(observations,self.harness)
                if done:
                    break
            if not done:
                raise ValueError('environment did not terminate within signed budget')
            return dict(schema=2,env_id=self.spec.id, environment_version=self.spec.version,
                        index=index, sample_index=index, seed=seed, env_seed=env_seed,
                        task_hash=initial['task_hash'], reward=reward, classification=classification,
                        turns=turns, **self.sampling_receipt(seed)), arrays
        finally:
            session.close()

    def verify(self, rollout, arrays):
        from .audit_policy import InvalidSample
        calibrated=getattr(self,'fast_sampling_calibration',None)
        from .fast_prefill_audit import THREEWAY_VERSION,NumericalAmbiguity
        threeway=(getattr(self,'sampling_context',None)or{}).get('contract',{}).get('version')==THREEWAY_VERSION
        uncertain=[]
        if getattr(self, 'probability_artifact_policy', None) is not None and getattr(self, 'sampling_context', None) is None:
            raise InvalidSample('compact probability artifacts require authenticated forced sampling')
        if getattr(self, 'sampling_context', None) is not None:
            try:
                expected = self.sampling_receipt(rollout.get('seed'))['sampling']
            except ValueError as error:
                raise InvalidSample(str(error)) from error
            if rollout.get('sampling') != expected:
                raise InvalidSample('sampling contract/attempt binding')
        if rollout.get('schema',1)>=2 and (rollout.get('sample_index')!=rollout.get('index') or rollout.get('env_id')!=self.spec.id or rollout.get('environment_version')!=self.spec.version):
            raise InvalidSample('required environment binding')
        if type(rollout.get('reward')) not in (int,float) or not math.isfinite(rollout['reward']):
            raise InvalidSample('reward type or finiteness')
        if rollout.get('env_id', self.spec.id) != self.spec.id or rollout.get('environment_version', self.spec.version) != self.spec.version:
            raise InvalidSample('environment binding')
        session = create_session(self.spec)
        try:
            env_seed = int(self.spec.config.get('seed', 0))
            if rollout.get('env_seed', env_seed) != env_seed:
                raise InvalidSample('environment seed')
            initial = session.reset(rollout['index'], env_seed)
            if rollout.get('task_hash', initial['task_hash']) != initial['task_hash']:
                raise InvalidSample('task hash')
            messages, tools = initial['messages'], initial.get('tools', [])
            turns = rollout['turns']
            if not 1 <= len(turns) <= self.spec.max_turns or len(arrays) != len(turns):
                raise InvalidSample('trajectory length')
            for i, (turn, claimed) in enumerate(zip(turns, arrays)):
                prompt = self.prompt(messages, tools)
                if turn['prompt'] != prompt:
                    raise InvalidSample('context')
                output = turn['output']
                if not 0 < len(output) <= self.spec.max_output_tokens or any(type(x) is not int or not 0 <= x < self.model.config.vocab_size for x in output):
                    raise InvalidSample('tokens')
                if len(prompt)+len(output) > min(getattr(self.model.config,'max_position_embeddings',8192),8192):
                    raise InvalidSample('model context budget')
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                if turn['text'] != text:
                    raise InvalidSample('text')
                if type(turn.get('done')) is not bool or type(turn.get('reward')) not in (int,float) or not math.isfinite(turn['reward']):
                    raise InvalidSample('turn outcome types')
                acts, probs = self.compute(prompt, output)
                from .probability_artifacts import verify_claim
                verify_claim(claimed, probs, output, getattr(self, 'probability_artifact_policy', None), atol=calibrated['logprob_atol']if calibrated else 1e-5)
                count = 1+math.ceil(len(output)/16)
                try:validate_framing(turn['proofs'],count)
                except ValueError as error:raise InvalidSample('proof framing') from error
                results = self.verify_proofs(acts, turn['proofs'], decode_batching_size=16, topk=128)
                if len(results) != count or any(r.exp_mismatches>(calibrated['toploc_exp_mismatches']if calibrated else 0) or r.mant_err_mean>(calibrated['toploc_mant_err_mean']if calibrated else 0) or r.mant_err_median>(calibrated['toploc_mant_err_median']if calibrated else 0) for r in results):
                    raise InvalidSample('TOPLOC')
                if getattr(self, 'sampling_context', None) is not None:
                    if calibrated is not None:
                        from .fast_prefill_audit import verify_sampling
                        try:verify_sampling(self,rollout,i,prompt,output,probs)
                        except NumericalAmbiguity as error:
                            if not threeway:raise
                            # Numerical uncertainty cannot hide later native or turn invalidity.
                            uncertain.append((i,error))
                    else:
                        selected = self.sample_output(prompt, rollout['seed'], messages, i, rollout['index'], initial['task_hash'])
                        if selected != output:raise InvalidSample('sampling replay mismatch')
                result = session.step(policy.action(text,self.harness))
                done, reward = result['done'], result['reward']
                observations = result['observations']
                expected_observations = turn.get('observations')
                if expected_observations is None and self.legacy:
                    expected_observations = [dict(role='user',content=turn.get('feedback',''))]
                if turn['done'] != done or expected_observations != observations or turn['reward'] != reward:
                    raise InvalidSample('environment replay')
                if turn.get('classification',result['classification']) != result['classification']:
                    raise InvalidSample('classification')
                if done and i != len(turns)-1:
                    raise InvalidSample('extra turns')
                messages = messages + [dict(role='assistant',content=text)] + policy.observations(observations,self.harness)
            if not done or rollout['reward'] != reward or rollout.get('classification',result['classification']) != result['classification']:
                raise InvalidSample('incomplete rollout or score')
            if uncertain:
                error=uncertain[0][1]
                error.environment_verification_complete=True
                error.uncertain_turns=[dict(turn=i,positions=getattr(e,'uncertain_positions',[]),count=getattr(e,'uncertain_position_count',0))for i,e in uncertain]
                error.uncertain_position_count=sum(row['count']for row in error.uncertain_turns)
                raise error
            return True
        finally:
            session.close()

    def sequence_logprob(self, rollout):
        total, count = 0, 0
        for turn in rollout['turns']:
            prompt, output = turn['prompt'], turn['output']
            ids = torch.tensor([prompt+output])
            logits = self.model(ids, use_cache=False).logits[0, len(prompt)-1:len(prompt)+len(output)-1]
            logprobs = torch.log_softmax(logits.float(), -1)
            total = total + logprobs.gather(1, torch.tensor(output)[:, None]).sum()
            count += len(output)
        return total/count

    def train(self, pairs, destination, steps=1):
        if not pairs:
            raise ValueError('no verified training pairs')
        # Reference-relative sequence preference objective. This is an explicit
        # mock training choice, not a claim of per-token credit assignment.
        references = []
        with torch.no_grad():
            for pos, neg in pairs:
                references.append(float(self.sequence_logprob(pos)-self.sequence_logprob(neg)))
        before = self.model.lm_head.weight.detach().clone()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        losses = []
        self.model.eval()  # gradients remain enabled; remove dropout randomness.
        for step in range(steps):
            pos, neg = pairs[step % len(pairs)]
            optimizer.zero_grad(set_to_none=True)
            margin = self.sequence_logprob(pos)-self.sequence_logprob(neg)-references[step % len(pairs)]
            loss = -torch.nn.functional.logsigmoid(.1*margin)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        changed = not torch.equal(before, self.model.lm_head.weight.detach())
        if not changed:
            raise ValueError('training did not change model weights')
        Path(destination).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination, safe_serialization=True)
        self.tokenizer.save_pretrained(destination)
        return dict(steps=steps, losses=losses, weights_changed=changed, objective='reference-relative sequence preference')


def check_runtime_profile(manifest):
    allowed={'MKL_CBWR','ATEN_CPU_CAPABILITY','ONEDNN_MAX_CPU_ISA','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM'}
    profile=manifest.get('runtime_profile',{})
    if not isinstance(profile,dict) or set(profile)-allowed:raise ValueError('unsupported runtime profile')
    if any(os.environ.get(k)!=v for k,v in profile.items()):raise ValueError('runtime profile mismatch; launch with signed manifest CPU environment')
