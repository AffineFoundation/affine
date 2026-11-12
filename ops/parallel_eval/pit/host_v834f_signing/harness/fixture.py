"""CPU fixtures for exercising the host-v834f launcher and its sharded update without a GPU or the live closure.

What is real and what is a stand-in:

* Host files: copied byte for byte from `stage/` (host-v834f) or `miner/validator/trainer-v834e/` (admitted).
* Source closure: the signed `source-final-v823u` bundle is not in this repository. The stand-in is the repository's
  `subnet/` package with `miner/validator/engine/subnet/mimo_grpo` (the published verbatim live engine) laid over
  it, plus the pinned verl vendor tree. In that engine `FP32Accumulator.__init__(self, named_parameters)` has the
  live signature (no `buffer_device`).
* `toploc`: its native wheel does not import under the CPU torch builds used here, and
  `subnet.mimo_runtime.runtime_fingerprint` imports it. The closure gets empty `toploc` modules (and its
  distribution metadata) so the fingerprint can run. The trainer path under test never calls toploc.
* Training kernels: `v83_kernels_policy.allow()` requires the installed flash-linear-attention / fla-core versions
  to equal the declared 0.6.0. The closure carries that distribution metadata, so the launcher's gate runs on a
  machine without the packages. The kernels themselves are CUDA-only and are NOT exercised: on the CPU the sharded
  update uses the stock transformers forward (see v835_zero3.make_train_update).
* Model: a tiny random `Qwen3_5ForConditionalGeneration` (the production class and layer pattern) under a
  `cpu_fixture` runtime profile, which `subnet.mimo_runtime` defines as fp32 on the CPU.

`subnet.*`, torch and transformers are imported inside the functions that need them: `subnet` must resolve to the
stand-in closure, which the driver puts on sys.path only after `source_root()` has built it.
"""
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIGNING = HERE.parent
STAGE = SIGNING / 'stage'
REPO = HERE.parents[4]
ADMITTED = REPO / 'miner' / 'validator' / 'trainer-v834e'
ENGINE = REPO / 'miner' / 'validator' / 'engine'
TOPLOC_MODULES = ('__init__.py', 'poly.py', 'utils.py', 'C/__init__.py', 'C/csrc/__init__.py', 'C/csrc/poly.py',
                  'C/csrc/ndd.py', 'C/csrc/utils.py')
STAND_IN_DISTRIBUTIONS = (('toploc', '0.1.6'), ('flash-linear-attention', '0.6.0'), ('fla-core', '0.6.0'))
# live engine block (trainer-config.public.json of host-v834e), shrunk only where the fixture needs it:
# dp_world 2 and dp_max_rows 2 so that a handful of short rollouts makes several micro-batches per rank
ENGINE_BLOCK = dict(
    batched_engine='packed', token_budget=40960, max_rows=64, fast_stage=True, stage_workers=2, part_concurrency=2,
    shard_concurrency=2, optimizer_step_threads=2, drop_unresolved=False, optimizer_persistence='none',
    loss_agg_mode='seq-mean-token-mean', loss_scale=0.5635, dp_world=2, dp_port=29583, dp_token_budget=40960,
    dp_max_rows=2, training_kernels={'flash-linear-attention': '0.6.0', 'fla-core': '0.6.0', 'causal-conv1d': None,
                                     'flash-attn': None},
    sgd_lr=5e-07, sgd_update_norm_cap=1.0, sgd_mode='fresh-adam', sgd_freeze=[], sgd_tensor_rel_cap=2e-05,
    sgd_fp32_master=True, sgd_direction='normgrad', support_renorm=dict(top_k=20, top_p=0.95, temperature=1.0))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_root(dest):
    """Build the stand-in source closure (see the module docstring) and return its path."""
    dest = Path(dest)
    if dest.exists():
        return dest
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', 'mimo_grpo')
    shutil.copytree(REPO / 'subnet', dest / 'subnet', ignore=ignore)
    shutil.copytree(ENGINE / 'subnet' / 'mimo_grpo', dest / 'subnet' / 'mimo_grpo',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ENGINE / 'vendor', dest / 'vendor', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for rel in TOPLOC_MODULES:
        path = dest / 'toploc' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""fixture stand-in: the toploc native wheel does not import under this CPU torch build"""\n')
    for name, version in STAND_IN_DISTRIBUTIONS:
        info = dest / ('%s-%s.dist-info' % (name.replace('-', '_'), version))
        info.mkdir()
        (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: %s\nVersion: %s\n' % (name, version))
    return dest


def host_dir(dest, source):
    """Copy a host package (its launcher's HOST_FILES, byte for byte) to `dest`, as the box holds it."""
    dest, source = Path(dest), Path(source)
    if dest.exists():
        return dest
    probe = load_module('fixture_launcher_probe', source / 'mimo_host_trainer_v83.py')
    for rel in probe.HOST_FILES:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, dest / rel)
    return dest


def original_methods(host):
    """The method pins the launcher checks, computed on the stand-in closure (the live pins bind the live one)."""
    import subnet.mimo_grpo.optimizer as optimizer
    import subnet.mimo_grpo.publication as publication
    import subnet.mimo_grpo.recovery as recovery
    import subnet.mimo_grpo.service as service
    import subnet.mimo_grpo.transaction as transaction
    import subnet.mimo_training_worker as worker
    objects = {
        'mimo_grpo.transaction._receipt': transaction._receipt,
        'mimo_grpo.transaction.Transaction.run': transaction.Transaction.run,
        'mimo_grpo.publication.validate_publication': publication.validate_publication,
        'mimo_grpo.service.execute_transaction': service.execute_transaction,
        'mimo_training_worker.validate_result': worker.validate_result,
        'mimo_grpo.optimizer.CPUAdamW.step': optimizer.CPUAdamW.step,
        'mimo_grpo.publication.R2ModelPublisher.__init__': publication.R2ModelPublisher.__init__,
        'mimo_grpo.recovery.restore_transaction_state': recovery.restore_transaction_state,
        'mimo_training_worker.prepare_signed_dispatch': worker.prepare_signed_dispatch,
    }
    return {site: host.method_sha(obj) for site, obj in objects.items()}


def signed_config(host, directory, root, engine, seed):
    """Write a ROOT-ADMITTED private config for `directory`, signed by a throwaway test key (never a real seed)."""
    from nacl.signing import SigningKey
    from subnet.storage import canonical, sha
    key = SigningKey(seed)
    authority = key.verify_key.encode().hex()
    proposal = dict(version=host.VERSION, scope=host.SCOPE, authority=authority, source_root=str(root),
                    host_files={n: host.file_sha(Path(directory) / n) for n in host.HOST_FILES},
                    original_methods=original_methods(host), engine=engine)
    payload = dict(version=host.VERSION, configuration_sha256=sha(canonical(proposal)), source_unchanged=True, scope=host.SCOPE)
    admission = dict(payload=payload, signer=authority,
                     signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())
    path = Path(directory) / 'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(canonical(dict(proposal, admission=admission)))
    return path


def tiny_checkpoint(directory, *, layers=4):
    """Tiny random production-class model + tokenizer + cpu_fixture profile (as tests/test_mimo_runtime.py)."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig
    from subnet.mimo_runtime import digest, file_hash, make_runtime_spec
    directory = Path(directory)
    directory.mkdir(parents=True)
    torch.manual_seed(9)
    text = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=layers, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=(['linear_attention'] * 3 + ['full_attention']) * (layers // 4),
        max_position_embeddings=128,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000, 'partial_rotary_factor': 1.0, 'mrope_section': [2, 2, 4]},
        eos_token_id=2, pad_token_id=0)
    config = Qwen3_5Config(text_config=text, vision_config={
        'depth': 1, 'hidden_size': 32, 'intermediate_size': 48, 'num_heads': 2, 'patch_size': 2, 'spatial_merge_size': 1,
        'temporal_patch_size': 1, 'out_hidden_size': 32, 'num_position_embeddings': 16})
    config._attn_implementation = 'eager'
    model = Qwen3_5ForConditionalGeneration(config).eval()
    model.generation_config.eos_token_id = [2, 3]
    model.save_pretrained(directory, safe_serialization=True)
    inner = Tokenizer(WordLevel({'<pad>': 0, '<unk>': 1, '<eos>': 2, '<eot>': 3, 'user': 4, 'assistant': 5, 'hello': 6,
                                 'think': 7}, unk_token='<unk>'))
    inner.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner, eos_token='<eos>', unk_token='<unk>', pad_token='<pad>')
    tokenizer.chat_template = ('{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}'
                               '{% if enable_thinking %}think {% endif %}{% if add_generation_prompt %}assistant{% endif %}')
    tokenizer.save_pretrained(directory)
    files = {p.name: file_hash(p) for p in directory.iterdir()}
    descriptor = {'id': digest(files), 'files': files}
    spec = make_runtime_spec(chat_template=tokenizer.chat_template, enable_thinking=True, max_context_tokens=128,
                             scope='cpu_fixture', eos_token_ids=[2, 3])
    return descriptor, spec.to_dict()


def free_groups(ff, *, groups=5, seed=11, capped_tokens=None):
    """Free-file groups as ff_trainer.prepare builds them: miner rewards, 1-3 rollouts, one capped rollout.

    The capped rollout has under 32 tokens, which is less than one derail window: the derail mask never cuts it.
    `capped_tokens` gives it that many tokens instead (every other rollout keeps its tokens), so that the mask
    has a full window to judge and a tail to keep."""
    import random
    rng = random.Random(seed)
    out = []
    for g in range(groups):
        prompt = tuple(rng.randrange(4, 64) for _ in range(rng.randrange(3, 8)))
        rows = []
        for nonce in range(1 + (g % 3)):
            body = [rng.randrange(4, 64) for _ in range(rng.randrange(4, 30))]
            capped = g == 2 and nonce == 1
            if capped and capped_tokens is not None:     # its own generator: the other rollouts stay as they are
                more = random.Random(seed * 1000003 + capped_tokens)
                body = [more.randrange(4, 64) for _ in range(capped_tokens)]
            stop = 'max_tokens' if capped else 'eos'
            reward = 0.0 if capped else float(nonce % 2 == 0)
            rows.append(ff.RewardedTrajectory(nonce=nonce, tokens=tuple(body if capped else body + [2]),
                                              outcome=ff.label(stop, reward), stop=stop, reward=reward))
        doc = dict(version=ff.DOCUMENT, group_index=g, prompt_tokens=list(prompt),
                   rollouts=[dict(tokens=list(r.tokens), stop=r.stop, reward=r.reward) for r in rows])
        out.append(ff.FreeGroup(task_id=hashlib.sha256(('fixture-task-%d' % g).encode()).hexdigest(), prompt=prompt,
                                trajectories=tuple(rows),
                                admission_sha256=hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()))
    return tuple(out)


def trainer_settings():
    """The signed GRPO settings the live trainer publishes (public copy of host-v834e)."""
    return json.loads((ADMITTED / 'trainer-config.public.json').read_text())['trainer_config']['settings']
