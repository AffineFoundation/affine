"""Pinned MiMo/Qwen3.5 text runtime and checkpoint-lifetime caches.

This module does not sign qualifications or activate a release. A matching
runtime fingerprint identifies an implementation; it is not numerical
qualification. Inference, audit and training callers share the same explicit
template, stop set and model loader. Training probabilities are raw T=1
probabilities, distinct from the protocol's top-k/top-p sampling distribution.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import threading
import uuid

VERSION = "qwen3_5-mimo-v1"
MODEL_ID = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
EOS_TOKEN_IDS = (248046, 248044)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def file_hash(path):
    """Hash one ordinary file, rejecting replacement or mutation during hashing."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    result = hashlib.sha256()
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        _need(stat.S_ISREG(before.st_mode), "ordinary checkpoint file required")
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(chunk)
        after = os.fstat(stream.fileno())
        _need(_stamp(before) == _stamp(after), "checkpoint file changed while hashing")
    return result.hexdigest()


def _stamp(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def validate_descriptor(descriptor):
    _need(type(descriptor) is dict and set(descriptor) == {"id", "files"}, "exact checkpoint descriptor")
    files = descriptor["files"]
    _need(type(files) is dict and 1 <= len(files) <= 256, "bounded checkpoint inventory")
    for name, checksum in files.items():
        _need(type(name) is str and Path(name).name == name and not name.startswith(".") and _sha(checksum),
              "safe checkpoint filename and sha256")
    _need(descriptor["id"] == digest(files), "checkpoint content identifier")
    _need("config.json" in files and any(n.endswith(".safetensors") for n in files),
          "config and safetensors checkpoint required")
    return descriptor


class CheckpointGuard:
    """Authenticate once on admission, then guard pinned local file identities.

    The private cache is owned by the worker. The cheap guard avoids rehashing
    18GB per job; it is not a defense against a privileged host adversary.
    """
    def __init__(self, directory, descriptor):
        self.directory = Path(directory)
        self.descriptor = json.loads(canonical(validate_descriptor(descriptor)))
        _need(self.directory.is_absolute() and self.directory.resolve() == self.directory,
              "ordinary absolute checkpoint directory")
        _need(set(p.name for p in self.directory.iterdir()) == set(descriptor["files"]),
              "exact checkpoint physical inventory")
        self.stamps = {}
        for name, expected in descriptor["files"].items():
            path = self.directory / name
            _need(not path.is_symlink() and file_hash(path) == expected, "checkpoint hash mismatch: " + name)
            self.stamps[name] = _stamp(path.stat(follow_symlinks=False))

    def __call__(self):
        _need(set(p.name for p in self.directory.iterdir()) == set(self.stamps), "checkpoint inventory changed")
        for name, stamp in self.stamps.items():
            path = self.directory / name
            _need(not path.is_symlink() and _stamp(path.stat(follow_symlinks=False)) == stamp,
                  "checkpoint local state changed: " + name)


def runtime_fingerprint():
    """Pin concrete HF hybrid model/cache implementations and dependency versions."""
    packages = ("torch", "transformers", "safetensors", "tokenizers", "toploc")
    versions = {name: importlib.metadata.version(name) for name in packages}
    for optional in ("flash-linear-attention", "fla-core", "causal-conv1d", "flash-attn"):
        try:
            versions[optional] = importlib.metadata.version(optional)
        except importlib.metadata.PackageNotFoundError:
            versions[optional] = None
    modules = ("transformers.models.qwen3_5.modeling_qwen3_5", "transformers.cache_utils",
               "transformers.modeling_utils", "transformers.generation.utils",
               "toploc", "toploc.poly", "toploc.utils",
               "toploc.C.csrc.poly", "toploc.C.csrc.ndd", "toploc.C.csrc.utils")
    sources = {name: file_hash(inspect.getfile(importlib.import_module(name))) for name in modules}
    # Importing each native extension also fails early on a torch/CUDA ABI
    # mismatch. Versions alone do not identify an independently rebuilt wheel.
    proof_root = Path(inspect.getfile(importlib.import_module("toploc"))).parent
    for path in sorted(proof_root.rglob("*")):
        if path.suffix in (".cpp", ".cu", ".h", ".hpp", ".cuh"):
            sources["toploc/source/" + str(path.relative_to(proof_root))] = file_hash(path)
    return {"versions": versions, "sources": sources}


@dataclass(frozen=True)
class MimoRuntimeSpec:
    version: str
    model_id: str
    runtime_fingerprint: dict
    chat_template_sha256: str
    enable_thinking: bool
    max_context_tokens: int
    eos_token_ids: tuple[int, ...]
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    attention: str = "eager"
    parameter_dtype: str = "bfloat16"
    recurrent_state_dtype: str = "float32"
    scope: str = "unqualified_cuda"

    def to_dict(self):
        result = asdict(self)
        result["eos_token_ids"] = list(self.eos_token_ids)
        return result

    @property
    def sha256(self):
        return digest(self.to_dict())


def validate_runtime_spec(value, *, allow_fixture=False):
    if isinstance(value, MimoRuntimeSpec):
        value = value.to_dict()
    expected = set(MimoRuntimeSpec.__dataclass_fields__)
    _need(type(value) is dict and set(value) == expected, "exact MiMo runtime profile")
    _need(value["version"] == VERSION and value["model_id"] == MODEL_ID, "MiMo model/runtime identity")
    _need(type(value["runtime_fingerprint"]) is dict and
          set(value["runtime_fingerprint"]) == {"versions", "sources"}, "pinned runtime fingerprint")
    _need(_sha(value["chat_template_sha256"]) and type(value["enable_thinking"]) is bool,
          "explicit template hash and thinking mode")
    _need(type(value["max_context_tokens"]) is int and 16 <= value["max_context_tokens"] <= 262144,
          "explicit bounded context")
    fixture = value["scope"] == "cpu_fixture"
    _need(value["scope"] in ("unqualified_cuda", "cpu_fixture") and (not fixture or allow_fixture),
          "CPU fixtures cannot be production profiles")
    _need(value["parameter_dtype"] == ("float32" if fixture else "bfloat16") and
          value["recurrent_state_dtype"] == "float32", "MiMo parameter/recurrent precision")
    _need(value["attention"] in ("eager", "sdpa"), "explicit supported attention backend")
    _need(type(value["eos_token_ids"]) is list and value["eos_token_ids"] and
          all(type(v) is int and v >= 0 for v in value["eos_token_ids"]) and
          len(set(value["eos_token_ids"])) == len(value["eos_token_ids"]), "nonempty unique EOS set")
    _need(fixture or tuple(value["eos_token_ids"]) == EOS_TOKEN_IDS, "full MiMo generation EOS set")
    _need(value["temperature"] == 1.0 and value["top_p"] == 0.95 and type(value["top_k"]) is int
          and value["top_k"] == 20, "reference RL sampler T1 top-p .95 top-k20")
    return MimoRuntimeSpec(**{**value, "eos_token_ids": tuple(value["eos_token_ids"])})


def make_runtime_spec(*, chat_template, enable_thinking, max_context_tokens,
                      attention="eager", scope="unqualified_cuda", eos_token_ids=EOS_TOKEN_IDS):
    """Prepare a candidate profile; caller must separately qualify and sign it."""
    _need(type(chat_template) is str and chat_template, "one explicit chat template")
    result = MimoRuntimeSpec(VERSION, MODEL_ID, runtime_fingerprint(),
        hashlib.sha256(chat_template.encode()).hexdigest(), enable_thinking,
        max_context_tokens, tuple(eos_token_ids), attention=attention,
        parameter_dtype="float32" if scope == "cpu_fixture" else "bfloat16", scope=scope)
    return validate_runtime_spec(result, allow_fixture=scope == "cpu_fixture")


class MimoRuntime:
    """One persistent actual HF hybrid model. No hidden checkpoint reloads."""
    @classmethod
    def load(cls, checkpoint_directory, checkpoint_descriptor, spec, *, device="cuda",
             trainable=False, allow_fixture=False):
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        spec = validate_runtime_spec(spec, allow_fixture=allow_fixture)
        _need((device == "cpu") == (spec.scope == "cpu_fixture"), "runtime device/scope pairing")
        _need(runtime_fingerprint() == spec.runtime_fingerprint, "executed runtime differs from profile")
        guard = CheckpointGuard(checkpoint_directory, checkpoint_descriptor)
        dtype = getattr(torch, spec.parameter_dtype)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_directory, local_files_only=True, trust_remote_code=False)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint_directory,
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=dtype, attn_implementation=spec.attention).to(device).eval()
        result = cls(model, tokenizer, checkpoint_descriptor["id"], spec, guard, trainable=trainable)
        guard()
        return result

    def __init__(self, model, tokenizer, checkpoint_id, spec, checkpoint_guard, *, trainable=False):
        import torch
        from transformers import Qwen3_5ForConditionalGeneration
        self.spec = validate_runtime_spec(spec, allow_fixture=spec.scope == "cpu_fixture")
        self._spec_digest = self.spec.sha256
        _need(isinstance(model, Qwen3_5ForConditionalGeneration), "actual Qwen3.5 conditional model")
        _need(_sha(checkpoint_id), "immutable checkpoint identity")
        self.model, self.tokenizer, self.checkpoint_id = model.eval(), tokenizer, checkpoint_id
        self._checkpoint_guard = checkpoint_guard
        self.eos_token_ids = self.spec.eos_token_ids
        self.vocab_size = model.config.text_config.vocab_size
        self.max_context_tokens = self.spec.max_context_tokens
        self.trainable = trainable
        self.closed = False
        self._staged_export = None
        self._model_id = id(model)
        self._config_digest = digest(model.config.to_dict())
        config = model.config.text_config
        _need(self.max_context_tokens <= config.max_position_embeddings, "context exceeds loaded model")
        _need(all(token < self.vocab_size for token in self.eos_token_ids), "EOS outside model vocabulary")
        expected_layers = ["linear_attention"] * 3 + ["full_attention"]
        _need(config.layer_types == expected_layers * (config.num_hidden_layers // 4), "MiMo hybrid layer pattern")
        _need(spec.scope == "cpu_fixture" or config.num_hidden_layers == 32, "MiMo 9B layer geometry")
        template = tokenizer.chat_template
        _need(type(template) is str and hashlib.sha256(template.encode()).hexdigest() == spec.chat_template_sha256,
              "loaded chat template differs from signed profile")
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        _need(eos == list(self.eos_token_ids), "generation stop set differs from profile")
        self.trainable_parameter_names = []
        for name, parameter in model.named_parameters():
            # Math-only updates do not touch the vision tower or optional MTP heads.
            active = bool(trainable and (name.startswith("model.language_model.") or name.startswith("lm_head.")))
            parameter.requires_grad_(active)
            if active:
                self.trainable_parameter_names.append(name)
        _need(not trainable or self.trainable_parameter_names, "nonempty text training parameter inventory")
        self.trainable_parameter_names = tuple(self.trainable_parameter_names)
        self._expected_dtype = getattr(torch, spec.parameter_dtype)
        self.gradient_checkpointing_layers = 0
        if trainable:
            self._checkpoint_text_layers()
        self.guard()

    def _checkpoint_text_layers(self):
        """Checkpoint decoder blocks while keeping dropout and model mode fixed.

        HF's default checkpoint switch runs only in training mode. This wrapper
        calls the very same layer forward under nonreentrant checkpointing, so
        OLD and actor both stay in eval mode and inference caches stay untouched.
        """
        import torch
        from torch.utils.checkpoint import checkpoint
        for layer in self.model.model.language_model.layers:
            original = layer.forward
            def make_forward(original):
                def forward(*args, **kwargs):
                    if torch.is_grad_enabled():
                        _need(not kwargs.get("use_cache") and kwargs.get("past_key_values") is None,
                              "training block checkpoint cannot share a generation cache")
                        return checkpoint(original, *args, use_reentrant=False, **kwargs)
                    return original(*args, **kwargs)
                return forward
            layer.forward = make_forward(original)
            self.gradient_checkpointing_layers += 1

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    def guard(self):
        _need(not self.closed and id(self.model) == self._model_id, "original open runtime model")
        _need(self.spec.sha256 == self._spec_digest, "runtime profile changed")
        self._checkpoint_guard()
        _need(digest(self.model.config.to_dict()) == self._config_digest, "runtime model config changed")
        _need(not any(module.training for module in self.model.modules()), "inference-stable eval mode required")
        _need(self.model.config.text_config._attn_implementation == self.spec.attention,
              "attention implementation changed")
        _need(hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest() == self.spec.chat_template_sha256,
              "runtime chat template changed")
        _need(self.model.get_input_embeddings().weight.dtype == self._expected_dtype and
              self.model.lm_head.weight.dtype == self._expected_dtype, "runtime parameter dtype changed")

    def _tokens(self, ids):
        _need(isinstance(ids, (tuple, list)) and ids and len(ids) <= self.max_context_tokens and
              all(type(x) is int and 0 <= x < self.vocab_size for x in ids), "bounded model token sequence")
        return list(ids)

    def prepare_prompt(self, messages, tools=()):
        self.guard()
        kwargs = dict(tokenize=True, return_dict=False, add_generation_prompt=True,
                      enable_thinking=self.spec.enable_thinking)
        if tools:
            kwargs["tools"] = list(tools)
        return self._tokens(self.tokenizer.apply_chat_template(messages, **kwargs))

    def next_token_logits(self, input_ids, past_key_values=None, *, use_cache=True):
        import torch
        self.guard()
        ids = self._tokens(input_ids)
        prefix = past_key_values.get_seq_length() if past_key_values is not None else 0
        _need(prefix + len(ids) <= self.max_context_tokens, "cached context limit")
        with torch.inference_mode():
            result = self.model(torch.tensor([ids], device=self.device), use_cache=use_cache,
                past_key_values=past_key_values, logits_to_keep=1)
        cache = result.past_key_values
        if cache is not None:
            self.check_recurrent_state(cache)
        return result.logits[0, -1].float(), cache

    @staticmethod
    def check_recurrent_state(cache):
        import torch
        count = 0
        for layer in cache.layers:
            states = getattr(layer, "recurrent_states", {})
            for value in states.values() if isinstance(states, dict) else states:
                if value is not None:
                    _need(value.dtype == torch.float32, "hybrid recurrent state must stay FP32")
                    count += 1
        _need(count > 0, "hybrid cache has no recurrent states")
        return count

    def response_log_probs(self, prompt_ids, output_ids, *, gradients=True, token_chunk_size=128):
        """Selected raw T1 probabilities; checkpoint the head to bound vocab RAM."""
        import torch
        from torch.utils.checkpoint import checkpoint
        self.guard()
        prompt, output = self._tokens(prompt_ids), self._tokens(output_ids)
        _need(len(prompt) + len(output) <= self.max_context_tokens, "response context limit")
        _need(type(token_chunk_size) is int and token_chunk_size >= 1, "positive token chunk size")
        _need(not gradients or self.trainable, "inference runtime cannot train")
        with torch.set_grad_enabled(gradients):
            hidden = self.model.model(torch.tensor([prompt + output], device=self.device),
                                      use_cache=False).last_hidden_state[0, len(prompt)-1:-1]
            targets = torch.tensor(output, device=self.device)
            def selected(states, tokens):
                logits = self.model.lm_head(states).float()
                return logits.gather(-1, tokens[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)
            chunks = []
            for start in range(0, len(output), token_chunk_size):
                hs, ts = hidden[start:start+token_chunk_size], targets[start:start+token_chunk_size]
                chunks.append(checkpoint(selected, hs, ts, use_reentrant=False) if gradients else selected(hs, ts))
            return torch.cat(chunks)

    def score(self, prompt, output, *, gradients=False):
        return self.response_log_probs(prompt, output, gradients=gradients)

    def score_support(self, prompt_ids, output_ids, *, top_k, top_p, token_chunk_size=128):
        """OLD selected log-probs plus the replayed sampler support (no grad).

        Used only when the trainer's sampler_correction flag is enabled. The
        support is recomputed from trusted prefill logits, never miner claims.
        """
        import torch
        from .mimo_grpo.sampler_correction import replay_support
        self.guard()
        prompt, output = self._tokens(prompt_ids), self._tokens(output_ids)
        _need(len(prompt) + len(output) <= self.max_context_tokens, "response context limit")
        with torch.no_grad():
            hidden = self.model.model(torch.tensor([prompt + output], device=self.device),
                                      use_cache=False).last_hidden_state[0, len(prompt)-1:-1]
            targets = torch.tensor(output, device=self.device)
            selected, indices, qs = [], [], []
            for start in range(0, len(output), token_chunk_size):
                logits = self.model.lm_head(hidden[start:start+token_chunk_size]).float()
                ts = targets[start:start+token_chunk_size]
                selected.append(logits.gather(-1, ts[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1))
                index, q = replay_support(logits, top_k, top_p)
                indices.append(index); qs.append(q)
            return torch.cat(selected), torch.cat(indices), torch.cat(qs)

    def score_candidates(self, prompt_ids, output_ids, indices, *, gradients=True, token_chunk_size=128):
        """Selected raw log-probs and logits restricted to replayed candidates."""
        import torch
        from torch.utils.checkpoint import checkpoint
        self.guard()
        prompt, output = self._tokens(prompt_ids), self._tokens(output_ids)
        _need(len(prompt) + len(output) <= self.max_context_tokens, "response context limit")
        _need(not gradients or self.trainable, "inference runtime cannot train")
        indices = torch.as_tensor(indices, device=self.device)
        _need(indices.ndim == 2 and indices.shape[0] == len(output), "candidate ids for every position")
        with torch.set_grad_enabled(gradients):
            hidden = self.model.model(torch.tensor([prompt + output], device=self.device),
                                      use_cache=False).last_hidden_state[0, len(prompt)-1:-1]
            targets = torch.tensor(output, device=self.device)
            def both(states, tokens, ids):
                logits = self.model.lm_head(states).float()
                return torch.cat([(logits.gather(-1, tokens[:, None]) -
                                   torch.logsumexp(logits, dim=-1, keepdim=True)), logits.gather(-1, ids)], -1)
            chunks = []
            for start in range(0, len(output), token_chunk_size):
                args = (hidden[start:start+token_chunk_size], targets[start:start+token_chunk_size],
                        indices[start:start+token_chunk_size])
                chunks.append(checkpoint(both, *args, use_reentrant=False) if gradients else both(*args))
            values = torch.cat(chunks)
            return values[:, 0], values[:, 1:]

    def prefill_log_probs(self, prompt, output):
        return self.response_log_probs(prompt, output, gradients=False)

    def compute(self, prompt, output):
        """Compatibility seam for TOPLOC. Full distributions stay local."""
        import torch
        self.guard()
        prompt, output = self._tokens(prompt), self._tokens(output)
        _need(len(prompt) + len(output) <= self.max_context_tokens, "proof context limit")
        with torch.inference_mode():
            result = self.model(torch.tensor([prompt+output], device=self.device),
                                output_hidden_states=True, use_cache=False)
            hidden = result.hidden_states[-1][0].to(dtype=torch.bfloat16, device="cpu").contiguous()
            logits = result.logits[0, len(prompt)-1:len(prompt)+len(output)-1]
            log_probs = torch.log_softmax(logits.float(), -1).cpu().numpy()
        acts = [hidden[:len(prompt)]] + [hidden[i:i+1] for i in range(len(prompt), len(prompt)+len(output))]
        return acts, log_probs

    def save_model(self, directory, *, max_shard_size="2GB"):
        """Stage only model/tokenizer state; caller verifies durable publication."""
        self.guard()
        path = Path(directory)
        _need(not path.exists() or (path.is_dir() and not any(path.iterdir())), "new empty export directory")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True, max_shard_size=max_shard_size)
        self.tokenizer.save_pretrained(path)
        files = {p.name: file_hash(p) for p in path.iterdir() if p.is_file()}
        _need(set(files) == set(p.name for p in path.iterdir()), "flat ordinary HF export")
        descriptor = {"id": digest(files), "files": files}
        self._staged_export = (descriptor, CheckpointGuard(path.absolute(), descriptor))
        return descriptor

    def acknowledge_model_publication(self, checkpoint_id, checkpoint_files):
        """Advance a trained actor without reloading, after caller checks R2 receipt.

        Only this runtime's most recent model-only export may become its parent.
        Durable upload/readback and transaction acknowledgement remain the
        publication worker's responsibility; this method never claims those.
        """
        _need(self.trainable and self._staged_export is not None, "trained model export required")
        descriptor = validate_descriptor({"id": checkpoint_id, "files": checkpoint_files})
        staged, guard = self._staged_export
        _need(staged == descriptor, "publication must match this actor's staged export")
        guard()
        self.checkpoint_id = checkpoint_id
        self._checkpoint_guard = guard
        self._staged_export = None

    def close(self):
        if not self.closed:
            self.closed = True
            self.model = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class CheckpointCache:
    """Private, single-process cache with reference leases and bounded bytes.

    ``fetch(name, destination)`` must download from the authenticated descriptor's
    durable store. No completion callback deletes the active checkpoint: only
    unreferenced old generations are evicted under admission pressure.
    """
    def __init__(self, root, *, max_bytes, max_checkpoints=2, disk_reserve_bytes=1024**3):
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _need(self.root.resolve() == self.root, "ordinary private cache root")
        _need(type(max_bytes) is int and max_bytes > 0 and type(max_checkpoints) is int and max_checkpoints > 0,
              "positive bounded cache limits")
        _need(type(disk_reserve_bytes) is int and disk_reserve_bytes >= 0, "nonnegative disk reserve")
        self.max_bytes, self.max_checkpoints, self.disk_reserve_bytes = max_bytes, max_checkpoints, disk_reserve_bytes
        self._lock_file = open(self.root / ".worker.lock", "a+b")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise ValueError("checkpoint cache already owned by a worker") from None
        self._lock = threading.RLock()
        self._entries = {}
        self._counter = 0
        self.closed = False
        # Reconcile only paths owned by this implementation after a process crash.
        for path in self.root.glob(".download-*"):
            _need(path.is_dir() and not path.is_symlink(), "ordinary stale download directory")
            shutil.rmtree(path)

    def _scan(self):
        result = []
        for path in self.root.iterdir():
            if _sha(path.name):
                _need(path.is_dir() and not path.is_symlink(), "ordinary cache generation")
                size = sum(p.stat().st_size for p in path.iterdir() if p.is_file())
                entry = self._entries.get(path.name, {})
                result.append((entry.get("last", 0), path, size, entry.get("references", 0)))
        return result

    def _room(self, incoming_bytes, incoming_count):
        rows = sorted(self._scan())
        total = sum(row[2] for row in rows)
        count = len(rows)
        for _, path, size, references in rows:
            if (total + incoming_bytes <= self.max_bytes and count + incoming_count <= self.max_checkpoints
                    and shutil.disk_usage(self.root).free >= incoming_bytes + self.disk_reserve_bytes):
                return
            if references:
                continue
            shutil.rmtree(path)
            self._entries.pop(path.name, None)
            total -= size
            count -= 1
        _need(total + incoming_bytes <= self.max_bytes and count + incoming_count <= self.max_checkpoints
              and shutil.disk_usage(self.root).free >= incoming_bytes + self.disk_reserve_bytes,
              "checkpoint cache capacity occupied by in-flight generations")

    @contextmanager
    def acquire(self, descriptor, *, fetch=None, expected_bytes=None):
        validate_descriptor(descriptor)
        key = descriptor["id"]
        with self._lock:
            _need(not self.closed, "cache closed")
            path = self.root / key
            entry = self._entries.get(key)
            if entry is None:
                if not path.exists():
                    _need(fetch is not None and type(expected_bytes) is int and 0 < expected_bytes <= self.max_bytes,
                          "bounded authenticated download size required")
                    self._room(expected_bytes, 1)
                    staging = self.root / (".download-" + uuid.uuid4().hex)
                    staging.mkdir(mode=0o700)
                    try:
                        total = 0
                        for name in descriptor["files"]:
                            destination = staging / name
                            fetch(name, destination)
                            _need(destination.is_file() and not destination.is_symlink(), "ordinary downloaded file")
                            total += destination.stat().st_size
                            _need(total <= expected_bytes, "download exceeds signed size bound")
                        CheckpointGuard(staging, descriptor)
                        os.rename(staging, path)
                    finally:
                        if staging.exists():
                            shutil.rmtree(staging)
                guard = CheckpointGuard(path, descriptor)
                entry = self._entries[key] = {"references": 0, "guard": guard, "last": 0}
            _need(entry["guard"].descriptor == descriptor, "cache descriptor conflict")
            entry["guard"]()
            self._counter += 1
            entry["last"] = self._counter
            entry["references"] += 1
            try:
                # A restarted worker may have stricter limits than its previous
                # process. Retain this admitted generation while reconciling
                # old disk entries; a warm cache hit must not bypass the bound.
                self._room(0, 0)
            except BaseException:
                entry["references"] -= 1
                raise
        try:
            yield path
        finally:
            with self._lock:
                entry["references"] -= 1

    def close(self):
        with self._lock:
            _need(not any(e["references"] for e in self._entries.values()), "in-flight checkpoint leases")
            if not self.closed:
                self.closed = True
                self._lock_file.close()


class MimoRuntimePool:
    """Reuse a loaded inference runtime across audit/mining jobs for an epoch."""
    def __init__(self, cache, *, max_loaded=1, loader=MimoRuntime.load):
        _need(type(max_loaded) is int and max_loaded > 0, "positive loaded runtime limit")
        self.cache, self.max_loaded, self.loader = cache, max_loaded, loader
        self._entries = {}
        self._lock = threading.RLock()

    @contextmanager
    def acquire(self, descriptor, spec, *, device="cuda", fetch=None, expected_bytes=None, allow_fixture=False):
        spec = validate_runtime_spec(spec, allow_fixture=allow_fixture)
        key = (descriptor["id"], spec.sha256, device)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                for old_key, old in list(self._entries.items()):
                    if len(self._entries) < self.max_loaded:
                        break
                    if old["references"] == 0:
                        old["runtime"].close()
                        old["lease"].__exit__(None, None, None)
                        del self._entries[old_key]
                _need(len(self._entries) < self.max_loaded, "all loaded runtimes are in flight")
                lease = self.cache.acquire(descriptor, fetch=fetch, expected_bytes=expected_bytes)
                path = lease.__enter__()
                try:
                    runtime = self.loader(path, descriptor, spec, device=device, trainable=False,
                                          allow_fixture=allow_fixture)
                except BaseException:
                    lease.__exit__(None, None, None)
                    raise
                entry = self._entries[key] = {"runtime": runtime, "lease": lease, "references": 0}
            entry["runtime"].guard()
            entry["references"] += 1
        try:
            yield entry["runtime"]
        finally:
            with self._lock:
                entry["references"] -= 1

    def close(self):
        with self._lock:
            _need(not any(e["references"] for e in self._entries.values()), "in-flight runtime leases")
            for entry in self._entries.values():
                entry["runtime"].close()
                entry["lease"].__exit__(None, None, None)
            self._entries.clear()


class PersistentMimoRuntimeFactory:
    """Worker-owned facade factory for successive complete-group audit jobs.

    Pass this object as ``runtime_factory`` to complete-group ``verify``. That
    entrypoint holds ``runtime_for_job`` throughout the audit. The raw backend
    is shared, while miner/task/challenge bindings get a fresh lightweight
    facade. Closing one job does not release GPU weights or checkpoint files.

    ``fetch_file(manifest, name, destination)`` may download directly from the
    worker's authenticated artifact store. Without it, the first job copies its
    already-downloaded checkpoint into this private cache once. It never reads
    that staging directory again while the checkpoint remains resident.
    """
    def __init__(self, cache_root, *, max_bytes, max_checkpoints=2,
                 max_checkpoint_bytes=64 * 1024**3, disk_reserve_bytes=1024**3,
                 device="cuda", fetch_file=None, loader=MimoRuntime.load,
                 facade_builder=None, allow_fixture=False):
        _need(type(max_checkpoint_bytes) is int and 0 < max_checkpoint_bytes <= max_bytes,
              "bounded individual checkpoint download")
        self.cache = CheckpointCache(cache_root, max_bytes=max_bytes,
            max_checkpoints=max_checkpoints, disk_reserve_bytes=disk_reserve_bytes)
        self.pool = MimoRuntimePool(self.cache, max_loaded=1, loader=loader)
        self.device, self.fetch_file = device, fetch_file
        self.max_checkpoint_bytes = max_checkpoint_bytes
        self.facade_builder = facade_builder
        self.allow_fixture = allow_fixture
        self.closed = False
        # The HF multimodal wrapper updates rotary/position bookkeeping even
        # during text inference. One persistent model may serve successive jobs
        # but concurrent callers must not interleave that mutable state.
        self._job_lock = threading.Lock()

    @contextmanager
    def runtime_for_job(self, checkpoint, manifest, environment, harness, *, miner):
        _need(not self.closed, "persistent runtime worker closed")
        _need(self._job_lock.acquire(blocking=False), "persistent runtime already in flight")
        try:
            with self._runtime_for_job(checkpoint, manifest, environment, harness, miner=miner) as runtime:
                yield runtime
        finally:
            self._job_lock.release()

    @contextmanager
    def _runtime_for_job(self, checkpoint, manifest, environment, harness, *, miner):
        descriptor = {key: manifest["checkpoint"][key] for key in ("id", "files")}
        spec = validate_runtime_spec(manifest["mimo_runtime"], allow_fixture=self.allow_fixture)
        copied_bytes = 0
        def fetch(name, destination):
            nonlocal copied_bytes
            if self.fetch_file is not None:
                self.fetch_file(manifest, name, destination)
            else:
                source = Path(checkpoint) / name
                _need(source.is_file() and not source.is_symlink(), "ordinary checkpoint staging file")
                copied_bytes += source.stat().st_size
                _need(copied_bytes <= self.max_checkpoint_bytes, "local checkpoint exceeds download bound")
                shutil.copyfile(source, destination)
        with self.pool.acquire(descriptor, spec, device=self.device, fetch=fetch,
                expected_bytes=self.max_checkpoint_bytes, allow_fixture=self.allow_fixture) as backend:
            builder = self.facade_builder
            if builder is None:
                from .complete_group_runtime import MimoGroupRuntime
                builder = MimoGroupRuntime
            yield builder(checkpoint, manifest, environment, harness, miner=miner, backend=backend)

    def close(self):
        if not self.closed:
            self.pool.close()
            self.cache.close()
            self.closed = True
