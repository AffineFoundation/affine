"""vLLM process lifecycle on the eval machine.

Serving slots, GPU allocation from affine.toml:
  teacher    — frozen reference. Local vLLM when [teacher].base_url is
               empty; otherwise a remote OpenAI-compatible endpoint (the
               teacher swarm router) and no local teacher process.
  king       — reigning champion, loaded on first duel, kept warm across
               duels, swapped only when the validator sends a new king ref.
               Optional king2 replica ([miner_serving].king_replica_gpus).
  challenger — loaded per duel, killed afterwards. Optional challenger2
               replica ([miner_serving].challenger_replica_gpus).

Models are served with `vllm serve <repo> --revision <sha>` so the snapshot
is pinned to the on-chain commitment (TOCTOU). HF cache lives under
/root/hf; a stale challenger snapshot is pruned lazily right before the next
challenger's download (not eagerly after the duel), so validator retries and
repeat duels of the same repo@revision skip the multi-GB re-download while
disk stays bounded to teacher + king + at most one challenger.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from . import r2store
from .vllm_client import Served

log = logging.getLogger("evalsrv.engine")

HF_HOME = os.environ.get("HF_HOME", "/root/hf")
LOG_DIR = Path(os.environ.get("AFFINE_LOG_DIR", "/root/logs"))

# Free-disk headroom (GB) a challenger download needs on top of its servable
# weights. Also the low-disk floor `diagnose_load_failure` treats as pod
# trouble, so the pre-download fit check and the post-failure diagnosis agree.
CHALLENGER_DISK_HEADROOM_GB = 20.0

# Kill a prefetch download that reports no progress for this long. Progress
# is the r2store child's heartbeat (every landed 8 MB chunk + every hashed
# chunk, see r2store.Progress) or, for HF downloads, on-disk bytes.
# hf_transfer can hang forever on a dead TCP connection (observed live:
# 0 B/s with ESTABLISHED sockets and no timeout); a hung in-process thread
# would block every future prefetch and stall the next load_challenger join.
# History: measuring on-disk bytes alone mis-fired on every first attempt
# (2026-09-05/06, "failed after 195s"): a preallocated `.incomplete` counted
# at full size until its first part landed, so the count dropped and could
# not recover inside the window. r2store's per-part retry budget (123 s) is
# kept under this window on purpose.
PREFETCH_STALL_S = 180.0
# Completed-but-unconsumed prefetch snapshots kept out of the cache prune.
# 2 covers "next" plus "next-next" when the validator prefetches ahead of a
# dispatch; each is one challenger (≤ submission.max_model_size_gb) of disk.
PREFETCH_KEEP = 2

# Warm-swap challengers (2026-09-07). A challenger vLLM start costs ~8.5
# min on the eval box (process/engine init ~2.5, weight read ~2, compile
# ~1, profiling + CUDA-graph capture ~2) — ~25% of a 35 min verdict. With
# the pinned architecture every challenger has the same tensor layout, so
# the challenger engines stay alive between duels and the next checkpoint's
# weights are loaded IN PLACE through the evalsrv.vllm_ext.WeightTools
# worker extension (raw `model.load_weights` from the new snapshot over the
# dev-mode /collective_rpc endpoint, then /reset_prefix_cache). Only the
# 72 GB weight read is paid (~10 s from page cache, ~2 min cold). Guarded by
# _swap_compatible: anything that could make a swapped engine score the
# miner differently from a fresh one (config.json beyond cosmetic keys —
# dtype, rope, norms —, tokenizer files, generation_config sampling
# defaults, the tensor-name set) forces the old full relaunch.
# Score-invariant by construction: same weights, same engine config, KV
# cache reset.
#
# Verification (2026-09-07, commit ffd870b + this one): on a 1x B200
# (TP1) and a 2x H200 (TP2, the eval pod's layout), swapping king ->
# challenger gave weights BIT-IDENTICAL to a fresh load of the challenger on
# every rank (948/948 tensors, sha256 of raw bytes) and identical logprobs
# (delta 0.0, prompts up to 39k tokens), for both the raw path used here and
# vLLM's own layerwise `reload_weights`; each swapped engine then served 10
# min of duel-like load (~1000 completions) with zero failures. vLLM's
# "Following weights were not loaded from checkpoint" warning (fused MoE
# routed_experts.w13/w2 names) appears on these correct swaps too: it is a
# bookkeeping artefact of the reload wrapper's return value, not data loss.
# (The same-day emergency disable was triggered by that warning plus two
# engine deaths that turned out to be the operator's own redeploy pkill.)
# Runtime invariant checked after every swap: the raw loader must report
# exactly the pinned architecture's expected counts (SWAP_EXPECTED_LOADED
# reported names, and the SWAP_EXPECTED_UNREPORTED fused-expert names that
# FusedMoE never reports) — any other shape of result -> relaunch.
WARM_SWAP = os.environ.get("AFFINE_CHALLENGER_WARM_SWAP", "1") != "0"
# Pinned-arch constants observed on the verified swaps (Qwen3.6-35B-A3B
# family with vision tower, 40 layers): 906 names reported loaded, 80 fused
# expert params (w13_weight + w2_weight x 40 layers) legitimately
# unreported. The admitted text-only variant (Qwen3_5MoeForCausalLM) has a
# different count, so it always takes the cold relaunch until verified.
SWAP_EXPECTED_LOADED = 906
SWAP_UNREPORTED_SUFFIXES = (".mlp.experts.routed_experts.w13_weight",
                            ".mlp.experts.routed_experts.w2_weight")
WORKER_EXTENSION = "evalsrv.vllm_ext.WeightTools"
# Fixed served-model alias for the challenger slots so requests keep
# resolving across swaps (the repo name is added too, for logs).
CHALLENGER_ALIAS = "challenger"
SWAP_CONFIG_IGNORE_KEYS = ("_name_or_path", "transformers_version")
# generation_config.json keys vLLM turns into request defaults (see
# ModelConfig.get_diff_sampling_param) plus the stop-token set. Anything
# else in that file (transformers_version, do_sample, duplicated eos ids)
# does not change how the engine samples. Overnight 2026-09-08 byte
# comparison forced 10/32 cold relaunches on files that differed only by a
# duplicated eos id and a version string.
SWAP_GEN_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                 "max_new_tokens", "max_length")
# tokenizer_config.json special-token strings the engine reads at launch
# (chat_template is client-side rendering here; model_max_length is
# overridden by max_model_len). Behavioural flags (add_bos_token,
# clean_up_tokenization_spaces, ...) are judged by the probes below, not by
# spelling: "false" vs absent is the same tokenizer.
SWAP_TOKCFG_KEYS = ("eos_token", "bos_token", "pad_token", "unk_token")
# Tokenizer equivalence is decided empirically: both tokenizers must encode
# this probe (plus every added-token string of either side, plus the chat
# markers the duel renders) to identical ids, and decode identically.
# tokenizer.json files from different transformers versions differ in
# serialisation (merges as pairs vs strings, decoder flags, regex text)
# while tokenizing identically; a real difference — extra added tokens,
# another pre-tokenizer — shows up on the probe and forces a relaunch.
SWAP_TOKENIZER_PROBE = (
    "def f(x):\n    return {'a': x ** 2, \"b\": [1, 2, 3]}  # comment\n"
    "SELECT id, name FROM users WHERE created_at >= '2026-01-01' AND id IN (1,22,333,4444);\n"
    "$ ls -la /usr/local/bin | grep -E '^-rwx' | awk '{print $9}'\n"
    "if err != nil {\n\treturn fmt.Errorf(\"wrap: %w\", err)\n}\n"
    "let x: Vec<u8> = vec![0u8; 1024]; println!(\"{:?}\", &x[..4]);\n"
    "Ünïcödé — 日本語のテキスト, русский текст, العربية, 🤖🚀 ✓ ∑∫√ 1234567890 3.14159 1e-9\n"
    "    \t  mixed   whitespace\n\n\n\r\n tabs\t\ttabs\n"
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
    "<|im_start|>assistant\n<think>\nthinking...\n</think>\n\n```bash\ncd /repo && make test\n```\n"
    "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"x\"}}\n</tool_call>\n\\boxed{42}\n"
    "camelCaseIdentifier snake_case_identifier CONSTANT_VALUE __dunder__ ->>= <<== != === ...\n"
)

# Best-effort second copy (king2 / challenger2). Primary ready is enough to
# start scoring; waiting the full vLLM launch (up to 20 min) for a slow or
# dead replica serialized the duel behind a GPU that may never come up.
# If the process is still alive after this grace, leave it warming — the
# next ensure/load can pick it up instead of killing a half-loaded engine.
REPLICA_READY_S = 90.0

# Pip nvidia-cuda-nvcc wheel ships nvcc under site-packages; Lium images often
# lack /usr/local/cuda (or ship an EMPTY stub of it). FlashInfer JIT needs both
# nvcc AND the toolkit headers (cuda_fp16.h, cublasLt.h), so a candidate only
# counts as a CUDA home when both are present — an nvcc-only stub caused JIT
# 'cuda_fp16.h: No such file or directory' king-launch failures on B300 pods.
def _cuda_complete(p: Path) -> bool:
    return (p / "bin" / "nvcc").exists() and (p / "include" / "cuda_fp16.h").exists()


def _cuda_home() -> str:
    if os.environ.get("CUDA_HOME") and _cuda_complete(Path(os.environ["CUDA_HOME"])):
        return os.environ["CUDA_HOME"]
    try:
        # Namespace package: __file__ is None, __path__ lists real roots.
        import nvidia  # type: ignore
        roots = [Path(p) for p in nvidia.__path__]
    except Exception:
        roots = []
    for root in roots:
        for cand in (root / "cu13", root / "cuda_runtime"):
            if _cuda_complete(cand):
                return str(cand)
    for p in (Path("/usr/local/cuda"), Path("/usr/lib/cuda")):
        if _cuda_complete(p):
            return str(p)
    return os.environ.get("CUDA_HOME", "/usr/local/cuda")


PUBLIC_MODELS_BUCKET_PREFIX = "r2://affine-models/"


def _is_public_king_cache(repo_dir: Path) -> bool:
    """True when a cache dir holds a verified snapshot served from the public
    bucket (only crowned kings live there). Kept out of the challenger prune:
    after an evalsrv restart king_slot.served is empty, so a /prefetch prune
    could evict the king's 72 GB and force a re-download before the next
    duel (2026-09-07 22:5x: the king snapshot vanished while the server sat
    without R2 credentials). A few crowned kings is bounded disk."""
    try:
        for marker in repo_dir.glob(f"snapshots/*/{r2store.COMPLETE_MARKER}"):
            repo = json.loads(marker.read_text()).get("repo") or ""
            if repo.startswith(PUBLIC_MODELS_BUCKET_PREFIX):
                return True
    except (OSError, ValueError):
        pass
    return False


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _generation_config_mismatch(old: Path, new: Path) -> str | None:
    """Sampling defaults and stop tokens the engine took from the loaded
    checkpoint's generation_config.json must be what the incoming one
    specifies. Semantic comparison (see SWAP_GEN_KEYS)."""
    ga = _load_json(old / "generation_config.json") or {}
    gb = _load_json(new / "generation_config.json") or {}
    for k in SWAP_GEN_KEYS:
        if ga.get(k) != gb.get(k):
            return f"generation_config {k}: {ga.get(k)!r} vs {gb.get(k)!r}"

    def ids(g, k):
        v = g.get(k)
        return frozenset(v) if isinstance(v, list) else frozenset([v] if v is not None else [])

    for k in ("eos_token_id", "pad_token_id", "bos_token_id"):
        if ids(ga, k) != ids(gb, k):
            return f"generation_config {k}: {ga.get(k)!r} vs {gb.get(k)!r}"
    return None


def _tokenizer_mismatch(old: Path, new: Path) -> str | None:
    """The engine keeps the tokenizer it was launched with; the incoming
    checkpoint's tokenizer must behave identically. Runtime-relevant
    tokenizer_config keys must match, and both tokenizers must encode a
    rich probe (plus every added token of either side) to identical ids and
    decode identically."""
    ta = _load_json(old / "tokenizer_config.json") or {}
    tb = _load_json(new / "tokenizer_config.json") or {}

    def norm(v):
        return v.get("content") if isinstance(v, dict) else v

    for k in SWAP_TOKCFG_KEYS:
        if norm(ta.get(k)) != norm(tb.get(k)):
            return f"tokenizer_config {k}: {norm(ta.get(k))!r} vs {norm(tb.get(k))!r}"
    if (old / "tokenizer.json").is_file() != (new / "tokenizer.json").is_file():
        return "tokenizer.json present on one side only"
    tok_a = AutoTokenizer.from_pretrained(str(old))
    tok_b = AutoTokenizer.from_pretrained(str(new))
    if len(tok_a) != len(tok_b):
        return f"tokenizer size {len(tok_a)} vs {len(tok_b)}"
    added = sorted(set(tok_a.get_added_vocab()) | set(tok_b.get_added_vocab()))
    probe = SWAP_TOKENIZER_PROBE + "\n".join(added) + "\n" + " ".join(added)
    for special in (False, True):
        ia = tok_a(probe, add_special_tokens=special)["input_ids"]
        ib = tok_b(probe, add_special_tokens=special)["input_ids"]
        if ia != ib:
            first = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y),
                         min(len(ia), len(ib)))
            return (f"tokenizer encodes probe differently (add_special_tokens="
                    f"{special}, first divergence at token {first}, lens {len(ia)}/{len(ib)})")
    if tok_a.decode(ia) != tok_b.decode(ia):
        return "tokenizer decodes probe differently"
    return None


def _vllm_env() -> dict[str, str]:
    cuda_home = _cuda_home()
    path = os.environ.get("PATH", "")
    bin_dir = str(Path(cuda_home) / "bin")
    if bin_dir not in path.split(":"):
        path = f"{bin_dir}:{path}"
    lib_dir = str(Path(cuda_home) / "lib")
    lib64_dir = str(Path(cuda_home) / "lib64")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    lib_path = os.environ.get("LIBRARY_PATH", "")
    for d in (lib_dir, lib64_dir):
        if Path(d).is_dir():
            if d not in ld.split(":"):
                ld = f"{d}:{ld}" if ld else d
            if d not in lib_path.split(":"):
                lib_path = f"{d}:{lib_path}" if lib_path else d
    return {
        "HF_HOME": HF_HOME,
        "CUDA_HOME": cuda_home,
        "CUDA_PATH": cuda_home,
        "PATH": path,
        "LD_LIBRARY_PATH": ld,
        "LIBRARY_PATH": lib_path,
        # vLLM 0.26: VLLM_ATTENTION_BACKEND env is ignored; use CLI flag in _launch.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
        # Opt out of FlashInfer MoE JIT. Do NOT set VLLM_FLASHINFER_MOE_BACKEND
        # — on vLLM 0.22 it only accepts throughput|latency|masked_gemm and an
        # invalid value crashes workers after weights load. MoE backend is
        # forced via `--moe-backend triton` in _launch instead.
        "VLLM_USE_FLASHINFER_MOE_FP16": "0",
        "VLLM_USE_FLASHINFER_MOE_FP8": "0",
        "VLLM_USE_FLASHINFER_MOE_FP4": "0",
    }


def _purge_broken_flashinfer_moe_cache() -> None:
    """Drop FlashInfer fused_moe_trtllm JIT artifacts that crash load.

    A failed PTX 9.3 compile leaves broken objects under ~/.cache/flashinfer;
    the next auto-select attempt can re-hit them. Safe no-op when absent.
    """
    roots = [
        Path.home() / ".cache" / "flashinfer",
        Path(HF_HOME).parent / ".cache" / "flashinfer",
        Path("/root/.cache/flashinfer"),
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*fused_moe_trtllm*"):
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                log.warning("could not purge flashinfer cache %s", path,
                            exc_info=True)


def _vllm_log_tail(slot_label: str, max_chars: int = 1200) -> str:
    """Last chunk of a slot's vllm log — used to surface load-failure cause."""
    path = LOG_DIR / f"vllm_{slot_label}.log"
    if not path.exists():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) > max_chars:
        data = data[-max_chars:]
    text = data.decode("utf-8", errors="replace")
    # Prefer the most informative failure markers when present.
    for needle in ("Unsupported .version", "ptxas fatal", "Engine core",
                   "CUDA out of memory", "RuntimeError", "Error"):
        idx = text.rfind(needle)
        if idx >= 0:
            return text[max(0, idx - 80):].strip()
    return text.strip()


# A vLLM launch whose log shows one of these has already lost its engine
# core or a TP worker. The API server process usually stays alive anyway
# (observed 2026-09-11: warmup CUDA assert on vLLM 0.29.0 killed both TP
# workers, the leader idled, and `_wait_ready` sat out the full 3600 s
# before failing the duel). Match on the text written AFTER this launch
# started — the per-slot log is append-only across launches.
VLLM_FATAL_MARKERS = (
    "EngineCore failed to start",
    "Engine core initialization failed",
    "WorkerProc failed to start",
    "WorkerProc hit an exception",
    "device-side assert triggered",
    "CUDA out of memory",
    "flashinfer-cubin version",
)


def _vllm_log_fatal(slot_label: str, since_offset: int) -> str:
    """First fatal marker written to the slot's log since `since_offset`,
    with a little context; '' when none."""
    path = LOG_DIR / f"vllm_{slot_label}.log"
    try:
        with path.open("rb") as fh:
            fh.seek(max(0, since_offset))
            data = fh.read()
    except OSError:
        return ""
    if not data:
        return ""
    text = data.decode("utf-8", errors="replace")
    best = None
    for needle in VLLM_FATAL_MARKERS:
        idx = text.find(needle)
        if idx >= 0 and (best is None or idx < best):
            best = idx
    if best is None:
        return ""
    return text[max(0, best - 80):best + 400].strip()


@dataclass
class Slot:
    label: str
    port: int
    gpus: str
    tp: int
    served: Served | None = None
    proc: subprocess.Popen | None = None
    ready: bool = False
    load_error: str = ""
    # Process group of the launched vllm, captured at spawn. Kept separately
    # from `proc` so orphaned workers can still be reaped after the leader
    # process has already crashed (proc.poll() is not None ⇒ getpgid fails).
    pgid: int | None = None
    # Byte offset of the slot's append-only vllm log when this launch
    # started; fatal-marker scans (`_vllm_log_fatal`) read from here so an
    # earlier launch's crash cannot fail the current one.
    log_offset: int = 0


@dataclass
class Engine:
    cfg: dict  # full affine.toml dict
    teacher_slot: Slot = field(init=False)
    # Optional second teacher replica ([teacher].replica_gpus). Best-effort:
    # duels fall back to the primary alone when it is down. Unused when
    # [teacher].base_url is set (remote cutover).
    teacher2_slot: Slot | None = field(init=False, default=None)
    king_slot: Slot = field(init=False)
    king2_slot: Slot | None = field(init=False, default=None)
    chall_slot: Slot = field(init=False)
    chal2_slot: Slot | None = field(init=False, default=None)
    # One reentrant lock owns every keep-set read and cache prune, every slot
    # launch, and every prefetch state transition. Callers run in three thread
    # contexts (FastAPI threadpool, duel/bench job threads, teacher warmup);
    # without the lock a /prefetch prune could race a duel's prune→launch
    # window and delete the snapshot the duel just decided to keep.
    _lock: threading.RLock = field(init=False, default_factory=threading.RLock,
                                   repr=False)
    # Serializes ensure_teacher: the startup warmup thread and a duel job
    # can call it concurrently (evalsrv bounce with a dispatch in flight),
    # and unserialized each caller sees not-ready, relaunches, and sweeps
    # the other's warming workers as GPU orphans — teachers never finish
    # warmup and the load-failure self-kill loops the server (2026-08-15).
    # Cannot ride _lock: ensure_teacher blocks for minutes in _wait_ready,
    # which must not hold up prefetch/prune/launch state transitions.
    _teacher_lock: threading.Lock = field(init=False,
                                          default_factory=threading.Lock,
                                          repr=False)
    # Next-challenger snapshot being warmed while the current duel scores
    # (download is network-bound, scoring is GPU-bound): (repo, worker,
    # cancel event). Protection from pruning is derived from thread liveness:
    # a dead worker confers none, so a stale prefetch target self-expires
    # instead of inflating the keep set forever (which would shrink
    # reclaimable disk and could wedge the pod into POD_CAPACITY faults).
    _prefetch: tuple[str, threading.Thread, threading.Event] | None = \
        field(init=False, default=None, repr=False)
    # r2 refs being fetched inline by _ensure_snapshot (kept from pruning).
    _materializing: set[str] = field(init=False, default_factory=set, repr=False)
    # Completed prefetch targets (repo -> revision) whose duel has not
    # started yet. A finished prefetch has no live worker, so before this it
    # was protected by nothing: the validator's /prefetch for the *following*
    # queue item arrived seconds after a verdict, its challenger_fits prune
    # ran with the challenger slot already empty, and the snapshot the next
    # duel was about to use was evicted and re-downloaded from R2 (observed
    # every duel on 2026-09-05: ~13 min at ~120 MB/s, 65 min when parts
    # stalled). Bounded to PREFETCH_KEEP entries, insertion-ordered; an entry
    # leaves when its snapshot is gone or its repo is launched.
    _prefetched_ready: dict[str, str] = field(init=False, default_factory=dict,
                                              repr=False)
    # Repos of the duel being prepared right now (king + challenger), kept
    # out of every prune from /duel acceptance until the job ends. Closes the
    # second eviction path seen 2026-09-05 18:12: a /prefetch for the NEXT
    # queue item arrived while this duel's king was still downloading; its
    # prune ran with no slot served yet and no live prefetch, and evicted this
    # duel's own (complete) challenger snapshot — re-downloaded minutes later.
    _pending_duel: set[str] = field(init=False, default_factory=set, repr=False)

    def __post_init__(self):
        t = self.cfg["teacher"]
        ms = self.cfg["miner_serving"]
        self.role = os.environ.get("AFFINE_ROLE", "duel")
        if self.role in ("bench", "chat"):
            # Single-miner-slot pods (SWE bench / public king chat): one slot
            # across the rented GPUs. Teacher/king slots exist for status
            # shape but are never launched.
            section = "bench_serving" if self.role == "bench" else "chat"
            bs = self.cfg.get(section) or {}
            gpus = str(bs.get("gpus", "0,1"))
            tp = int(bs.get("tp", ms.get("tp", 2)))
            port = int(bs.get("port", ms.get("challenger_port", 8002)))
            self.teacher_slot = Slot("teacher", 8000, "0", 1)
            self.king_slot = Slot("king", 8001, "0", 1)
            self.chall_slot = Slot("challenger", port, gpus, tp)
            return
        self.teacher_slot = Slot("teacher", int(t["port"]), t["gpus"], int(t["tp"]))
        # Local teacher replica is unused under remote cutover (base_url).
        if t.get("replica_gpus") and not str(t.get("base_url") or "").strip():
            self.teacher2_slot = Slot("teacher2", int(t.get("replica_port", 8003)),
                                      str(t["replica_gpus"]), int(t["tp"]))
        self.king_slot = Slot("king", int(ms["king_port"]), ms["king_gpus"],
                              int(ms["tp"]))
        if ms.get("king_replica_gpus"):
            self.king2_slot = Slot("king2", int(ms.get("king_replica_port", 8003)),
                                   str(ms["king_replica_gpus"]), int(ms["tp"]))
        self.chall_slot = Slot("challenger", int(ms["challenger_port"]),
                               ms["challenger_gpus"], int(ms["tp"]))
        if ms.get("challenger_replica_gpus"):
            self.chal2_slot = Slot(
                "challenger2", int(ms.get("challenger_replica_port", 8004)),
                str(ms["challenger_replica_gpus"]), int(ms["tp"]))

    def _slots(self) -> list[Slot]:
        slots = [self.teacher_slot, self.king_slot, self.chall_slot]
        if self.teacher2_slot is not None:
            slots.append(self.teacher2_slot)
        if self.king2_slot is not None:
            slots.append(self.king2_slot)
        if self.chal2_slot is not None:
            slots.append(self.chal2_slot)
        return slots

    # -- process control -------------------------------------------------------
    def _vllm_cmd(self, slot: Slot, repo: str, revision: str | None) -> list[str]:
        ms = self.cfg["miner_serving"]
        # Teacher-forcing sends echo+logprobs requests; vLLM materializes
        # a fp32 log_softmax over (prefill_chunk x vocab) per request, so the
        # duel pod keeps prefill chunks small or the engine OOMs under
        # concurrency. The bench pod serves plain generation only, so it uses
        # bigger chunks ([bench_serving].max_num_batched_tokens) for fast
        # long-context agent prefills.
        batched_tokens = int(ms["max_num_batched_tokens"])
        gpu_util = ms["gpu_memory_utilization"]
        max_len = int(ms["max_model_len"])
        if slot.label.startswith("teacher"):
            # Teachers absorb nearly all echo traffic, so they get bigger
            # chunks — but the fp32 log_softmax spike lives OUTSIDE vLLM's
            # budgeted pool, and at 0.80 util a 12288-chunk spike OOM'd
            # teacher2 mid-duel (2026-08-14). Teacher KV runs ~2-5% full, so
            # a lower util buys the spike headroom for free.
            batched_tokens = int(ms.get("teacher_max_num_batched_tokens",
                                        batched_tokens))
            gpu_util = ms.get("teacher_gpu_memory_utilization", gpu_util)
        if self.role == "bench":
            bs = self.cfg.get("bench_serving") or {}
            batched_tokens = int(bs.get("max_num_batched_tokens", 16384))
            # Official SWE-rebench protocol is 128k context; the duel pod's
            # 65k is sized for corpus prefixes, not for 300-step agent runs.
            max_len = int(bs.get("max_model_len", max_len))
        if self.role == "chat":
            # Chat serves short interactive contexts, not 64k corpus prefixes:
            # a smaller KV pool leaves the 2-GPU pod headroom, and no echo
            # traffic means the higher util + big chunks are safe.
            cs = self.cfg.get("chat") or {}
            batched_tokens = int(cs.get("max_num_batched_tokens", 16384))
            gpu_util = cs.get("gpu_memory_utilization", gpu_util)
            max_len = int(cs.get("max_model_len", max_len))
        # r2 refs are served from their verified local snapshot; the model is
        # still *named* by the ref so client requests (model=<ref>) match.
        cmd = [
            "vllm", "serve", r2store.model_path(repo, revision),
            "--port", str(slot.port),
            "--tensor-parallel-size", str(slot.tp),
            "--max-model-len", str(max_len),
            "--gpu-memory-utilization", str(gpu_util),
            "--max-num-batched-tokens", str(batched_tokens),
            # Avoid FlashInfer JIT (needs a coherent system CUDA toolkit; the
            # pip nvidia-cu13 wheel headers often trip B300/Blackwell builds).
            "--attention-backend", "FLASH_ATTN",
            "--attention-config.use_trtllm_attention", "0",
            "--compilation-config.pass_config.fuse_allreduce_rms", "false",
            # Qwen3 MoE on Blackwell: moe_backend=auto picks flashinfer_trtllm
            # and JIT-compiles fused_moe_trtllm_sm100; pip cu13 ptxas is 13.0
            # (PTX 9.0) but the generated PTX is 9.3 → king exits 1. Triton
            # skips that path for unquantized MoE.
            "--moe-backend", "triton",
        ]
        # Qwen GDN linear attention: gdn_prefill_backend=auto picks
        # FlashInfer on Hopper (SM90) and JIT-compiles gdn_prefill_sm90 at
        # the first request; pip cu13 nvcc vs mismatched CUDA headers makes
        # the ninja build fail and the engine dies mid-serve (passes the
        # /v1/models ready check, then drops every completion → ConnectError).
        # Triton/FLA needs no JIT. Applied to all roles since the duel pod
        # moved to H200 (2026-08-13). Since the 2026-08-27 teacher swap the
        # ranked logprobs themselves come from a GDN model (Qwen3.8-27B), so
        # this flag now picks the teacher's echo prefill kernel: fine —
        # temperature-0 echo scoring only needs within-duel consistency, and
        # every echo on a pod goes through the same engine + kernel. The
        # bootstrap also installs prebuilt flashinfer wheels (vLLM >= 0.28
        # hard-imports flashinfer for GDN models even with this override).
        cmd += ["--additional-config", '{"gdn_prefill_backend": "triton"}']
        # HF cache lives on FUSE.GOCRYPTFS. vLLM only auto-prefetches NFS/
        # Lustre, so shard reads were ~30s each (~8 min serial H2D) while
        # 360 GB RAM sat empty (chal-00074 king, 2026-08-30). Force
        # prefetch: warm the snapshot in the page cache, then load to GPU.
        # Process-private heap does not hold the 65 GB (read-and-discard).
        # Four copies share two repos ≈ 135 GB unique. Load-only, no score
        # effect. Not used on remote teacher.
        if not slot.label.startswith("teacher"):
            cmd += ["--safetensors-load-strategy", "prefetch"]
        if self.role == "chat":
            # The chat pod's wire plane serves agent clients (arbos, Cursor)
            # that drive tool loops over /v1/chat/completions. Qwen-family
            # kings emit hermes-style <tool_call> blocks. Never set on duel/
            # bench pods — scoring must see raw completions.
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
        served_names = [repo] if r2store.is_r2(repo) else []
        if not r2store.is_r2(repo) and revision:
            cmd += ["--revision", revision]
        if self._warm_swap_slot(slot):
            # Alias for successive checkpoints (the dev-mode RPC surface this
            # enables, VLLM_SERVER_DEV_MODE in _launch, is reachable only from
            # inside the pod: slot ports are not among the pod's mapped ports).
            served_names.append(CHALLENGER_ALIAS)
            cmd += ["--worker-extension-cls", WORKER_EXTENSION]
        if served_names:
            cmd += ["--served-model-name", *served_names]
        return cmd

    def _warm_swap_slot(self, slot: Slot) -> bool:
        return (WARM_SWAP and self.role == "duel"
                and slot in (self.chall_slot, self.chal2_slot))

    def _ensure_snapshot(self, repo: str, revision: str | None,
                         cancel: threading.Event | None = None) -> None:
        """r2 refs must be on disk and verified before vLLM starts (HF repos
        download inside vLLM). Raises IntegrityError when the bucket content
        is not the pinned digest; r2store.FetchCancelled when `cancel` fires
        mid-download (the duel was superseded); other exceptions are
        transport faults. Runs OUTSIDE _lock: a multi-GB download must not
        block /prefetch."""
        if not r2store.is_r2(repo) or not revision:
            return
        if r2store.snapshot_ready(repo, revision):
            return
        log.info("materializing %s@%s from R2", repo, revision[:12])
        # Registered so a concurrent prune (a /prefetch for the next queue
        # item) cannot delete the half-written snapshot: like an in-flight
        # prefetch, an in-flight inline fetch is part of the keep set.
        with self._lock:
            self._materializing.add(repo)
        try:
            r2store.fetch_snapshot(repo, revision, cancel=cancel)
        finally:
            with self._lock:
                self._materializing.discard(repo)

    def _sweep_slot_gpus(self, slot: Slot) -> None:
        """SIGKILL compute processes still resident on the slot's GPUs.

        Belt to _kill's pgid suspenders: anything left on these GPUs at launch
        time (escaped worker, resource tracker, a vllm from before a server
        restart) would make the new engine fail init on insufficient free
        memory. Processes belonging to other live slots are never touched."""
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,pci.bus_id",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
            bus_to_idx = {}
            for line in q.stdout.strip().splitlines():
                idx, bus = [x.strip() for x in line.split(",", 1)]
                bus_to_idx[bus.lower()] = idx
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
            want = {g.strip() for g in slot.gpus.split(",") if g.strip()}
            keep_pgids = {s.pgid for s in self._slots()
                          if s is not slot and s.pgid is not None}
            for line in apps.stdout.strip().splitlines():
                if "," not in line:
                    continue
                pid_s, bus = [x.strip() for x in line.split(",", 1)]
                if bus_to_idx.get(bus.lower()) not in want:
                    continue
                pid = int(pid_s)
                try:
                    if os.getpgid(pid) in keep_pgids:
                        continue
                    os.kill(pid, signal.SIGKILL)
                    log.warning("swept orphan GPU process %d off %s gpus (%s)",
                                pid, slot.label, slot.gpus)
                except ProcessLookupError:
                    continue
        except Exception:
            log.warning("gpu orphan sweep failed for %s", slot.label,
                        exc_info=True)

    def _kill_port_listener(self, slot: Slot) -> None:
        """SIGKILL whatever still listens on the slot's port.

        An evalsrv restart orphans the previous vLLM processes: the GPU
        sweep reaps the CUDA workers, but the CPU-side API-server parent
        keeps the port bound and keeps answering /v1/models. A fresh launch
        then cannot bind, _wait_ready's GET hits the zombie and declares
        "ready in 0s", and every completion ConnectErrors (2026-08-15,
        king + challenger, twice in one day). Slot ports are engine-owned,
        one engine per port, so any listener found here is stale by
        definition.
        """
        try:
            out = subprocess.run(
                ["ss", "-tlnpH", f"sport = :{slot.port}"],
                capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            return
        for pid_s in set(re.findall(r"pid=(\d+)", out or "")):
            pid = int(pid_s)
            if pid == os.getpid():
                continue
            log.warning("killing stale listener pid %d on %s port %d",
                        pid, slot.label, slot.port)
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def _launch(self, slot: Slot, repo: str, revision: str | None) -> None:
        with self._lock:
            self._kill(slot)
            self._sweep_slot_gpus(slot)
            self._kill_port_listener(slot)
            slot.load_error = ""
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            _purge_broken_flashinfer_moe_cache()
            # Per-slot vLLM cache root. Both copies of a fresh challenger
            # torch.compile the same kernels at the same time, and a shared
            # ~/.cache/vllm/torch_compile_cache lets one copy import a kernel
            # file the other is still writing ("@jit functions should be
            # defined in a Python file", 9 challenger-copy load failures in
            # 10h on 2026-09-01). Isolating the roots removes the race; the
            # only cost is that each copy compiles for itself.
            cache_root = Path(os.environ.get("VLLM_CACHE_ROOT",
                                             "/root/.cache/vllm"))
            slot_cache = str(cache_root.with_name(
                f"{cache_root.name}_{slot.label}"))
            env = dict(os.environ, **_vllm_env(), CUDA_VISIBLE_DEVICES=slot.gpus,
                       VLLM_CACHE_ROOT=slot_cache)
            warm = self._warm_swap_slot(slot)
            if warm:
                # /collective_rpc + /reset_prefix_cache for weight swaps.
                env["VLLM_SERVER_DEV_MODE"] = "1"
            logf = open(LOG_DIR / f"vllm_{slot.label}.log", "a")
            slot.log_offset = logf.tell()
            log.info("launching %s: %s (gpus=%s)", slot.label, repo, slot.gpus)
            slot.proc = subprocess.Popen(
                self._vllm_cmd(slot, repo, revision), env=env,
                stdout=logf, stderr=logf,
                stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                slot.pgid = os.getpgid(slot.proc.pid)
            except ProcessLookupError:
                slot.pgid = None
            slot.served = Served(name=slot.label, repo=repo, revision=revision,
                                 port=slot.port,
                                 model_name=CHALLENGER_ALIAS if warm else None)
            slot.ready = False

    # -- warm swap -------------------------------------------------------------
    def _swap_compatible(self, slot: Slot, repo: str, revision: str | None
                         ) -> str | None:
        """None when `repo@revision` can be loaded in place into the engine
        `slot` is running; otherwise the reason a full relaunch is needed."""
        if not self._warm_swap_slot(slot):
            return "warm swap disabled for slot"
        if not (slot.served and slot.ready and self._proc_alive(slot)):
            return "no warm engine"
        if not slot.served.model_name:
            return "engine not serving the alias"
        if not (r2store.is_r2(slot.served.repo) and r2store.is_r2(repo)
                and revision):
            return "not an r2 -> r2 swap"
        if (slot.served.repo, slot.served.revision) == (repo, revision):
            return None  # same checkpoint: nothing to do
        old = Path(r2store.model_path(slot.served.repo, slot.served.revision))
        new = Path(r2store.model_path(repo, revision))
        if not old.is_dir() or not new.is_dir():
            return "snapshot dir missing"
        try:
            reason = _generation_config_mismatch(old, new)
            if reason:
                return reason
            reason = _tokenizer_mismatch(old, new)
            if reason:
                return reason
            ca = json.loads((old / "config.json").read_text())
            cb = json.loads((new / "config.json").read_text())
            for k in SWAP_CONFIG_IGNORE_KEYS:
                ca.pop(k, None)
                cb.pop(k, None)
            if ca != cb:
                diff = sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
                return f"config.json differs: {diff[:6]}"
            ia, ib = old / "model.safetensors.index.json", new / "model.safetensors.index.json"
            if not (ia.is_file() and ib.is_file()):
                return "no safetensors index"
            ta = set(json.loads(ia.read_text()).get("weight_map", {}))
            tb = set(json.loads(ib.read_text()).get("weight_map", {}))
            if ta != tb:
                return (f"tensor set differs (+{len(tb - ta)} -{len(ta - tb)})")
        except (OSError, ValueError) as e:
            return f"compat check failed: {e}"
        return None

    def _swap_weights(self, slot: Slot, repo: str, revision: str) -> bool:
        """Load `repo@revision` into the running engine in place. On any
        failure the caller falls back to a full relaunch."""
        assert slot.served is not None
        path = r2store.model_path(repo, revision)
        base = f"http://localhost:{slot.port}"
        t0 = time.time()
        old = slot.served
        slot.ready = False
        try:
            r = httpx.post(f"{base}/collective_rpc",
                           json={"method": "affine_direct_load",
                                 "kwargs": {"weights_path": path},
                                 "timeout": 1500},
                           timeout=1800)
            r.raise_for_status()
            # One report per TP rank; every rank must show the pinned
            # architecture's exact shape of result.
            for rank, raw in enumerate(r.json().get("results") or []):
                rep = json.loads(raw) if isinstance(raw, str) else raw
                missing = rep.get("missing") or []
                if (rep.get("loaded") != SWAP_EXPECTED_LOADED
                        or rep.get("n_missing") != len(SWAP_UNREPORTED_SUFFIXES) * 40
                        or any(not m.endswith(SWAP_UNREPORTED_SUFFIXES) for m in missing)):
                    raise RuntimeError(
                        f"rank{rank} loader report off-shape: loaded="
                        f"{rep.get('loaded')} n_missing={rep.get('n_missing')} "
                        f"first={missing[:2]}")
            # KV blocks were computed with the old weights.
            httpx.post(f"{base}/reset_prefix_cache", timeout=120).raise_for_status()
            # The engine must still answer with the new weights in place.
            r = httpx.post(f"{base}/v1/completions",
                           json={"model": CHALLENGER_ALIAS, "prompt": "1, 2, 3,",
                                 "max_tokens": 4, "temperature": 0},
                           timeout=300)
            r.raise_for_status()
            text = r.json()["choices"][0]["text"]
            if "4" not in text:
                # Any Qwen-family fine-tune continues the count; garbage here
                # means the in-place load left the model inconsistent.
                raise RuntimeError(f"post-swap probe returned {text!r}")
        except Exception as e:  # noqa: BLE001 — any failure -> relaunch
            log.warning("%s warm swap to %s@%s failed after %.0fs (%s); "
                        "relaunching", slot.label, repo, revision[:12],
                        time.time() - t0, e)
            return False
        slot.served = Served(name=slot.label, repo=repo, revision=revision,
                             port=slot.port, model_name=CHALLENGER_ALIAS)
        slot.ready = True
        slot.load_error = ""
        log.info("%s warm-swapped %s@%s -> %s@%s in %.0fs (probe %r)",
                 slot.label, old.repo[-20:], (old.revision or "")[:12],
                 repo[-20:], revision[:12], time.time() - t0, text)
        return True

    def _launch_or_swap(self, slot: Slot, repo: str, revision: str | None
                        ) -> bool:
        """Challenger slot: swap weights into the warm engine when safe, else
        relaunch. Returns True when the slot is already ready (swapped or
        same checkpoint) and no _wait_ready is needed."""
        reason = self._swap_compatible(slot, repo, revision)
        if reason is None:
            if slot.served and (slot.served.repo, slot.served.revision) == (repo, revision):
                log.info("%s already serving %s@%s", slot.label, repo,
                         (revision or "")[:12])
                return True
            if self._swap_weights(slot, repo, revision or ""):
                return True
        elif slot.served is not None and self._warm_swap_slot(slot):
            log.info("%s: full relaunch for %s (%s)", slot.label, repo, reason)
        self._launch(slot, repo, revision)
        return False

    def _kill(self, slot: Slot) -> None:
        if slot.proc and slot.proc.poll() is None:
            log.info("killing %s (pid %s)", slot.label, slot.proc.pid)
            try:
                os.killpg(os.getpgid(slot.proc.pid), signal.SIGTERM)
                slot.proc.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(os.getpgid(slot.proc.pid), signal.SIGKILL)
                except Exception:
                    log.warning("could not kill %s process group", slot.label,
                                exc_info=True)
        elif slot.pgid is not None:
            # Leader already exited (crash / EngineDeadError). Its TP workers
            # can outlive it in the same process group, holding all the slot's
            # VRAM — the next load then dies at init with "Free memory ... is
            # less than desired GPU memory utilization" until they are reaped
            # (observed as a streak of 6 consecutive bench load failures).
            try:
                os.killpg(slot.pgid, signal.SIGKILL)
                log.warning("%s leader dead; SIGKILLed orphaned process group %d",
                            slot.label, slot.pgid)
            except ProcessLookupError:
                pass  # group fully gone — nothing leaked
            except Exception:
                log.warning("could not kill %s orphan group", slot.label,
                            exc_info=True)
        slot.pgid = None
        slot.proc = None
        slot.served = None
        slot.ready = False

    def _fail_load(self, slot: Slot, reason: str) -> bool:
        tail = _vllm_log_tail(slot.label)
        detail = f"{reason}" + (f" | {tail}" if tail else "")
        slot.load_error = detail[:1500]
        log.error("%s load failed: %s", slot.label, slot.load_error[:500])
        return False

    def _probe_http_ready(self, slot: Slot) -> bool:
        """One /v1/models hit. Sets slot.ready on success. No wait."""
        if slot.proc is None or slot.proc.poll() is not None:
            return False
        try:
            httpx.get(f"http://localhost:{slot.port}/v1/models", timeout=3)
            slot.ready = True
            slot.load_error = ""
            return True
        except httpx.HTTPError:
            return False

    def _launch_dead(self, slot: Slot) -> str:
        """Non-empty when this launch can no longer become ready: the leader
        exited, or its log shows a fatal engine/worker marker while the
        leader idles. The lingering leader is killed so the GPUs free up."""
        if slot.proc is not None and slot.proc.poll() is not None:
            return f"vllm process exited with {slot.proc.returncode}"
        fatal = _vllm_log_fatal(slot.label, slot.log_offset)
        if fatal:
            log.error("%s: fatal vllm marker while leader alive; killing: %s",
                      slot.label, fatal[:300])
            self._kill(slot)
            return f"vllm engine died during startup: {fatal[:400]}"
        return ""

    def _wait_ready(self, slot: Slot, timeout_s: int = 3600,
                    *, required: bool = True) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            dead = self._launch_dead(slot)
            if dead:
                return self._fail_load(slot, dead)
            if self._probe_http_ready(slot):
                log.info("%s ready in %.0fs", slot.label, time.time() - t0)
                return True
            time.sleep(10)
        if required:
            return self._fail_load(slot, f"not ready after {timeout_s}s")
        log.warning("%s not ready after %.0fs (optional replica)",
                    slot.label, timeout_s)
        return False

    def _wait_any_ready(self, slots: list[Slot | None],
                        timeout_s: int = 3600) -> bool:
        """True when any listed copy answers /v1/models.

        A hung primary (CUDA-graph capture deadlock, observed live on
        chal-00075: 0/83 FULL graphs, replica already serving) must not
        block the duel. Score-invariant: the ranked quantity is teacher-side.
        """
        live = [s for s in slots if s is not None]
        if not live:
            return False
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            for slot in live:
                if slot.ready and self._alive(slot):
                    return True
                if self._probe_http_ready(slot):
                    log.info("%s ready in %.0fs (first copy)",
                             slot.label, time.time() - t0)
                    return True
            dead = {s.label: self._launch_dead(s) for s in live}
            if all(dead.values()):
                for slot in live:
                    self._fail_load(slot, dead[slot.label])
                return False
            time.sleep(10)
        for slot in live:
            if not slot.ready:
                self._fail_load(slot, f"not ready after {timeout_s}s")
        return False

    def _wait_replica(self, slot: Slot | None, kind: str) -> None:
        """Include the second copy if it comes up quickly; never block a duel
        on it. A still-running engine is left warming (do not kill)."""
        if slot is None:
            return
        if self._wait_ready(slot, timeout_s=int(REPLICA_READY_S),
                            required=False):
            return
        if slot.proc is not None and slot.proc.poll() is None:
            log.warning("%s replica still warming after %.0fs; "
                        "scoring without it", slot.label, REPLICA_READY_S)
            return
        log.warning("%s replica failed to warm; running single-%s",
                    kind, kind)
        self._kill(slot)

    # -- public API --------------------------------------------------------------
    def _teacher_base_url(self) -> str:
        return str(self.cfg.get("teacher", {}).get("base_url") or "").rstrip("/")

    def _probe_remote_teacher(self, base_url: str, repo: str) -> bool:
        """Health-check a remote OpenAI-compatible teacher; no local vLLM."""
        models_url = f"{base_url}/models"
        try:
            r = httpx.get(models_url, timeout=5.0)
            r.raise_for_status()
            data = r.json().get("data") or []
            ids = {m.get("id") for m in data if isinstance(m, dict)}
            if ids and repo not in ids:
                log.warning("remote teacher at %s serves %s (expected %s)",
                            base_url, sorted(ids)[:5], repo)
            self.teacher_slot.served = Served(
                name="teacher", repo=repo, revision=None, port=0,
                base_url=base_url)
            self.teacher_slot.ready = True
            self.teacher_slot.load_error = ""
            # Remote mode: never keep a stale local replica in the pool.
            if self.teacher2_slot is not None:
                self._kill(self.teacher2_slot)
            log.info("remote teacher ready at %s (repo=%s)", base_url, repo)
            return True
        except Exception as e:
            self.teacher_slot.ready = False
            self.teacher_slot.served = None
            self.teacher_slot.load_error = f"remote teacher probe failed: {e}"[:1500]
            log.error("remote teacher probe failed (%s): %s", models_url, e)
            return False

    def ensure_teacher(self) -> bool:
        """Primary teacher is required; the replica is best-effort. Both are
        launched before either wait so a cold rewarm pays one warmup, not two
        (launch is just a Popen; readiness is what takes minutes).

        When [teacher].base_url is set, skip local launch and probe the remote
        OpenAI-compatible endpoint instead (dedicated teacher box)."""
        t = self.cfg["teacher"]
        base_url = self._teacher_base_url()
        if base_url:
            # Re-probe every ensure: a dead remote must fail the duel closed.
            return self._probe_remote_teacher(base_url, str(t["repo"]))
        # Serialized: a second caller waits out the first warmup and then
        # sees ready teachers instead of relaunching over them.
        with self._teacher_lock:
            primary_ok = self.teacher_slot.ready and self._alive(self.teacher_slot)
            if not primary_ok:
                self._launch(self.teacher_slot, t["repo"], None)
            replica_launched = False
            if self.teacher2_slot is not None and not (
                    self.teacher2_slot.ready and self._alive(self.teacher2_slot)):
                self._launch(self.teacher2_slot, t["repo"], None)
                replica_launched = True
            if not primary_ok and not self._wait_ready(self.teacher_slot):
                return False
            if replica_launched and not self._wait_ready(self.teacher2_slot,
                                                         timeout_s=1200):
                # Non-fatal: reap the half-dead process so its GPUs stay clean
                # and the duel routes everything to the primary.
                log.warning("teacher replica failed to warm; running single-teacher")
                self._kill(self.teacher2_slot)
            return True

    def _probe_extra_teacher(self, base_url: str) -> bool:
        """Stateless liveness probe of one additive remote teacher endpoint.

        Verifies the endpoint actually serves the contract teacher repo, not
        just that it answers: during the 2026-08-27 teacher swap the swarm
        briefly kept serving the old model, and a 200 from /models alone
        would have admitted GLM echoes into a Qwen-scored duel pool."""
        repo = str(self.cfg["teacher"]["repo"])
        try:
            r = httpx.get(f"{base_url}/models", timeout=5.0)
            r.raise_for_status()
            ids = [m.get("id") for m in r.json().get("data", [])]
            if repo not in ids:
                log.warning("extra teacher %s serves %s, not %s; skipping",
                            base_url, ids[:3], repo)
                return False
            return True
        except Exception as e:
            log.warning("extra teacher %s dark, skipping: %s", base_url, e)
            return False

    def teacher_serveds(self) -> list[Served]:
        """Teacher endpoints currently servable, primary first. The replica is
        re-probed here (cheap, once per duel): a replica that died since
        warmup must not be handed to the duel's round-robin pool."""
        base_url = self._teacher_base_url()
        if base_url:
            if self.teacher_slot.served and self.teacher_slot.ready:
                # Cheap liveness re-check so a mid-queue remote death does not
                # keep feeding a dead URL into the duel pool.
                if self._probe_remote_teacher(base_url, str(self.cfg["teacher"]["repo"])):
                    return [self.teacher_slot.served]
            return []
        out: list[Served] = []
        if self.teacher_slot.served and self.teacher_slot.ready:
            out.append(self.teacher_slot.served)
        s2 = self.teacher2_slot
        if (s2 is not None and s2.served and s2.ready and self._alive(s2)):
            out.append(s2.served)
        # Additive swarm capacity (2026-08-15 operator directive): extra
        # OpenAI-compatible teacher endpoints join the duel pool when alive.
        # Strictly fail-open — a dark extra is skipped and the local slots
        # above remain the mandatory floor (ensure_teacher is unchanged).
        # Same repo at temperature-0 echo scoring, so which endpoint answers
        # a call is score-invariant; duplicates in the list are allowed and
        # weight the pool's uniform turn hash toward the bigger backend.
        repo = str(self.cfg["teacher"]["repo"])
        extras = [str(u).rstrip("/")
                  for u in (self.cfg["teacher"].get("extra_urls") or [])]
        alive: dict[str, bool] = {}
        for u in extras:
            if u not in alive:
                alive[u] = self._probe_extra_teacher(u)
            if alive[u]:
                out.append(Served(name="teacher", repo=repo, revision=None,
                                  port=0, base_url=u))
        return out

    def _replica_matches(self, slot: Slot, repo: str, revision: str) -> bool:
        s = slot.served
        return bool(s and s.repo == repo and s.revision == revision
                    and slot.ready and self._alive(slot))

    def _heal_replica(self, slot: Slot | None, repo: str, revision: str,
                      kind: str) -> None:
        """Best-effort second copy: launch if down, do not block the duel."""
        if slot is None or self._replica_matches(slot, repo, revision):
            return
        loading = (slot.served is not None
                   and slot.served.repo == repo
                   and slot.served.revision == revision
                   and slot.proc is not None
                   and slot.proc.poll() is None)
        if not loading:
            self._launch(slot, repo, revision)
        self._wait_replica(slot, kind)

    def _serveds_of(self, primary: Slot, replica: Slot | None) -> list[Served]:
        out: list[Served] = []
        if primary.served and primary.ready and self._alive(primary):
            out.append(primary.served)
        if (replica is not None and replica.served and replica.ready
                and self._alive(replica)):
            out.append(replica.served)
        return out

    def king_serveds(self) -> list[Served]:
        return self._serveds_of(self.king_slot, self.king2_slot)

    def challenger_serveds(self) -> list[Served]:
        return self._serveds_of(self.chall_slot, self.chal2_slot)

    def ensure_king(self, repo: str, revision: str) -> bool:
        if self._replica_matches(self.king_slot, repo, revision):
            self._heal_replica(self.king2_slot, repo, revision, "king")
            return True
        if (self.king2_slot is not None
                and self._replica_matches(self.king2_slot, repo, revision)):
            self._heal_replica(self.king_slot, repo, revision, "king")
            return True
        try:
            self._ensure_snapshot(repo, revision)
        except Exception as e:
            # A king we cannot materialize is a pod/bucket fault for the
            # caller (KING_LAUNCH), never a dethrone signal — the validator
            # proves a king gone with its own probe.
            self._fail_load(self.king_slot, f"king snapshot: {e}")
            return False
        self._launch(self.king_slot, repo, revision)
        if self.king2_slot is not None:
            self._launch(self.king2_slot, repo, revision)
        if not self._wait_any_ready([self.king_slot, self.king2_slot]):
            return False
        for slot in (self.king_slot, self.king2_slot):
            if slot is not None and not slot.ready:
                self._wait_replica(slot, "king")
        return True

    def _settle_prefetch(self, incoming_repo: str | None,
                         incoming_revision: str | None = None) -> None:
        """Bring the prefetch slot to rest before a challenger download.
        A live prefetch of the incoming repo is joined (finishing beats
        racing vLLM's downloader for the same blob locks; the worker's stall
        watchdog bounds the wait in practice, the timeout is a backstop). A
        live prefetch of any OTHER repo is the next queue item (or stale).
        Cancel it only when incoming still needs the NIC — otherwise leave
        it running so the next download overlaps this duel’s GPU load.
        Runs OUTSIDE the lock: joining under it would block /prefetch,
        challenger_fits and every launch for the duration."""
        with self._lock:
            pf = self._prefetch
        if pf is None or not pf[1].is_alive():
            return
        if pf[0] == incoming_repo:
            log.info("waiting for in-flight prefetch of %s", pf[0])
            pf[1].join(timeout=1800)
            return
        snap_ready = False
        if incoming_repo and incoming_revision:
            snap_ready = r2store.snapshot_ready(incoming_repo, incoming_revision)
        if snap_ready:
            log.info("leaving prefetch of %s running; %s@%s already cached",
                     pf[0], incoming_repo, incoming_revision[:12])
            return
        log.info("cancelling stale prefetch of %s (incoming: %s)",
                 pf[0], incoming_repo)
        pf[2].set()
        pf[1].join(timeout=60)

    def _same_target(self, slot: Slot | None, repo: str,
                     revision: str) -> bool:
        return bool(slot is not None and slot.served
                    and slot.served.repo == repo
                    and slot.served.revision == revision)

    def _proc_alive(self, slot: Slot | None) -> bool:
        return bool(slot is not None and slot.proc is not None
                    and slot.proc.poll() is None)

    def prepare_miners(self, king_repo: str, king_revision: str,
                       chall_repo: str, chall_revision: str,
                       cancel: threading.Event | None = None) -> bool:
        """Launch king + challenger together; wait first ready copy per role.

        They sit on disjoint GPUs. After an evalsrv bounce both are cold —
        the old ensure_king-then-load_challenger path paid two vLLM starts
        back-to-back (~11 min each). One wait covers both. King already
        warm: only the challenger pair launches. Either copy of a role is
        enough (a hung primary must not block a live replica). Score-invariant.
        """
        king_ready = self._replica_matches(
            self.king_slot, king_repo, king_revision)
        king2_ready = (
            self.king2_slot is None
            or self._replica_matches(self.king2_slot, king_repo, king_revision))
        self._settle_prefetch(chall_repo, chall_revision)
        with self._lock:
            if self._prefetch is not None and self._prefetch[0] == chall_repo:
                self._prefetch = None
            self._prune_challenger_cache(
                keep_repo=chall_repo, keep_revision=chall_revision,
                extra_keep={king_repo})
        # R2 checkpoints land on disk (verified) before any vLLM start. King
        # first: its failure is a pod fault (return False with no king served
        # → KING_LAUNCH). The challenger's IntegrityError propagates: that is
        # the miner's checkpoint not matching its commitment.
        if not king_ready:
            try:
                self._ensure_snapshot(king_repo, king_revision, cancel)
            except r2store.FetchCancelled:
                raise
            except Exception as e:
                self._fail_load(self.king_slot, f"king snapshot: {e}")
                return False
        self._ensure_snapshot(chall_repo, chall_revision, cancel)
        with self._lock:
            # Re-pin the keep set: the prefetch prune above ran before the
            # r2 snapshots existed (a fresh download must not be pruned by a
            # /prefetch racing in between).
            self._prune_challenger_cache(
                keep_repo=chall_repo, keep_revision=chall_revision,
                extra_keep={king_repo})
            if not king_ready and not (
                    self._same_target(self.king_slot, king_repo, king_revision)
                    and self._proc_alive(self.king_slot)):
                self._launch(self.king_slot, king_repo, king_revision)
            if (self.king2_slot is not None and not king2_ready and not (
                    self._same_target(self.king2_slot, king_repo, king_revision)
                    and self._proc_alive(self.king2_slot))):
                self._launch(self.king2_slot, king_repo, king_revision)
            # Consumed: the served slot protects the snapshot from here on.
            self._prefetched_ready.pop(chall_repo, None)
        # Challenger copies: weight swap into the warm engines (both copies
        # in parallel — each reads the 72 GB itself) or full relaunch. Runs
        # outside _lock: a swap blocks for the weight read.
        self._prepare_challengers(chall_repo, chall_revision)

        results = {"king": False, "chall": False}

        def wait_king() -> None:
            if self._replica_matches(self.king_slot, king_repo, king_revision):
                results["king"] = True
                return
            if (self.king2_slot is not None and self._replica_matches(
                    self.king2_slot, king_repo, king_revision)):
                results["king"] = True
                return
            results["king"] = self._wait_any_ready(
                [self.king_slot, self.king2_slot])

        def wait_chall() -> None:
            results["chall"] = self._wait_any_ready(
                [self.chall_slot, self.chal2_slot])

        t_k = threading.Thread(target=wait_king, name="wait-king", daemon=True)
        t_c = threading.Thread(target=wait_chall, name="wait-chall", daemon=True)
        t_k.start()
        t_c.start()
        t_k.join()
        t_c.join()
        if not results["king"] or not results["chall"]:
            return False

        extras: list[threading.Thread] = []
        for slot, kind in (
                (self.king_slot, "king"),
                (self.king2_slot, "king"),
                (self.chall_slot, "challenger"),
                (self.chal2_slot, "challenger")):
            if slot is not None and not slot.ready:
                extras.append(threading.Thread(
                    target=self._wait_replica, args=(slot, kind),
                    name=f"wait-{slot.label}", daemon=True))
        for t in extras:
            t.start()
        for t in extras:
            t.join()
        return True

    def _prepare_challengers(self, repo: str, revision: str | None) -> None:
        """Bring both challenger copies onto repo@revision: warm swap where
        compatible, full relaunch otherwise; copies in parallel."""
        slots = [s for s in (self.chall_slot, self.chal2_slot) if s is not None]
        threads = [threading.Thread(target=self._launch_or_swap,
                                    args=(s, repo, revision),
                                    name=f"prep-{s.label}", daemon=True)
                   for s in slots]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def load_challenger(self, repo: str, revision: str,
                        cancel: threading.Event | None = None) -> bool:
        self._settle_prefetch(repo, revision)
        with self._lock:
            if self._prefetch is not None and self._prefetch[0] == repo:
                # Consumed: keep_repo protects the snapshot from here on and
                # the slot frees up for the next prefetch target.
                self._prefetch = None
            self._prune_challenger_cache(keep_repo=repo, keep_revision=revision)
        self._ensure_snapshot(repo, revision, cancel)  # IntegrityError propagates
        with self._lock:
            self._prefetched_ready.pop(repo, None)
        self._prepare_challengers(repo, revision)
        if not self._wait_any_ready([self.chall_slot, self.chal2_slot]):
            return False
        for slot in (self.chall_slot, self.chal2_slot):
            if slot is not None and not slot.ready:
                self._wait_replica(slot, "challenger")
        return True

    def challenger_fits(self, required_bytes: int, repo: str | None = None,
                        revision: str | None = None,
                        extra_keep: set[str] | None = None) -> tuple[bool, float]:
        """Prune non-essential caches, then report whether a challenger's
        servable weights (`required_bytes`) fit the serviceable disk with
        CHALLENGER_DISK_HEADROOM_GB to spare. Returns (fits, free_gb).

        Bytes already cached for this exact repo@revision (a retry of the
        previous challenger) count as credit against `required_bytes`: they
        both survive the prune and reduce what the download still has to
        fetch, so without the credit a kept snapshot would false-fail the
        check against the very space it occupies.

        Fits is True when the size is unknown (<=0) or the disk is unreadable —
        the real download + `diagnose_load_failure` remain the backstop. Sizing
        from the servable weight set (not the whole repo) means we only fail
        fast when even the weights cannot land, never on legal non-weight files
        vLLM does not download."""
        # A stale prefetch still downloading would both hold its bytes out of
        # the prune and keep growing after we measure — cancel it first so
        # the fit answer stays true through the download that follows.
        self._settle_prefetch(repo, revision)
        with self._lock:
            self._prune_challenger_cache(keep_repo=repo, keep_revision=revision,
                                         extra_keep=extra_keep)
            free_gb = self.free_disk_gb()
            if required_bytes <= 0 or free_gb < 0:
                return True, free_gb
            cached = self._cached_repo_bytes(repo) if repo else 0
            need_gb = max(0, required_bytes - cached) / 1e9
            return need_gb + CHALLENGER_DISK_HEADROOM_GB <= free_gb, free_gb

    def free_disk_gb(self) -> float:
        try:
            return shutil.disk_usage(HF_HOME).free / 1e9
        except Exception:
            return -1.0

    def diagnose_load_failure(self,
                              min_free_gb: float = CHALLENGER_DISK_HEADROOM_GB) -> str:
        """Classify a challenger load failure as 'infra' (our fault — requeue,
        don't burn the miner) or 'model' (the checkpoint's fault — reject).
        Low disk or a dead teacher/king means the pod is unhealthy, not the
        challenger."""
        free = self.free_disk_gb()
        if 0 <= free < min_free_gb:
            return "infra"
        if getattr(self, "role", "duel") != "bench":
            if not self._alive(self.teacher_slot):
                return "infra"
            if self.king_slot.served and not self._alive(self.king_slot):
                return "infra"
        return "model"

    def challenger_process_dead(self) -> bool:
        """True when the challenger vLLM process has exited. Unambiguous
        (unlike an HTTP probe timing out under load) — safe to act on
        immediately."""
        proc = self.chall_slot.proc
        return proc is None or proc.poll() is not None

    def challenger_alive(self, http_timeout: float = 3.0) -> bool:
        """Process + HTTP liveness of the served challenger. Used by the bench
        watchdog to fail fast when vLLM dies mid-run instead of letting the
        agent harness grind for hours against a dead endpoint."""
        return self._alive(self.chall_slot, http_timeout=http_timeout)

    def challenger_log_tail(self) -> str:
        return _vllm_log_tail(self.chall_slot.label)

    def unload_challenger(self) -> None:
        # Snapshot deliberately NOT evicted here: the pre-load prune bounds
        # disk identically, and keeping it makes retries of the same
        # checkpoint skip the re-download.
        for slot in (self.chall_slot, self.chal2_slot):
            if slot is None:
                continue
            if self._warm_swap_slot(slot) and self._proc_alive(slot):
                # Keep the engine (compile, CUDA graphs, KV pool) for the
                # next challenger's in-place weight swap. Nothing else uses
                # these GPUs between duels. Its snapshot stays in the prune
                # keep set via slot.served until the swap replaces it.
                log.info("%s kept warm for the next challenger (%s)",
                         slot.label, (slot.served.repo if slot.served else "?"))
                continue
            self._kill(slot)

    # -- prefetch ---------------------------------------------------------------
    def begin_duel(self, king_repo: str, chall_repo: str) -> None:
        """Protect this duel's repos from cache prunes until end_duel()."""
        with self._lock:
            self._pending_duel = {r for r in (king_repo, chall_repo) if r}

    def end_duel(self) -> None:
        with self._lock:
            self._pending_duel = set()

    def _note_prefetched(self, repo: str, revision: str) -> None:
        """Record a completed prefetch so the prune keeps it until its duel."""
        with self._lock:
            self._prefetched_ready.pop(repo, None)
            self._prefetched_ready[repo] = revision
            while len(self._prefetched_ready) > PREFETCH_KEEP:
                oldest = next(iter(self._prefetched_ready))
                del self._prefetched_ready[oldest]

    def start_prefetch(self, repo: str, revision: str,
                       weight_bytes: int = 0) -> tuple[bool, str]:
        """Warm the next challenger's snapshot in the background while the
        current duel scores. Best-effort and advisory: any decline reason just
        means the next load pays the download like before. One at a time.

        `weight_bytes` comes from the validator's metadata scan (the same
        number /duel receives) — the engine stays a dumb process manager with
        no HF-metadata dependency and no contract-policy knowledge. 0 means
        unknown: the disk-fit check passes trivially and the pre-duel
        challenger_fits remains the backstop."""
        with self._lock:
            if self._prefetch is not None and self._prefetch[1].is_alive():
                return False, f"busy with {self._prefetch[0]}"
            self._prefetch = None
            if self.chall_slot.served and self.chall_slot.served.repo == repo:
                return False, "already serving"
            if r2store.snapshot_ready(repo, revision):
                # A completed snapshot needs no worker, but it must survive
                # every prune between now and its duel (see _prefetched_ready).
                self._prefetched_ready.pop(repo, None)
                self._prefetched_ready[repo] = revision
                while len(self._prefetched_ready) > PREFETCH_KEEP:
                    del self._prefetched_ready[next(iter(self._prefetched_ready))]
                return True, "already cached"
            fits, free_gb = self.challenger_fits(weight_bytes, repo=repo,
                                                 revision=revision)
            if not fits:
                return False, (f"~{weight_bytes / 1e9:.0f}GB does not fit "
                               f"{free_gb:.0f}GB free")
            cancel = threading.Event()
            t = threading.Thread(target=self._prefetch_worker,
                                 args=(repo, revision, cancel),
                                 daemon=True, name="prefetch")
            self._prefetch = (repo, t, cancel)
            t.start()
            return True, "downloading"

    @staticmethod
    def _kill_downloader(proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("prefetch downloader pid %s did not reap in 30s",
                        proc.pid)

    def _prefetch_worker(self, repo: str, revision: str,
                         cancel: threading.Event) -> None:
        """Download in a child process, killed on stall or cancel. Subprocess
        isolation is what makes both recoverable: a Python thread stuck
        inside hf_transfer cannot be killed, but a process group can. Partial
        blobs stay on disk and resume on the next attempt (or vLLM's own
        download), so killing mid-transfer never loses the bytes already
        fetched. HF_HOME is pinned explicitly so the watchdog measures the
        same cache tree the child writes."""
        t0 = time.time()
        if r2store.is_r2(repo):
            # Same child-process isolation; r2store verifies as it downloads
            # and only writes the completion marker on a full match. Logging
            # configured explicitly: without it only WARNING+ reach stderr and
            # the per-file rates never land in prefetch.log.
            code = ("import logging; logging.basicConfig(level='INFO', "
                    "format='%(asctime)s %(name)s %(levelname)s %(message)s'); "
                    "from evalsrv import r2store; "
                    f"r2store.fetch_snapshot({repo!r}, {revision!r})")
        else:
            code = ("from huggingface_hub import snapshot_download; "
                    f"snapshot_download(repo_id={repo!r}, revision={revision!r})")
        # hf_transfer (multi-stream Rust downloader) only for this watchdogged
        # child: single-stream python pulls were observed at 26 MB/s (2647s for
        # one challenger, 2026-08-28) and its known failure mode — hanging on a
        # dead TCP connection — is exactly what the stall loop below kills.
        # The inline vLLM download path has no such watchdog, so it stays on
        # the slow-but-safe default; with prefetch retrying, it is rarely hit.
        env = dict(os.environ, HF_HOME=HF_HOME, HF_HUB_ENABLE_HF_TRANSFER="1")
        # The child's own log (per-file rates, part retries, integrity
        # errors) used to go to a TemporaryFile that died with the attempt;
        # every stall post-mortem was blind. Append to a persistent log.
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        child_log = LOG_DIR / "prefetch.log"
        for attempt in range(3):
            if attempt:
                # Partial blobs resume, so a retry only re-pays the tail.
                if cancel.wait(10.0 * attempt):
                    log.info("prefetch %s cancelled", repo)
                    return
                log.info("prefetch %s: retry %d", repo, attempt)
            try:
                stderr_f = open(child_log, "ab")
                stderr_f.write(
                    f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                    f"prefetch {repo}@{revision[:12]} attempt {attempt}\n".encode())
                stderr_f.flush()
                proc = subprocess.Popen(
                    [sys.executable, "-c", code], env=env,
                    stdout=subprocess.DEVNULL, stderr=stderr_f,
                    stdin=subprocess.DEVNULL, start_new_session=True)
            except Exception:
                log.warning("prefetch %s: could not spawn downloader", repo,
                            exc_info=True)
                return
            last_key: tuple[str, int] | None = None
            last_progress = time.time()
            stalled = False
            while proc.poll() is None:
                if cancel.wait(15.0):
                    log.info("prefetch %s cancelled", repo)
                    self._kill_downloader(proc)
                    stderr_f.close()
                    return
                key = self._prefetch_progress(repo, revision, proc.pid)
                # Progress = the same source grew, or the source changed
                # (heartbeat appearing after the disk fallback). Heartbeat and
                # disk bytes are different scales; never compare across them.
                if last_key is None or key[0] != last_key[0] or key[1] > last_key[1]:
                    last_key = key
                    last_progress = time.time()
                elif time.time() - last_progress > PREFETCH_STALL_S:
                    log.warning(
                        "prefetch %s stalled at %.1fGB (%s; no progress for "
                        "%.0fs); killing downloader", repo,
                        max(key[1], 0) / 1e9, key[0], PREFETCH_STALL_S)
                    self._kill_downloader(proc)
                    stalled = True
                    break
            stderr_f.close()
            if not stalled and proc.returncode == 0:
                log.info("prefetch %s@%s done in %.0fs",
                         repo, revision[:12], time.time() - t0)
                self._note_prefetched(repo, revision)
                return
            tail = self._log_tail(child_log)
            log.warning("prefetch %s@%s attempt %d failed after %.0fs "
                        "(exit %s)%s", repo, revision[:12], attempt,
                        time.time() - t0,
                        "stall" if stalled else proc.returncode,
                        f": ...{tail}" if tail else "")
        log.warning("prefetch %s@%s gave up after 3 attempts; the duel "
                    "will download inline", repo, revision[:12])

    def _prefetch_progress(self, repo: str, revision: str,
                           pid: int) -> tuple[str, int]:
        """(source, value) the stall watchdog compares between polls. Prefers
        the r2store child's heartbeat (work bytes: landed chunks + hashed
        chunks, pid-stamped so a dead child's file is ignored); falls back to
        on-disk bytes for HF downloads or a child that has not written yet."""
        if r2store.is_r2(repo):
            hb = r2store.read_progress(r2store.snapshot_dir(repo, revision, HF_HOME))
            if hb and hb.get("pid") == pid:
                return "heartbeat", int(hb.get("work_bytes", 0))
        return "disk", self._cached_repo_bytes(repo)

    @staticmethod
    def _log_tail(path: Path, n: int = 300) -> str:
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(f.tell() - 2000, 0))
                return f.read().decode(errors="replace").strip()[-n:]
        except OSError:
            return ""

    def _alive(self, slot: Slot, http_timeout: float = 3.0) -> bool:
        # Remote teacher (base_url): no local process — probe the endpoint.
        if slot.served and slot.served.base_url:
            try:
                httpx.get(f"{slot.served.base_url.rstrip('/')}/models",
                          timeout=http_timeout)
                return True
            except httpx.HTTPError:
                return False
        if slot.proc is None or slot.proc.poll() is not None:
            return False
        try:
            httpx.get(f"http://localhost:{slot.port}/v1/models",
                      timeout=http_timeout)
            return True
        except httpx.HTTPError:
            return False

    def status(self) -> dict:
        def one(slot: Slot) -> dict:
            return {
                "ready": slot.ready and self._alive(slot),
                "repo": slot.served.repo if slot.served else None,
                "revision": slot.served.revision if slot.served else None,
            }
        out = {"teacher": one(self.teacher_slot),
               "king": one(self.king_slot),
               "challenger": one(self.chall_slot)}
        if self.teacher2_slot is not None:
            out["teacher2"] = one(self.teacher2_slot)
        if self.king2_slot is not None:
            out["king2"] = one(self.king2_slot)
        if self.chal2_slot is not None:
            out["challenger2"] = one(self.chal2_slot)
        return out

    # -- disk hygiene ---------------------------------------------------------------
    @staticmethod
    def _repo_cache_dir(repo: str) -> Path:
        return r2store.repo_cache_dir(repo, HF_HOME)

    def _cached_repo_bytes(self, repo: str) -> int:
        """On-disk bytes already cached for a repo (blob payloads; snapshot
        symlinks contribute nothing). Includes *.incomplete partials, which
        resume rather than re-download."""
        d = self._repo_cache_dir(repo)
        if not d.exists():
            return 0
        total = 0
        for p in d.rglob("*"):
            if p.is_file() and not p.is_symlink():
                landed = r2store.incomplete_bytes(p)
                if landed is not None:
                    # Preallocated ranged download: st_size is the target,
                    # not progress.
                    total += landed
                    continue
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
        return total

    def _prune_challenger_cache(self, keep_repo: str | None = None,
                                keep_revision: str | None = None,
                                extra_keep: set[str] | None = None) -> None:
        """Free space before a new challenger download: drop every cached
        model that is not the teacher, the current king, or the incoming
        challenger itself. The incoming repo dir is kept only when it holds a
        snapshot of the exact requested revision — a leftover dir with only a
        different revision would give no download credit while accumulating
        stale blobs, so it is pruned like any other stranger.

        Always kept regardless of arguments: the actively serving challenger
        (a prefetch prune runs mid-duel) and a LIVE prefetch target (partial
        blobs, no snapshot dir yet). A dead prefetch worker confers no
        protection, so an abandoned prefetch is reclaimed here instead of
        shrinking usable disk forever."""
        with self._lock:
            keep = {self.cfg["teacher"]["repo"]}
            if self.king_slot.served:
                keep.add(self.king_slot.served.repo)
            if self.chall_slot.served:
                keep.add(self.chall_slot.served.repo)
            if self._prefetch is not None and self._prefetch[1].is_alive():
                keep.add(self._prefetch[0])
            keep.update(self._materializing)
            keep.update(self._pending_duel)
            # Finished prefetches awaiting their duel: kept while the
            # verified snapshot is actually on disk, forgotten otherwise.
            for r, rev in list(self._prefetched_ready.items()):
                if r2store.snapshot_ready(r, rev, HF_HOME):
                    keep.add(r)
                else:
                    del self._prefetched_ready[r]
            if extra_keep:
                keep.update(r for r in extra_keep if r)
            if keep_repo and keep_revision and r2store.snapshot_ready(
                    keep_repo, keep_revision, HF_HOME):
                keep.add(keep_repo)
            hub = Path(HF_HOME) / "hub"
            if not hub.exists():
                return
            keep_dirs = {r2store.cache_dir_name(r) for r in keep}
            # Rename under the lock (atomic — instantly invisible to every
            # snap.exists()/cache check), then rmtree outside it: deleting a
            # multi-GB tree takes long enough to stall launches otherwise.
            doomed: list[Path] = list(hub.glob("*.pruning"))
            for d in hub.iterdir():
                if (d.is_dir() and d.name.startswith("models--")
                        and d.name not in keep_dirs
                        and not _is_public_king_cache(d)):
                    log.info("pruning cached model %s", d.name)
                    target = hub / f"{d.name}.pruning"
                    try:
                        d.rename(target)
                        doomed.append(target)
                    except OSError:
                        doomed.append(d)
        for d in doomed:
            shutil.rmtree(d, ignore_errors=True)
