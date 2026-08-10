"""vLLM process lifecycle on the eval machine.

Three serving slots, GPU allocation from affine.toml:
  teacher    — frozen reference, launched at boot, always warm.
  king       — reigning champion, loaded on first duel, kept warm across
               duels, swapped only when the validator sends a new king ref.
  challenger — loaded per duel, killed afterwards.

Models are served with `vllm serve <repo> --revision <sha>` so the snapshot
is pinned to the on-chain commitment (TOCTOU). HF cache lives under
/root/hf; a stale challenger snapshot is pruned lazily right before the next
challenger's download (not eagerly after the duel), so validator retries and
repeat duels of the same repo@revision skip the multi-GB re-download while
disk stays bounded to teacher + king + at most one challenger.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .vllm_client import Served

log = logging.getLogger("evalsrv.engine")

HF_HOME = os.environ.get("HF_HOME", "/root/hf")
LOG_DIR = Path(os.environ.get("AFFINE_LOG_DIR", "/root/logs"))

# Free-disk headroom (GB) a challenger download needs on top of its servable
# weights. Also the low-disk floor `diagnose_load_failure` treats as pod
# trouble, so the pre-download fit check and the post-failure diagnosis agree.
CHALLENGER_DISK_HEADROOM_GB = 20.0

# Kill a prefetch download whose on-disk bytes have not grown for this long.
# hf_transfer can hang forever on a dead TCP connection (observed live:
# 0 B/s with ESTABLISHED sockets and no timeout); a hung in-process thread
# would block every future prefetch and stall the next load_challenger join.
PREFETCH_STALL_S = 180.0

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


@dataclass
class Engine:
    cfg: dict  # full affine.toml dict
    teacher_slot: Slot = field(init=False)
    # Optional second teacher replica ([teacher].replica_gpus). Best-effort:
    # duels fall back to the primary alone when it is down.
    teacher2_slot: Slot | None = field(init=False, default=None)
    king_slot: Slot = field(init=False)
    chall_slot: Slot = field(init=False)
    # One reentrant lock owns every keep-set read and cache prune, every slot
    # launch, and every prefetch state transition. Callers run in three thread
    # contexts (FastAPI threadpool, duel/bench job threads, teacher warmup);
    # without the lock a /prefetch prune could race a duel's prune→launch
    # window and delete the snapshot the duel just decided to keep.
    _lock: threading.RLock = field(init=False, default_factory=threading.RLock,
                                   repr=False)
    # Next-challenger snapshot being warmed while the current duel scores
    # (download is network-bound, scoring is GPU-bound): (repo, worker,
    # cancel event). Protection from pruning is derived from thread liveness:
    # a dead worker confers none, so a stale prefetch target self-expires
    # instead of inflating the keep set forever (which would shrink
    # reclaimable disk and could wedge the pod into POD_CAPACITY faults).
    _prefetch: tuple[str, threading.Thread, threading.Event] | None = \
        field(init=False, default=None, repr=False)

    def __post_init__(self):
        t = self.cfg["teacher"]
        ms = self.cfg["miner_serving"]
        self.role = os.environ.get("AFFINE_ROLE", "duel")
        if self.role == "bench":
            # Dedicated bench pod: one miner slot across the rented GPUs.
            # Teacher/king slots exist for status shape but are never launched.
            bs = self.cfg.get("bench_serving") or {}
            gpus = str(bs.get("gpus", "0,1"))
            tp = int(bs.get("tp", ms.get("tp", 2)))
            port = int(bs.get("port", ms.get("challenger_port", 8002)))
            self.teacher_slot = Slot("teacher", 8000, "0", 1)
            self.king_slot = Slot("king", 8001, "0", 1)
            self.chall_slot = Slot("challenger", port, gpus, tp)
            return
        self.teacher_slot = Slot("teacher", int(t["port"]), t["gpus"], int(t["tp"]))
        if t.get("replica_gpus"):
            self.teacher2_slot = Slot("teacher2", int(t.get("replica_port", 8003)),
                                      str(t["replica_gpus"]), int(t["tp"]))
        self.king_slot = Slot("king", int(ms["king_port"]), ms["king_gpus"],
                              int(ms["tp"]))
        self.chall_slot = Slot("challenger", int(ms["challenger_port"]),
                               ms["challenger_gpus"], int(ms["tp"]))

    def _slots(self) -> list[Slot]:
        slots = [self.teacher_slot, self.king_slot, self.chall_slot]
        if self.teacher2_slot is not None:
            slots.append(self.teacher2_slot)
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
        if self.role == "bench":
            bs = self.cfg.get("bench_serving") or {}
            batched_tokens = int(bs.get("max_num_batched_tokens", 16384))
        cmd = [
            "vllm", "serve", repo,
            "--port", str(slot.port),
            "--tensor-parallel-size", str(slot.tp),
            "--max-model-len", str(ms["max_model_len"]),
            "--gpu-memory-utilization", str(ms["gpu_memory_utilization"]),
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
        if self.role == "bench":
            # Qwen3.6 GDN linear attention: gdn_prefill_backend=auto picks
            # FlashInfer on Hopper (SM90) and JIT-compiles gdn_prefill_sm90 at
            # the first request; pip cu13 nvcc vs mismatched CUDA headers makes
            # the ninja build fail and the engine dies mid-serve. Triton/FLA
            # needs no JIT. Bench-only: the duel pod (Blackwell) resolves its
            # own working backend and its numerics stay as burned in.
            cmd += ["--additional-config", '{"gdn_prefill_backend": "triton"}']
        if revision:
            cmd += ["--revision", revision]
        return cmd

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

    def _launch(self, slot: Slot, repo: str, revision: str | None) -> None:
        with self._lock:
            self._kill(slot)
            self._sweep_slot_gpus(slot)
            slot.load_error = ""
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            _purge_broken_flashinfer_moe_cache()
            env = dict(os.environ, **_vllm_env(), CUDA_VISIBLE_DEVICES=slot.gpus)
            logf = open(LOG_DIR / f"vllm_{slot.label}.log", "a")
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
                                 port=slot.port)
            slot.ready = False

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

    def _wait_ready(self, slot: Slot, timeout_s: int = 3600) -> bool:
        t0 = time.time()
        url = f"http://localhost:{slot.port}/v1/models"
        while time.time() - t0 < timeout_s:
            if slot.proc is not None and slot.proc.poll() is not None:
                return self._fail_load(
                    slot, f"vllm process exited with {slot.proc.returncode}")
            try:
                httpx.get(url, timeout=3)
                slot.ready = True
                slot.load_error = ""
                log.info("%s ready in %.0fs", slot.label, time.time() - t0)
                return True
            except httpx.HTTPError:
                time.sleep(10)
        return self._fail_load(slot, f"not ready after {timeout_s}s")

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
        return out

    def ensure_king(self, repo: str, revision: str) -> bool:
        s = self.king_slot.served
        if (s and s.repo == repo and s.revision == revision
                and self.king_slot.ready and self._alive(self.king_slot)):
            return True
        self._launch(self.king_slot, repo, revision)
        return self._wait_ready(self.king_slot)

    def _settle_prefetch(self, incoming_repo: str | None) -> None:
        """Bring the prefetch slot to rest before a challenger download.
        A live prefetch of the incoming repo is joined (finishing beats
        racing vLLM's downloader for the same blob locks; the worker's stall
        watchdog bounds the wait in practice, the timeout is a backstop). A
        live prefetch of any OTHER repo is stale — the queue was reordered —
        so it is cancelled before it can contend for disk and bandwidth.
        Runs OUTSIDE the lock: joining under it would block /prefetch,
        challenger_fits and every launch for the duration."""
        with self._lock:
            pf = self._prefetch
        if pf is None or not pf[1].is_alive():
            return
        if pf[0] == incoming_repo:
            log.info("waiting for in-flight prefetch of %s", pf[0])
            pf[1].join(timeout=1800)
        else:
            log.info("cancelling stale prefetch of %s (incoming: %s)",
                     pf[0], incoming_repo)
            pf[2].set()
            pf[1].join(timeout=60)

    def load_challenger(self, repo: str, revision: str) -> bool:
        self._settle_prefetch(repo)
        with self._lock:
            if self._prefetch is not None and self._prefetch[0] == repo:
                # Consumed: keep_repo protects the snapshot from here on and
                # the slot frees up for the next prefetch target.
                self._prefetch = None
            self._prune_challenger_cache(keep_repo=repo, keep_revision=revision)
            self._launch(self.chall_slot, repo, revision)
        return self._wait_ready(self.chall_slot)

    def challenger_fits(self, required_bytes: int, repo: str | None = None,
                        revision: str | None = None) -> tuple[bool, float]:
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
        self._settle_prefetch(repo)
        with self._lock:
            self._prune_challenger_cache(keep_repo=repo, keep_revision=revision)
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
        self._kill(self.chall_slot)

    # -- prefetch ---------------------------------------------------------------
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
            snap = self._repo_cache_dir(repo) / "snapshots" / revision
            if snap.exists():
                # A completed snapshot needs no worker; the load-time
                # keep_repo prune protection covers it from here.
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
        code = ("from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id={repo!r}, revision={revision!r})")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                env=dict(os.environ, HF_HOME=HF_HOME),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, start_new_session=True)
        except Exception:
            log.warning("prefetch %s: could not spawn downloader", repo,
                        exc_info=True)
            return
        last_bytes = -1
        last_progress = time.time()
        while proc.poll() is None:
            if cancel.wait(15.0):
                log.info("prefetch %s cancelled", repo)
                self._kill_downloader(proc)
                return
            cur = self._cached_repo_bytes(repo)
            if cur > last_bytes:
                last_bytes = cur
                last_progress = time.time()
            elif time.time() - last_progress > PREFETCH_STALL_S:
                log.warning(
                    "prefetch %s stalled at %.1fGB (no progress for %.0fs); "
                    "killing downloader", repo, max(cur, 0) / 1e9,
                    PREFETCH_STALL_S)
                self._kill_downloader(proc)
                return
        if proc.returncode == 0:
            log.info("prefetch %s@%s done in %.0fs",
                     repo, revision[:12], time.time() - t0)
        else:
            log.warning("prefetch %s@%s failed after %.0fs (exit %s)",
                        repo, revision[:12], time.time() - t0, proc.returncode)

    def _alive(self, slot: Slot, http_timeout: float = 3.0) -> bool:
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
        return out

    # -- disk hygiene ---------------------------------------------------------------
    @staticmethod
    def _repo_cache_dir(repo: str) -> Path:
        return Path(HF_HOME) / "hub" / ("models--" + repo.replace("/", "--"))

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
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
        return total

    def _prune_challenger_cache(self, keep_repo: str | None = None,
                                keep_revision: str | None = None) -> None:
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
            if keep_repo and keep_revision:
                snap = self._repo_cache_dir(keep_repo) / "snapshots" / keep_revision
                if snap.exists():
                    keep.add(keep_repo)
            hub = Path(HF_HOME) / "hub"
            if not hub.exists():
                return
            keep_dirs = {"models--" + r.replace("/", "--") for r in keep}
            # Rename under the lock (atomic — instantly invisible to every
            # snap.exists()/cache check), then rmtree outside it: deleting a
            # multi-GB tree takes long enough to stall launches otherwise.
            doomed: list[Path] = list(hub.glob("*.pruning"))
            for d in hub.iterdir():
                if (d.is_dir() and d.name.startswith("models--")
                        and d.name not in keep_dirs):
                    log.info("pruning cached model %s", d.name)
                    target = hub / f"{d.name}.pruning"
                    try:
                        d.rename(target)
                        doomed.append(target)
                    except OSError:
                        doomed.append(d)
        for d in doomed:
            shutil.rmtree(d, ignore_errors=True)
