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
    }


@dataclass
class Slot:
    label: str
    port: int
    gpus: str
    tp: int
    served: Served | None = None
    proc: subprocess.Popen | None = None
    ready: bool = False


@dataclass
class Engine:
    cfg: dict  # full affine.toml dict
    teacher_slot: Slot = field(init=False)
    king_slot: Slot = field(init=False)
    chall_slot: Slot = field(init=False)

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
        self.king_slot = Slot("king", int(ms["king_port"]), ms["king_gpus"],
                              int(ms["tp"]))
        self.chall_slot = Slot("challenger", int(ms["challenger_port"]),
                               ms["challenger_gpus"], int(ms["tp"]))

    # -- process control -------------------------------------------------------
    def _launch(self, slot: Slot, repo: str, revision: str | None) -> None:
        self._kill(slot)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ms = self.cfg["miner_serving"]
        env = dict(os.environ, **_vllm_env(), CUDA_VISIBLE_DEVICES=slot.gpus)
        cmd = [
            "vllm", "serve", repo,
            "--port", str(slot.port),
            "--tensor-parallel-size", str(slot.tp),
            "--max-model-len", str(ms["max_model_len"]),
            "--gpu-memory-utilization", str(ms["gpu_memory_utilization"]),
            # Teacher-forcing sends echo+logprobs requests; vLLM materializes
            # a fp32 log_softmax over (prefill_chunk x vocab) per request, so
            # keep chunks small or the engine OOMs under concurrency.
            "--max-num-batched-tokens", str(ms["max_num_batched_tokens"]),
            # Avoid FlashInfer JIT (needs a coherent system CUDA toolkit; the
            # pip nvidia-cu13 wheel headers often trip B300/Blackwell builds).
            "--attention-backend", "FLASH_ATTN",
            "--attention-config.use_trtllm_attention", "0",
            "--compilation-config.pass_config.fuse_allreduce_rms", "false",
        ]
        if revision:
            cmd += ["--revision", revision]
        logf = open(LOG_DIR / f"vllm_{slot.label}.log", "a")
        log.info("launching %s: %s (gpus=%s)", slot.label, repo, slot.gpus)
        slot.proc = subprocess.Popen(
            cmd, env=env, stdout=logf, stderr=logf,
            stdin=subprocess.DEVNULL, start_new_session=True)
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
        slot.proc = None
        slot.served = None
        slot.ready = False

    def _wait_ready(self, slot: Slot, timeout_s: int = 3600) -> bool:
        t0 = time.time()
        url = f"http://localhost:{slot.port}/v1/models"
        while time.time() - t0 < timeout_s:
            if slot.proc is not None and slot.proc.poll() is not None:
                log.error("%s vllm process exited with %s", slot.label,
                          slot.proc.returncode)
                return False
            try:
                httpx.get(url, timeout=3)
                slot.ready = True
                log.info("%s ready in %.0fs", slot.label, time.time() - t0)
                return True
            except httpx.HTTPError:
                time.sleep(10)
        return False

    # -- public API --------------------------------------------------------------
    def ensure_teacher(self) -> bool:
        t = self.cfg["teacher"]
        if self.teacher_slot.ready and self._alive(self.teacher_slot):
            return True
        self._launch(self.teacher_slot, t["repo"], None)
        return self._wait_ready(self.teacher_slot)

    def ensure_king(self, repo: str, revision: str) -> bool:
        s = self.king_slot.served
        if (s and s.repo == repo and s.revision == revision
                and self.king_slot.ready and self._alive(self.king_slot)):
            return True
        self._launch(self.king_slot, repo, revision)
        return self._wait_ready(self.king_slot)

    def load_challenger(self, repo: str, revision: str) -> bool:
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

    def unload_challenger(self) -> None:
        # Snapshot deliberately NOT evicted here: the pre-load prune bounds
        # disk identically, and keeping it makes retries of the same
        # checkpoint skip the re-download.
        self._kill(self.chall_slot)

    def _alive(self, slot: Slot) -> bool:
        if slot.proc is None or slot.proc.poll() is not None:
            return False
        try:
            httpx.get(f"http://localhost:{slot.port}/v1/models", timeout=3)
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
        return {"teacher": one(self.teacher_slot),
                "king": one(self.king_slot),
                "challenger": one(self.chall_slot)}

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
        stale blobs, so it is pruned like any other stranger."""
        keep = {self.cfg["teacher"]["repo"]}
        if self.king_slot.served:
            keep.add(self.king_slot.served.repo)
        if keep_repo and keep_revision:
            snap = self._repo_cache_dir(keep_repo) / "snapshots" / keep_revision
            if snap.exists():
                keep.add(keep_repo)
        hub = Path(HF_HOME) / "hub"
        if not hub.exists():
            return
        keep_dirs = {"models--" + r.replace("/", "--") for r in keep}
        for d in hub.iterdir():
            if d.is_dir() and d.name.startswith("models--") and d.name not in keep_dirs:
                log.info("pruning cached model %s", d.name)
                shutil.rmtree(d, ignore_errors=True)
