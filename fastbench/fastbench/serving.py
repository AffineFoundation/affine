"""vLLM slot manager: one served model, kept warm across benches.

Keep-warm is the single biggest repeat-bench win: if the requested
repo@revision is already served and alive, ensure() is a no-op. Cold loads
pre-download with hf_transfer (parallel, near line rate) before `vllm serve`
so the engine starts from local cache.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .config import Config

log = logging.getLogger("fastbench.serving")

HF_HOME = os.environ.get("HF_HOME", "/root/hf")
LOG_DIR = Path(os.environ.get("FASTBENCH_LOG_DIR", "/root/logs"))


def _cuda_complete(p: Path) -> bool:
    return ((p / "bin" / "nvcc").exists()
            and (p / "include" / "cuda_fp16.h").exists())


def _cuda_home() -> str:
    """Find a complete CUDA toolkit (nvcc + headers) for FlashInfer JIT.

    Lium images often lack /usr/local/cuda; the pip nvidia/cu13 wheel ships a
    full toolkit under site-packages (proven fix from the evalsrv pods)."""
    env_home = os.environ.get("CUDA_HOME")
    if env_home and _cuda_complete(Path(env_home)):
        return env_home
    try:
        import nvidia  # type: ignore  # namespace package
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
    return env_home or "/usr/local/cuda"


def _cuda_env() -> dict[str, str]:
    cuda_home = _cuda_home()
    path = os.environ.get("PATH", "")
    bin_dir = str(Path(cuda_home) / "bin")
    if bin_dir not in path.split(":"):
        path = f"{bin_dir}:{path}"
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    lib_path = os.environ.get("LIBRARY_PATH", "")
    for d in (str(Path(cuda_home) / "lib"), str(Path(cuda_home) / "lib64")):
        if Path(d).is_dir():
            if d not in ld.split(":"):
                ld = f"{d}:{ld}" if ld else d
            if d not in lib_path.split(":"):
                lib_path = f"{d}:{lib_path}" if lib_path else d
    return {"CUDA_HOME": cuda_home, "CUDA_PATH": cuda_home, "PATH": path,
            "LD_LIBRARY_PATH": ld, "LIBRARY_PATH": lib_path}


def parse_model(model: str) -> tuple[str, str | None]:
    """'org/repo@rev' -> (repo, rev); 'org/repo' -> (repo, None)."""
    if "@" in model:
        repo, rev = model.split("@", 1)
        return repo.strip(), rev.strip() or None
    return model.strip(), None


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=30).stdout
        return len([ln for ln in out.splitlines() if ln.strip()])
    except Exception:
        return 0


@dataclass
class SlotState:
    repo: str | None = None
    revision: str | None = None
    proc: subprocess.Popen | None = None
    pgid: int | None = None
    ready: bool = False
    load_error: str = ""
    loaded_at: float = 0.0
    # True when this engine was started by an earlier process (adopted after a
    # service redeploy) rather than launched by us, so we hold no Popen for it.
    adopted: bool = False


@dataclass
class PrefetchState:
    repo: str
    revision: str | None
    state: str            # downloading | ready | error
    started: float
    finished: float = 0.0
    error: str = ""

    def as_dict(self) -> dict:
        out = asdict(self)
        out["elapsed_s"] = round((self.finished or time.time()) - self.started, 1)
        return out


class ModelSlot:
    """One vLLM serve process across all GPUs (TP x DP)."""

    KEEP_LOGS = 6

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = SlotState()
        self._lock = threading.RLock()
        self._log_path: Path | None = None
        # Weight prefetch runs outside self._lock: it only touches the HF disk
        # cache, so it can overlap a bench instead of queueing behind it.
        self._prefetch: PrefetchState | None = None
        self._prefetch_lock = threading.Lock()

    def prefetch(self, repo: str, revision: str | None = None) -> dict:
        """Download a model's weights ahead of time, without disturbing the
        engine that is currently serving.

        Switching models costs a download plus an engine boot. The download is
        the larger half for a model that was never benched here (tens of GB),
        and it needs no GPU, so it can happen while the current bench runs."""
        with self._prefetch_lock:
            cur = self._prefetch
            if cur and cur.repo == repo and cur.state in ("downloading", "ready"):
                return cur.as_dict()
            if cur and cur.state == "downloading":
                return {"state": "busy", "repo": cur.repo,
                        "error": f"already prefetching {cur.repo}"}
            pf = PrefetchState(repo, revision, "downloading", time.time())
            self._prefetch = pf
        threading.Thread(target=self._run_prefetch, args=(pf,),
                         daemon=True, name="fastbench-prefetch").start()
        return pf.as_dict()

    def prefetch_status(self) -> dict | None:
        pf = self._prefetch
        return pf.as_dict() if pf else None

    def _run_prefetch(self, pf: PrefetchState) -> None:
        err = self._predownload(pf.repo, pf.revision, None)
        pf.finished = time.time()
        pf.error = err or ""
        pf.state = "error" if err else "ready"
        log.info("prefetch %s finished in %.0fs (%s)", pf.repo,
                 pf.finished - pf.started, pf.state)

    # -- public ---------------------------------------------------------------
    def ensure(self, repo: str, revision: str | None,
               abort: threading.Event | None = None) -> dict:
        """Serve repo@revision; reuse the live engine when it already matches.

        Returns {"ok", "warm", "load_s", "error"}."""
        with self._lock:
            if self._matches(repo, revision) and self._alive():
                return {"ok": True, "warm": True, "load_s": 0.0, "error": ""}
            t0 = time.time()
            dl_err = self._predownload(repo, revision, abort)
            if dl_err:
                return {"ok": False, "warm": False,
                        "load_s": round(time.time() - t0, 1), "error": dl_err}
            if abort is not None and abort.is_set():
                return {"ok": False, "warm": False, "load_s": 0.0,
                        "error": "aborted"}
            self._launch(repo, revision)
            ok = self._wait_ready(self.cfg.model_load_timeout_s, abort)
            load_s = round(time.time() - t0, 1)
            if ok:
                self._prune_cache()
            return {"ok": ok, "warm": False, "load_s": load_s,
                    "error": self.state.load_error}

    def alive(self) -> bool:
        with self._lock:
            return self._alive()

    def status(self) -> dict:
        s = self.state
        return {"repo": s.repo, "revision": s.revision,
                "ready": s.ready and self._serving(),
                "adopted": s.adopted,
                "loaded_at": s.loaded_at, "load_error": s.load_error}

    def engine_counters(self) -> dict:
        """Cumulative engine totals, for live rollout progress.

        A finished rollout is the only thing that moves the done counter, so a
        run with 25 deep tasks reads 0/25 for many minutes even though every
        agent is stepping. Completed requests and generated tokens show that
        motion. Values are cumulative for the engine's whole lifetime, so
        callers must diff against a baseline."""
        try:
            r = httpx.get(f"http://127.0.0.1:{self.cfg.vllm_port}/metrics",
                          timeout=5)
            if r.status_code != 200:
                return {}
        except httpx.HTTPError:
            return {}
        totals = {"steps": 0.0, "gen_tokens": 0.0}
        for line in r.text.splitlines():
            # Summed over engines (data-parallel replicas) and finish reasons.
            if line.startswith("vllm:request_success_total"):
                key = "steps"
            elif line.startswith("vllm:generation_tokens_total"):
                key = "gen_tokens"
            else:
                continue
            try:
                totals[key] += float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
        return {k: int(v) for k, v in totals.items()}

    def _max_len_for(self, repo: str, revision: str | None) -> int:
        """Our context budget, clamped to what the model actually supports.

        vLLM refuses to start when --max-model-len exceeds the checkpoint's
        trained context, so a fixed budget turns every shorter-context model
        into a failed load (Qwen3-32B tops out at 40960, we ask for 65536)."""
        want = self.cfg.max_model_len
        try:
            if os.path.isdir(repo):
                # local checkpoint dir (scp'd merged weights): read straight
                # from disk; hf_hub_download rejects paths as repo ids
                path = os.path.join(repo, "config.json")
            else:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo, "config.json", revision=revision,
                                       cache_dir=os.path.join(HF_HOME, "hub"))
            with open(path) as f:
                conf = json.load(f)
        except Exception:
            log.warning("no config.json for %s; using max_model_len=%d",
                        repo, want)
            return want
        text = conf.get("text_config") or {}
        for key in ("max_position_embeddings", "model_max_length", "seq_length"):
            got = conf.get(key) or text.get(key)
            if isinstance(got, int) and got > 0:
                if got < want:
                    log.info("clamping max_model_len %d -> %d for %s",
                             want, got, repo)
                return min(want, got)
        return want

    def _serving(self) -> bool:
        """Cheap liveness for status calls: is something answering the port?

        /health must stay responsive while an engine boots. The strict check
        (can_generate) issues a real completion, which blocks for as long as
        the engine takes to come up, so it belongs on the bench path only."""
        s = self.state
        if s.proc is not None and s.proc.poll() is not None:
            return False
        try:
            r = httpx.get(f"http://127.0.0.1:{self.cfg.vllm_port}/v1/models",
                          timeout=3)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def stop(self) -> None:
        with self._lock:
            self._kill()

    @property
    def port(self) -> int:
        return self.cfg.vllm_port

    # -- internals -------------------------------------------------------------
    def _matches(self, repo: str, revision: str | None) -> bool:
        s = self.state
        if s.repo != repo or not s.ready:
            return False
        # Un-pinned request matches whatever revision is currently served
        # (the caller asked for "latest"; a warm hit beats a re-resolve).
        return revision is None or s.revision == revision

    def adopt_existing(self) -> bool:
        """Reuse an engine left running by a previous service process.

        Redeploys restart this service but not its vLLM child, so without
        adoption every deploy pays a full cold load for a model that is
        already resident."""
        with self._lock:
            if self._alive():
                return True
            try:
                r = httpx.get(f"http://127.0.0.1:{self.cfg.vllm_port}/v1/models",
                              timeout=10)
                served = [m.get("id") for m in (r.json().get("data") or [])]
            except (httpx.HTTPError, ValueError):
                return False
            repo = next((s for s in served if s), None)
            if not repo or not self.can_generate(repo):
                return False
            s = self.state
            s.repo, s.revision = repo, None
            s.proc, s.pgid = None, None
            s.adopted = True
            s.ready = True
            s.loaded_at = time.time()
            log.info("adopted running engine serving %s", repo)
            return True

    def _alive(self) -> bool:
        s = self.state
        if s.adopted and s.proc is None:
            return bool(s.repo) and self.can_generate(s.repo)
        if s.proc is None or s.proc.poll() is not None:
            return False
        return self.can_generate(s.repo or "")

    def can_generate(self, repo: str) -> bool:
        """Prove the endpoint can actually serve `repo`.

        A bare 200 from /v1/models is not enough: an orphaned API server whose
        engine cores are dead still answers it, and every rollout then fails
        against a 'ready' slot. This asks for one token instead."""
        base = f"http://127.0.0.1:{self.cfg.vllm_port}/v1"
        try:
            r = httpx.get(f"{base}/models", timeout=10)
            if r.status_code != 200:
                return False
            served = {m.get("id") for m in (r.json().get("data") or [])}
            if repo and repo not in served:
                log.warning("port %d serves %s, expected %s",
                            self.cfg.vllm_port, sorted(served), repo)
                return False
            g = httpx.post(f"{base}/completions",
                           json={"model": repo, "prompt": "ok",
                                 "max_tokens": 1, "temperature": 0},
                           timeout=120)
            return g.status_code == 200 and bool(g.json().get("choices"))
        except (httpx.HTTPError, ValueError, KeyError):
            return False

    def _predownload(self, repo: str, revision: str | None,
                     abort: threading.Event | None) -> str:
        """snapshot_download in a killable child process (Xet fast path)."""
        if os.path.isdir(repo):
            # Local checkpoint dir (scp'd merged weights): nothing to fetch,
            # and snapshot_download would reject the path as a repo id.
            if not os.path.exists(os.path.join(repo, "config.json")):
                return f"local model dir {repo} has no config.json"
            return ""
        rev_arg = f", revision={revision!r}" if revision else ""
        code = ("from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id={repo!r}{rev_arg})")
        env = dict(os.environ, HF_HOME=HF_HOME,
                   HF_XET_HIGH_PERFORMANCE="1")
        proc = subprocess.Popen([sys.executable, "-c", code], env=env,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE,
                                start_new_session=True)
        while proc.poll() is None:
            if abort is not None and abort.wait(5.0):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                return "aborted"
            time.sleep(0)
        if proc.returncode != 0:
            err = (proc.stderr.read() if proc.stderr else b"").decode(
                errors="replace")[-500:]
            return f"download failed: {err}"
        return ""

    def _vllm_cmd(self, repo: str, revision: str | None) -> list[str]:
        cfg = self.cfg
        n_gpus = gpu_count() or 1
        tp = max(1, cfg.tp)
        dp = cfg.dp if cfg.dp > 0 else max(1, n_gpus // tp)
        cmd = [
            "vllm", "serve", repo,
            "--port", str(cfg.vllm_port),
            "--tensor-parallel-size", str(tp),
            "--max-model-len", str(self._max_len_for(repo, revision)),
            "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
            "--max-num-batched-tokens", str(cfg.max_num_batched_tokens),
            "--max-num-seqs", str(cfg.max_num_seqs),
            # The agent resends the whole growing conversation every step;
            # cached prefills are the biggest serving win for this workload.
            "--enable-prefix-caching",
        ]
        if dp > 1:
            cmd += ["--data-parallel-size", str(dp)]
        if cfg.reasoning_parser:
            cmd += ["--reasoning-parser", cfg.reasoning_parser]
        if cfg.moe_backend:
            cmd += ["--moe-backend", cfg.moe_backend]
        if cfg.gdn_prefill_backend:
            cmd += ["--additional-config",
                    json.dumps({"gdn_prefill_backend":
                                cfg.gdn_prefill_backend})]
        if revision:
            cmd += ["--revision", revision]
        if cfg.vllm_extra_args:
            cmd += cfg.vllm_extra_args.split()
        return cmd

    @staticmethod
    def _purge_broken_moe_jit_cache() -> None:
        """Drop failed fused_moe_trtllm JIT artifacts; a broken object left by
        a failed compile can poison the next launch. Safe no-op when absent."""
        for root in (Path.home() / ".cache" / "flashinfer",
                     Path("/root/.cache/flashinfer")):
            if not root.is_dir():
                continue
            for path in root.rglob("*fused_moe_trtllm*"):
                try:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _launch(self, repo: str, revision: str | None) -> None:
        self._kill()
        self._free_port()
        self._sweep_gpus()
        self._purge_broken_moe_jit_cache()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        # One log per launch: a shared append-only file makes error tails (and
        # any external grep) report failures from earlier attempts as if they
        # were current, and it loses the engine stats of past runs that the
        # tuning work needs. `vllm_fastbench.log` symlinks to the newest.
        log_path = LOG_DIR / f"vllm_fastbench-{int(time.time())}.log"
        self._log_path = log_path
        latest = LOG_DIR / "vllm_fastbench.log"
        latest.unlink(missing_ok=True)
        try:
            latest.symlink_to(log_path.name)
        except OSError:
            pass
        old = sorted(LOG_DIR.glob("vllm_fastbench-*.log"))[:-self.KEEP_LOGS]
        for path in old:
            path.unlink(missing_ok=True)
        # FLASHINFER_SAMPLER=0: the sampling JIT hits the same cu13-vs-cccl
        # header clash as fused_moe on this stack; torch sampling is fine.
        env = dict(os.environ, **_cuda_env(), HF_HOME=HF_HOME,
                   HF_XET_HIGH_PERFORMANCE="1",
                   VLLM_USE_FLASHINFER_SAMPLER="0")
        logf = open(log_path, "w")
        cmd = self._vllm_cmd(repo, revision)
        log.info("launching vllm: %s", " ".join(cmd))
        s = self.state
        s.proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=logf,
                                  stdin=subprocess.DEVNULL,
                                  start_new_session=True)
        try:
            s.pgid = os.getpgid(s.proc.pid)
        except ProcessLookupError:
            s.pgid = None
        s.repo, s.revision = repo, revision
        s.ready = False
        s.load_error = ""

    def _wait_ready(self, timeout_s: int,
                    abort: threading.Event | None) -> bool:
        s = self.state
        t0 = time.time()
        url = f"http://127.0.0.1:{self.cfg.vllm_port}/v1/models"
        while time.time() - t0 < timeout_s:
            if abort is not None and abort.is_set():
                s.load_error = "aborted"
                self._kill()
                return False
            if s.proc is not None and s.proc.poll() is not None:
                s.load_error = (f"vllm exited {s.proc.returncode} | "
                                f"{self._log_tail()}")
                return False
            try:
                httpx.get(url, timeout=3)
            except httpx.HTTPError:
                time.sleep(5)
                continue
            # Port answers; only trust it once it really generates.
            if self.can_generate(s.repo or ""):
                s.ready = True
                s.loaded_at = time.time()
                log.info("model ready in %.0fs", time.time() - t0)
                return True
            time.sleep(5)
        s.load_error = f"not ready after {timeout_s}s | {self._log_tail()}"
        return False

    def _log_tail(self, max_chars: int = 800) -> str:
        path = self._log_path or (LOG_DIR / "vllm_fastbench.log")
        try:
            data = path.read_bytes()[-max_chars:]
            return data.decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _kill(self) -> None:
        s = self.state
        if s.adopted and s.proc is None:
            # No Popen to signal: the port owner is the only handle we have.
            self._free_port()
        if s.proc and s.proc.poll() is None:
            try:
                os.killpg(os.getpgid(s.proc.pid), signal.SIGTERM)
                s.proc.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(os.getpgid(s.proc.pid), signal.SIGKILL)
                except Exception:
                    log.warning("could not kill vllm group", exc_info=True)
        elif s.pgid is not None:
            # Leader died; TP/DP workers can outlive it holding all VRAM.
            try:
                os.killpg(s.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        s.proc = None
        s.pgid = None
        s.ready = False
        s.repo = None
        s.revision = None
        s.adopted = False

    def _free_port(self) -> None:
        """Kill whatever still owns the serving port.

        The service restarts (redeploys) without killing its vLLM child, so an
        orphan can hold the port across restarts. Taking ownership here is what
        makes 'is the slot serving the model I asked for' answerable."""
        port = self.cfg.vllm_port
        try:
            out = subprocess.run(["ss", "-tlnpH"], capture_output=True,
                                 text=True, timeout=30).stdout
        except Exception:
            out = ""
        pids: set[int] = set()
        for line in out.splitlines():
            if f":{port} " not in line and not line.strip().endswith(f":{port}"):
                continue
            for token in line.split("pid=")[1:]:
                digits = ""
                for ch in token:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    pids.add(int(digits))
        for pid in pids:
            if pid == os.getpid():
                continue
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                log.warning("killed stale port-%d owner pgid of pid %d",
                            port, pid)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if pids:
            time.sleep(3)

    def _sweep_gpus(self) -> None:
        """SIGKILL any compute process still resident on the GPUs (orphans
        from a crashed engine would fail the next init on free memory)."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30).stdout
            for pid_s in out.split():
                pid = int(pid_s)
                if pid == os.getpid():
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    log.warning("swept orphan GPU pid %d", pid)
                except ProcessLookupError:
                    pass
        except Exception:
            log.warning("gpu sweep failed", exc_info=True)

    def _prune_cache(self) -> None:
        """Keep the newest keep_snapshots model dirs (always the served one)."""
        hub = Path(HF_HOME) / "hub"
        if not hub.is_dir():
            return
        keep_dir = None
        if self.state.repo:
            keep_dir = "models--" + self.state.repo.replace("/", "--")
        dirs = [d for d in hub.iterdir()
                if d.is_dir() and d.name.startswith("models--")]
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        doomed = [d for d in dirs[self.cfg.keep_snapshots:]
                  if d.name != keep_dir]
        for d in doomed:
            log.info("pruning cached model %s", d.name)
            shutil.rmtree(d, ignore_errors=True)
