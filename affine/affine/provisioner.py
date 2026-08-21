"""GPU-machine lifecycle: keep duel + bench pods running evalsrv.

Two always-on machines are managed independently:
  * affine-eval  (role=duel)  — teacher + king + challenger, S* duels
  * affine-bench (role=bench) — single miner slot + SWE advisory suites

Providers (tried in `provider_order`) are encapsulated behind the `Provider`
interface so the manager holds one code path and each backend owns its own
provision/tunnel/terminate/push quirks:
  * LiumProvider   — the `lium` CLI (assumed installed + `lium init`-ed).
  * TargonProvider — the REST Workloads API (RENTAL type).

Each manager is called once per validator tick and NEVER blocks it:

    mgr.ensure()   # health-check; kick off provisioning in the background

Fault-tolerance contract (executors are assumed unreliable):
  * Provisioning (rent + tar upload + bootstrap + warmup, 30-60 min cold)
    runs in a daemon thread; ticks, weight-setting, and the watchdog keep
    running throughout.
  * `unhealthy_threshold` consecutive failed health checks ⇒ terminate +
    reprovision. Failed attempts back off exponentially (60s → 15 min cap).
  * A validator restart ADOPTS the existing pod (tunnel rebuilt from state,
    stale tunnel pid killed) instead of tearing down a healthy machine.
  * Termination verifies the provider actually released the pod before the
    state record is cleared, so a failed `lium rm` can't orphan a billing pod.
  * The price cap is enforced pre-rent: an executor/tier over budget is
    refused rather than rented.
  * Secrets are never placed on the remote command line — they are written to
    a 0600 env file on the pod and sourced by bootstrap.sh.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import shlex
import signal
import subprocess
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import httpx

log = logging.getLogger("affine.provisioner")

EVAL_POD_NAME = "affine-eval"
BENCH_POD_NAME = "affine-bench"
CHAT_POD_NAME = "affine-chat"
REMOTE_DIR = "/root/affine"
REMOTE_ENV_FILE = "/root/affine/.eval_env"
REMOTE_TAR = "/tmp/affine-src.tar.gz"

PROVISION_BACKOFF_BASE_S = 60
PROVISION_BACKOFF_CAP_S = 900

# Paths never worth shipping to the pod (bloat / secrets / host-only state).
# Prefer tar upload over `lium rsync` — rsync has been observed to drop *.py.
TAR_EXCLUDES = (".venv", ".venv-smoke", "logs", "state", ".git",
                "__pycache__", ".env", ".env.path", "affine.egg-info",
                "website/node_modules")

_SECRET_TOKENS: list[str] = []


def register_secret(value: str) -> None:
    """Redact this value from `_run` debug logging."""
    if value and value not in _SECRET_TOKENS:
        _SECRET_TOKENS.append(value)


def _redact(text: str) -> str:
    for s in _SECRET_TOKENS:
        text = text.replace(s, "***")
    return text


def _run(cmd: list[str], timeout: int = 300,
         input_text: str | None = None) -> subprocess.CompletedProcess:
    log.debug("run: %s", _redact(" ".join(shlex.quote(c) for c in cmd)))
    env = None
    # Doppler/pm2 may inject a stale LIUM_API_KEY that overrides the working
    # ~/.lium/config.ini profile and makes every `lium` call fail with
    # "Invalid API key". Prefer the on-disk CLI profile when present.
    if cmd and cmd[0] == "lium" and (Path.home() / ".lium" / "config.ini").is_file():
        env = os.environ.copy()
        env.pop("LIUM_API_KEY", None)
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, input=input_text, env=env)


def _kill_pid(pid: int) -> None:
    """Best-effort SIGTERM for a stale tunnel from a previous validator
    incarnation (spawned with start_new_session, so it outlives us)."""
    try:
        os.kill(pid, signal.SIGTERM)
        log.info("killed stale tunnel pid %d", pid)
    except ProcessLookupError:
        pass
    except Exception:
        log.warning("could not kill stale tunnel pid %d", pid, exc_info=True)


class Tunnel:
    """Keep a local-port forward to the pod's evalsrv alive."""

    def __init__(self, on_restart=None):
        self.proc: subprocess.Popen | None = None
        self.cmd: list[str] = []
        self._on_restart = on_restart

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def start(self, cmd: list[str]) -> None:
        self.stop()
        self.cmd = cmd
        log.info("starting tunnel: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)

    def ensure(self) -> None:
        if self.cmd and (self.proc is None or self.proc.poll() is not None):
            log.warning("tunnel died; restarting")
            self.start(self.cmd)
            if self._on_restart:
                self._on_restart(self.pid)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None
        self.cmd = []


# -- providers ------------------------------------------------------------------

class Provider(abc.ABC):
    name: str

    def __init__(self, cfg, em, repo_root: Path, pod_name: str):
        self.cfg = cfg
        self.em = em
        self.repo_root = repo_root
        self.pod_name = pod_name

    @abc.abstractmethod
    def provision(self) -> dict | None:
        """Rent a pod. Return a machine record (must include 'provider'), or
        None if unavailable / over budget."""

    @abc.abstractmethod
    def tunnel_cmd(self, machine: dict, port: int) -> list[str]:
        ...

    @abc.abstractmethod
    def terminate(self, machine: dict) -> bool:
        """Release the pod. Return True only when the provider confirms the
        pod is gone (so the caller may clear state)."""

    @abc.abstractmethod
    def push_and_bootstrap(self, machine: dict, env_file_contents: str) -> None:
        """Upload the repo, write the secret env file, launch bootstrap.sh.
        Raise on failure."""

    # shared helpers -----------------------------------------------------------
    def _rsync_args(self) -> list[str]:
        args = []
        for pat in TAR_EXCLUDES:
            args += ["--exclude", pat]
        return args


class LiumProvider(Provider):
    name = "lium"
    # How many distinct executors to try before giving up on this provider.
    # `lium exec` / `lium up` have been observed to hang on a live pod — when
    # that happens we rm and rent the next cheapest under the cap.
    MAX_EXECUTOR_ATTEMPTS = 3

    def __init__(self, cfg, em, repo_root: Path, pod_name: str):
        super().__init__(cfg, em, repo_root, pod_name)
        # Executors that failed up/ssh/bootstrap this process lifetime — never
        # re-rent them until the validator restarts.
        self._bad_executors: set[str] = set()

    def remember_bad(self, machine: dict) -> None:
        ex = machine.get("executor")
        if ex:
            self._bad_executors.add(str(ex))
            log.info("lium: blacklisting executor %s after failure", ex)

    def provision(self) -> dict | None:
        skipped: set[str] = set(self._bad_executors)
        for attempt in range(self.MAX_EXECUTOR_ATTEMPTS):
            pick = self._cheapest_executor_under_cap(exclude=skipped)
            if pick is None:
                return None
            huid, price, gpu = pick
            skipped.add(huid)
            machine = self._rent(huid, price, gpu)
            if machine is not None:
                return machine
            self._bad_executors.add(huid)
            log.warning("lium executor %s unusable; trying another machine (%d/%d)",
                        huid, attempt + 1, self.MAX_EXECUTOR_ATTEMPTS)
        return None

    def _rent(self, huid: str, price: float, gpu: str) -> dict | None:
        # Drop a leftover pod with our name before renting (failed prior attempt).
        _run(["lium", "rm", self.pod_name], timeout=120, input_text="y\n")
        cmd = ["lium", "up", huid, "--name", self.pod_name, "--yes"]
        if self.em.lium_template_id:
            cmd += ["--template_id", self.em.lium_template_id]
        # Hard timeout: `lium up` has hung indefinitely on some executors.
        # On timeout, still check whether the named pod appeared — Lium often
        # finishes the rent after the CLI stops responding.
        timed_out = False
        try:
            p = _run(cmd, timeout=240)
            out = (p.stdout or "") + (p.stderr or "")
            if p.returncode != 0 or "not found" in out.lower():
                log.error("lium up failed on %s: %s", huid, _redact(out[-800:]))
                # Pod may still exist under our name from a partial up.
                if not self._resolve_ssh(self.pod_name):
                    return None
                log.warning("lium up reported failure but pod %s exists; adopting",
                            self.pod_name)
            else:
                log.info("lium pod %s created on %s (%s, $%.2f/hr): %s",
                         self.pod_name, huid, gpu, price,
                         _redact(out.strip()[-300:]))
        except subprocess.TimeoutExpired:
            timed_out = True
            log.warning("lium up timed out on %s; checking if pod appeared", huid)

        ssh = None
        for _ in range(6):
            ssh = self._resolve_ssh(self.pod_name)
            if ssh:
                break
            time.sleep(10)
        if not ssh:
            log.error("no ssh for %s after up (timeout=%s); releasing",
                      self.pod_name, timed_out)
            _run(["lium", "rm", self.pod_name], timeout=120, input_text="y\n")
            return None
        # Probe SSH before returning — a pod that never answers is useless.
        # Retry: fresh pods take a minute before sshd is up.
        probe_ok = False
        for _ in range(6):
            probe = _ssh_run(ssh, "echo SSH_OK", timeout=45)
            if probe.returncode == 0 and "SSH_OK" in (probe.stdout or ""):
                probe_ok = True
                break
            time.sleep(10)
        if not probe_ok:
            log.error("ssh probe failed for %s (%s); releasing and trying another",
                      self.pod_name, ssh)
            _run(["lium", "rm", self.pod_name], timeout=120, input_text="y\n")
            return None
        # Verify the pod actually exposes the GPUs the listing promised —
        # executors have delivered fewer (e.g. 1× on a "2×" listing), which
        # passes health checks but breaks every TP>=2 vllm serve.
        smi = _ssh_run(ssh, "nvidia-smi -L | wc -l", timeout=45)
        try:
            n_gpus = int((smi.stdout or "0").strip().splitlines()[-1])
        except (ValueError, IndexError):
            n_gpus = 0
        if n_gpus < self.em.gpu_count:
            log.error("pod %s on %s exposes %d GPUs, need %d; releasing and "
                      "blacklisting", self.pod_name, huid, n_gpus,
                      self.em.gpu_count)
            _run(["lium", "rm", self.pod_name], timeout=120, input_text="y\n")
            return None
        return {"provider": "lium", "id": self.pod_name, "executor": huid,
                "gpu": gpu, "price_per_hour": price, "ssh": ssh,
                "created_at": time.time()}

    def _resolve_ssh(self, name: str) -> str | None:
        """Return `user@host -p PORT` for a running pod (from `lium ps --json`)."""
        p = _run(["lium", "ps", "--format", "json"], timeout=120)
        if p.returncode != 0:
            return None
        try:
            rows = json.loads(p.stdout or "[]")
        except json.JSONDecodeError:
            return None
        for row in rows if isinstance(rows, list) else []:
            if row.get("name") != name and row.get("huid") != name:
                continue
            cmd = (row.get("ssh_cmd") or "").strip()
            if cmd.startswith("ssh "):
                # "ssh root@host -p 40049" → "root@host -p 40049"
                return cmd[4:].strip()
            ip = row.get("ip")
            ports = row.get("ports") or {}
            port = ports.get("22") or ports.get(22)
            if ip and port:
                return f"root@{ip} -p {port}"
        return None

    def _cheapest_executor_under_cap(
            self, exclude: set[str] | None = None
            ) -> tuple[str, float, str] | None:
        """Pick the cheapest node under the cap, walking gpu_types preference
        order. Uses `lium ls --format json` so we get a real huid/UUID.
        `exclude` skips executors that already failed up/ssh/bootstrap."""
        exclude = exclude or set()
        gpu_types = list(getattr(self.em, "gpu_types", None) or [self.em.gpu_type])
        for gpu in gpu_types:
            # NOTE: combining `--gpu` and `--count` in `lium ls` has been
            # observed to return [] even when matching executors exist, so we
            # filter gpu_count locally instead (>= wanted; the price cap keeps
            # oversized pods out).
            p = _run(["lium", "ls", "--gpu", gpu, "--format", "json"],
                     timeout=120)
            if p.returncode != 0:
                log.error("lium ls --gpu %s failed: %s",
                          gpu, _redact(p.stderr[-300:]))
                continue
            ranked = _rank_lium_executors(p.stdout, self.em.max_price_per_hour,
                                          min_gpu_count=self.em.gpu_count)
            for huid, price in ranked:
                if huid in exclude:
                    continue
                log.info("selected lium %s executor %s at $%.2f/hr", gpu, huid, price)
                return huid, price, gpu
            log.info("no %s x%d under $%.2f/hr (after excludes=%s)",
                     gpu, self.em.gpu_count, self.em.max_price_per_hour,
                     sorted(exclude)[:3])
        log.error("no executor under $%.2f/hr for gpu_types=%s x%d",
                  self.em.max_price_per_hour, gpu_types, self.em.gpu_count)
        return None

    def tunnel_cmd(self, machine: dict, port: int) -> list[str]:
        # Prefer SSH -L: evalsrv binds loopback on the pod and is NOT in the
        # Lium "exposed ports" list, so `lium port-forward` rejects it.
        ssh = machine.get("ssh") or ""
        if ssh:
            # ssh field is "user@host -p PORT" (no leading `ssh`).
            parts = shlex.split(ssh)
            return ["ssh", "-N",
                    # accept-new: Lium reuses host:port across pod swaps; a
                    # fixed known_hosts entry then fails closed as a MITM.
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "ServerAliveInterval=30",
                    "-o", "ExitOnForwardFailure=yes",
                    "-L", f"{port}:127.0.0.1:{port}",
                    *parts]
        return ["lium", "port-forward", machine["id"], str(port)]

    def terminate(self, machine: dict) -> bool:
        name = machine.get("id") or self.pod_name
        # -y: non-interactive (stdin "y\n" is unreliable across lium versions).
        p = _run(["lium", "rm", name, "-y"], timeout=120)
        out = ((p.stdout or "") + (p.stderr or "")).strip()
        if p.returncode != 0:
            # lium exits non-zero when nothing matched — pod already gone,
            # so billing is already stopped. Treat as confirmed release
            # (mirrors targon DELETE 404). Otherwise we wedge forever on
            # terminate_failed and never rent a replacement.
            if "No pods match targets" in out:
                log.info("lium rm %s: already absent; treating as terminated",
                         name)
                return True
            log.error("lium rm %s failed (pod may still be billing): %s",
                      name, _redact(out[-300:]))
            return False
        return True

    def push_and_bootstrap(self, machine: dict, env_file_contents: str) -> None:
        """Upload + bootstrap over direct SSH.

        Prefer SSH over `lium exec` / `lium scp` — those have hung on live
        pods (control plane sticky). On failure the manager terminates and
        rents another executor (see MAX_EXECUTOR_ATTEMPTS).
        """
        ssh = machine.get("ssh") or ""
        if not ssh:
            raise RuntimeError("machine record missing ssh; cannot bootstrap")
        tar_path = _pack_affine_tar(self.repo_root)
        try:
            p = _scp_to(ssh, tar_path, REMOTE_TAR, timeout=1200)
            if p.returncode != 0:
                raise RuntimeError(
                    f"scp tar failed: {_redact((p.stderr or p.stdout or '')[-500:])}")
        finally:
            try:
                tar_path.unlink(missing_ok=True)
            except Exception:
                pass
        extract = (
            f"mkdir -p {REMOTE_DIR} && tar xzf {REMOTE_TAR} -C {REMOTE_DIR} "
            f"&& rm -f {REMOTE_TAR} && "
            f"test $(find {REMOTE_DIR}/affine {REMOTE_DIR}/evalsrv "
            f"-name '*.py' 2>/dev/null | wc -l) -ge 10 && echo EXTRACT_OK"
        )
        p = _ssh_run(ssh, extract, timeout=300)
        if p.returncode != 0 or "EXTRACT_OK" not in (p.stdout or ""):
            raise RuntimeError(
                f"extract on pod failed: {_redact((p.stderr or p.stdout or '')[-500:])}")
        with tempfile.NamedTemporaryFile("w", delete=False, prefix="affine-eval-env-") as f:
            f.write(env_file_contents)
            local_env = Path(f.name)
        try:
            os.chmod(local_env, 0o600)
            p = _scp_to(ssh, local_env, REMOTE_ENV_FILE, timeout=120)
            if p.returncode != 0:
                raise RuntimeError(
                    f"writing env file failed: {_redact((p.stderr or p.stdout or '')[-300:])}")
        finally:
            local_env.unlink(missing_ok=True)
        # The explicit ( ... & ) subshell fully detaches bootstrap from the
        # ssh session's pipes. Backgrounding the bare `cd && nohup ...` list
        # instead leaves an intermediate subshell holding the ssh stdout pipe
        # while it waits on the never-exiting supervise loop, so sshd never
        # sees EOF and the launch "times out" even though bootstrap started
        # (bash-version dependent: some images optimize the fork away).
        boot = (f"cd {REMOTE_DIR} && (nohup bash evalsrv/bootstrap.sh "
                f"</dev/null >> /root/bootstrap.log 2>&1 &) "
                f"&& echo BOOTSTRAP_STARTED")
        p = _ssh_run(ssh, boot, timeout=60)
        if "BOOTSTRAP_STARTED" not in (p.stdout or ""):
            raise RuntimeError(
                f"bootstrap launch failed: {_redact((p.stdout or '')[-300:])} "
                f"{_redact((p.stderr or '')[-300:])}")


class TargonProvider(Provider):
    name = "targon"
    BASE = "https://api.targon.com/tha/v2"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.cfg.secrets.targon_api_key}",
                "Content-Type": "application/json"}

    def provision(self) -> dict | None:
        if not self.cfg.secrets.targon_api_key:
            log.warning("no TARGON_API_KEY; skipping targon provider")
            return None
        if not self._resource_under_cap():
            return None
        body = {
            "name": self.pod_name,
            "image": self.em.targon_image,
            "resource_name": self.em.targon_resource,
            "type": "RENTAL",
            # Only DIRECT ssh (22) — evalsrv is reached through the ssh tunnel,
            # never exposed as a public PROXIED port.
            "ports": [{"port": 22, "protocol": "TCP", "routing": "DIRECT"}],
        }
        with httpx.Client(timeout=60) as http:
            r = http.post(f"{self.BASE}/workloads", headers=self._headers(), json=body)
            r.raise_for_status()
            uid = r.json().get("uid") or r.json().get("workload_uid")
            if not uid:
                log.error("targon workload create returned no uid: %s", r.text[:300])
                return None
            r = http.post(f"{self.BASE}/workloads/{uid}/deploy", headers=self._headers())
            r.raise_for_status()
            for _ in range(60):
                time.sleep(10)
                g = http.get(f"{self.BASE}/workloads/{uid}", headers=self._headers())
                if g.status_code != 200:
                    continue
                d = g.json()
                ssh = d.get("ssh") or d.get("ssh_host") or ""
                if d.get("state", "").upper() in {"RUNNING", "READY"} and ssh:
                    return {"provider": "targon", "id": uid, "ssh": ssh,
                            "created_at": time.time()}
        log.error("targon workload %s never reached RUNNING with ssh", uid)
        return None

    def _resource_under_cap(self) -> bool:
        try:
            with httpx.Client(timeout=30) as http:
                r = http.get(f"{self.BASE}/inventory?type=rental&gpu=true")
                r.raise_for_status()
                for item in r.json():
                    if item.get("name") == self.em.targon_resource:
                        price = float(item.get("cost_per_hour", 1e9))
                        if price > self.em.max_price_per_hour:
                            log.error("targon %s at $%.2f/hr exceeds cap $%.2f",
                                      self.em.targon_resource, price,
                                      self.em.max_price_per_hour)
                            return False
                        return True
        except Exception:
            log.warning("targon inventory price check failed; refusing to rent",
                        exc_info=True)
            return False
        log.error("targon resource %s not found in inventory", self.em.targon_resource)
        return False

    def tunnel_cmd(self, machine: dict, port: int) -> list[str]:
        return ["ssh", "-N",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=30",
                "-o", "ExitOnForwardFailure=yes",
                "-L", f"{port}:127.0.0.1:{port}",
                machine["ssh"]]

    def terminate(self, machine: dict) -> bool:
        uid = machine.get("id", "")
        try:
            with httpx.Client(timeout=60) as http:
                r = http.delete(f"{self.BASE}/workloads/{uid}", headers=self._headers())
                if r.status_code not in (200, 202, 204, 404):
                    log.error("targon delete %s returned %s (may still bill)",
                              uid, r.status_code)
                    return False
                return True
        except Exception:
            log.exception("targon terminate failed for %s", uid)
            return False

    def push_and_bootstrap(self, machine: dict, env_file_contents: str) -> None:
        ssh = machine["ssh"]
        p = _run(["rsync", "-az", *self._rsync_args(),
                  "-e", "ssh -o StrictHostKeyChecking=no",
                  str(self.repo_root) + "/", f"{ssh}:{REMOTE_DIR}/"],
                 timeout=1200)
        if p.returncode != 0:
            raise RuntimeError(f"rsync to targon rental failed: {_redact(p.stderr[-500:])}")
        p = _run(["ssh", "-o", "StrictHostKeyChecking=no", ssh,
                  f"umask 077 && cat > {REMOTE_ENV_FILE}"],
                 timeout=120, input_text=env_file_contents)
        if p.returncode != 0:
            raise RuntimeError(f"writing env file failed: {_redact(p.stderr[-300:])}")
        p = _run(["ssh", "-o", "StrictHostKeyChecking=no", ssh,
                  f"cd {REMOTE_DIR} && nohup bash evalsrv/bootstrap.sh "
                  f">> /root/bootstrap.log 2>&1 & echo BOOTSTRAP_STARTED"],
                 timeout=300)
        if "BOOTSTRAP_STARTED" not in (p.stdout or ""):
            raise RuntimeError(
                f"bootstrap launch failed: {_redact(p.stdout[-300:])} {_redact(p.stderr[-300:])}")


def _ssh_target(ssh: str) -> tuple[str, int]:
    """Parse `user@host -p PORT` → (user@host, port)."""
    parts = shlex.split(ssh)
    host = parts[0]
    port = 22
    if "-p" in parts:
        i = parts.index("-p")
        if i + 1 < len(parts):
            port = int(parts[i + 1])
    return host, port


_SSH_BASE = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=30",
]


def _ssh_run(ssh: str, remote_cmd: str, timeout: int = 300,
             input_text: str | None = None) -> subprocess.CompletedProcess:
    """Run a command on the pod over direct SSH (not `lium exec`)."""
    host, port = _ssh_target(ssh)
    cmd = ["ssh", *_SSH_BASE, "-p", str(port), host,
           f"bash -lc {shlex.quote(remote_cmd)}"]
    return _run(cmd, timeout=timeout, input_text=input_text)


def _scp_to(ssh: str, local: Path, remote: str,
            timeout: int = 600) -> subprocess.CompletedProcess:
    """scp a local file to the pod over direct SSH (not `lium scp`)."""
    host, port = _ssh_target(ssh)
    cmd = ["scp", *_SSH_BASE, "-P", str(port),
           str(local), f"{host}:{remote}"]
    return _run(cmd, timeout=timeout)


def _rank_lium_executors(ls_output: str, cap: float,
                         min_gpu_count: int | None = None
                         ) -> list[tuple[str, float]]:
    """Parse `lium ls --format json` → [(id, $/hr), ...] cheapest first under cap.

    `min_gpu_count` drops JSON rows with fewer GPUs (the CLI's own
    --gpu/--count combo filter is unreliable). Falls back to a table scrape
    for older smoke fixtures / CLI versions, but only accepts huid-looking
    tokens (never Config labels like '8×H200')."""
    import re
    text = (ls_output or "").strip()
    if not text:
        return []
    ranked: list[tuple[str, float]] = []
    if text.startswith("["):
        try:
            rows = json.loads(text)
        except json.JSONDecodeError:
            rows = None
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    price = float(row.get("price_per_hour")
                                  or row.get("price_total") or 1e9)
                except (TypeError, ValueError):
                    continue
                # Prefer UUID — `lium up <huid>` has been observed to 404 even
                # when the huid appears in `lium ls --format json`; UUID works.
                ident = str(row.get("id") or row.get("huid") or "").strip()
                if not ident or price > cap:
                    continue
                if min_gpu_count is not None:
                    try:
                        if int(row.get("gpu_count") or 0) < min_gpu_count:
                            continue
                    except (TypeError, ValueError):
                        continue
                ranked.append((ident, price))
            ranked.sort(key=lambda x: x[1])
            return ranked

    # Table fallback (smoke tests / legacy CLI). Prefer a huid-shaped token.
    price_re = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")
    huid_re = re.compile(r"^[a-z]+-[a-z]+-[a-z0-9]+$", re.I)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("index", "id", "#", "config")):
            continue
        prices = [float(m) for m in price_re.findall(line)]
        if not prices:
            continue
        price = max(prices)
        huids = [t for t in re.split(r"\s+", line) if huid_re.match(t)]
        if not huids or price > cap:
            continue
        ranked.append((huids[0], price))
    ranked.sort(key=lambda x: x[1])
    return ranked


def _parse_cheapest_lium(ls_output: str, cap: float) -> tuple[str | None, float]:
    """Back-compat helper for smoke tests: cheapest (id, price) under cap."""
    ranked = _rank_lium_executors(ls_output, cap)
    if not ranked:
        return None, float("inf")
    return ranked[0][0], ranked[0][1]


def _pack_affine_tar(repo_root: Path) -> Path:
    """Build a gzipped tar of the affine package for upload to the eval pod."""
    exclude_names = set(TAR_EXCLUDES)

    def _filter(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(ti.name).parts
        if any(p in exclude_names for p in parts):
            return None
        if ti.name.endswith((".pyc", ".pyo")):
            return None
        return ti

    fd, name = tempfile.mkstemp(prefix="affine-src-", suffix=".tar.gz")
    os.close(fd)
    path = Path(name)
    with tarfile.open(path, "w:gz") as tf:
        tf.add(repo_root, arcname=".", filter=_filter)
    return path


class MachineManager:
    """Keep one named GPU pod healthy (duel eval or SWE bench)."""

    def __init__(self, cfg, state, repo_root: Path, *,
                 em, state_attr: str, pod_name: str, role: str,
                 label: str):
        self.cfg = cfg
        self.em = em
        self.state = state
        self.repo_root = repo_root
        self.state_attr = state_attr
        self.pod_name = pod_name
        self.role = role
        self.label = label
        self.tunnel = Tunnel(on_restart=self._record_tunnel_pid)
        self._providers = {p.name: p for p in (
            LiumProvider(cfg, self.em, repo_root, pod_name),
            TargonProvider(cfg, self.em, repo_root, pod_name))}
        self._unhealthy = 0
        self._last_check = 0.0
        self._last_check_ok = False
        self._last_health_reason = ""
        self._provision_thread: threading.Thread | None = None
        self._provision_failures = 0
        self._next_provision_at = 0.0
        register_secret(cfg.secrets.hf_token)
        register_secret(cfg.secrets.eval_token)
        register_secret(cfg.secrets.targon_api_key)

    def _get_machine(self) -> dict:
        return getattr(self.state, self.state_attr) or {}

    def _set_machine(self, machine: dict) -> None:
        setattr(self.state, self.state_attr, machine)

    @property
    def local_url(self) -> str:
        return f"http://127.0.0.1:{self.em.port}"

    def _provider_for(self, machine: dict) -> Provider | None:
        return self._providers.get(machine.get("provider", ""))

    def _record_tunnel_pid(self, pid: int | None) -> None:
        machine = self._get_machine()
        if machine and pid:
            machine["tunnel_pid"] = pid
            self._set_machine(machine)
            self.state.flush()

    # -- public entry ----------------------------------------------------------
    def ensure(self) -> bool:
        """Health-check / (re)provision. Non-blocking; returns True only when
        the eval server is reachable and healthy right now."""
        if self._provisioning():
            return False

        now = time.monotonic()
        if now - self._last_check < self.em.health_interval_s:
            return self._last_check_ok
        self._last_check = now

        machine = self._get_machine()
        # A pod whose termination we could not confirm: retry releasing it
        # before renting anything new, so we never double-bill.
        if machine.get("terminate_failed"):
            self._last_check_ok = False
            self._terminate()
            machine = self._get_machine()
            if machine.get("terminate_failed"):
                return False
            # Cleared — fall through (usually into provisioning).

        if not machine:
            self._last_check_ok = False
            self._start_provisioning()
            return False

        self._adopt_tunnel_if_needed()
        self.tunnel.ensure()
        if self._healthy():
            if self._unhealthy:
                log.info("%s machine healthy again after %d failed checks",
                         self.label, self._unhealthy)
            self._unhealthy = 0
            self._provision_failures = 0
            self._last_check_ok = True
            return True

        self._last_check_ok = False
        self._unhealthy += 1
        log.warning("%s machine unhealthy (%d/%d): %s",
                    self.label, self._unhealthy, self.em.unhealthy_threshold,
                    self._last_health_reason or "unknown")
        if self._unhealthy >= self.em.unhealthy_threshold:
            log.error("%s machine failed %d consecutive health checks; reprovisioning",
                      self.label, self._unhealthy)
            self._terminate()
            self._unhealthy = 0
            self._start_provisioning()
        elif self._unhealthy % 3 == 0:
            # Cheap recovery first: a dead evalsrv/bootstrap on a live pod is
            # fixable in ~2 min over SSH, vs ~30-60 min for terminate + rent +
            # re-download weights. Only reprovision when SSH recovery keeps
            # failing all the way to the threshold.
            self._soft_restart()
        return False

    def _soft_restart(self) -> None:
        """Relaunch bootstrap over direct SSH if evalsrv died on a live pod.

        No-op when the server or a bootstrap supervisor is already running
        (then the problem is elsewhere — warmup, tunnel, model load — and
        killing processes would only make it worse)."""
        machine = self._get_machine()
        ssh = machine.get("ssh") or ""
        if not ssh:
            return
        # The explicit ( ... & ) subshell detaches bootstrap from the ssh
        # session's pipes (see push_and_bootstrap for the full story).
        # The [b]racket trick keeps pgrep -f from matching the `bash -lc`
        # wrapper that carries this very script in its own cmdline — without
        # it both checks always self-match and the relaunch branch is
        # unreachable (soft restart silently becomes a permanent no-op).
        remote = (
            "if pgrep -f '[p]ython -m evalsrv.server' >/dev/null "
            "|| pgrep -f '[e]valsrv/bootstrap.sh' >/dev/null; then "
            "echo ALREADY_RUNNING; "
            f"elif [ -d {REMOTE_DIR} ]; then "
            f"cd {REMOTE_DIR} && (nohup bash evalsrv/bootstrap.sh "
            "</dev/null >> /root/bootstrap.log 2>&1 &) && echo RELAUNCHED; "
            "else echo NO_REPO; fi"
        )
        try:
            p = _ssh_run(ssh, remote, timeout=60)
            out = (p.stdout or "").strip()
            log.warning("%s soft-restart attempt: %s", self.label,
                        out or _redact((p.stderr or "")[-200:]))
        except Exception:
            log.warning("%s soft-restart ssh failed", self.label, exc_info=True)

    def is_healthy_now(self) -> bool:
        """Live probe, bypassing the check-interval cache. Used by the
        validator to decide whether a mid-duel failure was the machine's
        fault (requeue without penalty) or the challenger's."""
        if self._provisioning():
            return False
        ok = self._healthy()
        self._last_check_ok = ok
        return ok

    def _healthy(self) -> bool:
        """Probe /health; on failure leave WHY in _last_health_reason so the
        unhealthy log line (and anyone tailing it) can tell a dead tunnel
        from a sick server without ssh'ing in."""
        try:
            headers = {}
            if self.cfg.secrets.eval_token:
                headers["X-Affine-Token"] = self.cfg.secrets.eval_token
            r = httpx.get(f"{self.local_url}/health", timeout=10, headers=headers)
            if r.status_code != 200:
                self._last_health_reason = f"HTTP {r.status_code}"
                return False
            body = r.json()
            if not body.get("ok", False):
                self._last_health_reason = f"ok=false: {json.dumps(body)[:200]}"
                return False
            # Role mismatch means we hit the wrong pod through a stale tunnel.
            role = body.get("role")
            if role and role != self.role:
                log.warning("%s health role mismatch: want %s got %s",
                            self.label, self.role, role)
                self._last_health_reason = f"role mismatch: want {self.role} got {role}"
                return False
            # Serving-stack disclosure: surface the pod's installed
            # vllm/transformers/torch versions through state → snapshot API so
            # miners can pre-flight checkpoints against the live stack.
            versions = body.get("versions")
            machine = self._get_machine()
            if versions and machine and machine.get("versions") != versions:
                machine["versions"] = versions
                self._set_machine(machine)
                self.state.flush()
            self._last_health_reason = ""
            return True
        except Exception as e:
            # ConnectError here usually means the SSH tunnel is down, not the pod.
            self._last_health_reason = f"{type(e).__name__}: {e}"
            return False

    def status_word(self) -> str:
        """One-word status for heartbeat logs: ok | provisioning | down."""
        if self._provisioning():
            return "provisioning"
        return "ok" if self._last_check_ok else "down"

    # -- tunnel adoption (validator restart survival) -----------------------------
    def _adopt_tunnel_if_needed(self) -> None:
        if self.tunnel.cmd:
            return
        machine = self._get_machine()
        stale_pid = machine.get("tunnel_pid")
        if stale_pid:
            _kill_pid(int(stale_pid))
            time.sleep(1)
        log.info("adopting existing %s machine after restart: %s",
                 self.label, {k: machine.get(k) for k in ("provider", "id")})
        self._start_tunnel(machine)

    def _start_tunnel(self, machine: dict) -> None:
        provider = self._provider_for(machine)
        if provider is None:
            raise ValueError(f"unknown provider {machine.get('provider')!r}")
        self.tunnel.start(provider.tunnel_cmd(machine, self.em.port))
        machine["tunnel_pid"] = self.tunnel.pid
        self._set_machine(machine)
        self.state.flush()

    # -- provisioning (background thread) -------------------------------------------
    def _provisioning(self) -> bool:
        return self._provision_thread is not None and self._provision_thread.is_alive()

    def _start_provisioning(self) -> None:
        if self._provisioning():
            return
        now = time.monotonic()
        if now < self._next_provision_at:
            log.info("%s provisioning backed off for another %.0fs",
                     self.label, self._next_provision_at - now)
            return
        self._provision_thread = threading.Thread(
            target=self._provision_and_record, daemon=True,
            name=f"{self.label}-provisioner")
        self._provision_thread.start()

    def _provision_and_record(self) -> None:
        try:
            ok = self._provision()
        except Exception:
            log.exception("%s provisioning thread crashed", self.label)
            ok = False
        if ok:
            self._provision_failures = 0
            self._next_provision_at = 0.0
            self._last_check_ok = True
        else:
            self._provision_failures += 1
            backoff = min(PROVISION_BACKOFF_CAP_S,
                          PROVISION_BACKOFF_BASE_S * 2 ** (self._provision_failures - 1))
            self._next_provision_at = time.monotonic() + backoff
            log.error("%s provisioning attempt %d failed; next attempt in %ds",
                      self.label, self._provision_failures, backoff)
        self.state.set_phase("tick")

    def _env_file_contents(self) -> str:
        kv = {
            "HF_TOKEN": self.cfg.secrets.hf_token,
            "AFFINE_EVAL_PORT": str(self.em.port),
            "AFFINE_EVAL_TOKEN": self.cfg.secrets.eval_token,
            "AFFINE_ROLE": self.role,
        }
        return "".join(f"export {k}={shlex.quote(v)}\n" for k, v in kv.items() if v)

    def _provision(self) -> bool:
        self.state.set_phase(f"provisioning_{self.label}_machine")
        # Per-provider retries: a sticky Lium executor (hung exec/up) must not
        # burn the whole provider — blacklist it and rent another machine.
        for name in self.em.provider_order:
            provider = self._providers.get(name)
            if provider is None:
                log.error("unknown provider %r in provider_order", name)
                continue
            attempts = getattr(provider, "MAX_EXECUTOR_ATTEMPTS", 1)
            for attempt in range(attempts):
                machine: dict | None = None
                try:
                    machine = provider.provision()
                    if not machine:
                        break
                    self._set_machine(machine)
                    self.state.flush()
                    self._start_tunnel(machine)
                    self.state.set_phase(f"bootstrapping_{self.label}_machine")
                    provider.push_and_bootstrap(machine, self._env_file_contents())
                    if self._wait_ready():
                        log.info("%s machine ready via %s: %s", self.label, name,
                                 {k: machine.get(k) for k in ("provider", "id")})
                        return True
                    log.error("%s machine on %s never became healthy; "
                              "trying another (%d/%d)",
                              self.label, name, attempt + 1, attempts)
                except Exception:
                    log.exception("%s provisioning via %s failed (attempt %d/%d)",
                                  self.label, name, attempt + 1, attempts)
                if machine is not None and hasattr(provider, "remember_bad"):
                    provider.remember_bad(machine)
                self._terminate()
        log.error("%s: all providers failed", self.label)
        return False

    def _wait_ready(self) -> bool:
        deadline = time.monotonic() + self.em.provision_timeout_s
        while time.monotonic() < deadline:
            self.tunnel.ensure()
            if self._healthy():
                return True
            time.sleep(30)
        return False

    def _terminate(self) -> None:
        machine = self._get_machine()
        self.tunnel.stop()
        if not machine:
            return
        provider = self._provider_for(machine)
        if provider is None:
            log.error("cannot terminate %s machine with unknown provider %r; "
                      "leaving state so it is retried, not orphaned",
                      self.label, machine.get("provider"))
            return
        if provider.terminate(machine):
            self._set_machine({})
            self.state.flush()
        else:
            # Keep the record and mark it dead so ensure() retries termination
            # on the next tick instead of renting a second (double-billed) pod.
            machine["terminate_failed"] = True
            self._set_machine(machine)
            self.state.flush()
            log.error("%s termination unconfirmed; will retry before provisioning anew",
                      self.label)


def EvalMachineManager(cfg, state, repo_root: Path) -> MachineManager:
    """Always-on duel eval pod (teacher + king + challenger)."""
    return MachineManager(
        cfg, state, repo_root,
        em=cfg.eval_machine,
        state_attr="eval_machine",
        pod_name=EVAL_POD_NAME,
        role="duel",
        label="eval",
    )


def BenchMachineManager(cfg, state, repo_root: Path) -> MachineManager:
    """Always-on dedicated SWE bench pod (no teacher)."""
    return MachineManager(
        cfg, state, repo_root,
        em=cfg.bench_machine,
        state_attr="bench_machine",
        pod_name=BENCH_POD_NAME,
        role="bench",
        label="bench",
    )


def ChatMachineManager(cfg, state, repo_root: Path) -> MachineManager:
    """Always-on public chat pod serving the current king (never scores)."""
    return MachineManager(
        cfg, state, repo_root,
        em=cfg.chat_machine,
        state_attr="chat_machine",
        pod_name=CHAT_POD_NAME,
        role="chat",
        label="chat",
    )
