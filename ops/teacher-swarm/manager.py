#!/usr/bin/env python3
"""Teacher-swarm manager: rent -> bootstrap -> probe -> heal -> state.json.

Reconcile loop (default every 30 s):
  1. List swarm pods (name prefix guard) and probe every replica endpoint.
  2. SSH-bootstrap pods that are reachable but not yet serving.
  3. Replace pods that are dark past the grace window or failed bootstrap
     (rm + executor blacklist + re-rent).
  4. Top up each type to its target while stock, price cap and the global
     budget allow. Shrink when over target.
  5. Write state/state.json: healthy replica URLs for the router.

Run:  python manager.py            (daemon)
      python manager.py --once     (single reconcile, for tests)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import threading
import time
import tomllib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import httpx

import lium_api

HERE = Path(__file__).resolve().parent
STATE_DIR = HERE / "state"
STATE_JSON = STATE_DIR / "state.json"
PODS_JSON = STATE_DIR / "pods.json"          # manager memory across restarts
BLACKLIST = STATE_DIR / "blacklist.txt"      # executor ids that burned us
KNOWN_HOSTS = STATE_DIR / "known_hosts"      # host:port reuse => key churn
ENV_FILE = HERE / ".swarm_env"               # HF_TOKEN=..., SWARM_KEY=...
ECHO_PLUGIN_DIR = HERE / "echo_cache_plugin"  # vLLM echo prefix-cache plugin

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
]


def log(msg: str) -> None:
    print(f"[swarm-mgr] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
          f"{msg}", flush=True)


def load_env_file() -> dict[str, str]:
    out: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


@dataclass
class TypePlan:
    name: str
    match: str
    gpu_count: int
    tp: int
    replicas: int
    target: int
    max_price: float
    est_tps: float
    max_num_batched_tokens: int
    gpu_util: float
    # Existing non-swarm pods (exact names) managed as members of this type.
    # They count toward target; lium_api.remove refuses non-prefixed names,
    # so the manager can bootstrap/heal them but never delete them.
    adopt: list[str] = field(default_factory=list)
    # When false the manager still boots/heals the box but the router never
    # sees its URLs. Used for eval-tune sandboxes under live load tests.
    advertise: bool = True
    # Extra `vllm serve` argv, space-separated. Empty = stock flags.
    vllm_extra: str = ""
    # Stand-in: this type is wanted (target, standin_max_price) only while
    # `standin_for` has no healthy replica; once that type has been healthy
    # for standin_release_min the effective target drops to 0 and the boxes
    # are released as "over target". Replaces hand-editing targets/caps
    # during a B200 x8 drought (2026-09-17) and undoing them by hand later.
    standin_for: str = ""
    standin_max_price: float = 0.0


@dataclass
class Config:
    model: str
    vllm_version: str
    pod_prefix: str
    template_id: str
    budget_usd_hr: float
    ttl_hours: int
    bootstrap_timeout_min: int
    unreachable_grace_min: int
    min_balance_usd: float
    max_model_len: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    # Type names tried in order when a target>0 type has no rentable stock
    # and the swarm holds no pod at all (2026-09-05: the single b200-8x died
    # 22:15 UTC, no 8x B200 <= $48 was in stock, and the manager sat at
    # "0 healthy replicas on 0 pods" for 10h45m while every duel failed
    # teacher_unservable). A fallback pod is released once the primary type
    # is healthy again.
    fallback_types: list[str] = field(default_factory=list)
    # Rent a same-type replacement this long before a pod's Lium
    # `removal_scheduled_at` (48 h TTL), admit it, then release the old box:
    # zero-gap rotation (kingctl does the same for the king seat). 0 = off.
    # 2026-09-17: the only teacher box expired on its TTL at 10:16 UTC and
    # the market had no replacement — 58 min at 0 replicas.
    rotate_before_ttl_hours: float = 0.0
    standin_release_min: float = 15.0
    types: dict[str, TypePlan] = field(default_factory=dict)


def load_config() -> Config:
    raw = tomllib.loads((HERE / "swarm.toml").read_text())
    s = raw["swarm"]
    cfg = Config(
        model=s["model"], vllm_version=s["vllm_version"],
        pod_prefix=s["pod_prefix"], template_id=s["template_id"],
        budget_usd_hr=float(s["budget_usd_hr"]), ttl_hours=int(s["ttl_hours"]),
        bootstrap_timeout_min=int(s["bootstrap_timeout_min"]),
        unreachable_grace_min=int(s["unreachable_grace_min"]),
        min_balance_usd=float(s["min_balance_usd"]),
        max_model_len=int(s["max_model_len"]),
        gpu_memory_utilization=float(s["gpu_memory_utilization"]),
        max_num_batched_tokens=int(s["max_num_batched_tokens"]),
        fallback_types=[str(x) for x in (s.get("fallback_types") or [])],
        rotate_before_ttl_hours=float(s.get("rotate_before_ttl_hours", 0) or 0),
        standin_release_min=float(s.get("standin_release_min", 15) or 15),
    )
    for name, t in raw.get("types", {}).items():
        cfg.types[name] = TypePlan(
            name=name, match=t["match"], gpu_count=int(t["gpu_count"]),
            tp=int(t["tp"]), replicas=int(t["replicas"]),
            target=int(t["target"]), max_price=float(t["max_price"]),
            est_tps=float(t["est_tps"]),
            max_num_batched_tokens=int(
                t.get("max_num_batched_tokens", cfg.max_num_batched_tokens)),
            gpu_util=float(t.get("gpu_util", cfg.gpu_memory_utilization)),
            adopt=[str(x) for x in (t.get("adopt") or [])],
            advertise=bool(t.get("advertise", True)),
            vllm_extra=str(t.get("vllm_extra", "")),
            standin_for=str(t.get("standin_for") or ""),
            standin_max_price=float(t.get("standin_max_price", 0) or 0),
        )
    return cfg


# ---------------------------------------------------------------- ssh helpers
def ssh_run(host: str, port: int, cmd: str, timeout: int = 60
            ) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd],
        capture_output=True, text=True, timeout=timeout)


def scp_put(host: str, port: int, local: Path, remote: str,
            timeout: int = 120) -> bool:
    p = subprocess.run(
        ["scp", *SSH_OPTS, "-P", str(port), str(local),
         f"root@{host}:{remote}"],
        capture_output=True, text=True, timeout=timeout)
    return p.returncode == 0


# ------------------------------------------------------------- persistence
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


def blacklist_ids() -> set[str]:
    if not BLACKLIST.exists():
        return set()
    return {ln.strip() for ln in BLACKLIST.read_text().splitlines()
            if ln.strip()}


_bl_lock = threading.Lock()


def blacklist_add(executor_id: str, reason: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with _bl_lock, BLACKLIST.open("a") as f:
        f.write(f"{executor_id}\n")
    log(f"blacklist += {executor_id} ({reason})")


STRIKES_JSON = STATE_DIR / "strikes.json"    # executor -> soft-failure count


def strike(executor_id: str, reason: str, limit: int = 2) -> None:
    """Soft failures (slow download, transient darkness) don't prove bad
    hardware, but an executor that keeps failing does: blacklist at `limit`."""
    with _bl_lock:
        try:
            strikes = json.loads(STRIKES_JSON.read_text())
        except (OSError, ValueError):
            strikes = {}
        strikes[executor_id] = strikes.get(executor_id, 0) + 1
        STRIKES_JSON.write_text(json.dumps(strikes, indent=1))
        n = strikes[executor_id]
    log(f"strike {n}/{limit} on {executor_id[:12]} ({reason})")
    if n >= limit:
        blacklist_add(executor_id, f"{reason} x{n}")


# ------------------------------------------------------------------ manager
class Manager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sess = lium_api.session()
        self.env = load_env_file()
        if not self.env.get("HF_TOKEN") or not self.env.get("SWARM_KEY"):
            raise SystemExit(f"{ENV_FILE} needs HF_TOKEN and SWARM_KEY")
        # memory: pod name -> {type, executor_id, rented_at, phase,
        #                      bootstrap_started, last_seen, last_healthy}
        self.mem: dict[str, dict] = load_json(PODS_JSON, {})
        try:
            self.pubkey = (Path.home() / ".ssh/id_ed25519.pub"
                           ).read_text().strip()
        except OSError:
            self.pubkey = ""
        # Optional Discord line sink (main wires Alerter.post); rotations and
        # stand-in releases are reported through it.
        self.notify = lambda text: None
        # type -> epoch when its backends were first seen healthy in the
        # current healthy streak (stand-in release hysteresis).
        self.type_healthy_since: dict[str, float] = {}

    # ---- stand-ins / rotation -------------------------------------------
    def effective_plan(self, plan: TypePlan, healthy_types: set[str],
                       now: float) -> TypePlan:
        """The plan as it applies this cycle: a stand-in type wants its
        boxes only while the covered type has not been healthy for
        standin_release_min; its price cap is the stand-in cap meanwhile."""
        if not plan.standin_for:
            return plan
        since = self.type_healthy_since.get(plan.standin_for)
        covered_ok = (since is not None
                      and now - since >= self.cfg.standin_release_min * 60)
        if covered_ok:
            return replace(plan, target=0)
        cap = plan.standin_max_price or plan.max_price
        return replace(plan, max_price=cap)

    @staticmethod
    def removal_epoch(pod: dict) -> float | None:
        """Lium `removal_scheduled_at` (naive ISO, UTC) -> epoch, or None."""
        raw = pod.get("removal_scheduled_at")
        if not raw:
            return None
        try:
            return (datetime.fromisoformat(str(raw).replace("Z", ""))
                    .replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            return None

    def rotate_expiring(self, swarm_pods: list[dict], backends: list[dict],
                        healthy_types: set[str], names_seen: set[str],
                        spend: float, stock: list[dict] | None,
                        now: float) -> tuple[float, list[dict] | None]:
        """Early TTL rotation. For every serving pod inside the rotation
        window rent one same-type replacement (marked `replaces`); once the
        replacement serves, release the old box. Returns updated spend/stock."""
        cfg = self.cfg
        if cfg.rotate_before_ttl_hours <= 0:
            return spend, stock
        window = cfg.rotate_before_ttl_hours * 3600
        healthy_pods = {b["pod"] for b in backends}
        live_names = {lium_api.pod_name(p) for p in swarm_pods}
        for pod in swarm_pods:
            name = lium_api.pod_name(pod)
            m = self.mem.get(name) or {}
            plan = self.plan_of(pod)
            if plan is None or not name.startswith(cfg.pod_prefix):
                continue
            # Old box whose replacement now serves: release it.
            repl = m.get("rotating_to")
            if repl:
                if repl in healthy_pods:
                    self.remove_pod(pod, f"ttl rotation complete -> {repl}")
                    self.notify(f"rotation: {name} released, {repl} serving "
                                f"({plan.name})")
                    # The replacement keeps its `replaces` marker until the
                    # old box has left the listing: this cycle's shrink step
                    # still counts both and must not drop the newer one.
                    self.mem.pop(name, None)
                elif repl not in live_names and repl not in self.mem:
                    m.pop("rotating_to", None)  # replacement vanished; retry
                continue
            if m.get("replaces"):
                if m["replaces"] not in live_names:
                    m.pop("replaces", None)  # old box gone; ordinary member now
                continue  # a replacement never rotates itself this cycle
            exp = self.removal_epoch(pod)
            if exp is None or exp - now > window or name not in healthy_pods:
                continue
            # Inside the window and serving: rent the replacement (one try
            # per cycle; a market with no stock retries until the TTL hits).
            if stock is None:
                stock = lium_api.executors(self.sess)
            eplan = self.effective_plan(plan, healthy_types, now)
            if eplan.target <= 0:
                continue  # stand-in no longer wanted; let the TTL take it
            cands = self.match_stock(stock, eplan)
            if not cands:
                if now - float(m.get("rotate_attempt_logged", 0)) > 3600:
                    m["rotate_attempt_logged"] = now
                    msg = (f"rotation: {name} expires in "
                           f"{(exp - now) / 60:.0f} min and no {plan.name} "
                           f"stock under ${eplan.max_price:.0f}/h — will run to TTL")
                    log(msg)
                    self.notify(msg)
                continue
            cand = cands[0]
            price = (cand.get("price_per_gpu") or 0) * plan.gpu_count
            if spend + price > cfg.budget_usd_hr:
                log(f"rotation: budget ${spend:.2f}+${price:.2f} > "
                    f"${cfg.budget_usd_hr} — {name} runs to TTL")
                continue
            new = self.new_name(plan, names_seen)
            pod_id = lium_api.rent(self.sess, str(cand["id"]), new,
                                   plan.gpu_count, cfg.template_id,
                                   cfg.ttl_hours, self.pubkey)
            if not pod_id or pod_id == "RATE_LIMITED":
                log(f"rotation: rent failed for {name} ({pod_id})")
                continue
            names_seen.add(new)
            spend += price
            self.mem[new] = {
                "type": plan.name, "executor_id": str(cand["id"]),
                "rented_at": now, "phase": "renting",
                "bootstrap_started": 0, "last_seen": now,
                "last_healthy": 0, "replaces": name}
            m["rotating_to"] = new
            msg = (f"rotation: {name} expires in {(exp - now) / 60:.0f} min — "
                   f"rented {new} ({cand.get('machine_name')} ${price:.2f}/h); "
                   f"old box released once it serves")
            log(msg)
            self.notify(msg)
            time.sleep(2.0)
        return spend, stock

    # ---- naming -------------------------------------------------------
    def new_name(self, plan: TypePlan, existing: set[str]) -> str:
        i = 1
        while True:
            name = f"{self.cfg.pod_prefix}{plan.name}-{i}"
            if name not in existing:
                return name
            i += 1

    def plan_of(self, pod: dict) -> TypePlan | None:
        name = lium_api.pod_name(pod)
        for plan in self.cfg.types.values():
            if name in plan.adopt:
                return plan
        m = self.mem.get(name)
        if m and m.get("type") in self.cfg.types:
            return self.cfg.types[m["type"]]
        # Fallback: parse type token out of the name.
        rest = name[len(self.cfg.pod_prefix):]
        for tname, plan in self.cfg.types.items():
            if rest.startswith(tname):
                return plan
        return None

    # ---- replica layout -------------------------------------------------
    def replica_specs(self, plan: TypePlan, ports: dict[int, int]
                      ) -> list[tuple[int, str, int]]:
        """[(internal_port, gpus, tp)] for one pod.

        Adapts to the pod's mapped-port count (templates vary: some boxes get
        12 data ports, some 2): pick the largest replica count n <= plan and
        <= available ports that divides gpu_count evenly, then widen TP to
        gpu_count/n so every GPU is used. plan.tp is the floor the model
        needs to fit, so TP only ever grows."""
        n_max = min(plan.replicas, len(ports))
        n = next((n for n in range(n_max, 0, -1)
                  if plan.gpu_count % n == 0
                  and plan.gpu_count // n >= plan.tp), 0)
        if n == 0:
            return []
        tp = plan.gpu_count // n
        internal = sorted(ports)[:n]
        specs = []
        for i, port in enumerate(internal):
            gpus = ",".join(str(g) for g in range(i * tp, (i + 1) * tp))
            specs.append((port, gpus, tp))
        return specs

    # Known-good canary values (verified identical across B300 / RTX PRO 6000
    # / Ada replicas, and matching production): greedy math answer plus the
    # summed echo logprob of a fixed sentence. A replica whose FP8 kernels
    # are numerically broken would corrupt Reason scores silently — never
    # admit one into the pool.
    CANARY_PROMPT = "Q: What is 2+2?\nA: 2+2="
    CANARY_ECHO = "The quick brown fox jumps over the lazy dog."
    CANARY_LP_RANGE = (-18.0, -14.0)  # measured ~-16.0 across types

    def canary_ok(self, base: str, headers: dict) -> bool:
        try:
            r = httpx.post(f"{base}/completions", headers=headers,
                           timeout=30.0, json={
                               "model": self.cfg.model,
                               "prompt": self.CANARY_PROMPT,
                               "max_tokens": 2, "temperature": 0})
            r.raise_for_status()
            text = r.json()["choices"][0]["text"]
            if not text.strip().startswith("4"):
                log(f"canary FAIL {base}: greedy text {text!r}")
                return False
            r = httpx.post(f"{base}/completions", headers=headers,
                           timeout=30.0, json={
                               "model": self.cfg.model,
                               "prompt": self.CANARY_ECHO,
                               "max_tokens": 1, "temperature": 0,
                               "echo": True, "logprobs": 0})
            r.raise_for_status()
            lps = r.json()["choices"][0]["logprobs"]["token_logprobs"]
            total = sum(x for x in lps if x is not None)
            lo, hi = self.CANARY_LP_RANGE
            if not lo <= total <= hi:
                log(f"canary FAIL {base}: echo lp {total:.2f} not in "
                    f"[{lo},{hi}]")
                return False
            return True
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as e:
            log(f"canary error {base}: {e!r}")
            return False

    def probe_replicas(self, pod: dict, plan: TypePlan) -> dict[int, bool]:
        """{external_port: healthy} — /v1/models liveness plus a one-time
        numerical canary per replica (cached in manager memory)."""
        name = lium_api.pod_name(pod)
        ports = lium_api.data_ports(pod)
        specs = self.replica_specs(plan, ports)
        ip = lium_api.pod_ip(pod)
        out: dict[int, bool] = {}
        headers = {"Authorization": f"Bearer {self.env['SWARM_KEY']}"}
        canary = self.mem.setdefault(name, {}).setdefault("canary", {})
        for internal, _gpus, _tp in specs:
            ext = ports[internal]
            base = f"http://{ip}:{ext}/v1"
            try:
                r = httpx.get(f"{base}/models", headers=headers, timeout=6.0)
                ok = r.status_code == 200
            except httpx.HTTPError:
                ok = False
            if ok and not canary.get(str(ext)):
                ok = self.canary_ok(base, headers)
                if ok:
                    canary[str(ext)] = True
                    log(f"{name}:{ext} canary passed — admitted")
            out[ext] = ok
        return out

    # ---- bootstrap -------------------------------------------------------
    def bootstrap(self, pod: dict, plan: TypePlan) -> bool:
        name = lium_api.pod_name(pod)
        ssh = lium_api.parse_ssh(pod)
        if not ssh:
            return False
        host, port = ssh
        ports = lium_api.data_ports(pod)
        specs = self.replica_specs(plan, ports)
        if not specs:
            log(f"{name}: no usable data-port/GPU layout "
                f"({len(ports)} ports) — cannot bootstrap")
            return False
        rep = ";".join(f"{p}:{g}:{t}" for p, g, t in specs)
        # Values are quoted: REPLICAS contains ';' which would otherwise
        # terminate the assignment when the pod sources this file.
        env_lines = [
            f'HF_TOKEN="{self.env["HF_TOKEN"]}"',
            f'SWARM_KEY="{self.env["SWARM_KEY"]}"',
            f'MODEL="{self.cfg.model}"',
            f'VLLM_VERSION="{self.cfg.vllm_version}"',
            f'REPLICAS="{rep}"',
            f'MAX_MODEL_LEN="{self.cfg.max_model_len}"',
            f'GPU_UTIL="{plan.gpu_util}"',
            f'BATCHED_TOKENS="{plan.max_num_batched_tokens}"',
            f"EXTRA_VLLM_ARGS='{plan.vllm_extra}'",
        ]
        # A fresh pod on a reused host:port has a new host key; accept-new
        # refuses ("REMOTE HOST IDENTIFICATION HAS CHANGED") and the env push
        # fails every cycle (2026-09-04 22:54-23:06, 24 cycles). The known_hosts
        # file is the manager's own throwaway: drop the stale entry first.
        subprocess.run(["ssh-keygen", "-R", f"[{host}]:{port}", "-f", str(KNOWN_HOSTS)],
                       capture_output=True, text=True, timeout=15)
        # Write env without putting secrets on a command line: pipe via stdin.
        try:
            p = subprocess.run(
                ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}",
                 "mkdir -p /root/swarm && umask 077 && cat > /root/swarm/env"],
                input="\n".join(env_lines) + "\n",
                capture_output=True, text=True, timeout=30)
            if p.returncode != 0:
                log(f"{name}: env push failed: {p.stderr[:200]}")
                return False
        except subprocess.SubprocessError as e:
            log(f"{name}: env push error: {e}")
            return False
        if not scp_put(host, port, HERE / "bootstrap_pod.sh",
                       "/root/swarm/bootstrap.sh"):
            log(f"{name}: bootstrap upload failed")
            return False
        # Echo prefix-cache plugin (a two-file pip package); bootstrap
        # installs it into the swarm venv before launching replicas.
        try:
            ssh_run(host, port, "mkdir -p /root/swarm/echo_cache_plugin")
        except subprocess.SubprocessError:
            pass
        for fname in ("pyproject.toml", "affine_vllm_echo_cache.py"):
            if not scp_put(host, port, ECHO_PLUGIN_DIR / fname,
                           f"/root/swarm/echo_cache_plugin/{fname}"):
                log(f"{name}: echo plugin upload failed ({fname})")
                return False
        # NB: a pgrep -f pattern would match this ssh command's own cmdline;
        # use a pidfile instead. setsid fully detaches from the ssh session.
        launch = (
            "if [ -f /root/swarm/boot.pid ] && "
            "kill -0 $(cat /root/swarm/boot.pid) 2>/dev/null; then "
            "echo already-running; else "
            "setsid nohup bash /root/swarm/bootstrap.sh "
            ">> /root/swarm/bootstrap.log 2>&1 < /dev/null & "
            "echo $! > /root/swarm/boot.pid; echo started; fi")
        try:
            p = ssh_run(host, port, launch)
            ok = p.returncode == 0
        except subprocess.SubprocessError:
            ok = False
        if ok:
            log(f"{name}: bootstrap launched ({len(specs)} replicas: {rep})")
        return ok

    def pod_failed_marker(self, pod: dict) -> str:
        ssh = lium_api.parse_ssh(pod)
        if not ssh:
            return ""
        try:
            p = ssh_run(*ssh, "cat /root/swarm/bootstrap.failed 2>/dev/null",
                        timeout=20)
            return p.stdout.strip() if p.returncode == 0 else ""
        except subprocess.SubprocessError:
            return ""

    # ---- economics -------------------------------------------------------
    def spend_usd_hr(self, swarm_pods: list[dict]) -> float:
        return sum(lium_api.pod_price(p) for p in swarm_pods)

    # ---- per-pod state machine -------------------------------------------
    def handle_pod(self, pod: dict, plan: TypePlan, now: float) -> list[dict]:
        """Advance one pod; returns its healthy backend entries."""
        name = lium_api.pod_name(pod)
        cfg = self.cfg
        m = self.mem.setdefault(name, {
            "type": plan.name, "executor_id": "",
            "rented_at": now, "phase": "new",
            "bootstrap_started": 0, "last_seen": now, "last_healthy": 0})
        m["last_seen"] = now
        if not m["executor_id"] and pod.get("executor_id"):
            m["executor_id"] = str(pod["executor_id"])

        status = str(pod.get("status") or "").upper()
        if status not in ("RUNNING",):
            # Still provisioning; give it the bootstrap window from rent.
            age_min = (now - m["rented_at"]) / 60
            if age_min > cfg.bootstrap_timeout_min:
                self.replace(pod, plan, f"stuck {status} {age_min:.0f}m")
            return []

        health = self.probe_replicas(pod, plan)
        n_up = sum(health.values())
        if n_up:
            m["last_healthy"] = now
            m["phase"] = "ready" if n_up == plan.replicas else "degraded"
            if not plan.advertise:
                return []
            ip = lium_api.pod_ip(pod)
            return [{"url": f"http://{ip}:{ext}/v1", "pod": name,
                     "type": plan.name, "est_tps": plan.est_tps / plan.replicas}
                    for ext, ok in health.items() if ok]

        # No replica up. Bootstrap not started / crashed / pod dark?
        failure = self.pod_failed_marker(pod)
        if failure:
            self.replace(pod, plan, f"bootstrap.failed={failure}")
            return []
        if not m["bootstrap_started"]:
            if self.bootstrap(pod, plan):
                m["phase"] = "bootstrapping"
                m["bootstrap_started"] = now
            else:
                # SSH not up yet is normal for a fresh pod; only replace when
                # the pod has been dark past the grace window.
                dark_min = (now - max(m["rented_at"],
                                      m["last_healthy"] or 0)) / 60
                if dark_min > cfg.bootstrap_timeout_min:
                    self.replace(pod, plan, f"unreachable {dark_min:.0f}m")
            return []
        boot_min = (now - m["bootstrap_started"]) / 60
        # Two distinct death modes, never mixed (2026-08-15 incident: healthy
        # B300s were killed by "bootstrap timeout 123m" after one replica
        # crash, because pod age kept counting past the boot window):
        #  - never healthy this epoch -> bootstrap window applies;
        #  - was healthy this epoch   -> dark grace applies, sized to cover a
        #    supervisor engine relaunch (110 GB weights load ~10-15 min).
        was_healthy_this_epoch = (m.get("last_healthy") or 0) > \
            m["bootstrap_started"]
        if not was_healthy_this_epoch:
            if boot_min > cfg.bootstrap_timeout_min:
                self.replace(pod, plan, f"bootstrap timeout {boot_min:.0f}m",
                             blame_executor=False)
        elif (now - m["last_healthy"]) / 60 > cfg.unreachable_grace_min:
            self.replace(pod, plan, "went dark after being healthy",
                         blame_executor=False)
        return []

    # ---- reconcile -------------------------------------------------------
    def reconcile(self) -> dict:
        cfg = self.cfg
        now = time.time()
        all_pods = lium_api.pods(self.sess)
        if all_pods is None:
            log("pods API failed — skipping cycle (no destructive action)")
            return {}
        adopted = {n for plan in cfg.types.values() for n in plan.adopt}
        swarm_pods = [p for p in all_pods
                      if lium_api.pod_name(p).startswith(cfg.pod_prefix)
                      or lium_api.pod_name(p) in adopted]
        by_type: dict[str, list[dict]] = {t: [] for t in cfg.types}
        backends: list[dict] = []
        # Seed with remembered names: a just-rented pod may lag the listing,
        # and reusing its name would create a duplicate-name pod.
        names_seen = set(self.mem)

        handled = []
        for pod in swarm_pods:
            name = lium_api.pod_name(pod)
            names_seen.add(name)
            plan = self.plan_of(pod)
            if plan is None:
                log(f"{name}: unknown type — leaving alone")
                continue
            by_type[plan.name].append(pod)
            handled.append((pod, plan))

        # Per-pod work (probe / bootstrap / replace) is SSH-bound: run it in
        # parallel. Each worker only touches its own mem[name] entry.
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            for pod_backends in ex.map(
                    lambda pp: self.handle_pod(pp[0], pp[1], now), handled):
                backends.extend(pod_backends)

        # Forget pods that no longer exist.
        for name in [n for n in self.mem if n not in names_seen]:
            del self.mem[name]

        # ---- scale up/down ------------------------------------------------
        live_pods = [p for p in swarm_pods if self.plan_of(p)]
        spend = self.spend_usd_hr(live_pods)
        stock = None
        healthy_types = {b["type"] for b in backends}
        for t in list(self.type_healthy_since):
            if t not in healthy_types:
                del self.type_healthy_since[t]
        # Hysteresis applies to transitions, not to a manager (re)start: a
        # type already healthy on the first cycle counts as long-healthy, or
        # every restart would rent the stand-ins for standin_release_min.
        seed = now - cfg.standin_release_min * 60 if not self.type_healthy_since             and not getattr(self, "_seen_cycle", False) else now
        self._seen_cycle = True
        for t in healthy_types:
            self.type_healthy_since.setdefault(t, seed)
        spend, stock = self.rotate_expiring(swarm_pods, backends, healthy_types,
                                            names_seen, spend, stock, now)
        unfilled: list[str] = []
        for tname, cplan in cfg.types.items():
            plan = self.effective_plan(cplan, healthy_types, now)
            if cplan.standin_for and plan.target == 0 and by_type[tname]:
                log(f"{tname}: stand-in released — {cplan.standin_for} healthy")
            have = len(by_type[tname])
            if have > plan.target:
                # Shrink: drop the newest boxes first (least sunk warmup). A
                # fallback pod stays until the type it covers for is healthy.
                extra = sorted(
                    by_type[tname],
                    key=lambda p: self.mem.get(lium_api.pod_name(p), {})
                                      .get("rented_at", 0),
                    reverse=True)[: have - plan.target]
                for pod in extra:
                    pm = self.mem.get(lium_api.pod_name(pod), {})
                    covering = pm.get("fallback_for")
                    if covering and covering not in healthy_types:
                        continue
                    if plan.target > 0 and (pm.get("replaces") or pm.get("rotating_to")):
                        continue  # ttl rotation pair; rotate_expiring owns it
                    why = "over target"
                    if covering:
                        why += f" ({covering} healthy again)"
                    elif cplan.standin_for and plan.target == 0:
                        why += f" (stand-in for {cplan.standin_for}, now healthy)"
                        self.notify(f"stand-in released: {lium_api.pod_name(pod)} "
                                    f"({tname}) — {cplan.standin_for} healthy again")
                    self.remove_pod(pod, why)
                continue
            missing = plan.target - have
            if missing <= 0:
                continue
            if stock is None:
                stock = lium_api.executors(self.sess)
            bal = lium_api.balance_usd()
            if bal is not None and bal < cfg.min_balance_usd:
                log(f"balance ${bal:.0f} < floor — no renting")
                break
            candidates = self.match_stock(stock, plan)
            if not candidates:
                unfilled.append(tname)
                log(f"{tname}: no rentable stock ({self.stock_summary(stock, plan)})")
            rate_limited = False
            for cand in candidates[:missing]:
                price = (cand.get("price_per_gpu") or 0) * plan.gpu_count
                if spend + price > cfg.budget_usd_hr:
                    log(f"budget: ${spend:.2f}+${price:.2f} > "
                        f"${cfg.budget_usd_hr} — skip {tname}")
                    break
                name = self.new_name(plan, names_seen)
                pod_id = lium_api.rent(self.sess, str(cand["id"]), name,
                                       plan.gpu_count, cfg.template_id,
                                       cfg.ttl_hours, self.pubkey)
                if pod_id == "RATE_LIMITED":
                    log("rent rate-limited — resuming next cycle")
                    rate_limited = True
                    break
                if pod_id:
                    log(f"rented {name} ({cand.get('machine_name')} "
                        f"${price:.2f}/h executor={str(cand['id'])[:12]})")
                    names_seen.add(name)
                    spend += price
                    self.mem[name] = {
                        "type": tname, "executor_id": str(cand["id"]),
                        "rented_at": now, "phase": "renting",
                        "bootstrap_started": 0, "last_seen": now,
                        "last_healthy": 0}
                else:
                    log(f"rent failed on {str(cand['id'])[:12]}")
                time.sleep(2.0)  # Lium cap: 3 requests / 5 s
            if rate_limited:
                break

        # Nothing rented at all and the primary has no stock: rent one box of
        # the first fallback type with stock so the teacher never sits empty.
        if unfilled and not live_pods and stock is not None:
            for tname in cfg.fallback_types:
                plan = cfg.types.get(tname)
                if plan is None or tname in unfilled:
                    continue
                cands = self.match_stock(stock, plan)
                if not cands:
                    continue
                cand = cands[0]
                price = (cand.get("price_per_gpu") or 0) * plan.gpu_count
                if spend + price > cfg.budget_usd_hr:
                    continue
                name = self.new_name(plan, names_seen)
                pod_id = lium_api.rent(self.sess, str(cand["id"]), name,
                                       plan.gpu_count, cfg.template_id,
                                       cfg.ttl_hours, self.pubkey)
                if pod_id and pod_id != "RATE_LIMITED":
                    log(f"FALLBACK for {unfilled[0]}: rented {name} "
                        f"({cand.get('machine_name')} ${price:.2f}/h "
                        f"executor={str(cand['id'])[:12]})")
                    names_seen.add(name)
                    spend += price
                    self.mem[name] = {
                        "type": tname, "executor_id": str(cand["id"]),
                        "rented_at": now, "phase": "renting",
                        "bootstrap_started": 0, "last_seen": now,
                        "last_healthy": 0, "fallback_for": unfilled[0]}
                    break
                log(f"fallback rent failed on {str(cand['id'])[:12]} ({pod_id})")
                time.sleep(2.0)

        state = {
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": cfg.model,
            "spend_usd_hr": round(spend, 2),
            "backends": backends,
            "pods": {n: {k: v for k, v in m.items() if k != "executor_id"}
                     for n, m in self.mem.items()},
        }
        save_json(STATE_JSON, state)
        save_json(PODS_JSON, self.mem)
        log(f"state: {len(backends)} healthy replicas on "
            f"{len(swarm_pods)} pods, ${spend:.2f}/h")
        return state

    def stock_summary(self, stock: list[dict], plan: TypePlan) -> str:
        """Why match_stock came back empty, for the log."""
        bl = blacklist_ids()
        n_match = n_full = n_price = n_bl = 0
        cheapest = None
        for n in stock:
            mn = (n.get("machine_name") or "").upper()
            if plan.match.upper() not in mn or (
                    plan.match.upper() == "B200" and "B300" in mn):
                continue
            if int(n.get("gpu_count") or 0) != plan.gpu_count:
                continue
            n_match += 1
            if str(n.get("id")) in bl:
                n_bl += 1
                continue
            avail = n.get("available_gpu_count")
            if avail is not None and int(avail) < plan.gpu_count:
                n_full += 1
                continue
            price = (n.get("price_per_gpu") or 1e9) * plan.gpu_count
            cheapest = price if cheapest is None else min(cheapest, price)
            if price > plan.max_price:
                n_price += 1
        return (f"{len(stock)} executors, {n_match} {plan.match} x{plan.gpu_count}: "
                f"{n_bl} blacklisted, {n_full} occupied, {n_price} above "
                f"${plan.max_price:.0f}" + (f", cheapest ${cheapest:.0f}" if cheapest else ""))

    def match_stock(self, stock: list[dict], plan: TypePlan) -> list[dict]:
        bl = blacklist_ids()
        out = []
        for n in stock:
            if str(n.get("id")) in bl:
                continue
            if int(n.get("gpu_count") or 0) != plan.gpu_count:
                continue
            avail = n.get("available_gpu_count")
            if avail is not None and int(avail) < plan.gpu_count:
                continue
            mn = (n.get("machine_name") or "").upper()
            if plan.match.upper() not in mn:
                continue
            # "B200 excludes B300" style disambiguation.
            if plan.match.upper() == "B200" and "B300" in mn:
                continue
            price = (n.get("price_per_gpu") or 1e9) * plan.gpu_count
            if price > plan.max_price:
                continue
            out.append(n)
        out.sort(key=lambda n: n.get("price_per_gpu") or 1e9)
        return out

    def remove_pod(self, pod: dict, reason: str) -> None:
        name = lium_api.pod_name(pod)
        log(f"rm {name} ({reason})")
        try:
            lium_api.remove(name, self.cfg.pod_prefix)
        except (ValueError, subprocess.SubprocessError) as e:
            log(f"rm {name} failed: {e}")

    def replace(self, pod: dict, plan: TypePlan, reason: str,
                blame_executor: bool = True) -> None:
        """Remove a pod; replacement happens next cycle via target top-up.

        blame_executor=False for deaths that do not prove bad hardware
        (slow downloads, transient darkness) — blacklisting those starves
        the type's supply for no reason."""
        name = lium_api.pod_name(pod)
        ex = self.mem.get(name, {}).get("executor_id", "")
        if ex:
            if blame_executor:
                blacklist_add(ex, reason)
            else:
                strike(ex, reason)
        self.remove_pod(pod, reason)
        self.mem.pop(name, None)


class Alerter:
    """One Discord line to the private ops channel when the swarm has served
    zero healthy replicas for `after_min` minutes, repeated every
    `repeat_min` while it lasts, and one line when it recovers.

    2026-09-17: the only b200-8x hit its 48 h TTL at 10:16 UTC, Lium had no
    B200 x8 in stock, and a Lium API change (422 on ?size=2000) made the
    fallback path see "0 executors" — 0 replicas for 50 min, every duel
    requeued teacher_unservable, and nothing paged anyone. Never raises."""

    def __init__(self, raw: dict):
        a = raw.get("alerts") or {}
        self.enabled = bool(a.get("enabled", False))
        self.channel = str(a.get("channel_id") or "")
        self.token_env = str(a.get("token_env") or "DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")
        self.after_s = float(a.get("zero_replicas_after_min", 5)) * 60
        self.repeat_s = float(a.get("repeat_min", 60)) * 60
        self.zero_since: float | None = None
        self.last_sent = 0.0
        self.alerting = False

    def _token(self) -> str:
        import os
        if os.environ.get(self.token_env):
            return os.environ[self.token_env]
        for path in (ENV_FILE, HERE.parents[1] / ".env"):
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                line = line.strip().removeprefix("export ").strip()
                if line.startswith(f"{self.token_env}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        return ""

    def post(self, text: str) -> None:
        log(f"alert: {text}")
        if not (self.enabled and self.channel):
            return
        token = self._token()
        if not token:
            log("alert: no discord token resolvable; not posted")
            return
        try:
            r = httpx.post(
                f"https://discord.com/api/v10/channels/{self.channel}/messages",
                headers={"Authorization": f"Bot {token}"},
                json={"content": f"[teacher swarm] {text}"[:1900]}, timeout=20)
            if r.status_code >= 300:
                log(f"alert: discord HTTP {r.status_code}")
        except httpx.HTTPError as e:
            log(f"alert: discord post failed: {e!r}")

    def observe(self, state: dict) -> None:
        now = time.time()
        healthy = len(state.get("backends") or [])
        # The manager's memory keeps months of removed pods; only boxes seen
        # on Lium in the last hour are relevant to the alert text.
        pods = {n: m for n, m in (state.get("pods") or {}).items()
                if float((m or {}).get("last_seen") or 0) > now - 3600}
        if healthy > 0:
            if self.alerting:
                self.post(f"recovered: {healthy} healthy replicas on "
                          f"{len(pods)} pod(s), ${state.get('spend_usd_hr')}/h")
            self.alerting = False
            self.zero_since = None
            return
        if self.zero_since is None:
            self.zero_since = now
            return
        if now - self.zero_since < self.after_s:
            return
        if self.alerting and now - self.last_sent < self.repeat_s:
            return
        phases = ", ".join(f"{n}:{m.get('phase')}" for n, m in pods.items()) or "no pods"
        mins = int((now - self.zero_since) / 60)
        self.post(f"0 healthy teacher replicas for {mins} min — every duel is "
                  f"requeuing teacher_unservable. Pods: {phases}. "
                  f"Check ops/teacher-swarm/state/manager.pm2.log (stock / caps / Lium API).")
        self.alerting = True
        self.last_sent = now


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=30.0)
    args = ap.parse_args()
    mgr = Manager(load_config())
    alerter = Alerter(tomllib.loads((HERE / "swarm.toml").read_text()))
    mgr.notify = alerter.post
    if args.once:
        mgr.reconcile()
        return 0
    log("daemon start")
    while True:
        try:
            state = mgr.reconcile()
            alerter.observe(state)
        except Exception as e:  # noqa: BLE001 — daemon must not die
            log(f"reconcile error: {e!r}")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
