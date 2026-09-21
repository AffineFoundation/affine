#!/usr/bin/env python
"""King-seat controller: serve the current SN120 king for datagen.

Loop (pm2 `affine-king-datagen`, every --interval seconds):

  1. read the king from the validator's state.json (repo / revision /
     reign). An R2 king is served from the PUBLIC copy on models.affine.io
     (only crowned models live there; no credentials); the HF genesis is
     served from its pinned revision. An unreadable state.json keeps the
     last known king for [king].state_stale_min before the seat is emptied.
  2. make sure a Lium box `king-dg-<ident>-<hex4>` serves that king: rent
     (first [[types]] entry with stock under its price cap), push env +
     bootstrap_king.sh, launch it, wait for every replica to answer. A box
     older than ttl_hours - rotate_before_ttl_hours gets a replacement
     rented while it still serves (zero-gap rotation past the Lium TTL).
  3. once the box answers a real completion, write /root/rollouts/.king_env
     on every `affine-datagen*` pod (base_url / model / key / digest /
     reign). rollouts/king.py re-reads that file each cycle, so the `king_*`
     policies start routing to the new king with no restart. After READY a
     canary completion runs every canary_every_min; three misses in a row
     count as dark even while /v1/models still answers.
  4. release boxes that are superseded (previous king; the published one only
     after the new one serves, an unpublished one at once), that never came
     up (bootstrap timeout -> executor strike), or that went dark after
     ready (unreachable grace). While no king box answers, .king_env is
     emptied so the king policies go idle instead of burning containers.
  5. keep the datagen pods alive: a pod whose rollouts supervisor AND its
     bootstrap loop are both gone for watchdog_relaunch_min is relaunched.

Memory: state/state.json (pods + what is published). A pod the controller
holds no memory for (state lost) is RECOVERED from the box's own
/root/king/env, not released. A pod missing from the Lium listing is
forgotten only after pod_forget_ticks consecutive misses. Secrets: the
per-box bearer is generated here and lives only in that state file (0600),
on the box and on the pods. Lium API key via ops/teacher-swarm/lium_api.py.
Never prints a secret. State changes are posted to Discord ([discord]).

  python kingctl.py --interval 60      # the service
  python kingctl.py --once             # one tick
  python kingctl.py status             # what is served / published
  python kingctl.py unpublish          # empty .king_env on the datagen pods
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "teacher-swarm"))

import lium_api  # noqa: E402
sys.path.insert(0, str(HERE.parents[1] / "ops" / "health"))
import podheal  # noqa: E402

sys.path.insert(0, str(HERE.parent / "pods"))
import registry as pod_registry  # noqa: E402

STATE_DIR = HERE / "state"
STATE_JSON = STATE_DIR / "state.json"
KNOWN_HOSTS = STATE_DIR / "known_hosts"
BLACKLIST = STATE_DIR / "blacklist.txt"
VALIDATOR_ENV = Path.home() / ".affine-validator.env"
REPO_ENV = REPO / ".env"
REPLICA_PORT_BASE = 31001          # loopback-only vLLM ports behind nginx
RENT_SETTLE_S = 10 * 60            # a rented pod may take this long to list
REPUBLISH_EVERY_S = 6 * 3600       # periodic full re-push (new datagen pods)
CANARY_DARK_FAILS = 3              # consecutive canary misses = dark
POD_SUPERVISOR_CMD = "/root/venv/bin/python -m rollouts.run"
POD_BOOTSTRAP_CMD = "bash /root/rollouts/bootstrap.sh"

# Host keys are not pinned: a Lium pod gets a new host key on every container
# restart (same host:port), and that is exactly when the watchdog must get in.
SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=15",
    "-o", "BatchMode=yes",
    "-o", "LogLevel=ERROR",
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[kingctl] {now_iso()} {msg}", flush=True)


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class TypePlan:
    name: str
    match: str
    gpu_count: int
    tp: int
    replicas: int
    max_price: float


@dataclass(frozen=True)
class Config:
    state_json: Path
    pod_prefix: str
    datagen_pod_prefix: str
    king_env_path: str
    template_id: str
    ttl_hours: int
    rotate_before_ttl_hours: float
    bootstrap_timeout_min: int
    unreachable_grace_min: int
    canary_every_min: int
    pod_forget_ticks: int
    seat_empty_alert_min: int
    prev_king_max_min: int
    datagen_disk_min_free_pct: int
    burst_enabled: bool
    burst_hours: float
    burst_budget_usd_hr: float
    burst_min_per_env: int
    burst_min_total: int
    burst_stats_url: str
    state_stale_min: int
    watchdog_every_min: int
    watchdog_relaunch_min: int
    min_balance_usd: float
    budget_usd_hr: float
    vllm_version: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    types: tuple[TypePlan, ...]
    discord_enabled: bool
    discord_channel_id: str
    discord_token_env: str


def load_config(path: Path = HERE / "king.toml") -> Config:
    raw = tomllib.loads(path.read_text())
    k = raw["king"]
    d = raw.get("discord") or {}
    types = tuple(TypePlan(
        name=t["name"], match=t["match"], gpu_count=int(t["gpu_count"]),
        tp=int(t["tp"]), replicas=int(t["replicas"]),
        max_price=float(t["max_price"])) for t in raw.get("types", []))
    if not types:
        raise SystemExit("king.toml: no [[types]]")
    for t in types:
        if t.tp * t.replicas > t.gpu_count:
            raise SystemExit(f"king.toml type {t.name}: tp*replicas > gpu_count")
    return Config(
        state_json=REPO / k["state_json"],
        pod_prefix=k["pod_prefix"],
        datagen_pod_prefix=k["datagen_pod_prefix"],
        king_env_path=k["king_env_path"],
        template_id=k["template_id"],
        ttl_hours=int(k["ttl_hours"]),
        rotate_before_ttl_hours=float(k.get("rotate_before_ttl_hours", 2)),
        bootstrap_timeout_min=int(k["bootstrap_timeout_min"]),
        unreachable_grace_min=int(k["unreachable_grace_min"]),
        seat_empty_alert_min=int(k.get("seat_empty_alert_min", 60)),
        prev_king_max_min=int(k.get("prev_king_max_min", 120)),
        datagen_disk_min_free_pct=int(k.get("datagen_disk_min_free_pct", 10)),
        burst_enabled=bool((raw.get("burst") or {}).get("enabled", False)),
        burst_hours=float((raw.get("burst") or {}).get("hours", 6)),
        burst_budget_usd_hr=float((raw.get("burst") or {}).get("budget_usd_hr", 15)),
        burst_min_per_env=int((raw.get("burst") or {}).get("min_rollouts_per_env", 24)),
        burst_min_total=int((raw.get("burst") or {}).get("min_rollouts_total", 200)),
        burst_stats_url=str((raw.get("burst") or {}).get("stats_url", "https://kings.affine.io/api/stats.json")),
        canary_every_min=int(k.get("canary_every_min", 10)),
        pod_forget_ticks=int(k.get("pod_forget_ticks", 5)),
        state_stale_min=int(k.get("state_stale_min", 30)),
        watchdog_every_min=int(k.get("watchdog_every_min", 5)),
        watchdog_relaunch_min=int(k.get("watchdog_relaunch_min", 10)),
        min_balance_usd=float(k["min_balance_usd"]),
        budget_usd_hr=float(k["budget_usd_hr"]),
        vllm_version=str(k["vllm_version"]),
        max_model_len=int(k["max_model_len"]),
        gpu_memory_utilization=float(k["gpu_memory_utilization"]),
        max_num_batched_tokens=int(k["max_num_batched_tokens"]),
        max_num_seqs=int(k["max_num_seqs"]),
        types=types,
        discord_enabled=bool(d.get("enabled", False)),
        discord_channel_id=str(d.get("channel_id") or ""),
        discord_token_env=str(d.get("token_env") or "DISCORD_BOT_TOKEN_ARBOS_BITTENSOR"),
    )


def env_file_value(name: str) -> str:
    """One value from the operator env snapshots (validator env first, then
    the repo .env). Missing file / key -> ""."""
    if os.environ.get(name):
        return os.environ[name]
    for path in (VALIDATOR_ENV, REPO_ENV):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def lium_ts(value) -> float | None:
    """Lium's naive ISO timestamps (UTC) -> epoch seconds, or None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def parse_env_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    return out


# ------------------------------------------------------------------- king
def king_from_repo(repo: str, rev: str, reign) -> dict:
    if repo.startswith("r2://"):
        return {"kind": "r2", "digest": rev, "ident": rev[:12],
                "served": f"king-{rev[:12]}", "reign": reign, "repo": repo}
    ident = hashlib.sha256(f"{repo}@{rev}".encode()).hexdigest()[:12]
    return {"kind": "hf", "hf_model": repo, "hf_rev": rev, "ident": ident,
            "served": f"king-{ident}", "reign": reign, "repo": repo}


def read_king(cfg: Config) -> dict | None:
    """Who is king right now per the validator's state.json, or None when the
    file is unreadable / has no king (the caller decides about staleness)."""
    try:
        st = json.loads(cfg.state_json.read_text())
    except (OSError, ValueError) as e:
        log(f"state.json unreadable: {e!r}")
        return None
    k = st.get("king") or {}
    repo, rev = str(k.get("repo") or ""), str(k.get("revision") or "")
    if not repo or not rev:
        return None
    king = king_from_repo(repo, rev, k.get("reign_number"))
    king["crowned_at"] = k.get("crowned_at")   # the previous-king cap counts from here
    return king


# -------------------------------------------------------------- ssh helpers
def ssh_run(host: str, port: int, cmd: str, *, input_text: str | None = None,
            timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd],
        input=input_text, capture_output=True, text=True, timeout=timeout)


def scp_put(host: str, port: int, local: Path, remote: str,
            timeout: int = 120) -> bool:
    p = subprocess.run(
        ["scp", *SSH_OPTS, "-P", str(port), str(local), f"root@{host}:{remote}"],
        capture_output=True, text=True, timeout=timeout)
    return p.returncode == 0


def forget_host_key(host: str, port: int) -> None:
    subprocess.run(["ssh-keygen", "-R", f"[{host}]:{port}", "-f", str(KNOWN_HOSTS)],
                   capture_output=True, text=True, timeout=15)


# ------------------------------------------------------------- persistence
def load_state() -> dict:
    st = {"pods": {}, "published": {}}
    if STATE_JSON.exists():
        st.update(json.loads(STATE_JSON.read_text()))
    st.setdefault("pods", {})
    st.setdefault("published", {})
    st.setdefault("datagen", {})
    return st


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    os.chmod(tmp, 0o600)
    tmp.replace(STATE_JSON)


def blacklist_ids() -> set[str]:
    if not BLACKLIST.exists():
        return set()
    return {l.split()[0] for l in BLACKLIST.read_text().splitlines() if l.strip()}


def blacklist_add(executor_id: str, reason: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BLACKLIST, "a") as f:
        f.write(f"{executor_id} {now_iso()} {reason}\n")


# --------------------------------------------------------------- controller

def king_crowned_ts(king: dict) -> float | None:
    """crowned_at of the current king as a unix time (state.json ISO), or None."""
    raw = king.get("crowned_at")
    if not raw:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None

class Controller:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sess = lium_api.session()
        self.state = load_state()
        try:
            self.pubkey = (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()
        except OSError:
            self.pubkey = ""

    # ---- notify ---------------------------------------------------------------
    def notify(self, text: str) -> None:
        """One Discord line per state change. Never raises; a failure is one
        log line."""
        log(f"event: {text}")
        cfg = self.cfg
        if not (cfg.discord_enabled and cfg.discord_channel_id):
            return
        token = env_file_value(cfg.discord_token_env)
        if not token:
            log("notify: no discord token in env; skipped")
            return
        try:
            r = httpx.post(
                f"https://discord.com/api/v10/channels/{cfg.discord_channel_id}/messages",
                headers={"Authorization": f"Bot {token}"},
                json={"content": f"[king seat] {text}"}, timeout=20)
            if r.status_code >= 300:
                log(f"notify: discord HTTP {r.status_code}")
        except httpx.HTTPError as e:
            log(f"notify: {e!r}")

    # ---- king (with staleness) ------------------------------------------------
    def king_now(self) -> dict | None:
        """The king to serve: state.json when readable, else the last known
        king for up to state_stale_min (a half-written file must not empty
        the seat)."""
        king = read_king(self.cfg)
        now = time.time()
        if king is not None:
            self.state["king_cache"] = {"king": king, "at": now}
            self.state.pop("king_unreadable_since", None)
            return king
        cache = self.state.get("king_cache") or {}
        since = self.state.setdefault("king_unreadable_since", now)
        if cache.get("king") and now - since <= self.cfg.state_stale_min * 60:
            log(f"state.json has no readable king for {int((now - since) / 60)} min; "
                f"keeping {cache['king']['served']}")
            return cache["king"]
        return None

    # ---- lium views -----------------------------------------------------------
    def my_pods(self, pods: list[dict]) -> dict[str, dict]:
        return {lium_api.pod_name(p): p for p in pods
                if lium_api.pod_name(p).startswith(self.cfg.pod_prefix)}

    def datagen_pods(self, pods: list[dict]) -> dict[str, dict]:
        return {lium_api.pod_name(p): p for p in pods
                if lium_api.pod_name(p).startswith(self.cfg.datagen_pod_prefix)
                and str(p.get("status", "")).upper() in ("RUNNING", "")}

    def spend(self, mine: dict[str, dict]) -> float:
        """$/h of every listed box; the rent-time price from memory when the
        listing carries none."""
        return sum(lium_api.pod_price(p)
                   or float((self.state["pods"].get(n) or {}).get("price") or 0)
                   for n, p in mine.items())

    def boxes_for(self, ident: str, mine: dict[str, dict]) -> list[str]:
        """Names of listed boxes (with memory) serving this king ident."""
        return [n for n, m in self.state["pods"].items()
                if n in mine and (m.get("king") or {}).get("ident") == ident]

    # ---- reconcile memory vs listing ------------------------------------------
    def reconcile(self, mine: dict[str, dict]) -> None:
        now = time.time()
        for name in list(self.state["pods"]):
            m = self.state["pods"][name]
            if name in mine:
                m["missing_ticks"] = 0
                continue
            if now - m.get("rented_at", 0) <= RENT_SETTLE_S:
                continue
            m["missing_ticks"] = m.get("missing_ticks", 0) + 1
            if m["missing_ticks"] >= self.cfg.pod_forget_ticks:
                log(f"{name}: gone from lium for {m['missing_ticks']} ticks; forgetting")
                self.notify(f"box {name} vanished from Lium; forgotten")
                self.state["pods"].pop(name)
        for name in list(mine):
            if name not in self.state["pods"]:
                if not self.recover(name, mine[name]):
                    self.remove(name, "no memory of this pod and /root/king/env unreadable")
                    mine.pop(name, None)

    def recover(self, name: str, pod: dict) -> bool:
        """Rebuild memory for a box we hold none for (controller state lost)
        from the box's own /root/king/env + ready marker."""
        ssh = lium_api.parse_ssh(pod)
        ports = lium_api.data_ports(pod)
        if not ssh or not ports:
            log(f"{name}: recover: no ssh/data ports listed yet")
            return False
        forget_host_key(*ssh)
        try:
            p = ssh_run(*ssh, "cat /root/king/env 2>/dev/null; echo '@@'; "
                              "stat -c %Y /root/king/ready 2>/dev/null; echo '@@'; "
                              "stat -c %Y /root/king/bootstrap.log 2>/dev/null", timeout=40)
        except subprocess.SubprocessError as e:
            log(f"{name}: recover: ssh error {e!r}")
            return False
        env_text, _, rest = p.stdout.partition("@@")
        ready_s, _, boot_s = rest.partition("@@")
        env = parse_env_lines(env_text)
        key, served = env.get("KING_KEY"), env.get("SERVED_NAME")
        if not key or not served:
            log(f"{name}: recover: /root/king/env has no KING_KEY/SERVED_NAME")
            return False
        if env.get("DIGEST"):
            king = king_from_repo(f"r2://recovered/{env['DIGEST']}/", env["DIGEST"], None)
        elif env.get("HF_MODEL") and env.get("HF_REV"):
            king = king_from_repo(env["HF_MODEL"], env["HF_REV"], None)
        else:
            log(f"{name}: recover: /root/king/env names no model")
            return False
        if king["served"] != served:
            log(f"{name}: recover: served name {served} != derived {king['served']}")
            return False
        try:
            front = int(env.get("FRONT_PORT") or sorted(ports)[0])
        except ValueError:
            front = sorted(ports)[0]
        if front not in ports:
            front = sorted(ports)[0]
        now = time.time()
        ready_at = float(ready_s.strip()) if ready_s.strip().isdigit() else None
        boot_started = float(boot_s.strip()) if boot_s.strip().isdigit() else now
        n_rep = len([s for s in (env.get("REPLICAS") or "").split(";") if s])
        created = lium_ts(pod.get("created_at")) or min(boot_started, now)
        self.state["pods"][name] = {
            "king": king, "type": f"recovered-{n_rep}rep", "executor_id": "",
            "machine": pod.get("machine_name") or (pod.get("executor") or {}).get("machine_name"),
            "price": lium_api.pod_price(pod), "rented_at": created,
            "key": key, "boot_started": boot_started, "front_internal": front,
            "base_url": f"http://{lium_api.pod_ip(pod)}:{ports[front]}/v1",
            "ready_at": ready_at, "last_ok": ready_at, "recovered_at": now,
        }
        log(f"{name}: recovered memory from the box ({king['served']}, "
            f"ready={'yes' if ready_at else 'no'})")
        self.notify(f"recovered memory of box {name} ({king['served']}) from the box itself")
        return True

    # ---- rent -----------------------------------------------------------------
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
            if plan.match.upper() == "B200" and "B300" in mn:
                continue
            price = (n.get("price_per_gpu") or 1e9) * plan.gpu_count
            if price > plan.max_price:
                continue
            out.append(n)
        out.sort(key=lambda n: n.get("price_per_gpu") or 1e9)
        return out

    def rent(self, king: dict, mine: dict[str, dict], why: str, *,
             burst: bool = False) -> None:
        name = f"{self.cfg.pod_prefix}{king['ident']}-{secrets.token_hex(2)}"
        bal = lium_api.balance_usd()
        if bal is not None and bal < self.cfg.min_balance_usd:
            log(f"rent {name}: balance ${bal:.0f} < ${self.cfg.min_balance_usd:.0f}; not renting")
            return
        spend = self.spend(mine)
        stock = lium_api.executors(self.sess)
        budget = self.cfg.burst_budget_usd_hr if burst else self.cfg.budget_usd_hr
        # A burst box is the type with stock that adds the MOST replicas
        # under the burst budget (primary + burst <= burst_budget_usd_hr),
        # cheapest first among equals; the primary follows the preference
        # order. Candidates over the remaining budget are skipped below.
        plans = list(self.cfg.types)
        if burst:
            room = budget - spend
            priced = []
            for plan in plans:
                for cand in self.match_stock(stock, plan):
                    price = (cand.get("price_per_gpu") or 0) * plan.gpu_count
                    if price <= room:
                        priced.append((-plan.replicas, price, plan, cand))
            priced.sort(key=lambda t: (t[0], t[1]))
            order = [(plan, [cand]) for _, _, plan, cand in priced]
        else:
            order = [(plan, self.match_stock(stock, plan)) for plan in plans]
        for plan, cands in order:
            for cand in cands:
                price = (cand.get("price_per_gpu") or 0) * plan.gpu_count
                if spend + price > budget:
                    log(f"rent {name}: ${spend:.2f}+${price:.2f} > budget "
                        f"${budget:.2f}; skipping {plan.name}")
                    continue
                res = lium_api.rent(self.sess, cand["id"], name, plan.gpu_count,
                                    self.cfg.template_id, self.cfg.ttl_hours,
                                    self.pubkey)
                if res == "RATE_LIMITED":
                    log("rent: rate limited; retry next tick")
                    return
                if res is None:
                    blacklist_add(str(cand["id"]), "rent-failed")
                    continue
                self.state["pods"][name] = {
                    "king": king, "type": plan.name, "executor_id": str(cand["id"]),
                    "machine": cand.get("machine_name"), "price": price,
                    "rented_at": time.time(), "key": secrets.token_hex(24),
                    "missing_ticks": 0, "burst": burst,
                }
                if burst:
                    self.state["burst"] = {"ident": king["ident"], "pod": name,
                                           "started_at": time.time(), "assignment": {}}
                pod_registry.register(
                    name, purpose="king_seat", owner="pm2:affine-king-datagen",
                    expected_hours=float(self.cfg.ttl_hours) + 1, price_usd_h=price,
                    ttl_hours=self.cfg.ttl_hours,
                    meta={"king": king.get("served"), "reign": king.get("reign"),
                          "type": plan.name})
                log(f"rented {name}: {plan.name} {cand.get('machine_name')} "
                    f"${price:.2f}/h executor={str(cand['id'])[:12]} ({why})")
                self.notify(f"rented {name} ({plan.name}, ${price:.2f}/h) for "
                            f"{king['served']} reign {king.get('reign')} — {why}")
                return
        self.state["last_rent_failure"] = (f"no stock under price caps for any type "
                                           f"({len(stock)} executors listed)")
        log(f"rent {name}: no stock under price caps for any type "
            f"({len(stock)} executors listed)")

    # ---- bootstrap ------------------------------------------------------------
    def replica_specs(self, plan: TypePlan) -> list[tuple[int, str, int]]:
        specs = []
        for i in range(plan.replicas):
            gpus = ",".join(str(g) for g in range(i * plan.tp, (i + 1) * plan.tp))
            specs.append((REPLICA_PORT_BASE + i, gpus, plan.tp))
        return specs

    def bootstrap(self, name: str, pod: dict, mem: dict) -> bool:
        ssh = lium_api.parse_ssh(pod)
        ports = lium_api.data_ports(pod)
        if not ssh or not ports:
            log(f"{name}: no ssh/data ports yet")
            return False
        host, port = ssh
        plan = next((t for t in self.cfg.types if t.name == mem["type"]), None)
        if plan is None:
            log(f"{name}: unknown type {mem['type']}; cannot bootstrap")
            return False
        front_internal = sorted(ports)[0]
        king = mem["king"]
        specs = self.replica_specs(plan)
        lines = [
            f'KING_KEY="{mem["key"]}"',
            f'SERVED_NAME="{king["served"]}"',
            f'FRONT_PORT="{front_internal}"',
            f'REPLICAS="{";".join(f"{p}:{g}:{t}" for p, g, t in specs)}"',
            f'MAX_MODEL_LEN="{self.cfg.max_model_len}"',
            f'GPU_UTIL="{self.cfg.gpu_memory_utilization}"',
            f'BATCHED_TOKENS="{self.cfg.max_num_batched_tokens}"',
            f'MAX_NUM_SEQS="{self.cfg.max_num_seqs}"',
            f'VLLM_VERSION="{self.cfg.vllm_version}"',
        ]
        if king["kind"] == "r2":
            lines.append(f'DIGEST="{king["digest"]}"')
            # A king crowned seconds ago still lives in the PRIVATE bucket
            # (the validator promotes it to models.affine.io in the
            # background, ~13 min for 70 GB). Fetch it from there with the
            # eval pods' read-only key so the seat swap keys off the crown,
            # not the promote (same path benchsuite/kingpod.py uses).
            repo = str(king.get("repo") or "")
            if repo.startswith("r2://affine-private-models/"):
                lines += [
                    f'KING_R2="{repo}"',
                    f'AFFINE_EVAL_R2_ENDPOINT="{env_file_value("AFFINE_EVAL_R2_ENDPOINT") or env_file_value("R2_ENDPOINT")}"',
                    f'AFFINE_EVAL_R2_ACCESS_KEY_ID="{env_file_value("AFFINE_EVAL_R2_ACCESS_KEY_ID")}"',
                    f'AFFINE_EVAL_R2_SECRET_ACCESS_KEY="{env_file_value("AFFINE_EVAL_R2_SECRET_ACCESS_KEY")}"',
                ]
                log(f"{name}: king {king['digest'][:12]} is still private — fetching via KING_R2")
        else:
            lines += [f'HF_MODEL="{king["hf_model"]}"', f'HF_REV="{king["hf_rev"]}"',
                      f'HF_TOKEN="{env_file_value("HF_TOKEN")}"']
        forget_host_key(host, port)
        try:
            p = ssh_run(host, port, "mkdir -p /root/king && umask 077 && cat > /root/king/env",
                        input_text="\n".join(lines) + "\n", timeout=40)
        except subprocess.SubprocessError as e:
            log(f"{name}: env push error {e!r}")
            return False
        if p.returncode != 0:
            log(f"{name}: env push failed: {p.stderr.strip()[:160]}")
            return False
        if not scp_put(host, port, HERE / "bootstrap_king.sh", "/root/king/bootstrap.sh"):
            log(f"{name}: bootstrap upload failed")
            return False
        launch = ("if [ -f /root/king/boot.pid ] && kill -0 $(cat /root/king/boot.pid) "
                  "2>/dev/null; then echo already-running; else setsid nohup bash "
                  "/root/king/bootstrap.sh >> /root/king/bootstrap.log 2>&1 < /dev/null & "
                  "echo $! > /root/king/boot.pid; echo started; fi")
        try:
            p = ssh_run(host, port, launch)
        except subprocess.SubprocessError as e:
            log(f"{name}: launch error {e!r}")
            return False
        if p.returncode != 0:
            log(f"{name}: launch failed: {p.stderr.strip()[:160]}")
            return False
        mem["boot_started"] = time.time()
        mem["front_internal"] = front_internal
        mem["base_url"] = f"http://{lium_api.pod_ip(pod)}:{ports[front_internal]}/v1"
        log(f"{name}: bootstrap launched ({len(specs)} replicas) -> {mem['base_url']}")
        return True

    # ---- probes ---------------------------------------------------------------
    def probe_models(self, mem: dict) -> bool:
        base, headers = mem.get("base_url"), {"Authorization": f"Bearer {mem['key']}"}
        if not base:
            return False
        try:
            r = httpx.get(f"{base}/models", headers=headers, timeout=8.0)
            r.raise_for_status()
            served = {m.get("id") for m in r.json().get("data", [])}
        except (httpx.HTTPError, ValueError, AttributeError):
            return False          # not up (yet) — the normal case while booting
        if mem["king"]["served"] not in served:
            log(f"probe {base}: serves {sorted(served)} not {mem['king']['served']}")
            return False
        return True

    def probe_canary(self, mem: dict) -> bool:
        base, headers = mem.get("base_url"), {"Authorization": f"Bearer {mem['key']}"}
        try:
            r = httpx.post(f"{base}/chat/completions", headers=headers, timeout=120.0,
                           json={"model": mem["king"]["served"], "max_tokens": 16,
                                 "temperature": 0,
                                 "messages": [{"role": "user", "content": "Say OK."}]})
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            if not (msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning")):
                log(f"canary {base}: empty reply")
                return False
            return True
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as e:
            log(f"canary {base}: {e!r}")
            return False

    def probe(self, mem: dict) -> bool:
        """Is this box serving? /v1/models every call; a canary completion
        before READY and every canary_every_min after it. CANARY_DARK_FAILS
        canary misses in a row make the box dark even while /models answers
        (a wedged engine)."""
        if not self.probe_models(mem):
            return False
        now = time.time()
        due = (not mem.get("ready_at")
               or now - mem.get("last_canary", 0) >= self.cfg.canary_every_min * 60)
        if due:
            mem["last_canary"] = now
            if self.probe_canary(mem):
                mem["canary_fails"] = 0
            else:
                mem["canary_fails"] = mem.get("canary_fails", 0) + 1
                if not mem.get("ready_at"):
                    return False
        return mem.get("canary_fails", 0) < CANARY_DARK_FAILS

    def boot_report(self, pod: dict) -> tuple[str, str]:
        """(failure reason from /root/king/bootstrap.failed or "", last log
        line) — read over ssh; ("", "") when the pod does not answer."""
        ssh = lium_api.parse_ssh(pod)
        if not ssh:
            return "", ""
        try:
            p = ssh_run(*ssh, "cat /root/king/bootstrap.failed 2>/dev/null; echo '@@'; "
                              "tail -1 /root/king/bootstrap.log 2>/dev/null", timeout=30)
        except subprocess.SubprocessError:
            return "", ""
        head, _, tail = p.stdout.partition("@@")
        return head.strip(), tail.strip()[-160:]

    def needs_rotation(self, name: str, pod: dict | None, now: float) -> bool:
        """Inside the pre-TTL window? Lium's own removal_scheduled_at when it
        is listed, else rented_at + ttl_hours."""
        mem = self.state["pods"][name]
        deadline = lium_ts((pod or {}).get("removal_scheduled_at")) or (
            mem.get("rented_at", now) + self.cfg.ttl_hours * 3600)
        return deadline - now < self.cfg.rotate_before_ttl_hours * 3600

    def choose_serving(self, serving: list[str], pub: dict, mine: dict[str, dict],
                       now: float) -> str | None:
        """Which serving box the pods should use: the published one while it
        serves (no flapping between two live boxes), unless it is inside
        its rotation window and a fresher box serves; otherwise the newest."""
        serving = [n for n in serving if not self.state["pods"][n].get("burst")] or serving
        if not serving:
            return None
        cur = pub.get("pod")
        if cur in serving:
            rotating = self.needs_rotation(cur, mine.get(cur), now)
            fresher = [n for n in serving if n != cur
                       and self.state["pods"][n].get("rented_at", 0)
                       > self.state["pods"][cur].get("rented_at", 0)]
            if not (rotating and fresher):
                return cur
            return max(fresher, key=lambda n: self.state["pods"][n]["ready_at"])
        return max(serving, key=lambda n: self.state["pods"][n]["ready_at"])

    # ---- one box, one tick ----------------------------------------------------
    def advance(self, name: str, pod: dict, mem: dict, now: float) -> bool:
        """Move one box along rent -> bootstrap -> ready -> serving. Returns
        True when it serves right now. Removes it on bootstrap failure /
        timeout or when dark past the grace."""
        cfg = self.cfg
        if not mem.get("boot_started"):
            self.bootstrap(name, pod, mem)
            return False
        if self.probe(mem):
            mem["last_ok"] = now
            if not mem.get("ready_at"):
                mem["ready_at"] = now
                mem["canary_fails"] = 0
                log(f"{name}: READY after {int((now - mem['boot_started']) / 60)} min")
                self.notify(f"{name} READY ({mem['king']['served']}) after "
                            f"{int((now - mem['boot_started']) / 60)} min at {mem['base_url']}")
            return True
        boot_age = now - mem["boot_started"]
        if not mem.get("ready_at"):
            failed, last = self.boot_report(pod)
            if failed:
                self.remove(name, f"bootstrap failed: {failed[:80]}", strike=True)
            elif boot_age > cfg.bootstrap_timeout_min * 60:
                self.remove(name, f"bootstrap timeout: {last[:100]}", strike=True)
            elif int(boot_age) % 300 < 60:
                log(f"{name}: booting {int(boot_age / 60)} min ({last or 'no log yet'})")
            return False
        dark = now - mem.get("last_ok", mem["ready_at"])
        if dark > 300 and now - mem.get("reboot_at", 0) > 2 * 3600:
            # same failure class as the driver pod: container back without
            # its volume -> sshd refuses every key. Reboot once (2 h throttle)
            # and re-run the bootstrap (weights persist on /root) before the
            # dark -> remove -> re-rent path (45 min gap) fires.
            ssh = lium_api.parse_ssh(pod)
            sstate = podheal.ssh_state(*ssh) if ssh else "unreachable"
            dsec = podheal.denied_for(name, sstate == "denied", now)
            if sstate == "denied" and dsec >= 10 * 60:
                res = podheal.maybe_reboot(name, listed_running=True, denied_s=max(dsec, podheal.DENIED_MIN * 60),
                                           reason="king box dark + ssh publickey denied", now=now)
                if res:
                    mem["reboot_at"] = now
                    mem.pop("boot_started", None)     # advance() re-runs bootstrap_king.sh
                    mem.pop("ready_at", None)
                    self.notify(f"{name}: dark {int(dark / 60)} min, ssh denied {int(dsec / 60)} min — {res}; re-bootstrapping")
                    return False
        if dark > cfg.unreachable_grace_min * 60:
            self.remove(name, f"dark {int(dark / 60)} min after ready")
        elif int(dark) % 300 < 60:
            log(f"{name}: dark {int(dark / 60)} min (canary fails "
                f"{mem.get('canary_fails', 0)})")
        return False

    # ---- publish --------------------------------------------------------------
    def king_env_text(self, mem: dict | None) -> str:
        if mem is None:
            return "# king seat: no king box is serving right now (kingctl)\n"
        k = mem["king"]
        burst = "1" if (self.state.get("burst") or {}).get("ident") == k["ident"] else "0"
        return (f"# written by ops/king-datagen/kingctl.py {now_iso()}\n"
                f"KING_BASE_URL={mem['base_url']}\n"
                f"KING_MODEL={k['served']}\n"
                f"KING_KEY={mem['key']}\n"
                f"KING_DIGEST={k.get('digest') or k.get('hf_rev')}\n"
                f"KING_REIGN={k.get('reign')}\n"
                f"KING_BURST={burst}\n")

    def push_king_env(self, datagen: dict[str, dict], mem: dict | None,
                      only: set[str] | None = None,
                      per_pod: dict[str, dict] | None = None) -> dict[str, bool]:
        """Write .king_env on the datagen pods. `per_pod` maps a pod name to
        the box memory it should use (the burst split); others get `mem`."""
        out: dict[str, bool] = {}
        for name, pod in sorted(datagen.items()):
            if only is not None and name not in only:
                continue
            text = self.king_env_text((per_pod or {}).get(name, mem))
            ssh = lium_api.parse_ssh(pod)
            if not ssh:
                out[name] = False
                continue
            host, port = ssh
            path = self.cfg.king_env_path
            try:
                p = ssh_run(host, port,
                            f"umask 077 && mkdir -p $(dirname {path}) && cat > {path}.tmp "
                            f"&& mv {path}.tmp {path}", input_text=text, timeout=40)
                out[name] = p.returncode == 0
                if p.returncode != 0:
                    log(f"push {name}: {p.stderr.strip()[:160]}")
            except subprocess.SubprocessError as e:
                log(f"push {name}: {e!r}")
                out[name] = False
        return out

    def publish(self, datagen: dict[str, dict], pod_name: str, mem: dict,
                per_pod: dict[str, dict] | None = None) -> None:
        res = self.push_king_env(datagen, mem, per_pod=per_pod)
        now = time.time()
        self.state["published"] = {
            "pod": pod_name, "ident": mem["king"]["ident"], "reign": mem["king"].get("reign"),
            "base_url": mem["base_url"], "at": now,
            "datagen": {n: now for n, ok in res.items() if ok},
            "split": {n: m["base_url"] for n, m in (per_pod or {}).items()},
        }
        log(f"published {mem['king']['served']} (reign {mem['king'].get('reign')}) at "
            f"{mem['base_url']} to {sum(res.values())}/{len(res)} datagen pods "
            f"{[n for n, ok in res.items() if not ok] or ''}")
        self.notify(f"published {mem['king']['served']} reign {mem['king'].get('reign')} "
                    f"({pod_name}) to {sum(res.values())}/{len(res)} datagen pods")

    def unpublish(self, datagen: dict[str, dict], reason: str) -> None:
        if not self.state.get("published"):
            return
        res = self.push_king_env(datagen, None)
        log(f"unpublished ({reason}); emptied .king_env on "
            f"{sum(res.values())}/{len(res)} datagen pods")
        self.notify(f"unpublished — {reason}; king policies idle on "
                    f"{sum(res.values())}/{len(res)} datagen pods")
        self.state["published"] = {}

    # ---- remove ---------------------------------------------------------------
    def remove(self, name: str, reason: str, *, strike: bool = False) -> None:
        mem = self.state["pods"].get(name) or {}
        log(f"removing {name} ({reason})")
        ok = lium_api.remove(name, self.cfg.pod_prefix)
        if not ok:
            log(f"{name}: lium rm failed; retry next tick")
            return
        if strike and mem.get("executor_id"):
            blacklist_add(mem["executor_id"], reason)
        self.state["pods"].pop(name, None)
        self.notify(f"removed {name} — {reason}")

    # ---- datagen pod watchdog -------------------------------------------------
    def datagen_watchdog(self, datagen: dict[str, dict], now: float) -> None:
        """A datagen pod whose rollouts supervisor is gone: if its bootstrap
        loop is also gone for watchdog_relaunch_min, relaunch the loop (it
        re-sources the env and starts the supervisor). Loop alive but
        supervisor absent that long = crash loop: alert only (a second loop
        would double the pod)."""
        cfg = self.cfg
        if now - self.state.get("watchdog_at", 0) < cfg.watchdog_every_min * 60:
            return
        self.state["watchdog_at"] = now
        mem_all = self.state.setdefault("datagen", {})
        for name, pod in sorted(datagen.items()):
            ssh = lium_api.parse_ssh(pod)
            if not ssh:
                continue
            m = mem_all.setdefault(name, {})
            try:
                p = ssh_run(*ssh,
                            f"pgrep -f -x '{POD_SUPERVISOR_CMD}' >/dev/null && echo sup; "
                            f"pgrep -f -x '{POD_BOOTSTRAP_CMD}' >/dev/null && echo loop; "
                            f"df --output=target,pcent,avail / /root 2>/dev/null | tail -n +2 | sed 's/^/df /'",
                            timeout=30)
                unreachable = p.returncode not in (0, 1)
                detail = p.stderr.strip()[:120]
            except subprocess.SubprocessError as e:
                unreachable, detail = True, repr(e)
            if unreachable:
                since = m.setdefault("ssh_fail_since", now)
                log(f"watchdog {name}: ssh failed ({int((now - since) / 60)} min): {detail}")
                # volume-unmounted container: sshd answers, every key refused,
                # Lium lists RUNNING -> lium reboot (podheal: > 20 min, 2 h throttle);
                # the supervisor relaunch below picks the pod up afterwards
                denied = "Permission denied" in detail
                dsec = podheal.denied_for(name, denied, now)
                if denied:
                    res = podheal.maybe_reboot(name, listed_running=True, denied_s=dsec,
                                               reason="datagen watchdog: ssh publickey denied", now=now)
                    if res:
                        self.notify(f"datagen pod {name}: ssh denied {int(dsec / 60)} min while RUNNING — {res}")
                if (now - since >= cfg.watchdog_relaunch_min * 60
                        and now - m.get("alerted_at", 0) > 3600):
                    m["alerted_at"] = now
                    self.notify(f"datagen pod {name}: ssh {'denied' if denied else 'unreachable'} for "
                                f"{int((now - since) / 60)} min ({detail[:60]})")
                continue
            podheal.denied_for(name, False, now)
            if m.pop("ssh_fail_since", None):
                self.notify(f"datagen pod {name}: ssh reachable again")
            sup, loop = "sup" in p.stdout.split(), "loop" in p.stdout.split()
            # disk: the host overlay (`/`, shared with docker's image store)
            # and the pod volume (`/root`). datagen-2 hit 98 % on 2026-09-20
            # with nothing paging; a full disk kills every rollout.
            for line in p.stdout.splitlines():
                if not line.startswith("df "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                mount, pct = parts[1], parts[2].rstrip("%")
                try:
                    used_pct = int(pct)
                except ValueError:
                    continue
                key = f"disk_alert_{mount}"
                if 100 - used_pct < cfg.datagen_disk_min_free_pct:
                    if now - m.get(key, 0) > 6 * 3600:
                        m[key] = now
                        self.notify(f"datagen pod {name}: {mount} is {used_pct}% full "
                                    f"({parts[3]} KB free) — under {cfg.datagen_disk_min_free_pct}% "
                                    f"headroom; diskgc stage 4 prunes task images below "
                                    f"{cfg.datagen_disk_min_free_pct}%, check /root/logs/rollouts.log")
                elif m.pop(key, None):
                    self.notify(f"datagen pod {name}: {mount} back to {100 - used_pct}% free")
            if sup:
                if m.get("down_since"):
                    self.notify(f"datagen pod {name}: supervisor back")
                m.pop("down_since", None)
                m["last_up"] = now
                continue
            since = m.setdefault("down_since", now)
            down_min = int((now - since) / 60)
            log(f"watchdog {name}: supervisor absent {down_min} min (loop "
                f"{'alive' if loop else 'gone'})")
            if now - since < cfg.watchdog_relaunch_min * 60:
                continue
            if loop:
                if now - m.get("alerted_at", 0) > 3600:
                    m["alerted_at"] = now
                    self.notify(f"datagen pod {name}: supervisor absent {down_min} min "
                                f"while its bootstrap loop lives — crash loop? check "
                                f"/root/logs/rollouts.log")
                continue
            try:
                r = ssh_run(*ssh, "install -m 0755 /root/rollouts/scripts/pod_post_start.sh /post_start.sh 2>/dev/null; "
                                  "cd /root/rollouts && setsid nohup bash "
                                  "/root/rollouts/bootstrap.sh >> "
                                  "/root/logs/rollouts_bootstrap.nohup 2>&1 < /dev/null & "
                                  "echo relaunched", timeout=30)
            except subprocess.SubprocessError as e:
                log(f"watchdog {name}: relaunch error {e!r}")
                continue
            if "relaunched" in r.stdout:
                m["down_since"] = now
                m["relaunched_at"] = now
                self.notify(f"datagen pod {name}: supervisor and bootstrap loop gone "
                            f"{down_min} min — relaunched bootstrap.sh")
            else:
                log(f"watchdog {name}: relaunch failed: {r.stderr.strip()[:120]}")

    # ---- tick -----------------------------------------------------------------
    def tick(self) -> None:
        cfg = self.cfg
        king = self.king_now()
        pods = lium_api.pods(self.sess)
        if pods is None:
            log("lium /pods failed; skipping tick")
            return
        mine = self.my_pods(pods)
        datagen = self.datagen_pods(pods)
        now = time.time()
        self.reconcile(mine)

        if king is None:
            log("no king in state.json")
            self.unpublish(datagen, "no king")
            self.datagen_watchdog(datagen, now)
            save_state(self.state)
            return

        ident = king["ident"]
        pub = self.state.get("published") or {}
        pub_before = dict(pub)      # who the pods used when this tick began
        self.seat_empty_alert(king, pub, now)
        # Advance every box of the current king; note which ones serve.
        serving: list[str] = []
        for name in self.boxes_for(ident, mine):
            mem = self.state["pods"][name]
            if mem["king"].get("reign") is None:      # recovered memory
                mem["king"]["reign"] = king.get("reign")
            if self.advance(name, mine[name], mem, now):
                serving.append(name)
        boxes = self.boxes_for(ident, mine)
        booting = [n for n in boxes if not self.state["pods"][n].get("ready_at")]
        newest = self.choose_serving(serving, pub, mine, now)
        # Pending rents (not listed yet) count as booting.
        pending = [n for n, m in self.state["pods"].items()
                   if n not in mine and (m.get("king") or {}).get("ident") == ident]

        # Rent: nothing alive for this king, or the serving box nears its TTL.
        if not boxes and not pending:
            self.rent(king, mine, "no box for this king")
        elif newest and not booting and not pending:
            if self.needs_rotation(newest, mine.get(newest), now):
                age_h = (now - self.state["pods"][newest].get("rented_at", now)) / 3600
                self.rent(king, mine, f"rotation: {newest} is {age_h:.1f} h old, "
                                      f"TTL {cfg.ttl_hours} h")

        # Publish the newest serving box; keep the pods' copy fresh.
        if newest:
            mem = self.state["pods"][newest]
            if pub.get("ident") != ident or pub.get("base_url") != mem["base_url"]:
                self.publish(datagen, newest, mem)
                pub = self.state["published"]
            else:
                missing = {n for n in datagen if n not in (pub.get("datagen") or {})}
                if missing or now - pub.get("at", 0) > REPUBLISH_EVERY_S:
                    res = self.push_king_env(datagen, mem, only=missing or None,
                                             per_pod=self.burst_per_pod())
                    pub.setdefault("datagen", {}).update({n: now for n, ok in res.items() if ok})
                    if not missing:
                        pub["at"] = now
                    log(f"re-pushed .king_env to {sum(res.values())}/{len(res)} pods")
        elif pub:
            # Nothing of the current king serves. The published box (the old
            # king during a swap, or the current one between probes) must keep
            # answering; if it is gone or dark past the grace, idle the king
            # policies rather than let them strike a dead endpoint.
            pmem = self.state["pods"].get(pub.get("pod") or "")
            if pmem is None:
                self.unpublish(datagen, "published box gone")
            elif pmem["king"].get("ident") != ident:
                if self.probe(pmem):
                    pmem["last_ok"] = now
                else:
                    dark = now - pmem.get("last_ok", pmem.get("ready_at", now))
                    if dark > cfg.unreachable_grace_min * 60:
                        self.unpublish(datagen, f"published box dark {int(dark / 60)} min")
            elif pub.get("pod") not in serving:
                self.unpublish(datagen, "published box not serving")

        self.burst_step(king, mine, datagen, newest, serving, now)

        # Retire: boxes of another king (the published one only once the new
        # king serves, or prev_king_max_min after the crown whatever happened;
        # never-published ones at once) and rotated-out boxes. Burst boxes
        # are the burst's business.
        pub = self.state.get("published") or {}
        crowned_at = king_crowned_ts(king)
        for name in list(mine):
            mem = self.state["pods"].get(name)
            if mem is None:
                continue
            if (mem.get("king") or {}).get("ident") != ident:
                if name in (pub.get("pod"), pub_before.get("pod")):
                    if newest:
                        self.remove(name, f"superseded by {newest}")
                    elif crowned_at and now - crowned_at > cfg.prev_king_max_min * 60:
                        self.remove(name, f"previous king's box {int((now - crowned_at) / 60)} min "
                                          f"after the crown of {ident} (cap {cfg.prev_king_max_min} min)")
                        self.unpublish(datagen, "previous king's box released (cap)")
                else:
                    self.remove(name, f"superseded (never published for king {ident})")
            elif mem.get("burst"):
                continue
            elif newest and name != newest and mem.get("ready_at") and name != pub.get("pod"):
                self.remove(name, f"rotated out; {newest} serves")

        self.datagen_watchdog(datagen, now)
        save_state(self.state)

    def seat_empty_alert(self, king: dict, pub: dict, now: float) -> None:
        """Page the ops channel when the current king has had no published
        box for more than seat_empty_alert_min, and every hour after that,
        with the last rent failure. 2026-09-17/18: reign 14 sat unserved for
        25 h because Lium's /executors started answering 422 and the running
        controller (old module in memory) logged "0 executors listed" once a
        minute, to the log only — nobody was paged."""
        cfg = self.cfg
        if pub.get("ident") == king["ident"] and pub.get("pod"):
            self.state.pop("seat_empty_since", None)
            self.state.pop("seat_empty_alerted_at", None)
            return
        since = self.state.setdefault("seat_empty_since", now)
        empty_min = int((now - since) / 60)
        if empty_min < cfg.seat_empty_alert_min:
            return
        if now - self.state.get("seat_empty_alerted_at", 0) < 3600:
            return
        self.state["seat_empty_alerted_at"] = now
        boxes = [n for n, m in self.state["pods"].items()
                 if (m.get("king") or {}).get("ident") == king["ident"]]
        self.notify(f"KING SEAT EMPTY {empty_min} min: {king['served']} reign "
                    f"{king.get('reign')} has no serving box on any datagen pod; "
                    f"boxes of this king: {boxes or 'none'}; last rent failure: "
                    f"{self.state.get('last_rent_failure') or 'none logged'}")

    # ---- burst: a second box for the first hours after a crown -------------------
    def row_counts(self, ident: str) -> tuple[int, int, int] | None:
        """(total rollouts, envs in the row, min rollouts over ALL the row's
        envs — an env the king has not touched counts 0) for the king on the
        kingboard; None when the board cannot be read."""
        try:
            r = httpx.get(self.cfg.burst_stats_url, timeout=30)
            st = r.json()
        except Exception as e:  # network / json — the 6 h cap still ends the burst
            log(f"burst: kingboard unreadable ({e!r})")
            return None
        universe = [e.get("source") for e in (st.get("envs") or []) if e.get("source")]
        for row in st.get("reigns") or []:
            if row.get("digest12") == ident:
                have = {e.get("source"): int(e.get("n") or 0) for e in row.get("envs") or []}
                envs = universe or list(have)
                return (int(row.get("n_rollouts") or 0), len(envs),
                        min((have.get(src, 0) for src in envs), default=0))
        return (0, len(universe), 0)

    def burst_per_pod(self) -> dict[str, dict] | None:
        """The live burst split (pod name -> box memory), or None."""
        burst = self.state.get("burst") or {}
        want = burst.get("assignment") or {}
        if not want or not all(b in self.state["pods"] for b in want.values()):
            return None
        return {n: self.state["pods"][b] for n, b in want.items()}

    def burst_step(self, king: dict, mine: dict[str, dict], datagen: dict[str, dict],
                   newest: str | None, serving: list[str], now: float) -> None:
        """Rent a second box when a king's first box starts serving; split the
        datagen pods across both (pod order alternates primary / burst) with
        KING_BURST=1 so the scheduler runs the king at KING_BURST_SHARE; end the
        burst after burst_hours or once the kingboard row has >= min_total
        rollouts and >= min_per_env on every env; then re-publish the primary."""
        cfg = self.cfg
        if not cfg.burst_enabled or not newest:
            return
        ident = king["ident"]
        burst = self.state.get("burst") or {}
        if burst and burst.get("ident") != ident:
            # a crown happened mid-burst: the old king's burst box goes with it
            if burst.get("pod") in self.state["pods"]:
                self.remove(burst["pod"], f"burst box of previous king {burst.get('ident')}")
            self.state.pop("burst", None)
            burst = {}
        if not burst:
            if self.state.get("burst_done_for") == ident:
                return
            pending_burst = [n for n, m in self.state["pods"].items()
                             if m.get("burst") and (m.get("king") or {}).get("ident") == ident]
            if not pending_burst:
                self.rent(king, mine, "burst: second box for the first hours after the crown", burst=True)
            return
        bname = burst["pod"]
        bmem = self.state["pods"].get(bname)
        if bmem is None:
            log("burst: box gone; burst over")
            self.state.pop("burst", None)
            self.state["burst_done_for"] = ident
            self.publish(datagen, newest, self.state["pods"][newest])
            return
        age_h = (now - burst["started_at"]) / 3600
        counts = self.row_counts(ident) if int(now) % 300 < 60 else None
        full = bool(counts and counts[0] >= cfg.burst_min_total and counts[1] > 0
                    and counts[2] >= cfg.burst_min_per_env)
        if age_h > cfg.burst_hours or full:
            why = (f"row full: {counts[0]} rollouts, min {counts[2]} per env over {counts[1]} envs"
                   if full else f"{age_h:.1f} h elapsed")
            self.remove(bname, f"burst over ({why})")
            self.state.pop("burst", None)
            self.state["burst_done_for"] = ident
            self.publish(datagen, newest, self.state["pods"][newest])
            self.notify(f"burst over for {king['served']} reign {king.get('reign')}: {why}; "
                        f"pods back on {newest}")
            return
        if bname not in serving:
            return
        # Both serve: split the pods (alternating) if not already published so.
        names = sorted(datagen)
        want = {n: (bname if i % 2 == 1 else newest) for i, n in enumerate(names)}
        pub = self.state.get("published") or {}
        if pub.get("split") != {n: self.state["pods"][b]["base_url"] for n, b in want.items()} \
                or pub.get("ident") != ident:
            per_pod = {n: self.state["pods"][b] for n, b in want.items()}
            self.publish(datagen, newest, self.state["pods"][newest], per_pod=per_pod)
            burst["assignment"] = want
            self.notify(f"burst: {king['served']} reign {king.get('reign')} on two boxes — "
                        f"{sum(1 for b in want.values() if b == newest)} pods on {newest}, "
                        f"{sum(1 for b in want.values() if b == bname)} on {bname} "
                        f"(${self.state['pods'][bname]['price']:.2f}/h); KING_BURST=1")

    def status(self) -> None:
        king = read_king(self.cfg)
        pub = self.state.get("published") or {}
        print(f"king: {king and king['served']} reign {king and king.get('reign')} ({king and king['kind']})")
        print(f"published: {pub.get('ident')} reign {pub.get('reign')} at {pub.get('base_url')} "
              f"-> {sorted((pub.get('datagen') or {}).keys())}")
        for name, m in sorted(self.state["pods"].items()):
            up = self.probe_models(m) if m.get("base_url") else False
            print(f"  {name}: {m.get('type')} {m.get('machine')} ${m.get('price', 0):.2f}/h "
                  f"rented {int((time.time() - m.get('rented_at', 0)) / 60)} min ago "
                  f"boot={'yes' if m.get('boot_started') else 'no'} "
                  f"ready={'yes' if m.get('ready_at') else 'no'} up_now={up} "
                  f"canary_fails={m.get('canary_fails', 0)} {m.get('base_url', '')}")
        for name, m in sorted((self.state.get("datagen") or {}).items()):
            if m.get("ssh_fail_since"):
                state = f"UNKNOWN (ssh unreachable {int((time.time() - m['ssh_fail_since']) / 60)} min)"
            elif m.get("down_since"):
                state = f"DOWN since {int((time.time() - m['down_since']) / 60)} min"
            else:
                state = "up"
            print(f"  datagen {name}: supervisor {state}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs="?", default="loop",
                    choices=["loop", "status", "unpublish", "pods"])
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    ctl = Controller(cfg)
    if args.cmd == "status":
        ctl.status()
        return 0
    if args.cmd == "pods":
        # `name host port` per datagen pod — for deploy_pods.sh --all.
        for name, pod in sorted(ctl.datagen_pods(lium_api.pods(ctl.sess) or []).items()):
            ssh = lium_api.parse_ssh(pod)
            if ssh:
                print(name, ssh[0], ssh[1])
        return 0
    if args.cmd == "unpublish":
        pods = lium_api.pods(ctl.sess) or []
        ctl.state["published"] = ctl.state.get("published") or {"forced": True}
        ctl.unpublish(ctl.datagen_pods(pods), "operator")
        save_state(ctl.state)
        return 0
    log(f"kingctl starting: prefix={cfg.pod_prefix} types={[t.name for t in cfg.types]} "
        f"budget=${cfg.budget_usd_hr}/h ttl={cfg.ttl_hours}h rotate@-{cfg.rotate_before_ttl_hours}h "
        f"canary every {cfg.canary_every_min} min discord={'on' if cfg.discord_enabled else 'off'}")
    while True:
        try:
            ctl.tick()
        except Exception as e:  # noqa: BLE001 — the loop must survive
            log(f"tick error: {e!r}")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
