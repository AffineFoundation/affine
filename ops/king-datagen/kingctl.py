#!/usr/bin/env python
"""King-seat controller: serve the current SN120 king for datagen.

Loop (pm2 `affine-king-datagen`, every --interval seconds):

  1. read the king from the validator's state.json (repo / revision /
     reign). An R2 king is served from the PUBLIC copy on models.affine.io
     (only crowned models live there; no credentials); the HF genesis is
     served from its pinned revision.
  2. make sure ONE Lium pod `king-dg-<ident>` exists for that king: rent
     (first [[types]] entry with stock under its price cap), push env +
     bootstrap_king.sh, launch it, wait for every replica to answer.
  3. once the box answers a real completion, write /root/rollouts/.king_env
     on every `affine-datagen*` pod (base_url / model / key / digest /
     reign). rollouts/king.py re-reads that file each cycle, so the `king_*`
     policies start routing to the new king with no restart.
  4. release the previous king's box, and any box that never came up
     (bootstrap timeout -> executor strike) or went dark after it did
     (unreachable grace). While no king box answers, .king_env is emptied
     so the king policies go idle instead of burning containers on a dead
     endpoint.

State: state/state.json (pods memory + what is published). Secrets: the
per-box bearer is generated here and lives only in that state file (0600)
and on the pods. Lium API key via ops/teacher-swarm/lium_api.py. Never
prints a secret.

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
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "teacher-swarm"))

import lium_api  # noqa: E402

STATE_DIR = HERE / "state"
STATE_JSON = STATE_DIR / "state.json"
KNOWN_HOSTS = STATE_DIR / "known_hosts"
BLACKLIST = STATE_DIR / "blacklist.txt"
VALIDATOR_ENV = Path.home() / ".affine-validator.env"
REPLICA_PORT_BASE = 31001          # loopback-only vLLM ports behind nginx
RENT_SETTLE_S = 10 * 60            # a rented pod may take this long to list
REPUBLISH_EVERY_S = 6 * 3600       # periodic full re-push (new datagen pods)

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
    "-o", "ConnectTimeout=15",
    "-o", "BatchMode=yes",
    "-o", "LogLevel=ERROR",
]


def log(msg: str) -> None:
    print(f"[kingctl] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}",
          flush=True)


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
    bootstrap_timeout_min: int
    unreachable_grace_min: int
    min_balance_usd: float
    budget_usd_hr: float
    vllm_version: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    types: tuple[TypePlan, ...]


def load_config(path: Path = HERE / "king.toml") -> Config:
    raw = tomllib.loads(path.read_text())
    k = raw["king"]
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
        bootstrap_timeout_min=int(k["bootstrap_timeout_min"]),
        unreachable_grace_min=int(k["unreachable_grace_min"]),
        min_balance_usd=float(k["min_balance_usd"]),
        budget_usd_hr=float(k["budget_usd_hr"]),
        vllm_version=str(k["vllm_version"]),
        max_model_len=int(k["max_model_len"]),
        gpu_memory_utilization=float(k["gpu_memory_utilization"]),
        max_num_batched_tokens=int(k["max_num_batched_tokens"]),
        max_num_seqs=int(k["max_num_seqs"]),
        types=types,
    )


def env_file_value(name: str) -> str:
    """One value from the validator env snapshot (HF_TOKEN for the genesis
    king). Missing file / key -> ""."""
    if not VALIDATOR_ENV.exists():
        return ""
    for line in VALIDATOR_ENV.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


# ------------------------------------------------------------------- king
def current_king(cfg: Config) -> dict | None:
    """Who is king right now, as the controller needs it:
    {ident, served, reign, kind: r2|hf, digest | hf_model+hf_rev}."""
    try:
        st = json.loads(cfg.state_json.read_text())
    except (OSError, ValueError) as e:
        log(f"state.json unreadable: {e!r}")
        return None
    k = st.get("king") or {}
    repo, rev = str(k.get("repo") or ""), str(k.get("revision") or "")
    if not repo or not rev:
        return None
    reign = k.get("reign_number")
    if repo.startswith("r2://"):
        return {"kind": "r2", "digest": rev, "ident": rev[:12],
                "served": f"king-{rev[:12]}", "reign": reign, "repo": repo}
    ident = hashlib.sha256(f"{repo}@{rev}".encode()).hexdigest()[:12]
    return {"kind": "hf", "hf_model": repo, "hf_rev": rev, "ident": ident,
            "served": f"king-{ident}", "reign": reign, "repo": repo}


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
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())
    return {"pods": {}, "published": {}}


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
        f.write(f"{executor_id} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {reason}\n")


# --------------------------------------------------------------- controller
class Controller:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sess = lium_api.session()
        self.state = load_state()
        try:
            self.pubkey = (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()
        except OSError:
            self.pubkey = ""

    # ---- lium views -----------------------------------------------------------
    def my_pods(self, pods: list[dict]) -> dict[str, dict]:
        return {lium_api.pod_name(p): p for p in pods
                if lium_api.pod_name(p).startswith(self.cfg.pod_prefix)}

    def datagen_pods(self, pods: list[dict]) -> dict[str, dict]:
        return {lium_api.pod_name(p): p for p in pods
                if lium_api.pod_name(p).startswith(self.cfg.datagen_pod_prefix)
                and str(p.get("status", "")).upper() in ("RUNNING", "")}

    def spend(self, mine: dict[str, dict]) -> float:
        return sum(lium_api.pod_price(p) for p in mine.values())

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

    def rent(self, king: dict, mine: dict[str, dict]) -> None:
        name = f"{self.cfg.pod_prefix}{king['ident']}"
        bal = lium_api.balance_usd()
        if bal is not None and bal < self.cfg.min_balance_usd:
            log(f"rent {name}: balance ${bal:.0f} < ${self.cfg.min_balance_usd:.0f}; not renting")
            return
        spend = self.spend(mine)
        stock = lium_api.executors(self.sess)
        for plan in self.cfg.types:
            for cand in self.match_stock(stock, plan):
                price = (cand.get("price_per_gpu") or 0) * plan.gpu_count
                if spend + price > self.cfg.budget_usd_hr:
                    log(f"rent {name}: ${spend:.2f}+${price:.2f} > budget "
                        f"${self.cfg.budget_usd_hr:.2f}; skipping {plan.name}")
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
                }
                log(f"rented {name}: {plan.name} {cand.get('machine_name')} "
                    f"${price:.2f}/h executor={str(cand['id'])[:12]}")
                return
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
        plan = next(t for t in self.cfg.types if t.name == mem["type"])
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
    def probe(self, mem: dict, *, canary: bool = False) -> bool:
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
        if not canary:
            return True
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

    # ---- publish --------------------------------------------------------------
    def king_env_text(self, mem: dict | None) -> str:
        if mem is None:
            return "# king seat: no king box is serving right now (kingctl)\n"
        k = mem["king"]
        return (f"# written by ops/king-datagen/kingctl.py {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                f"KING_BASE_URL={mem['base_url']}\n"
                f"KING_MODEL={k['served']}\n"
                f"KING_KEY={mem['key']}\n"
                f"KING_DIGEST={k.get('digest') or k.get('hf_rev')}\n"
                f"KING_REIGN={k.get('reign')}\n")

    def push_king_env(self, datagen: dict[str, dict], mem: dict | None,
                      only: set[str] | None = None) -> dict[str, bool]:
        text = self.king_env_text(mem)
        out: dict[str, bool] = {}
        for name, pod in sorted(datagen.items()):
            if only is not None and name not in only:
                continue
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

    def publish(self, datagen: dict[str, dict], pod_name: str, mem: dict) -> None:
        res = self.push_king_env(datagen, mem)
        now = time.time()
        self.state["published"] = {
            "pod": pod_name, "ident": mem["king"]["ident"], "reign": mem["king"].get("reign"),
            "base_url": mem["base_url"], "at": now,
            "datagen": {n: now for n, ok in res.items() if ok},
        }
        log(f"published {mem['king']['served']} (reign {mem['king'].get('reign')}) at "
            f"{mem['base_url']} to {sum(res.values())}/{len(res)} datagen pods "
            f"{[n for n, ok in res.items() if not ok] or ''}")

    def unpublish(self, datagen: dict[str, dict], reason: str) -> None:
        if not self.state.get("published"):
            return
        res = self.push_king_env(datagen, None)
        log(f"unpublished ({reason}); emptied .king_env on "
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

    # ---- tick -----------------------------------------------------------------
    def tick(self) -> None:
        cfg = self.cfg
        king = current_king(cfg)
        pods = lium_api.pods(self.sess)
        if pods is None:
            log("lium /pods failed; skipping tick")
            return
        mine = self.my_pods(pods)
        datagen = self.datagen_pods(pods)
        now = time.time()

        # Forget memory of pods Lium no longer lists (after the rent settles).
        for name in list(self.state["pods"]):
            m = self.state["pods"][name]
            if name not in mine and now - m.get("rented_at", 0) > RENT_SETTLE_S:
                log(f"{name}: gone from lium; forgetting")
                self.state["pods"].pop(name)
        # Pods we hold no memory for (controller restart with lost state):
        # we cannot know their key -> release them.
        for name in list(mine):
            if name not in self.state["pods"]:
                self.remove(name, "no memory of this pod")
                mine.pop(name, None)

        if king is None:
            log("no king in state.json")
            self.unpublish(datagen, "no king")
            save_state(self.state)
            return

        target = f"{cfg.pod_prefix}{king['ident']}"
        mem = self.state["pods"].get(target)
        pod = mine.get(target)
        serving = False

        if mem is None:
            self.rent(king, mine)
        elif pod is None:
            log(f"{target}: rented {int((now - mem['rented_at']) / 60)} min ago, not listed yet")
        elif not mem.get("boot_started"):
            self.bootstrap(target, pod, mem)
        else:
            up = self.probe(mem, canary=not mem.get("ready_at"))
            if up:
                mem["last_ok"] = now
                if not mem.get("ready_at"):
                    mem["ready_at"] = now
                    log(f"{target}: READY after {int((now - mem['boot_started']) / 60)} min")
                serving = True
            else:
                boot_age = now - mem["boot_started"]
                if not mem.get("ready_at"):
                    failed, last = self.boot_report(pod)
                    if failed:
                        self.remove(target, f"bootstrap failed: {failed[:80]}", strike=True)
                    elif boot_age > cfg.bootstrap_timeout_min * 60:
                        self.remove(target, f"bootstrap timeout: {last[:100]}", strike=True)
                    elif int(boot_age) % 300 < 60:
                        log(f"{target}: booting {int(boot_age / 60)} min ({last or 'no log yet'})")
                elif now - mem.get("last_ok", mem["ready_at"]) > cfg.unreachable_grace_min * 60:
                    self.remove(target, "unreachable after ready")

        pub = self.state.get("published") or {}
        if serving:
            mem = self.state["pods"][target]
            if pub.get("ident") != king["ident"] or pub.get("base_url") != mem["base_url"]:
                self.publish(datagen, target, mem)
            else:
                missing = {n for n in datagen if n not in (pub.get("datagen") or {})}
                if missing or now - pub.get("at", 0) > REPUBLISH_EVERY_S:
                    res = self.push_king_env(datagen, mem, only=missing or None)
                    pub.setdefault("datagen", {}).update({n: now for n, ok in res.items() if ok})
                    if not missing:
                        pub["at"] = now
                    log(f"re-pushed .king_env to {sum(res.values())}/{len(res)} pods")
            # The previous king's box is only released once the new one serves.
            for name in list(mine):
                if name != target:
                    self.remove(name, f"superseded by {target}")
        elif pub:
            # The published box (the old king during a swap, or the current one
            # between probes) must keep answering; if it is gone or dark past the
            # grace, idle the king policies rather than let them strike a dead
            # endpoint batch after batch.
            pmem = self.state["pods"].get(pub.get("pod") or "")
            if pmem is None:
                self.unpublish(datagen, "published box gone")
            elif self.probe(pmem):
                pmem["last_ok"] = now
            else:
                dark = now - pmem.get("last_ok", pmem.get("ready_at", now))
                if dark > cfg.unreachable_grace_min * 60:
                    self.unpublish(datagen, f"published box dark {int(dark / 60)} min")
        save_state(self.state)

    def status(self) -> None:
        king = current_king(self.cfg)
        pub = self.state.get("published") or {}
        print(f"king: {king and king['served']} reign {king and king.get('reign')} ({king and king['kind']})")
        print(f"published: {pub.get('ident')} reign {pub.get('reign')} at {pub.get('base_url')} "
              f"-> {sorted((pub.get('datagen') or {}).keys())}")
        for name, m in sorted(self.state["pods"].items()):
            up = self.probe(m) if m.get("base_url") else False
            print(f"  {name}: {m.get('type')} {m.get('machine')} ${m.get('price', 0):.2f}/h "
                  f"rented {int((time.time() - m.get('rented_at', 0)) / 60)} min ago "
                  f"boot={'yes' if m.get('boot_started') else 'no'} "
                  f"ready={'yes' if m.get('ready_at') else 'no'} up_now={up} {m.get('base_url', '')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs="?", default="loop", choices=["loop", "status", "unpublish"])
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    ctl = Controller(cfg)
    if args.cmd == "status":
        ctl.status()
        return 0
    if args.cmd == "unpublish":
        pods = lium_api.pods(ctl.sess) or []
        ctl.state["published"] = ctl.state.get("published") or {"forced": True}
        ctl.unpublish(ctl.datagen_pods(pods), "operator")
        save_state(ctl.state)
        return 0
    log(f"kingctl starting: prefix={cfg.pod_prefix} types={[t.name for t in cfg.types]} "
        f"budget=${cfg.budget_usd_hr}/h")
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
