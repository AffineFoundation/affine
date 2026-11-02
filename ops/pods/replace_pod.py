#!/usr/bin/env python3
"""SN120 Lium pod watch + one-command replacement (runs on the validator host; the Lium key never leaves it).

  replace_pod.py status [--json] [--quiet]     liveness of every role -> ~/subnet120-rewrite/state/pods/status.json (+ events.jsonl)
  replace_pod.py replace ROLE [--dry-run]      re-rent + provision + health + repoint + rm the exact dead pod id
                 [--node HUID] [--resume HUID] [--allow-manual]
  replace_pod.py snapshot ROLE                 golden copy of the role's code/runtime from the LIVE pod (no secrets) -> state/pods/golden/ROLE
  replace_pod.py verify ROLE                   read-only: does the live pod match what provisioning would produce?
  replace_pod.py repoint ROLE --to HUID [--dry-run]   only the host-side repoint (e.g. after a manual ROOT step)

Roles and their specs: roles.json next to this file. Safety:
  * never acts on excluded pods (Casper, Tobias, mech-1: roles.json _excluded) and never removes anything but the exact dead id
  * refuses when the role's current pod is up or only briefly unreachable (< dead_after_min), or when a reachable pod already
    carries the role's new name; at most max_rents_24h per role and _limits.max_rents_24h_global in total (ledger rents.jsonl)
  * no secrets are copied to a pod: code/data come from GitHub, the host's golden copy (secret patterns excluded and re-checked),
    and checkpoints from short-lived presigned R2 GETs (gate presign.py), deleted on the pod right after the fetch
  * every host file it edits is backed up first; each run writes runs/<ts>-<role>/revert.sh
"""
import argparse, fnmatch, glob, hashlib, json, os, re, shlex, subprocess, sys, tempfile, time
from pathlib import Path

HOME = Path.home()
HERE = Path(__file__).resolve().parent
RA = str(HOME / "subnet120-rewrite/state/root-audits")
CFG = json.loads(Path(os.environ.get("PODS_ROLES", HERE / "roles.json")).read_text())
LIUM = os.path.expanduser(os.environ.get("LIUM", "~/.lium/bin/lium"))
ST = Path(os.path.expanduser(os.environ.get("PODS_STATE", "~/subnet120-rewrite/state/pods")))
KH = HOME / ".lium/known_hosts"
HOST_PY = os.environ.get("HOST_PY", str(HOME / "subnet120-rewrite/.venv/bin/python"))
PRESIGN = f"{RA}/20261026-promotion-gate/code/presign.py"
GATE_STATUS = f"{RA}/20261026-promotion-gate/state/status.json"
LIFE = HOME / "subnet120-rewrite/state/mimo-mainnet-v1"
REPO_URL = "https://github.com/AffineFoundation/affine.git"
LIM = CFG["_limits"]; EXC = CFG["_excluded"]
ROLES = {k: v for k, v in CFG.items() if not k.startswith("_")}
# files that must never reach a pod. SECRET_TAR is excluded from every golden tar (narrow on purpose: venvs legitimately ship
# *.pem CA bundles etc.); SECRET_PATTERNS is checked on every single file pushed from the host.
SECRET_TAR = ["*.seed", "r2.env", "*.urls.private.json", "id_rsa*", "id_ed25519*", ".lium"]
SECRET_PATTERNS = SECRET_TAR + ["*.pem", "config.ini", "*credential*", "*secret*", "*.env"]
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def X(p): return os.path.expanduser(str(p).replace("{RA}", RA))
def now(): return time.time()
def cr(t=None): return time.strftime("%Y-%m-%d %H:%M CR", time.gmtime((t or now()) - 6 * 3600))
def scrub(s): return re.sub(r"sk_[A-Za-z0-9_\-…\.]{4,}", "sk_<redacted>", s or "")
def say(*a): print(*a, flush=True)


def wjson(path, obj, mode=0o644):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    t = path.with_name("." + path.name + ".tmp%d" % os.getpid()); t.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.chmod(t, mode); os.replace(t, path)


def rjson(path, default=None):
    try: return json.loads(Path(path).read_text())
    except Exception: return default


def append(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f: f.write(json.dumps(obj, sort_keys=True) + "\n")


# ---------------- lium ----------------
def lium(*a, timeout=180):
    p = subprocess.run([LIUM, *a], capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, scrub(p.stderr)


def lium_ps():
    rc, o, e = lium("ps", "--format", "json", timeout=90)
    if rc != 0: raise RuntimeError("lium ps failed: " + e[-300:])
    return json.loads(o)


def excluded(pod):
    if not pod: return False
    return bool(re.search(EXC["names_re"], pod.get("name") or "", re.I)) or pod.get("huid") in EXC["huids"] or pod.get("id") in EXC.get("ids", [])


# ---------------- current pod of a role (from the host file its service reads) ----------------
def current_ref(role):
    R = ROLES[role]; src = R["current_from"]
    if "glob_last" in src:
        fs = sorted(glob.glob(X(src["glob_last"])))
        if not fs: return None
        f = fs[-1]
    else: f = X(src["file"])
    try: txt = Path(f).read_text()
    except FileNotFoundError: return None
    if f.endswith(".json"):
        d = json.loads(txt); return dict(id=d.get("id") or d.get("lium_id"), huid=d.get("huid"), ip=d.get("ip"), port=d.get("port"), file=f)
    m = re.search(r"known_hosts/(" + UUID + ")", txt)
    e = re.search(r"""-p['"]?,?\s*['"]?(\d{2,5})['"]?,?\s*['"]?root@([\d.]+)""", txt)
    return dict(id=m.group(1) if m else None, ip=e.group(2) if e else None, port=int(e.group(1)) if e else None, file=f)


def ssh_argv(ip, port, pid, connect=15):
    kh = KH / pid
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=%s" % ("yes" if kh.exists() else "accept-new"),
            "-o", "ConnectTimeout=%d" % connect, "-o", "ServerAliveInterval=30", "-o", "UserKnownHostsFile=%s" % kh, "-p", str(port), "root@%s" % ip]


def ssh_ok(ip, port, pid, timeout=40):
    if not (ip and port and pid): return False
    try: return subprocess.run(ssh_argv(ip, port, pid) + ["-n", "true"], capture_output=True, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired: return False


class Pod:
    def __init__(s, pid, ip, port, huid=None, name=None):
        s.id, s.ip, s.port, s.huid, s.name = pid, ip, int(port), huid, name
    def run(s, cmd, inp=None, timeout=600, check=True):
        p = subprocess.run(ssh_argv(s.ip, s.port, s.id) + ([] if inp is not None else ["-n"]) + [cmd], input=inp, capture_output=True, timeout=timeout)
        out, err = p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")
        if check and p.returncode != 0: raise RuntimeError("pod rc=%d: %s" % (p.returncode, (err or out)[-600:]))
        return out
    def wrapper(s, role):
        return ("#!/bin/bash\n# SN120 %s pod %s (%s, lium %s), written by ops/pods/replace_pod.py %s\n"
                "exec ssh -T -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=60 -o ServerAliveInterval=30 "
                "-o UserKnownHostsFile=$HOME/.lium/known_hosts/%s -p %d root@%s \"$@\"\n") % (role, s.name, s.huid, s.id, cr(), s.id, s.port, s.ip)


# ---------------- status ----------------
def assess(role, ps, prev, probes=1):
    R = ROLES[role]; ref = current_ref(role) or {}
    listed = next((p for p in ps if ref.get("id") and p["id"] == ref["id"]), None)
    rec = dict(role=role, id=ref.get("id"), config=ref.get("file"), checked_at=now(), checked_cr=cr())
    if listed:
        rec.update(huid=listed["huid"], name=listed.get("name"), lium_status=listed.get("status"), price_h=listed.get("price_per_hour"),
                   gpu=listed.get("config"), ip=listed.get("ip") or ref.get("ip"), port=(listed.get("ports") or {}).get("22") or ref.get("port"))
    else: rec.update(ip=ref.get("ip"), port=ref.get("port"), huid=ref.get("huid"))
    if not ref.get("id"): rec["state"] = "no-config"; return rec
    if excluded(listed): rec["state"] = "excluded"; return rec
    if not listed: rec["state"] = "gone"; return rec
    if listed.get("status") != "RUNNING": rec["state"] = "down"; return rec
    ok = False
    for i in range(probes):
        ok = ssh_ok(rec["ip"], rec["port"], rec["id"])
        if ok: break
        if i + 1 < probes: time.sleep(20)
    p = (prev or {}) if (prev or {}).get("id") == rec["id"] else {}
    if ok: rec.update(state="up", unreachable_since=None, last_ok=now())
    else:
        since = p.get("unreachable_since") or now()
        rec.update(unreachable_since=since, last_ok=p.get("last_ok"),
                   state="dead" if now() - since >= 60 * LIM["dead_after_min"] else "unreachable")
    return rec


def cmd_status(a):
    ps = lium_ps(); old = rjson(ST / "status.json", {}) or {}
    out = dict(at=now(), at_cr=cr(), roles={}, unmapped=[])
    for role in ROLES:
        rec = assess(role, ps, (old.get("roles") or {}).get(role))
        R = ROLES[role]
        rec["action"] = None
        if rec["state"] in ("gone", "down", "dead"):
            rec["action"] = ("auto-replace: replace_pod.py replace %s" % role) if R.get("delegate") is None and R.get("auto") else \
                            ("delegated: %s" % R["delegate"]) if R.get("delegate") else ("manual: replace_pod.py replace %s --allow-manual (needs a ROOT step, see README)" % role)
            if R.get("optional") and not R.get("delegate"): rec["action"] = "optional role: replace only on request"
        out["roles"][role] = rec
        prev = (old.get("roles") or {}).get(role) or {}
        if prev.get("state") != rec["state"]:
            append(ST / "events.jsonl", dict(at=now(), at_cr=cr(), role=role, id=rec.get("id"), huid=rec.get("huid"), state=rec["state"], was=prev.get("state")))
    known = {r.get("id") for r in out["roles"].values()}
    for p in ps:
        if p["id"] not in known and not excluded(p) and (p.get("name") or "").startswith("affine-"):
            out["unmapped"].append(dict(huid=p["huid"], name=p.get("name"), id=p["id"], status=p.get("status"), price_h=p.get("price_per_hour")))
    wjson(ST / "status.json", out)
    lines = ["# SN120 pods %s" % out["at_cr"], "", "| role | pod | state | $/h | action |", "|---|---|---|---|---|"]
    for role, r in out["roles"].items():
        lines.append("| %s | %s %s | %s | %s | %s |" % (role, r.get("huid") or "-", (r.get("id") or "-")[:8], r["state"], r.get("price_h") or "-", r.get("action") or ""))
    if out["unmapped"]: lines += ["", "Unmapped affine-* pods: " + ", ".join("%s (%s)" % (u["huid"], u["name"]) for u in out["unmapped"])]
    (ST / "status.md").write_text("\n".join(lines) + "\n")
    if a.json: print(json.dumps(out, indent=1))
    elif not a.quiet: print("\n".join(lines))
    bad = [r for r, v in out["roles"].items() if v["state"] in ("gone", "down", "dead") and not ROLES[r].get("optional")]
    return 3 if bad else 0


# ---------------- node choice ----------------
def candidates(role, only=None, allow_gpu_change=False):
    R = ROLES[role]; out = []
    for spec in R["gpu_specs"]:
        if spec.get("gpu_change") and not allow_gpu_change: continue
        rc, o, e = lium("ls", "--gpu", spec["gpu"], "--format", "json", timeout=120)
        if rc != 0: say("  lium ls %s failed: %s" % (spec["gpu"], e[-200:])); continue
        rows = []
        for r in json.loads(o or "[]"):
            n = spec["count"]; avail = r.get("available_gpu_count", r["gpu_count"])
            if only and r["huid"] != only: continue
            if r.get("tier") not in spec.get("tiers", R.get("tiers", ["secure"])): continue
            if avail < n or (spec.get("whole_node") and r["gpu_count"] != n): continue
            if r.get("gpu_power_limited") is True: continue
            price = round(r["price_per_gpu_hour"] * n, 2)
            if price > R["max_price_h"] or (r.get("disk_gb") or 0) < R.get("min_disk_gb", 0): continue
            rows.append(dict(huid=r["huid"], gpu=spec["gpu"], count=n, node_gpus=r["gpu_count"], price_h=price, tier=r.get("tier"),
                             disk_gb=r.get("disk_gb"), country=r.get("country_code"), down=r.get("download_mbps")))
        rows.sort(key=lambda r: (r["price_h"], -(r["down"] or 0)))
        if rows: out.append((spec, rows))
        if rows and not only: break      # first spec with offers wins (fallbacks only when it has none)
    return [r for _, rows in out for r in rows]


def rents_24h(role=None):
    t0 = now() - 86400; n = 0
    try:
        for l in open(ST / "rents.jsonl"):
            r = json.loads(l)
            if r.get("event") == "rent" and r["at"] >= t0 and (role is None or r["role"] == role): n += 1
    except FileNotFoundError: pass
    return n


# ---------------- golden copy ----------------
def gdir(role):
    g = ROLES[role].get("golden") or {}
    return ST / "golden" / (g.get("from_role") or role)


def gspec(role):
    g = ROLES[role].get("golden") or {}
    return ROLES[g["from_role"]]["golden"] if g.get("from_role") else g


def listing(pod, g):
    """path -> size of what the golden tar holds, without reading file contents (GNU tar skips reads for /dev/null)."""
    ex = " ".join("--exclude=%s" % shlex.quote(x) for x in g.get("exclude", []) + SECRET_TAR)
    out = pod.run("cd / && tar -cvvf /dev/null %s %s 2>/dev/null; true" % (ex, " ".join(shlex.quote(p) for p in g["paths"])), timeout=1800)
    res = {}
    for l in out.splitlines():
        f = l.split(None, 5)
        if len(f) == 6: res[f[5].split(" -> ")[0]] = f[2]
    return res


def cmd_snapshot(a):
    role = a.role; R = ROLES[role]; g = gspec(role)
    if not g.get("paths"): say("%s has no golden paths" % role); return 0
    ref = current_ref(role); ps = lium_ps(); live = next((p for p in ps if p["id"] == ref["id"]), None)
    if not live or excluded(live): say("refuse: current pod not listed / excluded"); return 2
    pod = Pod(ref["id"], ref["ip"], ref["port"], live["huid"], live.get("name"))
    D = gdir(role); D.mkdir(parents=True, exist_ok=True); D.parent.chmod(0o700); ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ex = g.get("exclude", []) + SECRET_TAR
    exs = " ".join("--exclude=%s" % shlex.quote(x) for x in ex)
    paths = " ".join(shlex.quote(p) for p in g["paths"])
    lst = listing(pod, g); files = [(s, f) for f, s in lst.items()]
    bad = [f for _, f in files if any(fnmatch.fnmatch(os.path.basename(f.rstrip("/")), pat) for pat in SECRET_TAR) or "/private/" in "/" + f + "/"
           or os.path.basename(f.rstrip("/")) in ("r2.env", "config.ini") and ".lium" in f]
    if bad: say("refuse: secret-looking paths would be copied: %s" % bad[:5]); return 2
    out = D / ("software-%s.tar.zst" % ts); tmp = out.with_suffix(".partial")
    say("snapshot %s from %s (%s) -> %s (%d entries)" % (role, live["huid"], ref["id"][:8], out, len(files)))
    with open(tmp, "wb") as f:
        p1 = subprocess.Popen(ssh_argv(pod.ip, pod.port, pod.id) + ["-n", "cd / && nice -n 19 ionice -c3 tar -cf - %s %s" % (exs, paths)], stdout=subprocess.PIPE)
        p2 = subprocess.run(["zstd", "-q", "-T4", "-3", "-c"], stdin=p1.stdout, stdout=f); p1.wait()
    if p1.returncode not in (0, 1) or p2.returncode != 0: tmp.unlink(); say("snapshot failed rc=%s/%s" % (p1.returncode, p2.returncode)); return 1
    os.replace(tmp, out); os.chmod(out, 0o600)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    man = dict(role=role, pod=live["huid"], pod_id=ref["id"], at=now(), at_cr=cr(), tar=out.name, sha256=h.hexdigest(), bytes=out.stat().st_size,
               paths=g["paths"], exclude=ex, entries=len(files), listing="listing-%s.tsv" % ts)
    (D / man["listing"]).write_text("".join("%s\t%s\n" % (s, f) for s, f in files))
    wjson(D / "LATEST.json", man)
    for old in sorted(D.glob("software-*.tar.zst"))[:-2]:   # keep the newest two
        old.unlink(); (D / old.name.replace("software-", "listing-").replace(".tar.zst", ".tsv")).unlink(missing_ok=True)
    say("ok %s %.1f GB sha256 %s" % (out.name, man["bytes"] / 1e9, man["sha256"][:16])); return 0


# ---------------- provisioning ----------------
def head_and_files():
    head = (rjson(GATE_STATUS, {}) or {}).get("head")
    if not (isinstance(head, str) and len(head) == 64): raise RuntimeError("gate head unknown")
    d = rjson(LIFE / "CURRENT.json", {}) or {}
    if (d.get("checkpoint") or {}).get("id") == head: return head, d["checkpoint"]["files"]
    for p in sorted((p for p in (LIFE / "epochs").glob("*") if p.name.isdigit()), key=lambda p: -int(p.name))[:300]:
        r = rjson(p / "TRAINING-RESULT.json")
        if not r: continue
        desc = (((r.get("payload") or r).get("report") or {}).get("publication_receipt") or {}).get("payload", {}).get("descriptor") or {}
        if desc.get("id") == head: return head, desc["files"]
    raise RuntimeError("no descriptor for head %s" % head[:12])


def step(pod, role, s, ctx, dry):
    t = s["type"]; R = ROLES[role]
    if t == "clone":
        c = "command -v git >/dev/null || (apt-get -qq update && apt-get -qq install -y git) >/dev/null; D=%s; if [ -d $D/.git ]; then git -C $D pull -q --ff-only; else mkdir -p $(dirname $D) && git clone -q --depth 1 %s $D; fi; git -C $D log --oneline -1" % (s["dest"], REPO_URL)
        ctx["repo"] = s["dest"]
        return ("pod", c) if dry else pod.run(c, timeout=900)
    if t == "script":
        c = "cd %s && bash %s %s" % (ctx["repo"], s["script"], " ".join(shlex.quote(x) for x in s.get("args", [])))
        return ("pod", c) if dry else pod.run(c, timeout=s.get("timeout", 3600))
    if t in ("push_file", "push_dir"):
        src = X(s["src"])
        if s.get("optional") and not os.path.exists(src): return ("skip", "optional %s absent" % src) if dry else "optional %s absent: skipped" % src
        if any(fnmatch.fnmatch(os.path.basename(src), pat) for pat in SECRET_PATTERNS): raise RuntimeError("refuse secret-looking file %s" % src)
        if dry: return ("host->pod", "%s -> %s" % (src, s["dest"]))
        if t == "push_file":
            raw = Path(src).read_bytes()
            if s.get("sha256") and hashlib.sha256(raw).hexdigest() != s["sha256"]: raise RuntimeError("sha256 mismatch %s" % src)
            c = "mkdir -p $(dirname {d}) && {skip}cat > {d}.tmp && chmod {m} {d}.tmp && mv {d}.tmp {d} && sha256sum {d}".format(
                d=s["dest"], m=s.get("mode", "644"), skip=("[ -e %s ] && { cat >/dev/null; echo kept %s; exit 0; }; " % (s["dest"], s["dest"])) if s.get("if_missing") else "")
            return pod.run(c, inp=raw, timeout=1800)
        ex = sum((["--exclude", x] for x in s.get("exclude", []) + SECRET_PATTERNS), [])
        tar = subprocess.run(["tar", "-C", src, "-cf", "-", *ex, "."], capture_output=True, check=True).stdout
        return pod.run("mkdir -p %s && tar -C %s -xf - && echo pushed %s" % (s["dest"], s["dest"], s["dest"]), inp=tar, timeout=1800)
    if t == "pod_cmd":
        c = s["cmd"].replace("{head}", ctx.get("head") or "{head}")
        return ("pod", c) if dry else pod.run(c, timeout=s.get("timeout", 900))
    if t == "golden_restore":
        man = rjson(gdir(role) / "LATEST.json")
        if not man: raise RuntimeError("no golden copy for %s: run replace_pod.py snapshot %s while the pod is alive" % (role, role))
        g = gspec(role); links = g.get("links", {})
        lk = "; ".join("mkdir -p $(dirname %s); [ -L %s ] || [ ! -e %s ] && ln -sfn %s %s" % (k, k, k, v, k) for k, v in links.items())
        if dry: return ("host->pod", "zstd -dc %s | tar -C / -xpf -  (%.1f GB, from %s %s); links: %s" % (man["tar"], man["bytes"] / 1e9, man["pod"], man["at_cr"], lk or "-"))
        p1 = subprocess.Popen(["zstd", "-dc", str(gdir(role) / man["tar"])], stdout=subprocess.PIPE)
        p2 = subprocess.run(ssh_argv(pod.ip, pod.port, pod.id) + ["tar -C / -xpf - && echo restored"], stdin=p1.stdout, capture_output=True, timeout=7200); p1.wait()
        if p2.returncode != 0: raise RuntimeError("golden restore failed: %s" % p2.stderr.decode()[-300:])
        if lk: pod.run(lk + "; echo linked")
        return "restored %s (%s)" % (man["tar"], man["sha256"][:12])
    if t == "fetch_head":
        head, files = head_and_files() if not dry else ((rjson(GATE_STATUS, {}) or {}).get("head"), None)
        ctx["head"] = head; K = s["ckpt_dir"]
        if dry: return ("host+pod", "presign %s on host (gate presign.py) | pod: %s %s/<head>.descriptor.json %s <urls> --threads 32; rm urls" % ((head or "?")[:12], s["fetch_py"], K, K))
        if pod.run("test -f %s/%s/.verified && echo yes || true" % (K, head)).strip() == "yes": return "head %s already verified" % head[:12]
        blob = subprocess.run([HOST_PY, "-B", PRESIGN, head], input=json.dumps(files), capture_output=True, text=True, timeout=120)
        if blob.returncode != 0: raise RuntimeError("presign failed")
        c = ("set -e; umask 077; mkdir -p {K}/.work; cd {K}/.work && python3 -c \"import json,sys;d=json.load(sys.stdin);"
             "json.dump(d['descriptor'],open('{K}/{c}.descriptor.json','w'));json.dump(d['urls'],open('{c}.urls.private.json','w'))\" && "
             "(python3 {F} {K}/{c}.descriptor.json {K} {c}.urls.private.json --threads 32 >> {K}/{c}.fetch.log 2>&1; rc=$?; rm -f {c}.urls.private.json; "
             "tail -1 {K}/{c}.fetch.log; test -f {K}/{c}/.verified)").format(K=K, c=head, F=s["fetch_py"])
        return pod.run(c, inp=blob.stdout.encode(), timeout=5400)
    raise RuntimeError("unknown step " + t)


# ---------------- repoint (host) ----------------
def repoint_files(role):
    rp = ROLES[role].get("repoint") or {}
    fs = [X(f) for f in rp.get("files", [])]
    for d in rp.get("last_dropin", []):
        c = sorted(glob.glob(os.path.join(X(d), "*.conf")))
        if c: fs.append(c[-1])
    return [f for f in fs if os.path.exists(f)]


def substitute(text, old, new, subs=None):
    """Line-wise: only lines that name the old pod (its lium id, or its ip together with its port, or its huid) are changed."""
    out = []; n = 0
    for line in text.splitlines(keepends=True):
        hit = (old["id"] and old["id"] in line) or (old.get("ip") and old["ip"] in line and re.search(r"\b%s\b" % old["port"], line))
        if hit:   # huids are left alone on purpose: some files use the first pod's huid as a logical slot key (perf_service PODS)
            l2 = line.replace(old["id"], new["id"])
            if old.get("ip") and old.get("port") and (old["ip"] in l2):
                l2 = l2.replace(old["ip"], new["ip"]); l2 = re.sub(r"\b%s\b" % old["port"], str(new["port"]), l2)
            for rx, tpl in (subs or []):
                if new.get("gpu") and new.get("price_h") is not None: l2 = re.sub(rx, tpl.format(gpu=new["gpu"], price=new["price_h"]), l2)
            n += l2 != line; line = l2
        out.append(line)
    return "".join(out), n


def do_repoint(role, old, new, dry, run_dir):
    import difflib
    rp = ROLES[role].get("repoint") or {}; changed = []
    for f in repoint_files(role):
        t = Path(f).read_text(); t2, n = substitute(t, old, new, rp.get("line_subs"))
        if n == 0: say("  repoint %s: no line names the old pod" % f); continue
        if dry:
            say("".join(difflib.unified_diff(t.splitlines(True), t2.splitlines(True), f, f + " (new)", n=0)).rstrip()); continue
        b = Path(run_dir) / "backup" / f.lstrip("/"); b.parent.mkdir(parents=True, exist_ok=True); b.write_text(t); os.chmod(b, os.stat(f).st_mode & 0o777)
        tmp = f + ".tmp-replace-pod"; Path(tmp).write_text(t2); os.chmod(tmp, os.stat(f).st_mode & 0o777); os.replace(tmp, f); changed.append(f)
        say("  repointed %s (%d lines)" % (f, n))
    for r in rp.get("reset_json", []):
        f = X(r["file"]); cur = rjson(f, {}) or {}
        say("  %s %s: %s -> %s" % ("would set" if dry else "set", f, {k: cur.get(k) for k in r["set"]}, r["set"]))
        if not dry:
            b = Path(run_dir) / "backup" / f.lstrip("/"); b.parent.mkdir(parents=True, exist_ok=True)
            if os.path.exists(f): b.write_bytes(Path(f).read_bytes())
            cur.update(r["set"]); cur["reset_by"] = "replace_pod %s" % cr(); wjson(f, cur, 0o600); changed.append(f)
    return changed


def gpu_label(p):
    n = p.get("gpu_count") or 1; return "%sx%s" % (n, p.get("gpu_type") or (p.get("config") or "").split("\u00d7")[-1])


def systemctl(*a):
    p = subprocess.run(["systemctl", "--user", *a], capture_output=True, text=True); return p.returncode, (p.stdout + p.stderr).strip()


def write_revert(run_dir, changed, units):
    L = ["#!/bin/bash", "# revert replace_pod run %s: restore the host files it changed, reload, restart the units (the new pod is NOT removed)." % run_dir, "set -e"]
    for f in changed: L.append("cp -p %s %s" % (shlex.quote(str(Path(run_dir) / "backup" / f.lstrip("/"))), shlex.quote(f)))
    L.append("systemctl --user daemon-reload")
    for u in units: L.append("systemctl --user is-enabled --quiet %s && systemctl --user restart %s || true" % (u, u))
    L.append('echo reverted')
    p = Path(run_dir) / "revert.sh"; p.write_text("\n".join(L) + "\n"); os.chmod(p, 0o755); return p


def scan_leftovers(old):
    left = []
    for root in CFG.get("_scan_roots", []):
        r = subprocess.run(["grep", "-rlI", "--exclude=*.bak*", "--exclude-dir=backup", "--exclude-dir=__pycache__", "-e", old["id"], X(root)], capture_output=True, text=True)
        left += [l for l in r.stdout.splitlines() if ".bak" not in l and "before" not in l]
    return left


def cmd_repoint(a, old=None, new=None, run_dir=None):
    role = a.role; R = ROLES[role]
    old = old or current_ref(role)
    if new is None:
        p = next((p for p in lium_ps() if p["huid"] == a.to), None)
        if not p or excluded(p): say("refuse: %s not listed or excluded" % a.to); return 2
        new = dict(id=p["id"], ip=p["ip"], port=p["ports"]["22"], huid=p["huid"], gpu=gpu_label(p), price_h=p.get("price_per_hour"))
    if old.get("huid") is None:
        o = next((p for p in lium_ps() if p["id"] == old["id"]), None); old["huid"] = (o or {}).get("huid")
    rp = R.get("repoint") or {}; units = rp.get("units_restart", [])
    say("repoint %s: %s %s:%s -> %s %s:%s" % (role, (old["id"] or "?")[:8], old.get("ip"), old.get("port"), new["id"][:8], new["ip"], new["port"]))
    if a.dry_run: do_repoint(role, old, new, True, None); say("  units: stop %s; restart %s" % (rp.get("units_stop_first", []), units)); return 0
    run_dir = run_dir or ST / "runs" / ("%s-%s-repoint" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), role)); Path(run_dir).mkdir(parents=True, exist_ok=True)
    for u in rp.get("units_stop_first", []): say("  stop %s: %s" % (u, systemctl("stop", u)))
    changed = do_repoint(role, old, new, False, run_dir); rv = write_revert(run_dir, changed, units)
    systemctl("daemon-reload")
    for u in units:
        rc, _ = systemctl("is-enabled", "--quiet", u)
        if rc == 0: say("  restart %s: %s" % (u, systemctl("restart", u)[0]))
        else: say("  %s not enabled: left as is" % u)
    left = scan_leftovers(old)
    if left: say("  NOTE files still naming the old pod (not live config, review): %s" % left[:12])
    if getattr(a, "rm_old", False): rm_old(role, old)
    say("  revert: bash %s" % rv); return 0


def rm_old(role, old):
    """lium rm of the exact old id, only if Lium still lists it, it is not excluded, carries one of the role's names and is not reachable."""
    R = ROLES[role]; o = next((p for p in lium_ps() if p["id"] == old["id"]), None)
    if not o: say("  old pod %s no longer listed" % old["id"][:8]); return
    if excluded(o) or o.get("name") not in R["names"] + [R["name"]]: say("  old pod %s not removed (name %s)" % (o["huid"], o.get("name"))); return
    if o.get("status") == "RUNNING" and ssh_ok(o["ip"], o["ports"]["22"], o["id"]): say("  old pod %s is reachable again: NOT removed" % o["huid"]); return
    rc, out, e = lium("rm", o["id"], "-y"); say("  lium rm %s (exact id, %s): rc=%d %s" % (o["id"], o["huid"], rc, (out or e).strip()[-200:]))


# ---------------- replace ----------------
def cmd_replace(a):
    import fcntl
    role = a.role; R = ROLES.get(role)
    if not R: say("unknown role %s (have %s)" % (role, ", ".join(ROLES))); return 2
    if R.get("delegate"): say("%s: %s" % (role, R["delegate"])); return 0
    ST.mkdir(parents=True, exist_ok=True); lk = open(ST / ".replace.lock", "w")
    try: fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: say("refuse: another replace_pod run holds the lock"); return 2
    dry = a.dry_run; verdict = []
    ps = lium_ps(); old_status = (rjson(ST / "status.json", {}) or {}).get("roles", {}).get(role)
    cur = assess(role, ps, old_status, probes=1 if dry else 3)
    say("== %s (%s)\n current: %s %s state=%s lium=%s" % (role, R["what"], cur.get("huid"), (cur.get("id") or "")[:8], cur["state"], cur.get("lium_status")))
    if cur["state"] == "excluded": say("refuse: excluded pod"); return 2
    if cur["state"] == "up": verdict.append("current pod is up (live) -> nothing to replace")
    if cur["state"] == "unreachable": verdict.append("unreachable for %d min < dead_after_min %d" % ((now() - cur["unreachable_since"]) / 60, LIM["dead_after_min"]))
    resume = None
    for p in ps:
        if p.get("name") == R["name"] and p["id"] != cur.get("id") and not excluded(p):
            if a.resume == p["huid"]: resume = p
            elif ssh_ok(p["ip"], p["ports"]["22"], p["id"]): verdict.append("a reachable pod %s already carries the name %s (use --resume %s)" % (p["huid"], R["name"], p["huid"]))
    if R.get("needs_secrets") and not a.allow_manual: verdict.append("role needs a ROOT secret-staging step: rerun with --allow-manual to rent + restore the public runtime only")
    if not resume:
        if rents_24h(role) >= R.get("max_rents_24h", 2): verdict.append("role rent cap %d/24h reached" % R.get("max_rents_24h", 2))
        if rents_24h() >= LIM["max_rents_24h_global"]: verdict.append("global rent cap %d/24h reached" % LIM["max_rents_24h_global"])
    cands = [] if resume else candidates(role, a.node, a.allow_gpu_change)
    say(" spec: %s, <= $%s/h, disk >= %s GB, tiers %s" % (", ".join("%dx%s%s%s" % (s["count"], s["gpu"], " (whole node)" if s.get("whole_node") else "",
        " [only with --allow-gpu-change]" if s.get("gpu_change") else "") for s in R["gpu_specs"]), R["max_price_h"], R.get("min_disk_gb"), R.get("tiers")))
    if not a.allow_gpu_change and any(s.get("gpu_change") for s in R["gpu_specs"]) and not resume:
        alt = candidates(role, a.node, True)
        if alt and not cands: say("   (with --allow-gpu-change: %s %dx%s $%s/h %s)" % (alt[0]["huid"], alt[0]["count"], alt[0]["gpu"], alt[0]["price_h"], alt[0]["tier"]))
    for c in cands[:5]: say("   candidate %(huid)s %(count)dx%(gpu)s of %(node_gpus)d  $%(price_h)s/h  %(tier)s disk %(disk_gb)s GB %(country)s" % c)
    if not cands and not resume: verdict.append("no node offered within the spec")
    say(" rents last 24h: role %d, global %d" % (rents_24h(role), rents_24h()))
    if dry:
        fake = dict(id="NEW-LIUM-ID-0000-0000-000000000000", ip="203.0.113.1", port=40000, huid=(cands[0]["huid"] if cands else "new-huid"),
                    gpu=("%dx%s" % (cands[0]["count"], cands[0]["gpu"])) if cands else None, price_h=cands[0]["price_h"] if cands else None)
        ctx = {}
        say(" plan:\n  1 rent: lium up %s -c %s --name %s --verify-gpus --strict-gpus --no-volume-encryption -y --json" % (fake["huid"], cands[0]["count"] if cands else "?", R["name"]))
        say("  2 ssh wrapper -> %s" % (ST / "ssh" / ("%s.sh" % role)))
        for i, s in enumerate(R.get("provision", [])):
            try: r = step(None, role, s, ctx, True)
            except Exception as e: r = ("ERROR", str(e))
            say("  3.%d %s: %s" % (i + 1, r[0], r[1] if isinstance(r, tuple) else r))
        say("  4 health: %s" % R.get("health"))
        if R.get("needs_secrets"): say("  5 STOP: ROOT stages the private files (README 'trainer/evaluator'), then: replace_pod.py repoint %s --to <huid>" % role)
        else:
            say("  5 repoint host config (diff with a placeholder new pod):")
            old = dict(id=cur.get("id"), ip=cur.get("ip"), port=cur.get("port"), huid=cur.get("huid"))
            if old["id"]: do_repoint(role, old, fake, True, None)
            rp = R.get("repoint") or {}; say("    units: stop first %s; restart %s" % (rp.get("units_stop_first", []), rp.get("units_restart", [])))
        say("  6 remove the old pod by exact id if Lium still lists it: lium rm %s -y" % cur.get("id"))
        say(" verdict: %s" % ("; ".join(verdict) if verdict else "WOULD REPLACE (pod is dead and guards pass)"))
        return 0
    if verdict: say("refuse: " + "; ".join(verdict)); return 2
    run_dir = ST / "runs" / ("%s-%s" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), role)); run_dir.mkdir(parents=True, exist_ok=True)
    append(ST / "rents.jsonl", dict(at=now(), at_cr=cr(), role=role, event="start", old=cur.get("id"), old_state=cur["state"]))
    if resume: new = resume
    else:
        new = None
        for c in cands[:3]:
            say(" renting %s %dx%s $%s/h as %s" % (c["huid"], c["count"], c["gpu"], c["price_h"], R["name"]))
            append(ST / "rents.jsonl", dict(at=now(), at_cr=cr(), role=role, event="rent", node=c["huid"], price_h=c["price_h"], gpu="%dx%s" % (c["count"], c["gpu"])))
            rc, o, e = lium("up", c["huid"], "-c", str(c["count"]), "--name", R["name"], "--verify-gpus", "--strict-gpus", "--no-volume-encryption",
                            "-y", "--json", "--timeout", "900", timeout=1000)
            if rc == 0:
                try: new = json.loads(o)["pod"]; break
                except Exception: pass
            say("  rent failed rc=%d %s" % (rc, (e or o)[-300:]))
        if not new: say("no rent succeeded"); return 1
    pod = Pod(new["id"], new["ip"], (new.get("ports") or {}).get("22"), new.get("huid"), new.get("name"))
    append(ST / "rents.jsonl", dict(at=now(), at_cr=cr(), role=role, event="pod", id=pod.id, huid=pod.huid, ip=pod.ip, port=pod.port))
    w = ST / "ssh" / ("%s.sh" % role); w.parent.mkdir(parents=True, exist_ok=True); w.write_text(pod.wrapper(role)); os.chmod(w, 0o755)
    say(" new pod %s %s:%s (wrapper %s)" % (pod.huid, pod.ip, pod.port, w))
    for _ in range(40):
        if ssh_ok(pod.ip, pod.port, pod.id): break
        time.sleep(15)
    else: say("new pod never reachable over ssh; left running for inspection: %s" % pod.id); return 1
    ctx = {}
    try:
        for i, s in enumerate(R.get("provision", [])):
            t0 = now(); r = step(pod, role, s, ctx, False)
            say("  3.%d %s ok (%ds) %s" % (i + 1, s["type"], now() - t0, str(r).strip().splitlines()[-1][-160:] if str(r).strip() else ""))
        h = ""
        for _ in range(10):
            try: h = pod.run(R["health"], timeout=300); break
            except Exception as e: h = "FAIL " + str(e)[-200:]; time.sleep(30)
        say("  health: %s" % h.strip()[-200:])
        if h.startswith("FAIL"): raise RuntimeError("health check failed")
    except Exception as e:
        append(ST / "rents.jsonl", dict(at=now(), at_cr=cr(), role=role, event="provision-failed", id=pod.id, err=str(e)[-300:]))
        say("PROVISION FAILED (%s). Pod %s kept for inspection; rerun with --resume %s (idempotent)." % (str(e)[-300:], pod.huid, pod.huid)); return 1
    if R.get("needs_secrets"):
        say(" STOP: public runtime restored on %s. ROOT must stage the private files (README), then: replace_pod.py repoint %s --to %s" % (pod.huid, role, pod.huid))
        append(ST / "rents.jsonl", dict(at=now(), at_cr=cr(), role=role, event="awaiting-root", id=pod.id)); return 0
    old = dict(id=cur.get("id"), ip=cur.get("ip"), port=cur.get("port"), huid=cur.get("huid"))
    a.dry_run = False
    gp = next((p for p in lium_ps() if p["id"] == pod.id), {})
    cmd_repoint(a, old, dict(id=pod.id, ip=pod.ip, port=pod.port, huid=pod.huid, gpu=gpu_label(gp) if gp else None, price_h=gp.get("price_per_hour")), run_dir)
    if old["id"] != pod.id: rm_old(role, old)
    append(ST / "rents.jsonl", dict(at=now(), at_cr=cr(), role=role, event="done", id=pod.id, huid=pod.huid, run_dir=str(run_dir)))
    say("done: %s -> %s; revert host config: bash %s/revert.sh" % (role, pod.huid, run_dir)); return 0


# ---------------- verify (read-only) ----------------
def cmd_verify(a):
    role = a.role; R = ROLES[role]; ref = current_ref(role)
    ps = lium_ps(); live = next((p for p in ps if p["id"] == ref["id"]), None)
    if not live or excluded(live): say("refuse: not listed / excluded"); return 2
    pod = Pod(ref["id"], ref["ip"], ref["port"], live["huid"]); rc = 0
    say("== verify %s on %s (%s)" % (role, live["huid"], ref["id"][:8]))
    v = R.get("verify") or {}
    if v.get("venv"):
        # same comparison vllm_venv.sh CHECK=1 does, with the requirements streamed from this host's checkout
        req = (HOME / "bench-work/affine/bench/deploy/requirements-vllm.txt").read_text()
        want = sorted({l.strip().lower().replace("_", "-") for l in req.splitlines() if l.strip() and not l.startswith("#")} | {x.lower() for x in v.get("extra_pins", [])})
        have = pod.run("uv pip freeze -q --python %s/bin/python 2>/dev/null || %s/bin/python -m pip freeze" % (v["venv"], v["venv"]), timeout=120)
        have = {l.strip().lower().replace("_", "-") for l in have.splitlines() if "==" in l}
        miss = [w for w in want if w not in have]
        say(" venv %s: %d required pins, %d missing/different %s" % (v["venv"], len(want), len(miss), miss[:8]))
        rc |= bool(miss)
    for dst, src in (v.get("files") or {}).items():
        h1 = hashlib.sha256(Path(X(src)).read_bytes()).hexdigest(); h2 = (pod.run("sha256sum %s 2>/dev/null || true" % dst).split() or ["missing"])[0]
        say(" %s %s vs host %s" % ("same" if h1 == h2 else "DIFF", dst, src)); rc |= h1 != h2
    man = rjson(gdir(role) / "LATEST.json")
    if gspec(role).get("paths"):
        if not man: say(" golden: NONE (run snapshot %s)" % role); rc = 1
        else:
            now_ = listing(pod, gspec(role))
            then = dict(l.split("\t", 1)[::-1] for l in (gdir(role) / man["listing"]).read_text().splitlines() if "\t" in l)
            add = sorted(set(now_) - set(then)); gone = sorted(set(then) - set(now_)); chg = sorted(k for k in set(now_) & set(then) if now_[k] != then[k])
            say(" golden %s (%s, %s): live has %d new, %d removed, %d size-changed files %s" % (man["tar"], man["pod"], man["at_cr"], len(add), len(gone), len(chg), (add + chg)[:6]))
            if add or chg or gone: rc = 1
    try: say(" health: %s" % pod.run(R["health"], timeout=300).strip()[-200:])
    except Exception as e: say(" health FAIL %s" % str(e)[-200:]); rc = 1
    say(" verdict: %s" % ("MATCH" if rc == 0 else "DRIFT (see above)")); return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("status"); s.add_argument("--json", action="store_true"); s.add_argument("--quiet", action="store_true")
    s = sp.add_parser("replace"); s.add_argument("role"); s.add_argument("--dry-run", action="store_true"); s.add_argument("--node")
    s.add_argument("--resume"); s.add_argument("--allow-manual", action="store_true")
    s.add_argument("--allow-gpu-change", action="store_true", help="also use the role's fallback GPU types marked gpu_change (probe: B300/H200 instead of B200)")
    s = sp.add_parser("snapshot"); s.add_argument("role")
    s = sp.add_parser("verify"); s.add_argument("role")
    s = sp.add_parser("repoint"); s.add_argument("role"); s.add_argument("--to", required=True); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--rm-old", action="store_true", help="also lium rm the old pod's exact id if it is listed and unreachable")
    a = ap.parse_args()
    if getattr(a, "role", None) and a.role not in ROLES: say("unknown role %s (have %s)" % (a.role, ", ".join(ROLES))); return 2
    return dict(status=cmd_status, replace=cmd_replace, snapshot=cmd_snapshot, verify=cmd_verify, repoint=cmd_repoint)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
