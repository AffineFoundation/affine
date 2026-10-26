#!/usr/bin/env python3
"""Sign v829i: v829h-collect1200 + promotion gate wrapper (flag staged/PROMOTION-GATE.json, default OFF).
One release: release-v829i-gate-signed. Revert = flag off (no restart) or reactivate release-v829h-collect1200-signed.
Operator config = exact v829h field set/values except host_files hashes. Seed never printed."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT_ID = "3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd"
RA = Path("/home/const/subnet120-rewrite/state/root-audits")
SRC = RA / "20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822"
V828 = RA / "20261021-v828-audit-off-path/release-v828-signed"
AUDIT = RA / "20261023-toploc-credit-train-gate"
G = AUDIT / "release-v829h-collect1200-signed"
STAGE = Path("/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/release/stage")
VERSION = "affine-host-v829-train-authenticity-gate-v1"
SCOPE = "host-train-authenticity-gate-and-full-verify-v1"
OPN = "v829-operator.ROOT-ADMITTED.private.json"

def need(ok, why):
    if not ok:
        raise SystemExit("REFUSED: " + why)
hf = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    for d in (V828, G):
        need(subprocess.run(["sha256sum", "--quiet", "-c", "SIGNED-MANIFEST.sha256"], cwd=d).returncode == 0, "manifest " + d.name)
    sys.path.insert(0, str(SRC))
    from subnet.storage import Identity, canonical, sha
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import sign
    gop = json.loads((G / OPN).read_text())
    life = json.loads((RA / "20261017-v824/release-v824-signed/lifecycle.v8.2.4.ACTIVE.private.json").read_text())
    seed_path = Path(life["authority_seed_file"])
    need(seed_path.is_file() and not seed_path.is_symlink() and seed_path.stat().st_mode & 0o077 == 0, "seed mode")
    actor = Identity(bytes.fromhex(seed_path.read_text().strip()))
    need(actor.id == ROOT_ID, "ROOT id")
    need(hf(STAGE / "train_gate_authenticity.py") == gop["host_files"]["train_gate_authenticity.py"], "auth file unchanged")
    results = {}
    for seconds in (1200,):
        out = AUDIT / "release-v829i-gate-signed"
        need(not out.exists(), "output exists " + out.name)
        st = STAGE
        host_files = {n: hf(st / n) for n in ("mimo_v829_train_gate_launcher.py", "train_gate_authenticity.py")}
        prop = {k: v for k, v in gop.items() if k != "admission"}
        prop["host_files"] = host_files
        need(set(prop) | {"admission"} == set(gop), "exact v829h field set")
        cfg = dict(prop, admission=sign(actor, dict(version=VERSION, configuration_sha256=sha(canonical(prop)),
                                                     source_unchanged=True, scope=SCOPE)))
        out.mkdir(mode=0o700)
        def put(name, raw):
            fd = os.open(out / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
        for n in host_files:
            put(n, (st / n).read_bytes())
        put(OPN, canonical(cfg))
        r = subprocess.run(["/home/const/subnet120-rewrite/.venv/bin/python", "-B", str(out / "mimo_v829_train_gate_launcher.py"),
                            "--operator-config", str(out / OPN), "--v828-operator-config", str(V828 / "v828-operator.ROOT-ADMITTED.private.json"),
                            "--check"], capture_output=True, text=True, timeout=1200, cwd=str(out))
        last = (r.stdout.strip().splitlines() or [""])[-1]
        try:
            j = json.loads(last); v = j["v829"]
            ok = (r.returncode == 0 and j.get("v829_check") is True and v.get("v829g") is True and v.get("train_job_seconds") == 14400
                  and v.get("audit_job_seconds") == 7200 and v["v829h"] == dict(v829h=True, collection_seconds=seconds, accepted=[1200, 1800]) and v["v829i"]["v829i"] is True
                  and j["v828"]["v828"] is True)
        except Exception as e:
            ok = False; last = last + " PARSE " + repr(e)
        summary = dict(version="mimo-v829i-promotion-gate-root-signing-result-v1", authority=ROOT_ID,
            operator=str(out / OPN), operator_sha256=hf(out / OPN),
            configuration_sha256=signed(cfg["admission"], ROOT_ID)["configuration_sha256"],
            launcher_check=dict(ok=ok, out=last[:1200], err=(r.stderr or "")[-800:]), checks_passed=ok,
            change=dict(promotion_gate="v829i wrapper over _training; operator flag staged/PROMOTION-GATE.json (absent/enabled!=true = v829h behaviour)", collection_seconds="1200 (unchanged from v829h)",
                        max_tasks="256 (unchanged)", train_gate="v829c (unchanged)", train_job_seconds="14400 (unchanged)",
                        audit_job_seconds="7200 (unchanged)", base="release-v829h-collect1200-signed",
                        authorized="Jacob pt682u 2026-10-26 14:25 CR design approval; t684u 14:39 CR go live"))
        put("SIGNING-RESULT.json", (json.dumps(summary, indent=1, sort_keys=True) + "\n").encode())
        subprocess.run("find . -type f ! -name SIGNED-MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > SIGNED-MANIFEST.sha256 && chmod 600 SIGNED-MANIFEST.sha256",
                       shell=True, cwd=out, check=True)
        results[seconds] = dict(dir=str(out), ok=ok, operator_sha256=summary["operator_sha256"], err=summary["launcher_check"]["err"][-300:] if not ok else "")
    print(json.dumps(results, indent=1))
    sys.exit(0 if all(r["ok"] for r in results.values()) else 2)

if __name__ == "__main__":
    main()
