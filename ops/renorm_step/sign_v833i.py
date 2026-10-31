#!/usr/bin/env python3
"""SN120 trainer host v8.3.3i = live v8.3.3h (fp32 head master) + Allan normgrad direction step + support renorm (RA/20261031-renorm-step-live).
Changed host files: mimo_host_trainer_v83.py (engine sgd_direction / support_renorm wiring + normgrad step), v8211_batched.py,
v83_fast_kernels.py, v83_dp.py (Allan patch/support_renorm.diff as written). Engine: + sgd_direction 'normgrad',
+ support_renorm {top_k 20, top_p 0.95, temperature 1.0}; everything else (lr 5e-7, 2e-5 rel cap, global cap 1.0, fp32 master) unchanged.
Source root and original-method pins unchanged. Authorized by Jacob 2026-10-31 07:00 CR. ROOT seed read from the live lifecycle
config path, never printed. O_EXCL output."""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, importlib.util
from pathlib import Path
sys.dont_write_bytecode = True
ROOT_ID = "3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd"
RA = Path("/home/const/subnet120-rewrite/state/root-audits"); D = RA/"20261031-renorm-step-live"
SRC = RA/"20261029-v829u-125task/deployment-final-source-v823u"
B = RA/"20261030-free-file/release-v833h-signed"; H = B/"trainer-host-v833h"
STAGE = D/"stage/host-v833i"; CHANGED = ("mimo_host_trainer_v83.py", "v8211_batched.py", "v83_fast_kernels.py", "v83_dp.py")
LIFE = RA/"20261017-v824/release-v824-signed/lifecycle.v8.2.4.ACTIVE.private.json"
ENGINE_ADD = dict(sgd_direction="normgrad", support_renorm=dict(top_k=20, top_p=0.95, temperature=1.0))
def need(ok, why):
    if not ok: raise SystemExit("REFUSED: " + why)
hf = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    need(subprocess.run(["sha256sum", "--quiet", "-c", "SIGNED-MANIFEST.sha256"], cwd=B).returncode == 0, "v833h manifest")
    sys.path.insert(0, str(SRC))
    from subnet.storage import Identity, canonical, sha
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import sign
    spec = importlib.util.spec_from_file_location("v833i_launcher_probe", STAGE/"mimo_host_trainer_v83.py")
    L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
    old = json.loads((H/"TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json").read_text()); signed(old["admission"], ROOT_ID)
    need(L.VERSION == old["version"] and L.SCOPE == old["scope"] and set(L.HOST_FILES) == set(old["host_files"]), "same identity/inventory")
    files = {n: (hf(STAGE/n) if n in CHANGED else hf(H/n)) for n in L.HOST_FILES}
    need(all(hf(STAGE/n) == hf(H/n) for n in L.HOST_FILES if n not in CHANGED), "stage unchanged files = v833h")
    need(all(hf(STAGE/"patched"/n) == hf(H/"patched"/n) for n in os.listdir(H/"patched")) and sorted(os.listdir(STAGE/"patched")) == sorted(os.listdir(H/"patched")), "patched/ = v833h")
    need(sorted(n for n in files if files[n] != old["host_files"].get(n)) == sorted(CHANGED), "only the 4 named host files changed")
    prop = {k: v for k, v in old.items() if k != "admission"}; prop["host_files"] = files
    need(not (set(ENGINE_ADD) & set(old["engine"])), "engine keys new"); prop["engine"] = dict(old["engine"], **ENGINE_ADD)
    need(prop["original_methods"] == old["original_methods"] and prop["source_root"] == old["source_root"], "pins/source unchanged")
    need({k: v for k, v in prop["engine"].items() if k not in ENGINE_ADD} == old["engine"], "only the two engine keys added")
    L.direction_settings(prop["engine"]); L.support_settings(prop["engine"]); L.fp32_settings(prop["engine"]); L.sgd_settings(prop["engine"])
    if a.dry_run: actor = Identity(); A = actor.id
    else:
        life = json.loads(LIFE.read_text()); seed_path = Path(life["authority_seed_file"])
        need(seed_path.is_file() and not seed_path.is_symlink() and seed_path.stat().st_mode & 0o077 == 0, "seed mode")
        actor = Identity(bytes.fromhex(seed_path.read_text().strip())); A = ROOT_ID; need(actor.id == ROOT_ID, "authority id")
    pay = dict(signed(old["admission"], ROOT_ID)); pay["configuration_sha256"] = sha(canonical(prop))
    cfg = dict(prop, admission=sign(actor, pay)); actor = None
    need(signed(cfg["admission"], A)["configuration_sha256"] == sha(canonical(prop)), "admission binds proposal")
    OUT = Path(tempfile.mkdtemp(prefix="v833i-dry-")) if a.dry_run else D/"release-v833i-signed"
    if not a.dry_run: need(not OUT.exists(), "output exists"); OUT.mkdir(mode=0o700)
    (OUT/"trainer-host-v833i").mkdir(mode=0o700); (OUT/"trainer-host-v833i/patched").mkdir(mode=0o700)
    def put(rel, raw):
        fd = os.open(OUT/rel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        return OUT/rel
    for n in L.HOST_FILES: put("trainer-host-v833i/" + n, ((STAGE/n) if n in CHANGED else (H/n)).read_bytes())
    p = put("trainer-host-v833i/TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json", canonical(cfg))
    put("SIGNING-RESULT.json", (json.dumps(dict(version="mimo-v833i-root-signing-result-v1", dry_run=a.dry_run, authority=A, base="release-v833h-signed",
        trainer_host=dict(sha256=hf(p), configuration_sha256=sha(canonical(prop))), changed_host_files=list(CHANGED),
        host_file_sha256={n: files[n] for n in CHANGED}, engine_added=ENGINE_ADD, engine_changed=True, services_changed=False,
        allan_patch="support_renorm.diff sha256 138323858108b58869c2d0a38ac4ec7ad8d26dbfc799948f828a60e3cdc5145e (catoneone/sn120-k1-fixes@230b111)",
        authorized="Jacob 2026-10-31 07:00 CR via parent agent (ship all three once Allan confirms the formula; confirmed)"), indent=1, sort_keys=True) + "\n").encode())
    if not a.dry_run:
        subprocess.run("find . -type f ! -name SIGNED-MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > SIGNED-MANIFEST.sha256 && chmod 600 SIGNED-MANIFEST.sha256", shell=True, cwd=OUT, check=True)
    print(json.dumps(dict(done=True, dry_run=a.dry_run, out=str(OUT), config_sha256=hf(p))))
    if a.dry_run: shutil.rmtree(OUT)
if __name__ == "__main__": main()
