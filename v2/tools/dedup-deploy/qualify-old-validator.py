#!/usr/bin/env python3
"""Prove the physically restored package with ordinary ``python -m``.

The staged package comes from restore_validator.sh. A fresh ``python -m``
subprocess resolves the complete package from the staging parent. Only an
external offline sitecustomize fixture replaces chain connectivity with an
in-memory chain; no kit module or source-import shim is installed into the
package. No network, R2, chain submission, writer lock or writer state is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_root", type=Path, help="qualify-rollback.py output directory")
    parser.add_argument("--old-v2", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    kit = Path(__file__).resolve().parent
    fixture_root = args.fixture_root.resolve(strict=True)
    root = (fixture_root / "repin").resolve(strict=True)
    store = (fixture_root / "fixture/store").resolve(strict=True)
    old_v2 = args.old_v2.resolve(strict=True)
    expect = json.loads((kit / "expect.json").read_text())
    old_pin, new_pin = expect["old_params_hash"], expect["new_params_hash"]
    stage = fixture_root / "physical-validator-stage"
    tarball = fixture_root / "physical-validator-stage.tar.gz"
    restore = subprocess.run(
        [str(kit / "restore_validator.sh"), str(old_v2 / "sn120_v2"), str(stage),
         "--tar", str(tarball)],
        cwd=fixture_root, env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1",
                              "DEDUP_RESTORE_PY": args.python},
        capture_output=True, text=True, timeout=90)
    if restore.returncode:
        raise AssertionError(restore.stderr + restore.stdout)
    print(restore.stdout, end="", flush=True)
    summary = json.loads(restore.stdout.splitlines()[-1])
    manifest = json.loads((stage / "restore-validator-manifest.json").read_text())
    staged = {path.relative_to(stage).as_posix(): path for path in stage.rglob("*") if path.is_file()}
    for name, expected in manifest["files"].items():
        path = staged[f"sn120_v2/{name}"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    assert not list(stage.rglob("__pycache__")) and not list(stage.rglob("*.pyc"))
    with tarfile.open(tarball, "r:gz") as archive:
        archived = {info.name: archive.extractfile(info).read()
                    for info in archive.getmembers() if info.isfile()}
    assert set(archived) == {f"sn120_v2/{name}" for name in manifest["files"]} | {"restore-validator-manifest.json"}
    for name, raw in archived.items():
        assert hashlib.sha256(raw).hexdigest() == hashlib.sha256((stage / name).read_bytes()).hexdigest(), name
    before = (root / "journal.jsonl").read_bytes()
    # sitecustomize belongs only to the external offline fixture. It replaces
    # chain I/O, not sn120_v2 source modules. The real ``python -m`` process
    # resolves every package module from the staged physical directory.
    fixture = fixture_root / "physical-validator-fixture"
    fixture.mkdir()
    (fixture / "sitecustomize.py").write_text('''import atexit,json,os,pathlib,sys
root=pathlib.Path(os.environ["DEDUP_FIXTURE_STATE"])
stage=pathlib.Path(os.environ["DEDUP_FIXTURE_STAGE"]).resolve()
kit=pathlib.Path(os.environ["DEDUP_FIXTURE_KIT"]).resolve()
assert stage not in (pathlib.Path.cwd().resolve(), kit)
assert not any(pathlib.Path(p).resolve()==kit for p in sys.path if p)
from sn120_v2.chain import FakeChain,SubtensorClient
from sn120_v2.state import load
assert pathlib.Path(sys.modules["sn120_v2"].__file__).resolve().is_relative_to(stage)
state=load(root/"journal.jsonl")
king_hotkeys={k.uid:k.hotkey for k in state.kings}
hotkeys={uid:f"dry-unused-{uid}" for uid in range(max(king_hotkeys)+1)}|king_hotkeys
hotkeys[0]=state.genesis.validator_hotkey
chain=FakeChain(hotkeys,state.genesis.validator_hotkey)
SubtensorClient.connect=classmethod(lambda cls,*args:chain)
def finish():
    pathlib.Path(os.environ["DEDUP_FIXTURE_RESULT"]).write_text(json.dumps({"chain_submissions":chain.submissions,"staged_params_hash":state.params.params_hash,"staged_z_bar":state.params.z_bar,"staged_kings":len(state.kings),"package_file":sys.modules["sn120_v2"].__file__}))
atexit.register(finish)
''')
    result_path = fixture_root / "physical-validator-fixture.json"
    if result_path.exists():
        raise AssertionError("fixture result unexpectedly exists before the dry run")
    environment = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1",
                   "DEDUP_FIXTURE_STATE": str(root), "DEDUP_FIXTURE_STAGE": str(stage),
                   "DEDUP_FIXTURE_KIT": str(kit), "DEDUP_FIXTURE_RESULT": str(result_path),
                   "PYTHONPATH": f"{stage}:{fixture}"}
    completed = subprocess.run(
        [args.python, "-m", "sn120_v2.weights_writer", "--state-dir", str(root),
         "--network", "offline-fixture", "--dry-run"],
        cwd=fixture, env=environment, capture_output=True, text=True, timeout=90)
    if completed.returncode:
        raise AssertionError(completed.stderr + completed.stdout)
    result = json.loads(completed.stdout.splitlines()[-1])
    fake = json.loads(result_path.read_text())
    assert Path(fake["package_file"]).resolve().is_relative_to(stage)
    assert result["dry_run"] is True and result["status"] == "would set"
    assert result["kings"] == fake["staged_kings"] > 0
    assert fake["staged_params_hash"] == old_pin and fake["staged_z_bar"] == 1.0
    assert fake["chain_submissions"] == 0 and (root / "journal.jsonl").read_bytes() == before
    assert not (root / "weights_state.json").exists() and not (root / "weights.lock").exists()
    # The external fixture repeats the real reader checks in a separate process;
    # the ordinary ``python -m`` invocation above stays unchanged.
    detail = fixture_root / "physical-validator-history.py"
    detail.write_text('''import json,pathlib,sys
sys.path.insert(0,sys.argv[1])
from sn120_v2.dedup_compat import digest_params
from sn120_v2.journal import read_journal
from sn120_v2.params import Params
from sn120_v2.state import load
from sn120_v2.store import LocalStore
from sn120_v2.types import TrainReport
root=pathlib.Path(sys.argv[2]); store=pathlib.Path(sys.argv[3])
old_pin,historical_pin,repaired_pin,new_pin=sys.argv[4:8]
rows=read_journal(root/"journal.jsonl")
history=[r for r in rows if r.kind=="params_set" and r.data["params"].get("dedup_enabled")]
assert history and {digest_params(r.data["params"]) for r in history}=={historical_pin,repaired_pin,new_pin}
assert all(r.data["params"]["z_bar"]==(0.0 if r.data["params"].get("crown_gate")=="field_median" else 1.5) and r.data["params"]["eval_merged"] is True for r in history)
assert {Params.from_dict(r.data["params"]).params_hash for r in history}=={old_pin}
state=load(root/"journal.jsonl")
assert state.params.params_hash==old_pin and state.params.z_bar==1.0 and state.round is None
assert not hasattr(state.params,"dedup_enabled") and not hasattr(state.params,"train_seed_scope")
assert state.kings and any(e.status=="duplicate" for r in state.rounds for e in r.decision.entries)
settled=next(r for r in state.rounds if any(e.status=="null_update" for e in r.decision.entries))
assert any(r.data["round"].get("duplicate_decisions") for r in rows if r.kind=="round_opened")
raw=json.loads(LocalStore(store).get(f"qualification/sealed/{settled.round_id}/{settled.leases[0].lease_id}.json"))
report=next(e["lora"]["report"] for e in raw["entries"] if e["lora"]["report"].get("step_norm_nonfinite"))
assert TrainReport.from_dict(report).step_norm_nonfinite=="nan"
print(json.dumps({"kings":len(state.kings),"king_hotkeys":{str(k.uid):k.hotkey for k in state.kings}}))
''')
    details = subprocess.run([args.python, str(detail), str(stage), str(root), str(store), old_pin,
                              "d5979c301c7acf8a56cb6a9989e5b212e5887ec1bf0117f9250202af6a95f35c", "8caddd9c6fc276a9669dff9e01e1bb7e437d98bed5b12a605141e6025b9f9ec4", new_pin],
                             cwd=fixture, env=environment | {"PYTHONPATH": str(stage)}, capture_output=True, text=True, timeout=90)
    if details.returncode:
        raise AssertionError(details.stderr + details.stdout)
    detail_result = json.loads(details.stdout)
    assert set(result["weights"]) == set(detail_result["king_hotkeys"])
    assert result["kings"] == detail_result["kings"]
    # Exercise the physical journal.py itself. The old weights-only path takes
    # no Journal writer lock; controller rollback must still refuse open dedup
    # work before a leader fence or any local/durable store write.
    guard_script = fixture_root / "physical-validator-journal-guard.py"
    guard_script.write_text('''import hashlib,json,os,pathlib,sys
from sn120_v2 import layout
from sn120_v2.journal import Journal,Mirror,read_journal
from sn120_v2.store import LocalStore
source=pathlib.Path(sys.argv[1]); destination=pathlib.Path(sys.argv[2]); store_root=pathlib.Path(sys.argv[3])
destination.mkdir(); store_root.mkdir()
path=destination/"journal.jsonl"; path.write_bytes((source/"journal.jsonl").read_bytes())
rows=read_journal(path); store=LocalStore(store_root)
for seq,row in enumerate(rows): store.put(layout.journal_record(seq),row.line())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
before_local=sha(path); before_store={key:sha(store_root/key) for key in store.list("")}
writes=[]
def forbidden(*args,**kwargs):
    writes.append(args); raise AssertionError("open-dedup refusal attempted a store write")
store.put=forbidden
try: Journal(path,"physical-open-refusal",2000000000,Mirror(store))
except Exception as error: assert "finish it with reviewed new-code run" in str(error), error
else: raise AssertionError("physical journal accepted open dedup history")
assert not writes and sha(path)==before_local
assert {key:sha(store_root/key) for key in store.list("")}==before_store
print(json.dumps({"physical_journal_open_refusal":True,"store_writes":len(writes),"journal_unchanged":True}))
''')
    for name in ("open-undecided", "open-decided"):
        guarded = subprocess.run([args.python, str(guard_script), str(fixture_root / "fixture" / name),
                                  str(fixture_root / f"physical-{name}"), str(fixture_root / f"physical-{name}-store")],
                                 cwd=fixture, env=environment | {"PYTHONPATH": str(stage)},
                                 capture_output=True, text=True, timeout=90)
        if guarded.returncode:
            raise AssertionError(guarded.stderr + guarded.stdout)
    # The ordinary physical package also rejects a missing module, rather than
    # borrowing a kit or another installed package implicitly.
    broken = fixture_root / "physical-validator-broken-stage"
    shutil.copytree(stage, broken)
    (broken / "sn120_v2/canonical.py").unlink()
    refusal = subprocess.run(
        [args.python, "-m", "sn120_v2.weights_writer", "--state-dir", str(root),
         "--network", "offline-fixture", "--dry-run"],
        cwd=fixture, env=environment | {"DEDUP_FIXTURE_STAGE": str(broken), "PYTHONPATH": f"{broken}:{fixture}"},
        capture_output=True, text=True, timeout=90)
    assert refusal.returncode and "ModuleNotFoundError" in refusal.stderr and "sn120_v2.canonical" in refusal.stderr
    print("ROLLBACK QUALIFICATION PASS: c5252865 physical package runs python -m sn120_v2.weights_writer --dry-run: would set; no runtime shim, zero chain writes, journal unchanged")
    print("ROLLBACK QUALIFICATION PASS: physical journal.py refuses undecided/decided open dedup rounds before fence; local and mirrored journals unchanged")
    print(completed.stderr, end="", flush=True)
    print(completed.stdout, end="", flush=True)
    print(json.dumps({"runtime_import_shim": False, "old_modules_physically_overlaid": manifest["physical_overlay"],
                      "old_modules_physically_overlaid_count": len(manifest["physical_overlay"]),
                      "final_params_hash": old_pin, "z_bar": fake["staged_z_bar"], "kings": fake["staged_kings"],
                      "weights_dry_run": result["status"], "weights": result["weights"], "journal_unchanged": True,
                      "chain_submissions": fake["chain_submissions"], "restorer": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
