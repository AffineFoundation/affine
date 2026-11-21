#!/usr/bin/env python3
"""Isolated rollback compatibility qualification; no R2, chain or production state.

The fixture is generated with the real new controller in an operator-selected
new-code worktree. Old code is read from a separate archive of the pinned old
commit. All writes stay beneath a fresh scratch output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile

KIT = Path(__file__).resolve().parent
EXPECT = json.loads((KIT / "expect.json").read_text())
NEW_PIN = EXPECT["new_params_hash"]
REVIEWED_FIELD_GATE_PINS = EXPECT["reviewed_field_gate_params_hashes"]
HISTORICAL_PIN = "d5979c301c7acf8a56cb6a9989e5b212e5887ec1bf0117f9250202af6a95f35c"
OLD_PIN = EXPECT["old_params_hash"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def out(message: str) -> None:
    print(f"ROLLBACK QUALIFICATION PASS: {message}", flush=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fail_message(completed: subprocess.CompletedProcess, expected: str) -> None:
    if completed.returncode == 0 or expected not in completed.stderr + completed.stdout:
        raise AssertionError({"expected_refusal": expected, "returncode": completed.returncode,
                              "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--new-v2", type=Path, default=KIT.parents[1])
    parser.add_argument("--old-v2", type=Path, required=True,
                        help="operator-created archive of old commit c5252865; never a production checkout")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--new-params", type=Path, default=KIT / "new-params.json")
    args = parser.parse_args()
    new_params_path = args.new_params.resolve(strict=True)
    new_v2 = args.new_v2.resolve()
    sys.path.insert(0, str(new_v2))
    from sn120_v2.params import Params
    selected = Params.from_dict(json.loads(new_params_path.read_text()))
    assert selected.params_hash in REVIEWED_FIELD_GATE_PINS
    NEW_PIN = selected.params_hash
    if args.output.exists():
        raise ValueError("output must be a new isolated directory")
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    new_v2, old_v2 = new_v2, args.old_v2.resolve()
    # Select the old checkout before loading the overlay. The installed process
    # otherwise resolves the kit's new-code parent as its implicit old package.
    os.environ["DEDUP_OLD_V2"] = str(old_v2)
    for path in (new_v2 / "sn120_v2", old_v2 / "sn120_v2"):
        if not path.is_dir():
            raise ValueError(f"missing package {path}")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(new_v2), DEDUP_REVIEWED_PARAMS=str(new_params_path), DEDUP_REVIEWED_PARAMS_HASH=selected.params_hash)
    fixture = output / "fixture"
    subprocess.run([args.python, "-I", "-c",
                    "import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')",
                    str(new_v2), str(KIT / "make-rollback-fixture.py"), str(fixture)],
                   cwd=new_v2.parent, env=env, check=True, timeout=900)
    summary = json.loads((fixture / "fixture.json").read_text())
    assert summary["synthetic"] and summary["params_hash"] == NEW_PIN
    assert set(map(tuple, summary["queue"])) == set(map(tuple, summary["infra_queue"]))
    out("real new-controller fixture: failed lease, duplicate-only, null-update, Stage-B duplicate, sealed nonfinite evidence")

    # Read-only full-history compatibility replay, with exact queue accounting.
    compat = load_module("dedup_rollback_qualification", KIT / "rollback_cli.py")
    os.environ["DEDUP_OLD_V2"] = str(old_v2)
    compat.install(KIT)
    from sn120_v2.state import load, replay
    from sn120_v2.journal import read_journal
    from sn120_v2.dedup_compat import require_closed_dedup_history, digest_params
    from sn120_v2.params import Params
    from sn120_v2 import layout
    from sn120_v2.store import LocalStore
    from sn120_v2.types import TrainReport
    all_incident_rows = read_journal(fixture / "incident-seq601-603/journal.jsonl")
    incident_rows = all_incident_rows[601:604]
    assert [(row.seq, row.kind) for row in incident_rows] == [(601, "round_opened"), (602, "lease_failed"), (603, "round_closed")]
    assert incident_rows[0].data["round"]["round_id"] == "r000092"
    assert incident_rows[0].data["round"]["params_hash"] == HISTORICAL_PIN
    assert incident_rows[1].data["lease_id"] == "r000092-l0"
    assert incident_rows[1].data["reason"].startswith("InfraError: GPUs [4]")
    assert incident_rows[2].data["outcome"] == "no_decision"
    assert all(not incident_rows[2].data[name] for name in ("served", "front", "attempted", "held", "released"))
    incident_state = replay(all_incident_rows)
    assert incident_state.round is None and incident_state.rounds[-2].round_id == "r000092"
    assert incident_state.rounds[-2].decision is None and all(king.round_id != "r000092" for king in incident_state.kings)
    assert incident_state.rounds[-1].round_id == "r000093" and incident_state.rounds[-1].params_hash == NEW_PIN
    out("real new-controller seq601-603 tail: historical-pin r000092-l0 InfraError, then repaired-pin r000093, no settlement, charge or king")

    # This exception is not a blanket acceptance of unsettled no_decision closes.
    from dataclasses import replace
    for field, value in (("reason", "synthetic non-infrastructure failure"), ("served", [[99, 1099, "not-charged"]])):
        damaged = replace(incident_rows[-1], data=incident_rows[-1].data | {field: value})
        try:
            require_closed_dedup_history([*all_incident_rows[:603], damaged], incident_state)
        except Exception as error:
            assert "closes without exactly one valid decision" in str(error), error
        else:
            raise AssertionError(f"unsettled close with changed {field} was accepted")
    try:
        require_closed_dedup_history([*all_incident_rows, incident_rows[-1]], incident_state)
    except Exception as error:
        assert "closes without exactly one valid decision" in str(error), error
    else:
        raise AssertionError("a duplicate infrastructure close was accepted")
    out("unsettled no_decision without InfraError, with accounting, or with a duplicate close still refuses")
    rows = read_journal(fixture / "incident-seq601-603/journal.jsonl")
    state = replay(rows)
    require_closed_dedup_history(rows, state)
    assert state.params.params_hash == OLD_PIN and state.round is None and state.params.z_bar == 1.0
    assert {digest_params(row.data.get("params")) for row in rows if row.kind == "params_set" and row.data["params"].get("dedup_enabled")} == {HISTORICAL_PIN, NEW_PIN}
    assert {row.data["round"]["params_hash"] for row in rows if row.kind == "round_opened" and row.data["round"]["params_hash"] in (HISTORICAL_PIN, NEW_PIN)} == {HISTORICAL_PIN, NEW_PIN}
    by_id = {record.round_id: record for record in state.rounds}
    failed = by_id[summary["failed_lease"]]
    assert [entry.status for entry in failed.decision.entries] == ["no_decision", "duplicate", "no_decision"]
    # At the failed-lease boundary only the duplicate is charged. Both genuine
    # survivors return uncharged; the following round may consume and charge them.
    boundary = []
    for row in rows:
        boundary.append(row)
        if row.kind == "round_closed" and row.data.get("round_id") == failed.round_id:
            break
    at_failure = replay(boundary)
    assert set(at_failure.served_regs) == {(2, 1002)}
    assert set(at_failure.queue) == {(1, 1001), (3, 1003)}
    charged = {(entry.uid, entry.registration_block) for record in state.rounds
               for entry in (record.decision.entries if record.decision else ()) if entry.charged}
    assert state.served_regs == charged and len(charged) == 8, {"served": sorted(state.served_regs), "charged": sorted(charged)}
    assert set(state.queue) == set(map(tuple, summary["infra_queue"]))
    assert {(1, 1001), (2, 1002), (3, 1003)} <= state.served_regs
    assert (4, 1004) in state.served_regs and (5, 1005) in state.served_regs
    assert (6, 1006) in state.served_regs and (8, 1008) in state.served_regs
    assert by_id[summary["duplicate_only"]].leases == ()
    infra = by_id["r000092"]
    assert infra.outcome == "no_decision" and infra.decision is None and not infra.duplicate_decisions
    incident_tail = [row for row in rows if row.data.get("round_id") == infra.round_id
                     or row.data.get("round", {}).get("round_id") == infra.round_id]
    assert [row.kind for row in incident_tail] == ["round_opened", "lease_failed", "round_closed"]
    assert all(not incident_tail[-1].data[name] for name in ("served", "front", "attempted", "held", "released"))
    assert not any((entry.reg.uid, entry.reg.registration_block) in state.served_regs
                   for lease in infra.leases for entry in lease.entries)
    out("unsettled dedup infrastructure no_decision returns every seat uncharged, with no king or settlement")
    settled = by_id[summary["settled_null_stage_b"]]
    assert [entry.status for entry in settled.decision.entries] == ["null_update", "winner", "duplicate"]
    sealed_key = f"qualification/sealed/{settled.round_id}/{settled.leases[0].lease_id}.json"
    sealed = LocalStore(fixture / "store").get(sealed_key)
    assert b'"step_norm_nonfinite":"nan"' in sealed and b'"norm_before_cap_nonfinite":"inf"' in sealed
    report = TrainReport.from_dict(json.loads(sealed)["entries"][0]["lora"]["report"])
    assert report.step_norm_nonfinite == "nan" and report.norm_before_cap_nonfinite == "inf"
    out("closed dedup history replays with charged duplicates/null update and uncharged failed-lease survivors")

    # Independently exercise the bare-old decoder on each new record class.
    bare = output / "bare-old.py"
    bare.write_text('''import json,sys
sys.path.insert(0,sys.argv[1])
from sn120_v2.journal import read_journal
from sn120_v2.params import Params
from sn120_v2.state import RoundRecord
from sn120_v2.types import BoxResult, EntryDecision, RoundDecision, TrainReport
rows=read_journal(sys.argv[2])
mode=sys.argv[3]
if mode == "params":
    row=next(r for r in rows if r.kind=="params_set" and r.data["params"].get("dedup_enabled"))
    Params.from_dict(row.data["params"])
elif mode in ("duplicate", "null_update"):
    row=next(r for r in rows if r.kind=="round_settled" and any(e["status"]==mode for e in r.data["decision"]["entries"]))
    # Strip the separately tested new cluster field so the old status decoder
    # itself must reject this charged terminal class.
    decision=dict(row.data["decision"]); decision.pop("duplicate_clusters",None)
    RoundDecision.from_dict(decision)
elif mode == "round_fields":
    row=next(r for r in rows if r.kind=="round_opened" and r.data["round"].get("duplicate_decisions"))
    RoundRecord.from_dict(row.data["round"])
elif mode == "nonfinite":
    raw=json.load(open(sys.argv[4]))
    report=next(e["lora"]["report"] for e in raw["entries"] if e["lora"]["report"].get("step_norm_nonfinite"))
    TrainReport.from_dict(report)
else: raise AssertionError(mode)
''')
    old_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(old_v2))
    old_env.pop("DEDUP_OLD_V2", None)
    modes = {
        "params": "unknown field(s)",
        "duplicate": "unknown decision status",
        "null_update": "unknown decision status",
        "round_fields": "unknown field(s)",
        "nonfinite": "unknown field(s)",
    }
    from sn120_v2.store import LocalStore
    sealed_path = output / "sealed-result.json"
    sealed_path.write_bytes(LocalStore(fixture / "store").get(sealed_key))
    for mode, expected in modes.items():
        result = subprocess.run([args.python, "-I", str(bare), str(old_v2), str(fixture / "journal.jsonl"), mode, str(sealed_path)],
                                cwd=output, env=old_env, capture_output=True, text=True, timeout=90)
        fail_message(result, expected)
    out("bare old code separately rejects new params, duplicate/null_update, new round fields and tagged nonfinite values")

    # Old Params.from_dict accepts only the complete reviewed pin, never an
    # unknown/newer dedup variant. The re-pin then uses the real old CLI.
    changed = json.loads(new_params_path.read_text()) | {"dedup_rollout_overlap": 0.500001}
    try:
        Params.from_dict(changed)
    except Exception as error:
        assert "reviewed dedup" in str(error) or "historical params" in str(error), error
    else:
        raise AssertionError("a different dedup pin was accepted")
    repin = output / "repin"
    repin.mkdir()
    shutil.copyfile(fixture / "incident-seq601-603/journal.jsonl", repin / "journal.jsonl")
    helper = output / "loop_helper.py"
    helper.write_text('''import os
from sn120_v2.controller import Stack
from sn120_v2.sim import SimAdmission
from sn120_v2.store import LocalStore
class NoWorkRunner:
    def run(self, ticket):
        raise AssertionError("bounded empty-queue replay unexpectedly requested worker execution")
def stack(context):
    store = LocalStore(os.environ["DEDUP_LOOP_STORE"])
    return Stack(store, SimAdmission(store), None, NoWorkRunner())
''')
    command = [str(KIT / "rollback-python"), "set-params", "--state-dir", str(repin), "--stack", "loop_helper:stack", "--params", str(KIT / "old-params.json")]
    completed = subprocess.run(command, cwd=old_v2.parent, env=old_env | {"DEDUP_OLD_V2": str(old_v2), "DEDUP_ROLLBACK_PY": args.python, "PYTHONPATH": str(output), "DEDUP_LOOP_STORE": str(output / "repin-store")},
                               capture_output=True, text=True, timeout=180)
    if completed.returncode:
        raise AssertionError(completed.stderr + completed.stdout)
    change = json.loads(completed.stdout.splitlines()[-1])
    # Before re-pinning, compatibility status still reports the stripped old
    # execution hash; the subsequent CLI record carries the explicit old pin.
    assert change["params_hash_before"] == OLD_PIN and change["params_hash"] == OLD_PIN
    status = subprocess.run([str(KIT / "rollback-python"), "status", "--state-dir", str(repin)], cwd=old_v2.parent,
                            env=old_env | {"DEDUP_OLD_V2": str(old_v2), "DEDUP_ROLLBACK_PY": args.python},
                            capture_output=True, text=True, timeout=90)
    assert json.loads(status.stdout)["params_hash"] == OLD_PIN, status.stderr
    out(f"old-code compatibility CLI re-pins bba7a409 after historical {HISTORICAL_PIN[:8]} and repaired {NEW_PIN[:8]}; unknown/newer params refused")

    # The independent validator stays on native release code after this old-code
    # controller rollback. Prove that its journal/weights reader needs no shim:
    # all dedup fields, decisions and tagged evidence have real release decoders.
    # Subprocesses start from a scratch directory so no installed package or
    # compatibility environment can silently replace the selected new-v2 tree.
    native = output / "release-read-after-rollback.py"
    native.write_text('''import contextlib,io,json,os,pathlib,sys
sys.path.insert(0,sys.argv[1])
from sn120_v2.chain import FakeChain,SubtensorClient
from sn120_v2.journal import read_journal
from sn120_v2.state import load
from sn120_v2.store import LocalStore
from sn120_v2.types import TrainReport
from sn120_v2.weights_writer import main,read_view
root=pathlib.Path(sys.argv[2]); old_pin=sys.argv[3]; historical_pin=sys.argv[4]; new_pin=sys.argv[5]; new_z_bar=float(sys.argv[7]); selected_z_field_bar=float(sys.argv[8])
before=(root/"journal.jsonl").read_bytes()
rows=read_journal(root/"journal.jsonl")
new_params=next(r.data["params"] for r in rows if r.kind=="params_set" and r.data["params"].get("crown_gate")=="field_median")
history=[r for r in rows if r.kind=="params_set" and r.data["params"].get("dedup_enabled")]
assert history and all(r.data["params"]["eval_merged"] is True for r in history)
assert [r.data["params"]["z_bar"] for r in history]==[new_z_bar,1.5,new_z_bar]
assert new_params["crown_gate"]=="field_median" and new_params["z_field_bar"]==selected_z_field_bar
from sn120_v2.params import Params
assert {Params.from_dict(r.data["params"]).params_hash for r in history}=={historical_pin,new_pin}
assert rows[-1].kind=="params_set" and not rows[-1].data["params"].get("dedup_enabled",False)
state=load(root/"journal.jsonl"); view=read_view(root)
assert state.params.params_hash==view.params.params_hash==old_pin
assert state.params.z_bar==1.0 and not state.params.dedup_enabled
assert state.params.train_seed_scope=="batch" and state.round is None
assert state.kings and any(e.status=="duplicate" for r in state.rounds for e in r.decision.entries)
settled=next(r for r in state.rounds if any(e.status=="null_update" for e in r.decision.entries))
assert any(r.data["round"].get("duplicate_decisions") for r in rows if r.kind=="round_opened")
store=LocalStore(pathlib.Path(sys.argv[6]))
raw=json.loads(store.get(f"qualification/sealed/{settled.round_id}/{settled.leases[0].lease_id}.json"))
report=next(e["lora"]["report"] for e in raw["entries"] if e["lora"]["report"].get("step_norm_nonfinite"))
assert TrainReport.from_dict(report).step_norm_nonfinite=="nan"
king_hotkeys={k.uid:k.hotkey for k in view.kings}
hotkeys={uid:f"dry-unused-{uid}" for uid in range(max(king_hotkeys)+1)}|king_hotkeys
hotkeys[0]=view.validator_hotkey
chain=FakeChain(hotkeys,view.validator_hotkey)
SubtensorClient.connect=classmethod(lambda cls,*args:chain)
stdout=io.StringIO()
with contextlib.redirect_stdout(stdout):
    assert main(["--state-dir",str(root),"--network","offline-fixture","--dry-run"])==0
result=json.loads(stdout.getvalue())
assert result["status"]=="would set" and result["kings"]==len(view.kings)
assert set(result["weights"])=={str(uid) for uid in king_hotkeys}
assert chain.submissions==0 and (root/"journal.jsonl").read_bytes()==before
assert not (root/"weights_state.json").exists() and not (root/"weights.lock").exists()
print(json.dumps({"native_release_read":True,"compatibility_overlay":False,
                  "final_params_hash":old_pin,"z_bar":state.params.z_bar,
                  "kings":len(view.kings),"weights_dry_run":result["status"],
                  "journal_unchanged":True,"chain_submissions":chain.submissions},sort_keys=True))
''')
    native_env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1",
                  "DEDUP_REVIEWED_PARAMS": str(new_params_path),
                  "DEDUP_REVIEWED_PARAMS_HASH": selected.params_hash}
    completed = subprocess.run([args.python, "-I", str(native), str(new_v2), str(repin), OLD_PIN, HISTORICAL_PIN, NEW_PIN, str(fixture / "store"), str(selected.z_bar), str(selected.z_field_bar)],
                               cwd=output, env=native_env, capture_output=True, text=True, timeout=90)
    if completed.returncode:
        raise AssertionError(completed.stderr + completed.stdout)
    report = json.loads(completed.stdout)
    assert report["native_release_read"] and not report["compatibility_overlay"]
    assert report["weights_dry_run"] == "would set" and report["journal_unchanged"] and not report["chain_submissions"]
    (output / "release-read-after-rollback.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    out("native release reads post-rollback dedup history at old bba7a409/z1.0; validator weights dry-run would set without shim or chain write")

    # The existing unmodifiable systemd invocation requires a physical package.
    # Stage the complete old tree plus seven reviewed source modules, then run
    # ordinary ``python -m sn120_v2.weights_writer`` with no runtime import shim.
    old_validator = subprocess.run([args.python, "-I", str(KIT / "qualify-old-validator.py"), str(output), "--old-v2", str(old_v2), "--python", args.python],
                                   cwd=output, env=native_env, capture_output=True, text=True, timeout=90)
    if old_validator.returncode:
        raise AssertionError(old_validator.stderr + old_validator.stdout)
    print(old_validator.stdout, end="", flush=True)
    (output / "old-validator-read-after-rollback.json").write_text(old_validator.stdout.splitlines()[-1] + "\n")

    # Mirror catch-up precedes the refusal. No local leader fence, no durable
    # journal append, and no object write may occur for either open variant.
    from sn120_v2.journal import Journal, Mirror
    from sn120_v2.store import LocalStore
    for name in ("open-undecided", "open-decided"):
        state_dir = output / name
        state_dir.mkdir()
        shutil.copyfile(fixture / name / "journal.jsonl", state_dir / "journal.jsonl")
        store_root = output / f"{name}-store"
        store_root.mkdir()
        mirror_store = LocalStore(store_root)
        for seq in range(len(read_journal(state_dir / "journal.jsonl"))):
            mirror_store.put(layout.journal_record(seq), read_journal(state_dir / "journal.jsonl")[seq].line())
        before_local = sha(state_dir / "journal.jsonl")
        before_store = {key: sha(store_root / key) for key in mirror_store.list("")}
        writes = []
        original_put = mirror_store.put
        def forbidden(*args, **kwargs):
            writes.append(args)
            raise AssertionError("open-dedup refusal attempted a store write")
        mirror_store.put = forbidden
        try:
            Journal(state_dir / "journal.jsonl", "qualification-open-refusal", 2000000000, Mirror(mirror_store))
        except Exception as error:
            assert "finish it with reviewed new-code run" in str(error), error
        else:
            raise AssertionError(f"old writer accepted {name}")
        assert not writes and sha(state_dir / "journal.jsonl") == before_local
        assert {key: sha(store_root / key) for key in mirror_store.list("")} == before_store
    out("undecided/decided open dedup rounds refuse before fence; local and mirrored journals unchanged")

    # The unmodified old loop accepts one executable and uses it for status,
    # end-round and run. Prove that all its entry points reach this loader.
    # A bounded --max-idle=1 run performs the real CLI path without opening a
    # new round (the fixture's queue is empty). No runner or network is needed.
    loop_state = output / "loop-state"
    loop_state.mkdir()
    shutil.copyfile(repin / "journal.jsonl", loop_state / "journal.jsonl")
    loop_work = output / "loop-work"
    loop_work.mkdir()
    loop_store = output / "loop-store"
    loop_store.mkdir()
    # The isolated store has no executable miner inputs. Missing files must leave
    # this bounded run IDLE rather than ask the NoWorkRunner for a new round.
    loop_env = old_env | {
        "DEDUP_OLD_V2": str(old_v2), "DEDUP_ROLLBACK_PY": args.python,
        "DEDUP_LOOP_STORE": str(loop_store), "PYTHONPATH": f"{output}:{old_v2}",
        "SN120_WORK_DIR": str(loop_work), "SN120_LOOP_PAUSE_S": "0",
        "PATH": os.environ.get("PATH", ""),
    }
    command = [str(KIT / "rollback-loop.sh"), "--state-dir", str(loop_state), "--stack", "loop_helper:stack", "--max-idle", "1", "--idle-sleep", "0"]
    completed = subprocess.run(command, cwd=old_v2.parent, env=loop_env, capture_output=True, text=True, timeout=180)
    if completed.returncode:
        raise AssertionError(completed.stderr[-4000:] + completed.stdout[-4000:])
    runlog = (loop_work / "logs/run.log").read_text()
    assert runlog.count("run-loop: loop started") == 1, runlog
    summaries = [json.loads(match.group(0)) for match in re.finditer(r"(?ms)^\{\n.*?^\}$", runlog)]
    assert summaries, runlog[-4000:]
    run_summary = summaries[-1]
    assert run_summary["phase"] == "IDLE" and run_summary["params_hash"] == OLD_PIN, run_summary
    assert run_summary["last_outcomes"][-1] == "no_decision", run_summary
    final_status = subprocess.run([str(KIT / "rollback-python"), "status", "--state-dir", str(loop_state)], cwd=old_v2.parent,
                                  env=old_env | {"DEDUP_OLD_V2": str(old_v2), "DEDUP_ROLLBACK_PY": args.python},
                                  capture_output=True, text=True, timeout=90)
    assert final_status.returncode == 0, final_status.stderr
    assert json.loads(final_status.stdout)["params_hash"] == OLD_PIN
    assert not (loop_state / "journal.jsonl").is_symlink()
    out("old end-round/run starts after r000092: bounded run is IDLE at bba7a409; no new round or worker execution")
    (output / "qualification.json").write_text(json.dumps({
        "synthetic": True, "historical_params_hash": HISTORICAL_PIN, "new_params_hash": NEW_PIN, "old_params_hash": OLD_PIN,
        "old_v2": str(old_v2), "journal_sha256": sha(fixture / "journal.jsonl"),
        "closed_records": len(state.rounds), "charged_seats": len(state.served_regs),
        "validator_physical_package": True, "validator_runtime_import_shim": False,
        "physical_journal_open_refusal": True,
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
