#!/usr/bin/env python3
"""Offline validator rollback reader proof on the post-repin rollback fixture.

Use only with qualify-rollback.py's synthetic repin directory and store. The
runtime loader keeps the old checkout at c5252865 and verifies every overlay
file. No network, R2, chain submission, writer lock or writer state is allowed.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys

# python -I excludes this kit's directory. Import its reviewed loader by path.
_kit = Path(__file__).resolve().parent
if str(_kit) not in sys.path:
    sys.path.insert(0, str(_kit))
from rollback_cli import install


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_root", type=Path, help="qualify-rollback.py output directory")
    parser.add_argument("--old-v2", type=Path, required=True)
    args = parser.parse_args()
    kit = Path(__file__).resolve().parent
    root = (args.fixture_root / "repin").resolve(strict=True)
    store = (args.fixture_root / "fixture/store").resolve(strict=True)
    old_v2 = args.old_v2.resolve(strict=True)
    os.environ["DEDUP_OLD_V2"] = str(old_v2)
    install(kit)
    from sn120_v2.chain import FakeChain, SubtensorClient
    from sn120_v2.dedup_compat import digest_params
    from sn120_v2.journal import read_journal
    from sn120_v2.params import Params
    from sn120_v2.state import load
    from sn120_v2.store import LocalStore
    from sn120_v2.types import TrainReport
    from sn120_v2.weights_writer import main as weights_main, read_view

    expect = json.loads((kit / "expect.json").read_text())
    old_pin, new_pin = expect["old_params_hash"], expect["new_params_hash"]
    before = (root / "journal.jsonl").read_bytes()
    rows = read_journal(root / "journal.jsonl")
    history = [r for r in rows if r.kind == "params_set" and r.data["params"].get("dedup_enabled")]
    assert history and {digest_params(r.data["params"]) for r in history} == {new_pin}
    assert all(r.data["params"]["z_bar"] == 1.5 and r.data["params"]["eval_merged"] is True for r in history)
    # The overlay intentionally translates exact historical new pins to the
    # old-compatible view. It does not implement new behavior in old Params.
    assert {Params.from_dict(r.data["params"]).params_hash for r in history} == {old_pin}
    assert rows[-1].kind == "params_set" and not rows[-1].data["params"].get("dedup_enabled", False)
    state, view = load(root / "journal.jsonl"), read_view(root)
    assert state.params.params_hash == view.params.params_hash == old_pin
    assert state.params.z_bar == 1.0 and state.round is None
    assert not hasattr(state.params, "dedup_enabled")
    assert not hasattr(state.params, "train_seed_scope")
    assert state.kings and any(e.status == "duplicate" for r in state.rounds for e in r.decision.entries)
    settled = next(r for r in state.rounds if any(e.status == "null_update" for e in r.decision.entries))
    assert any(r.data["round"].get("duplicate_decisions") for r in rows if r.kind == "round_opened")
    raw = json.loads(LocalStore(store).get(f"qualification/sealed/{settled.round_id}/{settled.leases[0].lease_id}.json"))
    report = next(e["lora"]["report"] for e in raw["entries"] if e["lora"]["report"].get("step_norm_nonfinite"))
    assert TrainReport.from_dict(report).step_norm_nonfinite == "nan"
    king_hotkeys = {k.uid: k.hotkey for k in view.kings}
    hotkeys = {uid: f"dry-unused-{uid}" for uid in range(max(king_hotkeys) + 1)} | king_hotkeys
    hotkeys[0] = view.validator_hotkey
    chain = FakeChain(hotkeys, view.validator_hotkey)
    SubtensorClient.connect = classmethod(lambda cls, *args: chain)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert weights_main(["--state-dir", str(root), "--network", "offline-fixture", "--dry-run"]) == 0
    result = json.loads(stdout.getvalue())
    assert result["status"] == "would set" and result["kings"] == len(view.kings)
    assert set(result["weights"]) == {str(uid) for uid in king_hotkeys}
    assert chain.submissions == 0 and (root / "journal.jsonl").read_bytes() == before
    assert not (root / "weights_state.json").exists() and not (root / "weights.lock").exists()
    print("ROLLBACK QUALIFICATION PASS: c5252865 validator weights_writer dry-run would set through reviewed runtime overlay; zero chain writes, journal unchanged")
    print(json.dumps({"runtime_import_shim": True, "old_modules_physically_overlaid": False,
                      "final_params_hash": old_pin, "z_bar": state.params.z_bar, "kings": len(view.kings),
                      "weights_dry_run": result["status"], "journal_unchanged": True,
                      "chain_submissions": chain.submissions}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
