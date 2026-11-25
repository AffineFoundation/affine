#!/usr/bin/env python3
"""Rehearse deployment and exact-old rollback on disposable simulation state.

Run with the new checkout on PYTHONPATH, then with the exact old checkout and
the verified rollback loader. No production path, service or wallet is used.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.journal import JOURNAL_FILE, read_journal
from sn120_v2.params import load_params
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.store import put_record
from sn120_v2.types import BoxResult, EntryDecision, EntryResult, SCORED


def rewrite_records(path: Path, transform) -> None:
    """Build a synthetic fixture; never use this helper on an operator journal."""
    output = []
    previous = ""
    for record in read_journal(path):
        data = transform(record.kind, json.loads(json.dumps(record.data)))
        record = dataclasses.replace(record, data=data, prev=previous)
        output.append(record.line())
        previous = record.checksum
    path.write_bytes(b"\n".join(output) + b"\n")


def simulate_activation(repo: Path, root: Path, python: str) -> dict:
    """Run the real set-params CLI on a copy of a closed live-pin state."""
    old_params = repo / "v2" / "ops" / "train64k_params_t3600_prune_open.json"
    new_params = repo / "v2" / "ops" / "train64k_params_t3600_prune_open_kl8k.json"
    source = root / "live-pin-source"
    world = SimWorld(source, load_params(old_params))
    world.register(1, quality=0.0)
    assert world.round() is not None
    assert world.controller.state.round is None
    assert world.controller.params.params_hash == "073ef680cf338de5e085100704e60f80ba96657d6d4e19d12995e4cb971ef6f8"
    original = (source / JOURNAL_FILE).read_bytes()
    copied = root / "live-pin-copy"
    shutil.copytree(source, copied)

    def cli(*args):
        proc = subprocess.run(
            [python, "-m", "sn120_v2", *map(str, args)],
            env=dict(os.environ, PYTHONPATH=str(repo / "v2")),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise AssertionError(f"new-code {args}: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
        return proc

    result = json.loads(cli(
        "set-params", "--state-dir", copied, "--stack", "sn120_v2.sim:stack",
        "--params", new_params,
    ).stdout)
    expected = {
        "min_kl_seq": {"before": 0.0, "after": 0.000225},
        "kl_max_tokens": {"before": 0, "after": 8192},
        "recipe.code_digest": {
            "before": "22b2d191b32e394d4621ce477e702b50410a6ef7706cce6cfd20c60ff64e42e1",
            "after": "5e3aef5849ef6903c9fc1ab67f89eddbd2ea2cac9748621fc42b75224bf58419",
        },
    }
    assert result == {
        "params_hash_before": "073ef680cf338de5e085100704e60f80ba96657d6d4e19d12995e4cb971ef6f8",
        "params_hash": "dcab97d47d2f4b02ef7a14d5d8786a0f5a14885ddbf2634845435612fb9b4b1e",
        "changed": expected,
    }
    status = json.loads(cli("status", "--state-dir", copied).stdout)
    from sn120_v2.state import replay
    copied_state = replay(read_journal(copied / JOURNAL_FILE))
    assert status["params_hash"] == result["params_hash"] and status["phase"] == "IDLE"
    assert copied_state.round is None and copied_state.params.params_hash == result["params_hash"]
    assert (source / JOURNAL_FILE).read_bytes() == original
    return result


def main() -> int:
    kit = Path(__file__).resolve().parent
    repo = kit.parents[2]
    old_v2 = Path(os.environ["KL_OLD_V2"]).resolve(strict=True)
    python = os.environ.get("KL_QUALIFY_PY", sys.executable)
    root = Path(tempfile.mkdtemp(prefix="kl-gate-qualification-"))
    activation = simulate_activation(repo, root, python)
    closed = root / "closed"
    new_params = repo / "v2" / "ops" / "train64k_params_t3600_prune_open_kl8k.json"
    old_params = repo / "v2" / "ops" / "train64k_params_t3600_prune_open.json"
    pinned = load_params(new_params)

    world = SimWorld(closed, sim_params())
    world.register(1, quality=0.0)
    record = world.round()
    assert record is not None and record.decision is not None
    lease = record.leases[0]
    result = BoxResult.from_json(world.store.get(layout.sealed_result(record.round_id, lease.lease_id)))
    entry = dataclasses.replace(
        result.entries[0], status=SCORED,
        kl_seq=0.5, kl_seq_se=0.02, kl_tok=0.01, kl_k=8, kl_tokens=50,
    )
    result = dataclasses.replace(
        result, entries=(entry,), null_kl_seq=0.0, null_kl_seq_se=0.0,
        null_kl_tok=0.0, null_kl_k=8, null_kl_tokens=50,
    )
    put_record(world.store, layout.sealed_result(record.round_id, lease.lease_id), result)

    def production_pin(kind, data):
        if kind == "params_set":
            data["params"] = pinned.to_dict()
        elif kind == "round_opened":
            data["round"]["params_hash"] = pinned.params_hash
            data["round"]["recipe_id"] = pinned.recipe_id
        elif kind == "round_settled":
            data["results"][lease.lease_id] = result.digest()
            rows = []
            for row in data["decision"]["entries"]:
                rows.append(row | dict(kl_seq=0.5, kl_seq_se=0.02, kl_tok=0.01, kl_k=8, kl_tokens=50))
            data["decision"]["entries"] = rows
        return data

    rewrite_records(closed / JOURNAL_FILE, production_pin)
    historical = (closed / JOURNAL_FILE).read_bytes()
    mirrored = {}
    for seq, record in enumerate(read_journal(closed / JOURNAL_FILE)):
        path = closed / "store" / layout.journal_record(seq)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(record.line())
        mirrored[path.name] = record.line()

    loader = ["python3", "-I", str(kit / "rollback.py"), "--kit", str(kit)]
    env = dict(os.environ, KL_OLD_V2=str(old_v2), PYTHONPATH=str(old_v2))

    def old(*args, expect=0):
        proc = subprocess.run(
            [python, "-I", str(kit / "rollback.py"), "--kit", str(kit), *map(str, args)],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != expect:
            raise AssertionError(f"rollback {args}: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
        return proc

    # Exact-old replay retains the sealed KL fields without changing history.
    probe = old("status", "--state-dir", closed)
    assert json.loads(probe.stdout)["params_hash"] == "073ef680cf338de5e085100704e60f80ba96657d6d4e19d12995e4cb971ef6f8"
    from sn120_v2.state import replay
    replayed = replay(read_journal(closed / JOURNAL_FILE))
    assert replayed.params.params_hash == pinned.params_hash
    assert replayed.rounds[-1].decision.entries[0].kl_seq == 0.5

    assert (closed / JOURNAL_FILE).read_bytes() == historical

    # Even end-round takes the journal and must not bypass the active pin.
    probe = old("end-round", "--state-dir", closed, "--stack", "sn120_v2.sim:stack", expect=2)
    assert "KL params are still pinned" in probe.stderr, probe.stderr

    # A closed round under the new pin still refuses the old loop.
    work = root / "work"
    (work / "logs").mkdir(parents=True)
    (work / "logs" / "run-loop.stop-after-round").touch()
    blocked = subprocess.run(
        ["bash", str(kit / "rollback-loop.sh"), "--state-dir", str(closed),
         "--stack", "sn120_v2.sim:stack"],
        env=dict(env, SN120_WORK_DIR=str(work), KL_ROLLBACK_PY=python,
                 PATH=os.environ.get("PATH", "")),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if blocked.returncode != 2 or "KL params are still pinned" not in (blocked.stdout + blocked.stderr):
        raise AssertionError(f"pinned rollback loop did not fail closed: {blocked.returncode}\n{blocked.stdout}\n{blocked.stderr}")
    assert (closed / JOURNAL_FILE).read_bytes() == historical

    # Only set-params may cross from a closed KL state to the reviewed old pin.
    old("set-params", "--state-dir", closed, "--stack", "sn120_v2.sim:stack", "--params", old_params)
    final_set = old("set-params", "--state-dir", closed, "--stack", "sn120_v2.sim:stack", "--params", old_params)
    assert json.loads(final_set.stdout)["changed"] == {}
    old("verify-params", old_params)
    final = old("status", "--state-dir", closed)
    assert json.loads(final.stdout)["params_hash"] == "073ef680cf338de5e085100704e60f80ba96657d6d4e19d12995e4cb971ef6f8"
    updated = (closed / JOURNAL_FILE).read_bytes()
    assert updated.startswith(historical)
    appended = read_journal(closed / JOURNAL_FILE)[len(read_journal_bytes(historical)):]
    assert [record.kind for record in appended] == ["leader", "params_set", "leader", "params_set"]
    for name, raw in mirrored.items():
        assert (closed / "store" / "private" / "journal" / name).read_bytes() == raw

    # The unmodified old run-loop, through the loader, honors stop-after-round.
    (work / "logs" / "run-loop.stop-after-round").touch()
    subprocess.run(
        ["bash", str(kit / "rollback-loop.sh"), "--state-dir", str(closed),
         "--stack", "sn120_v2.sim:stack"],
        env=dict(env, SN120_WORK_DIR=str(work), KL_ROLLBACK_PY=python,
                 PATH=os.environ.get("PATH", "")), check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert not (work / "logs" / "run-loop.stop-after-round").exists()

    # Any open round under the KL pin is refused before a leader is appended.
    # Build this from the original closed fixture, not from the transitioned
    # old-pin state, and restore the original queue and round settlement.
    opened = root / "open"
    shutil.copytree(closed, opened)
    records = read_journal_bytes(historical)

    def reopen(kind, data):
        if kind in ("round_settled", "round_closed"):
            return None
        return data

    raw = []
    mirrored_open = {}
    previous = ""
    for seq, record in enumerate(records):
        data = reopen(record.kind, record.data)
        if data is not None:
            record = dataclasses.replace(record, data=data, prev=previous)
            line = record.line()
            raw.append(line)
            path = opened / "store" / layout.journal_record(seq)
            path.write_bytes(line)
            mirrored_open[path.name] = line
            previous = record.checksum
    for path in (opened / "store" / "private" / "journal").iterdir():
        if path.name not in mirrored_open:
            path.unlink()
    (opened / JOURNAL_FILE).write_bytes(b"\n".join(raw) + b"\n")
    opened_state = replay(read_journal(opened / JOURNAL_FILE))
    assert opened_state.round is not None and opened_state.round.decision is None
    assert opened_state.round.params_hash == pinned.params_hash
    assert opened_state.params.params_hash == pinned.params_hash
    before = (opened / JOURNAL_FILE).read_bytes()
    refused = old("set-params", "--state-dir", opened, "--stack", "sn120_v2.sim:stack",
                  "--params", old_params, expect=2)
    assert "under KL params" in refused.stderr
    assert (opened / JOURNAL_FILE).read_bytes() == before

    print(json.dumps({
        "fixture": str(root), "old_code_replay": "passed",
        "open_kl_round_refusal": "passed", "pinned_old_loop_refusal": "passed",
        "copied_live_params_transition": activation,
        "copied_source_journal": "unchanged",
        "old_params_transition": "passed", "historical_journal_prefix": "unchanged",
        "historical_mirror_records": "unchanged", "old_loop_stop_after_round": "passed",
        "note": "Synthetic CPU control-plane qualification only; no GPU or production operation.",
    }, indent=2))
    return 0


def read_journal_bytes(raw: bytes):
    path = Path(tempfile.mkdtemp(prefix="kl-journal-read-")) / JOURNAL_FILE
    path.write_bytes(raw)
    return read_journal(path)


if __name__ == "__main__":
    raise SystemExit(main())
