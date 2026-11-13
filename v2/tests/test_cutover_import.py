"""Cutover import, against v1's real public state.

The two files in tests/data/cutover were read from affine.io on 13 Nov 2026,
when v1 had stopped at epoch 1310 with 14 turns taken and not judged.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from sn120_v2 import cli, cutover, layout
from sn120_v2.canonical import manifest_digest
from sn120_v2.cutover import CutoverError, build_genesis, build_import, chain_report, copy_miner_files, preview_queue
from sn120_v2.errors import ConfigError
from sn120_v2.params import GENESIS_BASE_HASH, GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION, GenesisSpec, Params
from sn120_v2.payout import weights
from sn120_v2.queue import waiting
from sn120_v2.sim import JOURNAL_FILE
from sn120_v2.state import load as load_state
from sn120_v2.store import LocalStore
from sn120_v2.types import ChainSnapshot, Registration, V1Import

DATA = Path(__file__).parent / "data" / "cutover"
RAW = {name: json.loads((DATA / f"{name}.json").read_text()) for name in ("k1", "mining")}
NOW = RAW["k1"]["generated_at"] + 600
KING_EPOCHS = [1121, 1122, 1162, 1165, 1166, 1167, 1169, 1170, 1174, 1211]  # v1's 10 payees, oldest first
KING_UIDS = [955, 956, 942, 821, 851, 909, 822, 852, 918, 920]
VALIDATOR = "5G8McktkLiwrkhtmFJuS79CCYmjeLVCbRzqWCJ8jDXp2bPbn"  # any valid address: the fixtures do not name the real one
OWNER = "5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8"  # the owner hotkey that ops/HOURLY_BURN.md expects at UID 0
DIGEST = "ab" * 32
FILES = ["--k1", str(DATA / "k1.json"), "--mining", str(DATA / "mining.json"), "--max-age-s", str(10**10)]


@pytest.fixture
def v1() -> dict:
    return copy.deepcopy(RAW)


def load(v1: dict, undecided: str | None = "front", earlier: tuple[dict, ...] = ()) -> V1Import:
    return build_import(v1["k1"], v1["mining"], now=NOW, undecided=undecided, earlier=earlier)


def turn(k1: dict, epoch: int) -> dict:
    return next(row for row in k1["turns"] if row["epoch_index"] == epoch)


def chain(v1: dict, extra: tuple[Registration, ...] = ()) -> ChainSnapshot:
    """A registration snapshot from the fixture: every seat v1 names, UID 0 and ``extra``. The newest seat of a UID wins."""
    queue = v1["k1"]["queue"]
    rows = [(r["uid"], r["hotkey"], r["reg_block"]) for r in queue["served"] if r["reg_block"]]
    rows += [(r["uid"], r["hotkey"], r["registration_block"]) for r in queue["expired_back_of_queue"]]
    seats = {uid: Registration(uid, hotkey, block) for uid, hotkey, block in sorted(rows, key=lambda row: row[2])}
    seats |= {reg.uid: reg for reg in (Registration(0, "owner", 1), *extra)}
    return ChainSnapshot(120, queue["chain_block"] + 10, int(NOW), tuple(seats.values()))


def test_kings_are_v1s_last_ten_payees_oldest_first(v1):
    imp = load(v1)
    assert [king.round_id for king in imp.kings] == [f"v1-e{epoch}" for epoch in KING_EPOCHS]
    assert [king.uid for king in imp.kings] == KING_UIDS
    assert all(king.head_version_id == "" and king.applied_at > 0 and king.z > 1.0 for king in imp.kings)
    table = weights(imp.kings, chain(v1).hotkeys(), Params())
    assert sorted(table) == sorted(KING_UIDS) and set(table.values()) == {0.1} and sum(table.values()) == pytest.approx(1.0)
    assert V1Import.from_json(imp.to_json()) == imp


def test_a_start_that_would_burn_all_weight_is_refused(v1):
    imp, snapshot = load(v1), chain(v1)
    assert any("first weights" in line for line in chain_report(imp, None, snapshot, Params()))
    nobody = ChainSnapshot(120, snapshot.block, 0, tuple(reg for reg in snapshot.registrations if reg.uid not in KING_UIDS))
    with pytest.raises(CutoverError, match="burn"):
        chain_report(imp, None, nobody, Params())
    with pytest.raises(CutoverError, match="validator hotkey"):
        chain_report(imp, build_genesis(DIGEST, "5HoHVmesgr4F7Trw1ow5wNyFTrm4zs3rLHZir7LAVSHUvFi9"), nobody, Params())


def test_turns_without_a_verdict_need_the_operators_decision(v1):
    with pytest.raises(CutoverError, match="14 v1 turn"):
        load(v1, undecided=None)
    front, served = load(v1, "front"), load(v1, "served")
    assert len(served.served_regs) == len(served.served_hotkeys) == 778 and served.front == ()
    assert len(front.front) == 14 and len(front.served_regs) == len(front.served_hotkeys) == 764
    assert set(front.front) | set(front.served_regs) == set(served.served_regs)
    assert list(front.front) == sorted(front.front, key=cutover.fifo)
    assert len(front.demoted) == 194 and not set(front.demoted) & set(served.served_regs)


def test_a_pending_turn_that_left_v1s_window_is_kept_by_an_earlier_copy(v1):
    """k1.json shows 200 epochs. Epoch 1157 (pending) is the first to leave; the newer copy here is made up from the saved one."""
    saved, lost, full = copy.deepcopy(v1["k1"]), (923, 9474708), load(v1)
    saved["generated_at"] -= 3600
    v1["k1"]["turns"] = [row for row in v1["k1"]["turns"] if row["epoch_index"] >= 1158]
    alone = load(v1)
    assert len(alone.front) == 13 and lost in alone.served_regs  # without the saved copy, the miner is charged in silence
    assert [king.applied_at > 0 for king in alone.kings] == [False, False] + [True] * 8  # two kings left the window too
    assert load(v1, earlier=(saved,)) == full
    assert lost in load(v1, "served", (saved,)).served_regs
    with pytest.raises(CutoverError, match="1157"):
        load(v1, None, (saved,))
    judged, reopened = copy.deepcopy(saved), copy.deepcopy(saved)
    judged["generated_at"] += 60
    turn(judged, 1157)["verdict"] = "red"  # a newer saved copy shows the verdict: the turn stays judged
    turn(reopened, 1214)["verdict"] = "pending"  # the newest copy shows e1214 as RED: it stays RED
    assert load(v1, earlier=(judged, saved, reopened)) == replace(alone, kings=full.kings)
    other_miner, from_the_future = copy.deepcopy(saved), copy.deepcopy(saved)
    turn(other_miner, 1157)["uid"] = 924
    from_the_future["generated_at"] += 7200
    for bad in (other_miner, from_the_future, {**saved, "version": "sn120-k1-public-v0"}):
        with pytest.raises(CutoverError, match="earlier k1.json"):
            load(v1, earlier=(saved, bad))


def test_v1_retry_grants_go_to_the_front_in_block_then_uid_order(v1):
    """Made-up data: v1's public retry-grant list was empty on 13 Nov 2026. A grant is one more turn after a RED turn."""
    queue = v1["k1"]["queue"]
    reds = [turn(v1["k1"], 1208), turn(v1["k1"], 1214)]  # UID 8 at block 9493273, UID 992 at block 9488898
    queue["grants"] = [
        dict(position=i - 2, uid=red["uid"], hotkey=red["hotkey"], registration_block=red["reg_block"], turns=1, grant=dict(red_epoch=red["epoch_index"]))
        for i, red in enumerate(reds)
    ]
    waits = queue["expired_back_of_queue"][0]
    queue["next"] = [*queue["grants"], dict(position=0, uid=waits["uid"], hotkey=waits["hotkey"], registration_block=waits["registration_block"])]
    imp = load(v1, "served")
    assert imp.front == ((992, 9488898), (8, 9493273))  # the earlier registration block goes first, not the lower UID
    assert not set(imp.front) & set(imp.served_regs) and not {red["hotkey"] for red in reds} & set(imp.served_hotkeys)
    assert len(imp.served_regs) == len(imp.served_hotkeys) == 776
    both = load(v1, "front")
    assert len(both.front) == 16 and list(both.front) == sorted(both.front, key=cutover.fifo)
    assert [seat.reg.key for seat in preview_queue(both, chain(v1), Params())][:16] == list(both.front)
    queue["next"].append(dict(position=1, uid=8, hotkey=reds[0]["hotkey"], registration_block=9493273))  # a served seat that waits, no grant
    with pytest.raises(CutoverError, match="also listed as served"):
        load(v1)
    queue["next"].pop()
    queue["grants"][0]["hotkey"] = VALIDATOR[:-1] + "m"
    with pytest.raises(CutoverError, match="never had a turn"):
        load(v1)


def test_ingest_against_a_real_registration_snapshot_keeps_the_paid_queue(v1):
    """One turn per registration: v2 queues only seats that v1 did not serve, owed seats first, in block-then-UID order."""
    imp = load(v1)
    fresh = (Registration(990, "fresh-a", imp.chain_block + 5), Registration(991, "fresh-b", imp.chain_block + 2))
    reused = Registration(992, imp.served_hotkeys[0], imp.chain_block + 3)  # a served hotkey on a new seat: no second turn
    snapshot = chain(v1, (*fresh, reused))
    seats = [seat.reg.key for seat in preview_queue(imp, snapshot, Params())]
    on_chain = {reg.key for reg in snapshot.registrations}
    owed = [seat for seat in imp.front if seat in on_chain]
    back = [seat for seat in imp.demoted if seat in on_chain]
    assert len(owed) >= 10 and len(back) >= 150
    assert seats == owed + [fresh[1].key, fresh[0].key] + back
    assert not set(seats) & set(imp.served_regs) and all(uid != 0 for uid, _ in seats)
    with pytest.raises(CutoverError, match="older"):
        preview_queue(imp, ChainSnapshot(120, imp.chain_block - 1, 0, snapshot.registrations), Params())


def _served_row_lost(v1):
    v1["k1"]["queue"]["served"].pop()


def _served_seat_lost(v1):
    v1["k1"]["queue"]["served_registrations"].pop()


def _served_lists_empty(v1):  # the bad state read that would give every paid seat a second turn
    v1["k1"]["queue"].update(served=[], served_registrations=[], served_hotkeys=[], registrations_served=0, hotkeys_served=0)


def _payee_was_not_green(v1):
    v1["k1"]["payees"][0]["uid"] = 919


def _timeout_is_not_a_win(v1):
    turn(v1["k1"], 1211)["verdict"] = "timeout-promote"


def _demoted_seat_also_served(v1):
    queue = v1["k1"]["queue"]
    row = queue["expired_back_of_queue"][0]
    queue["served_registrations"].append([row["uid"], row["registration_block"]])
    queue["registrations_served"] += 1


def _uid_zero_served(v1):
    v1["k1"]["queue"]["served_registrations"][0][0] = 0


MUTATIONS = [
    _served_row_lost,
    _served_seat_lost,
    _served_lists_empty,
    _payee_was_not_green,
    _timeout_is_not_a_win,
    _demoted_seat_also_served,
    _uid_zero_served,
    lambda v1: v1["k1"].update(payees=[]),
    lambda v1: v1["k1"].update(version="sn120-k1-public-v0"),
    lambda v1: v1["k1"]["queue"].update(enabled=False),
    lambda v1: v1["k1"]["queue"].pop("served_hotkeys"),
    lambda v1: v1["k1"]["queue"].update(expired_back_of_queue_count=193),
    lambda v1: v1["k1"].update(generated_at=NOW - 7200),
    lambda v1: v1["k1"].update(generated_at=NOW + 7200),
    lambda v1: v1["mining"].update(netuid=81),
    lambda v1: v1["mining"].update(epoch="affine-mimo9b-grpo-20261013-e001400"),
    lambda v1: v1["mining"].update(checkpoint="cd" * 32),
    lambda v1: v1["mining"].pop("checkpoint"),
]


@pytest.mark.parametrize("mutate", MUTATIONS, ids=lambda fn: fn.__name__.strip("_<>") or "field")
def test_incomplete_or_inconsistent_v1_data_fails_closed(v1, mutate):
    mutate(v1)
    with pytest.raises(CutoverError):
        load(v1)


def test_genesis_is_fresh_mimo_with_the_given_hotkey_and_no_defaults():
    spec = build_genesis(DIGEST, VALIDATOR)
    assert (spec.model_repo, spec.model_revision, spec.base_hash) == (GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION, GENESIS_BASE_HASH)
    assert (spec.base_fp32_master_digest, spec.validator_hotkey) == (DIGEST, VALIDATOR)
    for digest, hotkey in ((None, VALIDATOR), ("AB" * 32, VALIDATOR), (DIGEST, None), (DIGEST, ""), (DIGEST, VALIDATOR[:-1] + "m")):
        with pytest.raises(ConfigError):
            build_genesis(digest, hotkey)


def test_dry_run_prints_and_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cutover.main([*FILES, "--max-age-s", "60", "--dry-run"]) == 2  # the fixture is older than a minute
    assert "CUTOVER IMPORT REFUSED" in capsys.readouterr().err
    snapshot = tmp_path / "chain.json"
    snapshot.write_bytes(chain(RAW).to_json())
    assert cutover.main([*FILES, "--chain", str(snapshot), "--dry-run"]) == 0  # the 14 open turns go to the front by default
    out = capsys.readouterr().out
    assert "v1-e1211 uid 920" in out and "front of the queue: 14 seats" in out and "first weights" in out and "DRY RUN" in out
    assert "UID 0 is owner: it gets weight only if no king is registered; not checked" in out and "e1290 took 125 prompts, 1000 rollouts; 15 v1 turns in all" in out
    assert sorted(path.name for path in tmp_path.iterdir()) == ["chain.json"]
    assert cutover.main([*FILES, "--undecided", "front", "--owner-hotkey", OWNER, "--dry-run"]) == 2  # nothing to check it against
    assert "--owner-hotkey needs --chain" in capsys.readouterr().err


def test_a_uid_0_that_is_not_the_owner_hotkey_is_refused(tmp_path, capsys):
    """v2 sends unearned weight to UID 0. The operator names the owner hotkey; another hotkey at UID 0 stops the cutover."""
    args = [*FILES, "--undecided", "front", "--dry-run", "--chain"]
    for name, holder in (("owner.json", OWNER), ("other.json", VALIDATOR)):
        (tmp_path / name).write_bytes(chain(RAW, (Registration(0, holder, 1),)).to_json())
    assert cutover.main([*args, str(tmp_path / "owner.json"), "--owner-hotkey", OWNER]) == 0
    assert f"UID 0 is {OWNER}: it gets weight only if no king is registered; it is the given owner hotkey" in capsys.readouterr().out
    assert cutover.main([*args, str(tmp_path / "other.json"), "--owner-hotkey", OWNER]) == 2
    assert f"UID 0 is {VALIDATOR}, not the owner hotkey {OWNER}" in capsys.readouterr().err
    assert cutover.main([*args, str(tmp_path / "owner.json"), "--owner-hotkey", OWNER[:-1] + "9"]) == 2  # a typing error
    assert "ConfigError" in capsys.readouterr().err


def test_real_run_writes_both_records_once(tmp_path, capsys):
    out = tmp_path / "import"
    genesis = ["--master-digest", DIGEST, "--validator-hotkey", VALIDATOR]
    assert cutover.main([*FILES, "--undecided", "front", "--out", str(out)]) == 2  # no genesis inputs, no defaults
    assert cutover.main([*FILES, "--master-digest", DIGEST, "--validator-hotkey", VALIDATOR[:-1] + "m", "--out", str(out)]) == 2
    assert not out.exists()
    assert cutover.main([*FILES, *genesis, "--undecided", "front", "--earlier-k1", str(DATA / "k1.json"), "--out", str(out)]) == 0
    assert sorted(path.name for path in out.iterdir()) == ["earlier-1-k1.json", "genesis.json", "k1.json", "mining.json", "v1_import.json"]
    imp = V1Import.from_json((out / "v1_import.json").read_bytes())
    assert imp.kings == load(copy.deepcopy(RAW)).kings and "sha256=" in imp.source and "undecided=front" in imp.source
    assert (out / "k1.json").read_bytes() == (DATA / "k1.json").read_bytes()
    assert cutover.main([*FILES, *genesis, "--undecided", "front", "--out", str(out)]) == 2  # never replaces an import
    assert "exists already" in capsys.readouterr().err


def test_the_controller_starts_from_the_written_files_with_the_previewed_queue(tmp_path, capsys):
    """One real run, then the foundation's ``bootstrap`` on its files: kings, served seats and queue order are the import's."""
    out, state_dir = tmp_path / "import", tmp_path / "state"
    snapshot = chain(RAW, (Registration(0, OWNER, 1), Registration(23, "a-new-hotkey", 2)))
    (tmp_path / "snapshot.json").write_bytes(snapshot.to_json())
    base_files = {"model.bin": DIGEST}  # made-up model files: the real base hash needs the real model
    (tmp_path / "genesis_files.json").write_text(json.dumps({"base_files": base_files, "master_files": {"master.bin": DIGEST}}))
    inputs = ["--master-digest", DIGEST, "--validator-hotkey", VALIDATOR, "--undecided", "front", "--owner-hotkey", OWNER]
    assert cutover.main([*FILES, *inputs, "--chain", str(tmp_path / "snapshot.json"), "--out", str(out)]) == 0
    assert (out / "chain.json").read_bytes() == snapshot.to_json()
    genesis = replace(GenesisSpec.from_json((out / "genesis.json").read_bytes()), base_hash=manifest_digest(base_files))
    (tmp_path / "genesis.json").write_bytes(genesis.to_json())
    files = [f"--genesis={tmp_path / 'genesis.json'}", f"--genesis-files={tmp_path / 'genesis_files.json'}"]
    files += [f"--v1-import={out / 'v1_import.json'}", f"--chain={out / 'chain.json'}"]
    assert cli.main(["bootstrap", "--state-dir", str(state_dir), "--stack", "sn120_v2.sim:stack", *files]) == 0
    imp, state = V1Import.from_json((out / "v1_import.json").read_bytes()), load_state(state_dir / JOURNAL_FILE)
    assert tuple(state.kings) == imp.kings and state.genesis.validator_hotkey == VALIDATOR
    assert state.served_regs == set(imp.served_regs) and state.served_hotkeys == set(imp.served_hotkeys)
    seats = waiting(state.queue)
    assert seats == preview_queue(imp, snapshot, Params()) and [seat.reg.key for seat in seats[:14]] == list(imp.front)
    assert not set(state.queue) & state.served_regs and (23, 2) in state.queue  # a seat v1 never saw waits behind the 14


def test_queued_miner_files_are_copied_in_queue_order_and_the_source_is_only_read(tmp_path):
    src, dst = LocalStore(tmp_path / "v1"), LocalStore(tmp_path / "v2")
    files = {"hkA": b'{"format": "affine-batch-v1"}', "hkC": b'{"train": {"lr": 1}}', "hkD": b"not json", "hkE": b"{}"}
    for hotkey, data in files.items():
        src.put(layout.miner_object(hotkey), data)
    before = {key: src.get(key) for key in src.list("")}
    lines = copy_miner_files(src, dst, ["hkA", "hkB", "hkC", "hkD", "hkE"], limit=3)  # hkB has no file; hkE is past the limit
    assert [line.split()[1] for line in lines] == ["hkA", "hkC", "hkD"] and dst.list("") == [layout.miner_object(h) for h in ("hkA", "hkC", "hkD")]
    assert lines[0].endswith("default recipe") and lines[1].endswith('train={"lr": 1}') and "not a JSON object" in lines[2]
    assert all(dst.get(key) == before[key] for key in dst.list("")) and {key: src.get(key) for key in src.list("")} == before
