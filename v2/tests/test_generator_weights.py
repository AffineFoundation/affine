"""Mechanism-1 Generator setter: every gate and SDK path with local fakes."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sn120_v2.canonical import canonical_json
from sn120_v2.generator import weights
from sn120_v2.generator.constants import PAYOUT_MAX_AGE_SECONDS, PAYOUT_SCHEMA_VERSION
from sn120_v2.generator.payout import payout_document, Claim

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 11, 19, 13, 35, 20, tzinfo=UTC).timestamp()
OWNER = "owner-hotkey"
REGISTERED = {0: OWNER, 7: "hk7", 8: "hk8", 9: "hk9"}
T1, T2 = "01" * 10, "02" * 10


def chain(*, block=1000, last_update=760, hotkeys=REGISTERED):
    return weights.ChainState(block, dict(hotkeys), last_update)


def claims(*rows):
    return list(rows) or [Claim(T1, 7, "hk7", NOW - 100, "a" * 64)]


def document(*rows, updated_at=NOW - 10):
    return payout_document(claims(*rows), now=NOW, updated_at=updated_at)


def decide(state=None, doc="default", **options):
    return weights.decide(
        state or chain(),
        document() if doc == "default" else doc,
        now=options.pop("now", NOW),
        mech0_writer_active=options.pop("mech0_writer_active", False),
        **options,
    )


def submitted(decision):
    assert decision.status == "submit"
    assert decision.weights and sum(decision.weights.values()) == pytest.approx(1.0)
    assert any(value > 0 for value in decision.weights.values())
    return decision


def test_valid_claims_receive_equal_task_shares_and_mech1_is_fixed():
    two = Claim(T2, 7, "hk7", NOW - 90, "b" * 64)
    three = Claim("03" * 10, 8, "hk8", NOW - 80, "c" * 64)
    decision = submitted(decide(doc=document(claims()[0], two, three), execute=True))
    assert decision.weights == pytest.approx({7: 2 / 3, 8: 1 / 3})
    assert decision.uids == [7, 8] and decision.values == [decision.weights[7], decision.weights[8]]
    assert not decision.burn and decision.block == 1000
    assert (weights.NETUID, weights.MECH_ID, weights.VERSION_KEY) == (120, 1, 0)
    assert weights.TIMER_SECONDS_UTC == {35 * 60 + 20, 50 * 60 + 20}


@pytest.mark.parametrize("minute,second", [(35, 20), (50, 20)])
def test_exact_deployment_timer_instants_are_accepted(minute, second):
    now = NOW - 35 * 60 - 20 + minute * 60 + second
    submitted(decide(now=now, execute=True))


@pytest.mark.parametrize(
    "minute,second,reason",
    [(24, 59, "before 25"), (25, 0, "four-minute"), (35, 19, "four-minute"), (50, 19, "four-minute")],
)
def test_time_gate_rejects_early_minute_and_outside_windows(minute, second, reason):
    now = dt.datetime(2026, 11, 19, 13, minute, second, tzinfo=UTC).timestamp()
    result = decide(now=now, execute=True)
    assert result.status == "skip" and reason in result.reason
    assert not result.weights


@pytest.mark.parametrize("instant", [35 * 60 + 20, 50 * 60 + 20])
@pytest.mark.parametrize("offset,allowed", [(0, True), (1, True), (10, True), (239, True), (240, False)])
def test_timer_window_tolerates_startup_jitter(instant, offset, allowed):
    base = dt.datetime(2026, 11, 19, 13, 35, 20, tzinfo=UTC)
    now = base.replace(hour=instant // 3600, minute=(instant % 3600) // 60, second=instant % 60).timestamp() + offset
    gate = weights._gates(chain(), now=now, mech0_writer_active=False, last_success_at=None)
    assert (gate is None) is allowed


def test_ignore_window_skips_only_time_gates():
    now = dt.datetime(2026, 11, 19, 13, 10, 5, tzinfo=UTC).timestamp()
    assert weights._gates(None, now=now, mech0_writer_active=False, last_success_at=None) is not None
    assert weights._gates(None, now=now, mech0_writer_active=False, last_success_at=None, ignore_window=True) is None
    assert decide(now=now, ignore_window=True, execute=True).status == "submit"
    active = decide(now=now, ignore_window=True, execute=True, mech0_writer_active=True)
    recent = decide(now=now, ignore_window=True, execute=True, last_success_at=now - 1)
    blocks = decide(now=now, ignore_window=True, execute=True, state=chain(last_update=900))
    assert active.status == recent.status == blocks.status == "skip"
    assert not active.weights and not recent.weights and not blocks.weights


def test_ignore_window_cli_is_forwarded(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(weights, "run_once", lambda **options: calls.append(options) or weights._base_report(NOW, False))
    payout = tmp_path / "payout.json"
    write_payout(payout)
    assert weights.main(["--payout", str(payout), "--ignore-window"]) == 0
    assert calls[0]["ignore_window"] is True and calls[0]["execute"] is False
    capsys.readouterr()


def test_mech0_activity_gate_has_priority_and_never_burns():
    result = decide(doc=None, mech0_writer_active=True, execute=True)
    assert result.status == "skip" and result.weights == {} and not result.burn
    assert "mechanism-0" in result.reason


@pytest.mark.parametrize("blocks,allowed", [(239, False), (240, True), (241, True)])
def test_mech1_last_update_240_block_boundary(blocks, allowed):
    result = decide(chain(last_update=1000 - blocks), execute=True)
    if allowed:
        submitted(result)
    else:
        assert result.status == "skip" and "240" in result.reason and result.weights == {}


@pytest.mark.parametrize("age,allowed", [(0, False), (45 * 60 - 0.001, False), (45 * 60, True), (46 * 60, True)])
def test_own_successful_write_45_minute_boundary(age, allowed):
    result = decide(last_success_at=NOW - age, execute=True)
    if allowed:
        submitted(result)
    else:
        assert result.status == "skip" and "45 minutes" in result.reason


@pytest.mark.parametrize("doc", [None, [], {}, {"claims": []}, {"version": "wrong"}, float("nan")])
def test_missing_or_malformed_payout_burns_uid0_and_never_all_zero(doc):
    result = submitted(decide(doc=doc, execute=True))
    assert result.weights == {0: 1.0} and result.burn and "burning UID 0" in result.reason


@pytest.mark.parametrize("age", [6 * 3600, 6 * 3600 - 0.001])
def test_freshness_boundary_and_future_timestamp(age):
    assert submitted(decide(doc=document(updated_at=NOW - age), execute=True)).weights == {7: 1.0}
    stale = submitted(decide(doc=document(updated_at=NOW - 6 * 3600 - 0.001), execute=True))
    future = submitted(decide(doc=document(updated_at=NOW + 0.001), execute=True))
    assert stale.weights == future.weights == {0: 1.0}
    assert stale.payout_updated_at == NOW - PAYOUT_MAX_AGE_SECONDS - 0.001


def test_any_changed_or_missing_earning_hotkey_burns_instead_of_renormalizing():
    original, other = claims()[0], Claim(T2, 8, "hk8", NOW - 90, "b" * 64)
    doc = document(original, other)
    changed = decide(doc=doc, state=chain(hotkeys={0: OWNER, 7: "new-key", 8: "hk8"}), execute=True)
    missing = decide(doc=doc, state=chain(hotkeys={0: OWNER, 8: "hk8"}), execute=True)
    for result in (changed, missing):
        submitted(result)
        assert result.weights == {0: 1.0} and result.burn and "hotkey changed" in result.reason


def test_empty_or_expired_valid_claims_use_primitive_burn():
    empty = submitted(decide(doc=payout_document([], now=NOW, updated_at=NOW), execute=True))
    old = Claim(T1, 7, "hk7", NOW - 24 * 3600, "a" * 64)
    expired = submitted(decide(doc=document(old, updated_at=NOW), execute=True))
    assert empty.weights == expired.weights == {0: 1.0} and empty.burn and expired.burn


@pytest.mark.parametrize(
    "damage",
    [
        lambda doc: doc.update(claims=doc["claims"] * 2),
        lambda doc: doc["claims"][0].update(uid=True),
        lambda doc: doc["claims"][0].update(task_id="not-a-task"),
        lambda doc: doc["claims"][0].update(file_sha256="bad"),
        lambda doc: doc["claims"][0].update(hotkey=""),
        lambda doc: doc.update(window_seconds=12),
        lambda doc: doc.update(max_age_seconds=12),
        lambda doc: doc.update(extra="not accepted"),
        lambda doc: doc["claims"][0].update(uploaded_at=NOW + 1),
    ],
)
def test_claim_and_schema_validation_fail_to_burn(damage):
    doc = document()
    damage(doc)
    result = submitted(decide(doc=doc, execute=True))
    assert result.weights == {0: 1.0} and "invalid" in result.reason


def test_unknown_chain_owner_or_wrong_wallet_skips_not_burns():
    no_owner = decide(chain(hotkeys={7: "hk7"}), execute=True)
    wrong_wallet = decide(execute=True, wallet_hotkey="someone-else")
    assert no_owner.status == wrong_wallet.status == "skip"
    assert no_owner.weights == wrong_wallet.weights == {}


def test_dry_run_is_default_and_never_becomes_submission():
    result = decide()
    assert result.status == "dry_run" and result.weights == {7: 1.0}
    assert decide(doc=None).status == "dry_run" and decide(doc=None).weights == {0: 1.0}


def test_negative_and_boolean_observations_fail_closed():
    assert decide(chain(block=True)).status == "skip"
    assert decide(chain(last_update=True)).status == "skip"
    assert decide(chain(last_update=1001)).status == "skip"
    assert decide(last_success_at=float("nan")).status == "skip"
    assert decide(mech0_writer_active="inactive").status == "skip"


class FakeChain:
    block = 1000

    def __init__(self, *, plan_ok=True, success=True):
        self.plan_ok, self.success = plan_ok, success
        self.queries, self.intents, self.executions = [], [], []

    def query_map(self, item, keys):
        self.queries.append(("map", item, keys))
        return {uid: REGISTERED.get(uid, f"hk{uid}") for uid in range(max(REGISTERED) + 1)}

    def query(self, item, keys):
        self.queries.append(("query", item, keys))
        assert item == "LastUpdate"
        assert keys == [weights.storage_index(weights.NETUID, weights.MECH_ID)]
        assert keys == [4216]
        return [0 if uid == 0 else 999 for uid in range(max(REGISTERED) + 1)]

    def plan(self, intent, wallet):
        self.intents.append(intent)
        return SimpleNamespace(ok=self.plan_ok, signer_address=OWNER)

    def execute(self, intent, wallet, **options):
        self.executions.append((intent, wallet, options))
        return SimpleNamespace(success=self.success, block_hash="0xcommit" if self.success else "")


class FakeSDK:
    class Wallet:
        def __init__(self, **options):
            self.options = options
            self.hotkey = SimpleNamespace(ss58_address=OWNER)

    class Subtensor:
        def __init__(self, network):
            raise AssertionError("injected chain should prevent SDK network construction")

    storage = SimpleNamespace(SubtensorModule=SimpleNamespace(Keys="Keys", LastUpdate="LastUpdate"))

    def __init__(self):
        self.intents = []

    def SetWeights(self, **options):
        self.intents.append(options)
        return options


def write_payout(path: Path, doc=None) -> None:
    path.write_bytes(canonical_json(document() if doc is None else doc))


def runner(tmp_path, sdk=None, fake_chain=None, **options):
    payout = tmp_path / "payout.json"
    if "payout_path" not in options and not payout.exists():
        write_payout(payout)
    sdk = sdk or FakeSDK()
    fake_chain = fake_chain or FakeChain()
    report = weights.run_once(
        payout_path=options.pop("payout_path", payout),
        runs_path=tmp_path / "runs.jsonl",
        now=options.pop("now", NOW),
        sdk=sdk,
        chain=fake_chain,
        wallet=options.pop("wallet", SimpleNamespace(hotkey=SimpleNamespace(ss58_address=OWNER))),
        mech0_active=options.pop("mech0_active", False),
        **options,
    )
    return report, sdk, fake_chain


def test_execute_builds_exact_mech1_setweights_and_one_timelocked_commit(tmp_path):
    report, sdk, fake = runner(tmp_path, execute=True)
    assert report["status"] == "submitted" and report["block_hash"] == "0xcommit"
    assert sdk.intents == [{
        "netuid": 120, "uids": [7], "weights": [1.0], "mechid": 1, "version_key": 0,
    }]
    assert fake.intents == sdk.intents
    [(intent, wallet, options)] = fake.executions
    assert intent["mechid"] == 1 and intent["version_key"] == 0
    assert options == {"wait_for_inclusion": True, "wait_for_finalization": True, "retries": 0}
    assert report["execute"] and report["netuid"] == 120 and report["mechid"] == 1


def test_default_dry_run_has_no_wallet_intent_plan_or_chain_execution(tmp_path):
    report, sdk, fake = runner(tmp_path)
    assert report["status"] == "dry_run", report
    assert not report["execute"]
    assert sdk.intents == fake.intents == fake.executions == []


def test_missing_payout_dry_run_reaches_uid0_plan(tmp_path):
    report, sdk, fake = runner(tmp_path, payout_path=tmp_path / "missing-payout.json")
    assert report["status"] == "dry_run" and report["burn"]
    assert report["uids"] == [0] and report["weights"] == {"0": 1.0}
    assert "missing or unreadable" in report["reason"]
    assert sdk.intents == fake.intents == fake.executions == []
    assert ("query", "LastUpdate", [4216]) in fake.queries


def test_storage_index_is_single_mech_netuid_key():
    assert weights.storage_index(120, 1) == 4216
    assert weights.storage_index(120, 0) == 120
    for netuid, mechid in ((-1, 1), (120, -1), (True, 1), (120, False)):
        with pytest.raises(ValueError):
            weights.storage_index(netuid, mechid)


def test_skip_gates_run_before_chain_and_do_not_fallback_to_burn(tmp_path):
    fake = FakeChain()
    for options in (
        {"mech0_active": True},
        {"now": dt.datetime(2026, 11, 19, 13, 24, 59, tzinfo=UTC).timestamp()},
        {"now": dt.datetime(2026, 11, 19, 13, 35, 19, tzinfo=UTC).timestamp()},
        {"successful_at": NOW - 44 * 60},
    ):
        report, sdk, _ = runner(tmp_path, fake_chain=fake, **options)
        assert report["status"] == "skip" and sdk.intents == [] and fake.queries == []


def test_execute_wallet_is_constructed_lazily_with_default_default(tmp_path):
    class WalletSDK(FakeSDK):
        def __init__(self):
            super().__init__()
            self.wallet_options = None

        def Wallet(self, **options):
            self.wallet_options = options
            return SimpleNamespace(hotkey=SimpleNamespace(ss58_address=OWNER))

    sdk = WalletSDK()
    report, _, _ = runner(tmp_path, sdk=sdk, execute=True, wallet=None, wallet_path="/operator/wallets")
    assert report["status"] == "submitted" and sdk.wallet_options == {
        "name": "default", "hotkey": "default", "path": "/operator/wallets",
    }
    sdk = WalletSDK()
    report, _, _ = runner(tmp_path, sdk=sdk, execute=False, wallet=None, successful_at=NOW - 46 * 60)
    assert report["status"] == "dry_run" and sdk.wallet_options is None


def test_mech1_last_update_gate_blocks_execution(tmp_path):
    class Recent(FakeChain):
        def query(self, item, keys):
            return [761 if uid == 0 else 999 for uid in range(max(REGISTERED) + 1)]

    report, sdk, fake = runner(tmp_path, fake_chain=Recent(), execute=True)
    assert report["status"] == "skip" and "240" in report["reason"]
    assert sdk.intents == fake.executions == []


@pytest.mark.parametrize("plan_ok,success", [(False, True), (True, False)])
def test_plan_refusal_or_unsuccessful_execute_never_counts_as_success(tmp_path, plan_ok, success):
    report, sdk, fake = runner(tmp_path, fake_chain=FakeChain(plan_ok=plan_ok, success=success), execute=True)
    assert report["status"] == "failed"
    assert weights.last_success_at(tmp_path / "runs.jsonl") is None
    assert len(fake.executions) == (1 if plan_ok else 0)


def test_missing_invalid_stale_and_hotkey_churn_execute_burn_via_mech1(tmp_path):
    for index, doc in enumerate((None, {"broken": True}, document(updated_at=NOW - 6 * 3600 - 1))):
        directory = tmp_path / str(index)
        directory.mkdir()
        path = directory / "payout.json"
        if doc is not None:
            write_payout(path, doc)
        report, sdk, fake = runner(directory, execute=True, payout_path=path)
        assert report["status"] == "submitted" and report["burn"]
        assert sdk.intents[0] | {"uids": [0], "weights": [1.0]} == sdk.intents[0]
        assert sdk.intents[0]["uids"] == [0] and sdk.intents[0]["weights"] == [1.0]

    directory = tmp_path / "churn"
    directory.mkdir()
    churn = FakeChain()
    churn.query_map = lambda item, keys: {
        uid: ("new-hotkey" if uid == 7 else f"hk{uid}") for uid in range(max(REGISTERED) + 1)
    } | {0: OWNER}
    report, sdk, _ = runner(directory, fake_chain=churn, execute=True)
    assert report["status"] == "submitted" and report["burn"]
    assert sdk.intents[0]["uids"] == [0] and sdk.intents[0]["weights"] == [1.0]


def test_runs_jsonl_records_utc_timestamp_and_success_cursor_atomically(tmp_path):
    path = tmp_path / "runs.jsonl"
    first, _, _ = runner(tmp_path, execute=False)
    second, _, _ = runner(tmp_path, execute=True, successful_at=NOW - 46 * 60)
    lines = path.read_bytes().splitlines()
    assert len(lines) == 2 and all(line and not line.startswith(b" ") for line in lines)
    rows = [json.loads(line) for line in lines]
    assert rows == [first, second]
    assert rows[0]["timestamp"] == "2026-11-19T13:35:20Z"
    assert weights.last_success_at(path) == NOW
    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / "payout.json").exists()


def test_success_cursor_ignores_dry_run_failed_and_other_mechanism_rows(tmp_path):
    path = tmp_path / "runs.jsonl"
    rows = [
        {"status": "dry_run", "execute": False, "netuid": 120, "mechid": 1, "observed_at": NOW - 100},
        {"status": "failed", "execute": True, "netuid": 120, "mechid": 1, "observed_at": NOW - 90},
        {"status": "submitted", "execute": True, "netuid": 120, "mechid": 0, "observed_at": NOW - 80},
        {"status": "submitted", "execute": True, "netuid": 120, "mechid": 1, "observed_at": NOW - 70},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert weights.last_success_at(path) == NOW - 70
    assert weights.run_once(
        payout_path=tmp_path / "payout.json", runs_path=path, now=NOW,
        sdk=FakeSDK(), chain=FakeChain(), mech0_active=False,
    )["reason"].endswith("less than 45 minutes ago")


def test_json_parser_rejects_duplicate_keys_nan_and_non_object(tmp_path):
    for raw in (b'{"version":1,"version":2}', b'{"updated_at":NaN}', b'[]'):
        path = tmp_path / "payout.json"
        path.write_bytes(raw)
        loaded = weights.load_payout(path)
        assert loaded.document is None and loaded.error == "missing or invalid"


def test_atomic_append_preserves_previous_rows_and_refuses_partial_record(tmp_path):
    path = tmp_path / "runs.jsonl"
    first = {"status": "dry_run", "execute": False, "observed_at": NOW}
    second = {"status": "failed", "execute": True, "observed_at": NOW + 1}
    weights._atomic_append(path, first)
    weights._atomic_append(path, second)
    assert [json.loads(line) for line in path.read_bytes().splitlines()] == [first, second]
    before = path.read_bytes()
    path.write_bytes(before + b'{"partial":')
    with pytest.raises(ValueError, match="row boundary"):
        weights._atomic_append(path, {"status": "skip"})
    assert path.read_bytes() == before + b'{"partial":' and not list(tmp_path.glob("*.tmp"))


def test_future_last_success_and_unsupported_timestamp_skip_safely():
    assert decide(last_success_at=NOW + 1).status == "skip"
    assert decide(now=1e300).status == "skip"


def test_mech0_activity_probe_only_checks_units_and_fails_closed(monkeypatch):
    calls = []

    def inactive(command, **options):
        calls.append(command)
        return SimpleNamespace(stdout="inactive\n")

    monkeypatch.setattr(weights.subprocess, "run", inactive)
    assert weights._mech0_writer_active() is False
    assert len(calls) == len(weights.MECH0_WRITER_UNITS)
    assert all(call[:3] == ["systemctl", "--user", "is-active"] for call in calls)

    monkeypatch.setattr(
        weights.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="activating\n")
    )
    assert weights._mech0_writer_active() is True
    monkeypatch.setattr(weights.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no systemctl")))
    with pytest.raises(RuntimeError, match="cannot establish"):
        weights._mech0_writer_active()


def test_local_activity_probe_failure_is_logged_without_wallet_error_details(tmp_path, monkeypatch):
    def fail():
        raise RuntimeError("secret wallet key material must not be logged")

    report, sdk, fake = runner(tmp_path, mech0_active=fail, execute=True)
    assert report["status"] == "failed" and "RuntimeError" in report["reason"]
    assert "secret wallet" not in json.dumps(report) and sdk.intents == fake.queries == []


def test_no_sdk_import_for_time_or_activity_skip(tmp_path, monkeypatch):
    def forbidden(_):
        raise AssertionError("SDK import must be lazy")

    monkeypatch.setattr(weights, "_load_sdk", forbidden)
    report = weights.run_once(
        payout_path=tmp_path / "payout.json", runs_path=tmp_path / "runs.jsonl",
        now=NOW - 20 * 60, mech0_active=False,
    )
    assert report["status"] == "skip"


def test_cli_dry_run_is_default_and_execute_is_explicit(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_run(**options):
        calls.append(options)
        return weights._base_report(NOW, options["execute"]) | {"status": "dry_run" if not options["execute"] else "submitted"}

    monkeypatch.setattr(weights, "run_once", fake_run)
    payout = tmp_path / "payout.json"
    write_payout(payout)
    assert weights.main(["--payout", str(payout)]) == 0
    assert calls[-1]["execute"] is False and calls[-1]["wallet_name"] == "default" and calls[-1]["wallet_hotkey_name"] == "default"
    assert calls[-1]["runs_path"] == tmp_path / "runs.jsonl"
    assert weights.main(["--payout", str(payout), "--execute"]) == 0 and calls[-1]["execute"] is True
    for line in capsys.readouterr().out.splitlines():
        output = json.loads(line)
        assert output["mechid"] == 1 and output["version_key"] == 0 and "wallet" not in output
