"""Old-code rollback compatibility for the exact reviewed field-gate pins.

Every fixture comes from the real synthetic controller. All copies and source
archives live beneath pytest's scratch directory; no service or durable store.
"""
from __future__ import annotations

import dataclasses
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from sn120_v2.journal import JournalRecord, _parse, read_journal
from sn120_v2.params import Params

ROOT = Path(__file__).parents[2]
KIT = ROOT / "v2/tools/dedup-deploy"
OLD_COMMIT = "c525286580c84d795b3ca7cbdf2302e96fce31ba"
OLD_PIN = "bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a"
DEDUP_PIN = "8caddd9c6fc276a9669dff9e01e1bb7e437d98bed5b12a605141e6025b9f9ec4"
HISTORICAL_PIN = "d5979c301c7acf8a56cb6a9989e5b212e5887ec1bf0117f9250202af6a95f35c"
FIELD_PINS = {
    "17c20f822fb9c97b397d24272e18eaebf8de9574a9d0056361c8d529856696d4": "field_gate_params.json",
    "471b74f4a306efbb1e9444069ea832b5e02d68ec410f5ffebb06353f2deb9c5a": "field_gate_params_k1f1.json",
    "848707723628c85dc13f8c93e39d132432fcb5851076165490ac01693176236e": "field_gate_params_king_led.json",
}
OLD_FIELD_PINS = {
    "346b99a875be5a9d5e2467bf846f20d124229397b67935c6563f1e7c2409b8b1": "field_gate_params.json",
    "05a1b55556a14cd001f926f774a4231023851e43c2393ec41bcd23646fe6ca4b": "field_gate_params_k1f1.json",
    "0eb4867f6c08aa179015c9e927bb38d6ade14a022c406401a026280845a8e098": "field_gate_params_king_led.json",
}


def _git(path: str, revision: str = "14ae80f5") -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


def _params(path: str, revision: str = "14ae80f5") -> dict:
    data = json.loads(_git(path, revision))
    assert Params.from_dict(data).params_hash in {OLD_PIN, DEDUP_PIN, HISTORICAL_PIN}
    return data


def _field_fixture(tmp_path: Path, field_name: str) -> Path:
    """Run the real controller fixture at the exact reviewed field pin."""
    output = tmp_path / "fixture"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               DEDUP_REVIEWED_PARAMS=str(ROOT / "ops" / field_name),
               DEDUP_REVIEWED_PARAMS_HASH=str(Params.from_dict(json.loads((ROOT / "ops" / field_name).read_text())).params_hash))
    subprocess.run([
        sys.executable, "-I", "-c",
        "import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')",
        str(ROOT / "v2"), str(KIT / "make-rollback-fixture.py"), str(output),
    ], cwd=tmp_path, env=env, check=True, capture_output=True, text=True, timeout=120)
    return output


def _copy_with_transition(tmp_path: Path, fixture: Path, field_name: str) -> tuple[Path, list[JournalRecord]]:
    """Rebuild a byte-valid chain: old -> reviewed dedup -> field closed rounds."""
    root = tmp_path / "journal"
    root.mkdir()
    source = read_journal(fixture / "journal.jsonl")
    rows = [next(row for row in source if row.kind == "genesis")]
    field = json.loads((ROOT / "ops" / field_name).read_text())
    for data in (_params("v2/tools/dedup-deploy/old-params.json"),
                 _params("v2/tools/dedup-deploy/new-params.json"), field):
        rows.append(JournalRecord(len(rows), 1, len(rows) + 1, "params_set", {"params": data}, rows[-1].checksum))
    for row in source:
        if row.kind not in ("leader", "genesis", "params_set"):
            rows.append(JournalRecord(len(rows), 1, len(rows) + 1, row.kind, row.data, rows[-1].checksum))
    copied = rows
    rows = []
    for row in copied:
        rows.append(dataclasses.replace(
            row, seq=len(rows), fence=1,
            prev=rows[-1].checksum if rows else "",
        ))
    path = root / "journal.jsonl"
    path.write_bytes(b"".join(row.line() + b"\n" for row in rows))
    assert [row.data["params"].get("z_bar") for row in rows if row.kind == "params_set"] == [1.0, 1.5, field["z_bar"]]
    assert any(row.kind == "round_settled" for row in rows)
    return path, rows


def _old_v2(tmp_path: Path) -> Path:
    """Extract the exact old source without executing anything from the archive."""
    root = tmp_path / "old-source"
    root.mkdir()
    archive = subprocess.check_output(["git", "archive", OLD_COMMIT, "v2/sn120_v2"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            assert target.is_relative_to(root.resolve()) and not member.issym() and not member.islnk()
        tar.extractall(root, filter="data")
    return root / "v2"


def _install_old_overlay(old_v2: Path) -> None:
    sys.path.insert(0, str(KIT))
    import rollback_cli
    os.environ["DEDUP_OLD_V2"] = str(old_v2)
    rollback_cli.install(KIT)


def test_old_dedup_field_closed_journal_replays_with_legacy_behavior(tmp_path):
    fixture = _field_fixture(tmp_path, "field_gate_params_k1f1.json")
    journal, rows = _copy_with_transition(tmp_path, fixture, "field_gate_params_k1f1.json")
    _install_old_overlay(_old_v2(tmp_path))
    from sn120_v2.state import replay
    from sn120_v2.dedup_compat import digest_params, historical_params, require_closed_dedup_history
    from sn120_v2.params import Params as CompatParams

    original = [row.data["params"] for row in rows if row.kind == "params_set"]
    assert [digest_params(data) for data in original] == [OLD_PIN, DEDUP_PIN, "471b74f4a306efbb1e9444069ea832b5e02d68ec410f5ffebb06353f2deb9c5a"]
    state = replay(read_journal(journal))
    require_closed_dedup_history(read_journal(journal), state)
    assert state.round is None
    assert state.params.params_hash == OLD_PIN and state.params.z_bar == 1.0
    assert [CompatParams.from_dict(data).params_hash for data in original] == [OLD_PIN] * 3
    assert [CompatParams.from_dict(data).z_bar for data in original] == [1.0] * 3
    assert [historical_params(data).params_hash for data in original] == [OLD_PIN] * 3

    # The two original combined-release pins retain their old translation too.
    for revision, pin in (("14ae80f5", DEDUP_PIN), ("6e73e551^", HISTORICAL_PIN)):
        record = _params("v2/tools/dedup-deploy/new-params.json", revision)
        assert digest_params(record) == pin
        decoded = CompatParams.from_dict(record)
        assert decoded.params_hash == OLD_PIN and decoded.z_bar == 1.0


def test_open_field_round_blocks_rollback_before_any_write(tmp_path):
    fixture = _field_fixture(tmp_path, "field_gate_params_k1f1.json")
    _install_old_overlay(_old_v2(tmp_path))
    from sn120_v2.journal import Journal
    from sn120_v2.state import replay

    source = fixture / "open-undecided/journal.jsonl"
    rows = read_journal(source)
    assert rows[-1].kind == "round_opened" and rows[-1].data["round"]["params_hash"] in FIELD_PINS
    assert replay(rows).round is not None
    path = tmp_path / "open/journal.jsonl"
    path.parent.mkdir()
    path.write_bytes(source.read_bytes())
    before = path.read_bytes()
    with pytest.raises(Exception, match="finish it with reviewed new-code run"):
        Journal(path, "rollback-regression", 2_000_000_000)
    assert path.read_bytes() == before


def test_every_reviewed_field_pin_restores_the_same_legacy_params(tmp_path):
    _install_old_overlay(_old_v2(tmp_path))
    from sn120_v2.params import Params as CompatParams
    # Pre-render field-gate records keep their exact old worker identity.
    for pin, name in {**OLD_FIELD_PINS, **FIELD_PINS}.items():
        data = json.loads((ROOT / "ops" / name).read_text())
        if pin in OLD_FIELD_PINS:
            data["recipe"]["code_digest"] = "2f7fd4ecaa8fa26800d8a3f54063d6286d614e309fcdffd9af59bfffeb7824fa"
        assert Params.from_dict(data).params_hash == pin
        old = CompatParams.from_dict(data)
        assert old.params_hash == OLD_PIN and old.z_bar == 1.0
        assert old.recipe.code_digest == "5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47"
