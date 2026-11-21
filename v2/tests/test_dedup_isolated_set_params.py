"""The production assembly can pin a copied journal without R2 or a live mirror."""
from __future__ import annotations

import importlib.util
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("math_verify")

from sn120_v2 import assembly  # noqa: E402
from sn120_v2.cli import main, proposed_params  # noqa: E402
from sn120_v2.errors import ConfigError  # noqa: E402
from sn120_v2.journal import JOURNAL_FILE  # noqa: E402
from sn120_v2.params import Params  # noqa: E402
from sn120_v2.r2 import R2Store  # noqa: E402
from sn120_v2.sim import SimWorld, sim_params  # noqa: E402
from sn120_v2.state import load  # noqa: E402
from sn120_v2.store import LocalStore  # noqa: E402

SCRIPT = Path(__file__).parents[1] / "ops" / "dedup_isolated_set_params.py"
SPEC = importlib.util.spec_from_file_location("dedup_isolated_set_params", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
STEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP)


def copied_world(tmp_path: Path, monkeypatch) -> tuple[Path, Params]:
    source = SimWorld(tmp_path / "source", sim_params(irt_draw_rule="sqrt_fisher", null_samples_per_task=2))
    pinned = load(source.root / JOURNAL_FILE).params
    assert pinned is not None
    state = tmp_path / "copy"
    shutil.copytree(source.root, state)
    for name, value in (("SN120_PUBLIC_URL", "none"), ("SN120_V2_CHAIN_NETWORK", "none")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("SN120_V2_ISOLATED", "1")
    return state, pinned


def test_production_params_reject_legacy_file_and_malformed_overrides(tmp_path):
    legacy = Path(__file__).parents[1] / "tools" / "dedup-deploy" / "old-params.json"
    with pytest.raises(ConfigError, match="never apply the legacy king-x2 file with new code"):
        proposed_params(Params(), legacy, False, production=True)
    for raw in ("[]", "null", "{bad json"):
        path = tmp_path / "bad.json"
        path.write_text(raw)
        with pytest.raises(ConfigError):
            proposed_params(Params(), path, False, production=True)
    with pytest.raises(ConfigError, match="cannot read params file"):
        proposed_params(Params(), tmp_path / "missing.json", False, production=True)


def test_production_builder_is_local_unmirrored_and_the_guard_stops_r2_and_sockets(tmp_path, monkeypatch):
    state, _ = copied_world(tmp_path, monkeypatch)

    def no_r2(*args, **kwargs):
        raise AssertionError("R2 must never be constructed in isolated mode")

    monkeypatch.setattr(R2Store, "from_env", classmethod(no_r2))
    stack = assembly.stack(assembly.StackContext(state, Params(), "set-params"))
    assert not stack.mirror
    assert isinstance(stack.store, LocalStore) and stack.store.root == (state / "isolated-store").resolve()
    assert not list(stack.store.list("journal/"))
    with pytest.raises(AssertionError, match="network connection"), STEP.install_guard(stack.store.root):
        socket.create_connection(("192.0.2.1", 443), timeout=0.01)
    with pytest.raises(AssertionError, match="non-local store write"), STEP.install_guard(stack.store.root):
        LocalStore(tmp_path / "not-the-copy").put("must/not/write", b"bad")
    # The local store itself is not blocked: its object writes are disposable,
    # never the durable R2 journal.
    stack.store.put("local/evidence", b"copy")
    assert stack.store.get("local/evidence") == b"copy"


def test_open_round_refusal_does_not_change_even_the_local_journal(tmp_path, monkeypatch):
    state, pinned = copied_world(tmp_path, monkeypatch)
    # A fresh source with an open round: refusal must precede all appends,
    # including the leader fence of an ordinary controller command.
    source = SimWorld(tmp_path / "open-source", sim_params(irt_draw_rule="sqrt_fisher", null_samples_per_task=2))
    source.round()
    source.register(1, 0.1)
    record = source.controller.open_round(source.clock(), entropy="open-refusal-test")
    assert record is not None and source.controller.state.round is not None
    shutil.rmtree(state)
    shutil.copytree(source.root, state)
    journal = state / JOURNAL_FILE
    before = journal.read_bytes()
    output = tmp_path / "full.json"
    output.write_text(json.dumps(pinned.to_dict() | {"dedup_enabled": True, "train_seed_scope": "round"}))
    assert main(["set-params", "--state-dir", str(state), "--stack", "sn120_v2.assembly:stack",
                 "--params", str(output)]) == 2
    assert journal.read_bytes() == before


def test_real_script_uses_production_assembly_under_its_guard(tmp_path):
    root = Path(__file__).parents[1]
    source = SimWorld(tmp_path / "source", sim_params(irt_draw_rule="sqrt_fisher", null_samples_per_task=2))
    source.root.joinpath(JOURNAL_FILE).read_bytes()
    state = tmp_path / "state"
    state.mkdir()
    shutil.copyfile(source.root / JOURNAL_FILE, state / JOURNAL_FILE)
    pinned = load(state / JOURNAL_FILE).params
    full = tmp_path / "new.json"
    full.write_text(json.dumps(pinned.to_dict() | {name: getattr(Params(), name) for name in Params.LEGACY_FIELDS}
                               | {"dedup_enabled": True, "train_seed_scope": "round",
                                  "min_signal_rollouts": 64, "min_step_norm": 0.02}))
    completed = subprocess.run([sys.executable, "-I", str(SCRIPT), "--state-dir", str(state), "--params", str(full)],
                               cwd=root, text=True, capture_output=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report["isolated"] and report["network_connections"] == 0 and report["r2_writes"] == 0
    assert report["live_journal_writes"] == 0 and report["exit"] == 0
    assert load(state / JOURNAL_FILE).params.params_hash == report["after_params_hash"]
    assert not (state / "isolated-store" / "journal").exists()


def test_real_script_refuses_symlinked_inputs_before_importing_or_writing(tmp_path):
    source = SimWorld(tmp_path / "source", sim_params(irt_draw_rule="sqrt_fisher", null_samples_per_task=2))
    state = tmp_path / "state"
    state.mkdir()
    shutil.copyfile(source.root / JOURNAL_FILE, state / JOURNAL_FILE)
    before = (state / JOURNAL_FILE).read_bytes()
    params = tmp_path / "params.json"
    params.write_text(json.dumps(load(state / JOURNAL_FILE).params.to_dict()))
    linked = tmp_path / "linked.json"
    linked.symlink_to(params)
    completed = subprocess.run([sys.executable, "-I", str(SCRIPT), "--state-dir", str(state), "--params", str(linked)],
                               text=True, capture_output=True, timeout=120)
    assert completed.returncode != 0
    assert "full params file must be a regular non-symlink file" in completed.stderr
    assert not (state / "isolated-store").exists()
    outside = tmp_path / "outside-store"
    outside.mkdir()
    (state / "isolated-store").symlink_to(outside, target_is_directory=True)
    completed = subprocess.run([sys.executable, "-I", str(SCRIPT), "--state-dir", str(state), "--params", str(params)],
                               text=True, capture_output=True, timeout=120)
    assert completed.returncode != 0
    assert "isolated store must not be a symlink" in completed.stderr
    assert not list(outside.iterdir())
    assert (state / JOURNAL_FILE).read_bytes() == before
