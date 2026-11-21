#!/usr/bin/env python3
"""Apply a full Params file to an isolated copy of a production journal.

The source journal must already have been copied by the operator. This command
writes a leader record and, when permitted, one params_set record to that copy.
It never pulls, mirrors, or connects to a network. The real production stack
builder is retained; its store is explicitly redirected by SN120_V2_ISOLATED=1.

  python -I v2/ops/dedup_isolated_set_params.py --state-dir COPY --params FULL.json

A socket attempt, construction of R2Store, any journal Mirror push, or a local
store write outside COPY/isolated-store fails closed. Symlinked state or input
files are refused so a seemingly copied path cannot name the live journal.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
from contextlib import contextmanager
from pathlib import Path


def _package() -> None:
    directory = Path(__file__).resolve().parents[1] / "sn120_v2"
    spec = importlib.util.spec_from_file_location(
        "sn120_v2", directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules["sn120_v2"] = package
    spec.loader.exec_module(package)


def _plain_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


@contextmanager
def install_guard(root: Path):
    """Guard the process before importing the production assembly."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    _package()
    from sn120_v2.journal import Mirror
    from sn120_v2.r2 import R2Store
    from sn120_v2.store import LocalStore

    original_r2_init = R2Store.__init__
    original_r2_from_env = R2Store.__dict__["from_env"]
    original_mirror_push = Mirror.push

    def blocked(*args, **kwargs):
        raise AssertionError("ISOLATION GUARD: network connection attempted")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked

    def no_r2(*args, **kwargs):
        raise AssertionError("ISOLATION GUARD: R2Store construction attempted")

    def no_mirror(self, record):
        raise AssertionError(f"ISOLATION GUARD: journal mirror write attempted at seq {record.seq}")

    original_write = LocalStore._write

    def local_only(self, key, fill):
        if self.root != root:
            raise AssertionError(f"ISOLATION GUARD: non-local store write {self.root} key {key!r}")
        return original_write(self, key, fill)

    R2Store.from_env = classmethod(no_r2)
    R2Store.__init__ = no_r2
    Mirror.push = no_mirror
    LocalStore._write = local_only
    try:
        yield LocalStore
    finally:
        # Tests use the context manager; the standalone process retains the
        # guard for its entire CLI and then restores it only during shutdown.
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.create_connection = original_create_connection
        R2Store.from_env = original_r2_from_env
        R2Store.__init__ = original_r2_init
        Mirror.push = original_mirror_push
        LocalStore._write = original_write


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--params", required=True)
    args = parser.parse_args(argv)
    state = Path(args.state_dir)
    if state.is_symlink() or not state.is_dir():
        raise ValueError(f"state directory must be a real non-symlink directory: {state}")
    state = state.resolve()
    journal = state / "journal.jsonl"
    _plain_file(journal, "copied journal")
    params = Path(args.params)
    _plain_file(params, "full params file")
    params = params.resolve(strict=True)
    # Refuse an input file inside the state directory: the command must not
    # mistake one of its own prior outputs for an independently reviewed file.
    if state in params.parents:
        raise ValueError("the reviewed full params file must be outside the copied state directory")
    copied_store = state / "isolated-store"
    if copied_store.is_symlink():
        raise ValueError("the isolated store must not be a symlink")
    expected = copied_store.resolve()
    _package()
    from sn120_v2 import assembly
    from sn120_v2.cli import main as cli_main
    from sn120_v2.controller import StackContext
    from sn120_v2.state import load

    before = load(journal)
    if before.params is None:
        raise ValueError("copied journal has no pinned params")
    with install_guard(expected) as local_store:
        os.environ["SN120_V2_ISOLATED"] = "1"
        os.environ["SN120_PUBLIC_URL"] = "none"
        os.environ["SN120_V2_CHAIN_NETWORK"] = "none"
        # Credentials are neither needed nor inherited by this local operation.
        for name in list(os.environ):
            if name.startswith("SN120_R2_"):
                os.environ.pop(name)
        # Build the actual production assembly once under the guard. The CLI builds
        # it again for its ordinary command path; neither build may read R2.
        probe = assembly.stack(StackContext(state, before.params, "set-params"))
        if probe.mirror or not isinstance(probe.store, local_store) or probe.store.root != expected:
            raise AssertionError("ISOLATION GUARD: production builder did not select its unmirrored local store")
        result = cli_main([
            "set-params", "--state-dir", str(state), "--stack", "sn120_v2.assembly:stack",
            "--params", str(params),
        ])
        after = load(journal)
        print(json.dumps({
            "isolated": True,
            "network_connections": 0,
            "r2_writes": 0,
            "live_journal_writes": 0,
            "state_dir": str(state),
            "before_params_hash": before.params.params_hash,
            "after_params_hash": after.params.params_hash if after.params else None,
            "journal_records_before": before.last_seq + 1,
            "journal_records_after": after.last_seq + 1,
            "exit": result,
        }, sort_keys=True))
        return result


if __name__ == "__main__":
    raise SystemExit(main())
