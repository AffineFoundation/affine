#!/usr/bin/env python3
"""Old-code CLI with reviewed dedup replay compatibility.

Usage after the operator checks out c5252865:
  python -I rollback_cli.py --kit /path/to/dedup-deploy status --state-dir ...

The kit must remain outside the checkout. Before importing the old package,
this loader verifies every compatibility file against manifest.json and makes
sn120_v2.canonical/types/params/state/controller/dedup_compat/journal resolve
to those copies. All other modules, including gpu/**, remain the old checkout's
exact files.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

OLD_COMMIT = "c525286580c84d795b3ca7cbdf2302e96fce31ba"
OLD_GPU_DIGEST = "5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47"
OVERLAY = ("canonical", "types", "params", "state", "controller", "dedup_compat", "journal")


def _load(name: str, path: Path, *, package: bool = False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(kit: Path) -> dict[str, str]:
    path = kit / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing regular compatibility manifest {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    files = rows.get("files")
    if not isinstance(files, dict) or rows.get("old_commit") != OLD_COMMIT or rows.get("old_gpu_digest") != OLD_GPU_DIGEST:
        raise RuntimeError("compatibility manifest does not pin the required old release")
    if set(files) != {f"rollback-compat/{name}.py" for name in (*OVERLAY, "__init__")}:
        raise RuntimeError("compatibility manifest does not cover exactly the reviewed overlay")
    for name, expected in files.items():
        target = kit / name
        if not isinstance(expected, str) or len(expected) != 64 or target.is_symlink() or not target.is_file():
            raise RuntimeError(f"invalid compatibility file {name}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"compatibility file changed: {name}")
    return files


def _checkout(kit: Path) -> Path:
    explicit = os.environ.get("DEDUP_OLD_V2")
    v2 = Path(explicit).resolve() if explicit else kit.parent.parent
    if not (v2 / "sn120_v2/__init__.py").is_file():
        raise RuntimeError(f"DEDUP_OLD_V2 must contain sn120_v2/__init__.py, got {v2}; the default is the kit's parent checkout")
    return v2


def install(kit: Path) -> None:
    _manifest(kit)
    package = _load("sn120_v2", _checkout(kit) / "sn120_v2/__init__.py", package=True)
    canonical = _load("sn120_v2.canonical", kit / "rollback-compat/canonical.py")
    digest = canonical.manifest_digest({path.name: canonical.file_sha256(path)
                                       for path in sorted(Path(package.__path__[0], "gpu").glob("*.py"))})
    if digest != OLD_GPU_DIGEST:
        raise RuntimeError(f"old GPU code digest is {digest}, not {OLD_GPU_DIGEST}")
    for name in ("types", "params", "state", "controller", "dedup_compat", "journal"):
        module = _load(f"sn120_v2.{name}", kit / f"rollback-compat/{name}.py")
        setattr(package, name, module)
    # Journal.__init__ appends a leader fence. Refuse new-code-only work first,
    # after mirror catch-up and under the writer lock, so even a refusal leaves
    # both local and durable journals byte-identical.
    from sn120_v2.journal import Journal
    from sn120_v2.params import REVIEWED_NEW_PARAMS_HASH
    from sn120_v2.state import replay
    original = Journal.__init__

    def guarded(journal, path, owner, at, mirror=None):
        import fcntl
        from sn120_v2.journal import JournalRecord, LEADER, _parse
        journal.path, journal.mirror = Path(path), mirror
        journal.path.parent.mkdir(parents=True, exist_ok=True)
        with open(journal.path, "a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(0)
                if mirror is not None:
                    mirror.catch_up(handle, backfill=True)
                journal.records, covered = _parse(handle.read())
                handle.truncate(covered)
                journal._size = covered
                journal.fence = (journal.records[-1].fence if journal.records else 0) + 1
                if any(row.kind == "params_set" and row.data.get("params", {}).get("params_hash") == REVIEWED_NEW_PARAMS_HASH
                       or row.kind == "round_opened" and row.data.get("round", {}).get("params_hash") == REVIEWED_NEW_PARAMS_HASH
                       for row in journal.records):
                    from sn120_v2.dedup_compat import require_closed_dedup_history
                    require_closed_dedup_history(journal.records, replay(journal.records))
                journal._write(handle, LEADER, {"owner": owner}, at)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    Journal.__init__ = guarded


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[1] != "--kit":
        raise RuntimeError("usage: rollback_cli.py --kit KIT COMMAND [arguments]")
    kit = Path(sys.argv[2]).resolve(strict=True)
    # ``python -I`` deliberately drops the operator's PYTHONPATH. The unmodified
    # old run-loop's stack is an explicit reviewed operator module, so retain
    # that opt-in path while still excluding the script/kit directory itself.
    for directory in reversed(os.environ.get("PYTHONPATH", "").split(os.pathsep)):
        if directory and Path(directory).resolve() not in (kit, kit / "rollback-compat"):
            sys.path.insert(0, directory)
    install(kit)
    if sys.argv[3] == "verify-params":
        if len(sys.argv) != 5:
            raise RuntimeError("usage: rollback_cli.py --kit KIT verify-params PARAMS.json")
        from sn120_v2.params import REVIEWED_OLD_PARAMS_HASH, load_params
        actual = load_params(Path(sys.argv[4]).resolve(strict=True)).params_hash
        if actual != REVIEWED_OLD_PARAMS_HASH:
            raise RuntimeError(f"legacy params hash is {actual}, not the reviewed old pin {REVIEWED_OLD_PARAMS_HASH}")
        print(actual)
        return 0
    from sn120_v2.cli import main as cli_main
    return cli_main(sys.argv[3:])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
