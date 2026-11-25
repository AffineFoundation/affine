#!/usr/bin/env python3
"""Run the exact pre-KL checkout with bounded, verified KL replay compatibility.

Usage (from any directory, after checking out 917ad548):
  KL_OLD_V2=/path/to/917ad548/v2 python3 -I rollback.py --kit /path/to/kit status --state-dir ...
  KL_OLD_V2=/path/to/917ad548/v2 python3 -I rollback.py --kit /path/to/kit set-params ...
  KL_OLD_V2=/path/to/917ad548/v2 python3 -I rollback.py --kit /path/to/kit run ...

The loader verifies its manifest and all old GPU files before importing any
compatibility module. It never rewrites historical journal or result JSON.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

OLD_COMMIT = "917ad5484ff38ba96924fb7631fcad79720151f2"
OLD_GPU_DIGEST = "22b2d191b32e394d4621ce477e702b50410a6ef7706cce6cfd20c60ff64e42e1"
OVERLAY = ("types", "params", "state", "controller", "journal", "kl_compat")


def _load(name: str, path: Path, *, package: bool = False):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)] if package else None)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(kit: Path) -> None:
    path = kit / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing regular compatibility manifest {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("old_commit") != OLD_COMMIT or manifest.get("old_gpu_digest") != OLD_GPU_DIGEST:
        raise RuntimeError("compatibility manifest does not pin the required old release")
    expected = {f"rollback-compat/{name}.py" for name in OVERLAY}
    if set(manifest.get("files") or {}) != expected:
        raise RuntimeError("compatibility manifest must cover exactly the reviewed overlay")
    for name, wanted in manifest["files"].items():
        target = kit / name
        if target.is_symlink() or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != wanted:
            raise RuntimeError(f"compatibility file is missing or changed: {name}")


def install(kit: Path, *, transition: bool = False) -> None:
    verify(kit)
    v2 = Path(os.environ.get("KL_OLD_V2", "")).resolve()
    package_path = v2 / "sn120_v2"
    if not (package_path / "__init__.py").is_file():
        raise RuntimeError("KL_OLD_V2 must name the v2 directory of the exact old checkout")
    package = _load("sn120_v2", package_path / "__init__.py", package=True)
    # Old modules bind Params while importing types. Install the complete
    # overlay before any old module can resolve that name.
    params_module = _load("sn120_v2.params", kit / "rollback-compat" / "params.py")
    types_module = _load("sn120_v2.types", kit / "rollback-compat" / "types.py")
    if len(dataclasses.fields(params_module.Params)) < 80:
        raise RuntimeError("compatibility Params did not load its complete dataclass definition")
    # cli.open_controller constructs Journal before Controller. This flag is
    # set only for the explicit set-params command, never for run/end-round.
    params_module.ROLLBACK_TRANSITION = transition
    from sn120_v2.canonical import _hints
    _hints.cache_clear()
    gpu = manifest_digest({p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in sorted((package_path / "gpu").glob("*.py"))})
    if gpu != OLD_GPU_DIGEST:
        raise RuntimeError(f"old GPU digest is {gpu}, not {OLD_GPU_DIGEST}")
    setattr(package, "params", params_module)
    setattr(package, "types", types_module)
    for name in ("journal", "state", "controller", "kl_compat"):
        module = _load(f"sn120_v2.{name}", kit / "rollback-compat" / f"{name}.py")
        setattr(package, name, module)
    from sn120_v2.types import Params


def manifest_digest(files):
    from sn120_v2.canonical import canonical_json
    return hashlib.sha256(canonical_json(files)).hexdigest()


def main() -> int:
    argv = sys.argv[1:]
    # Accept the old CLI's optional -m form without an ambient shim.
    if len(argv) >= 2 and argv[:2] == ["-m", "sn120_v2"]:
        argv = argv[2:]
    if len(argv) < 3 or argv[0] != "--kit":
        raise RuntimeError("usage: rollback.py --kit KIT COMMAND [arguments]")
    kit = Path(argv[1]).resolve(strict=True)
    for directory in reversed(os.environ.get("PYTHONPATH", "").split(os.pathsep)):
        if directory and Path(directory).resolve() not in (kit, kit / "rollback-compat"):
            sys.path.insert(0, directory)
    install(kit, transition=argv[2] == "set-params")
    if argv[2] == "verify-params":
        if len(argv) != 4:
            raise RuntimeError("usage: rollback.py --kit KIT verify-params OLD_PARAMS.json")
        from sn120_v2.params import OLD_PARAMS_HASH, load_params
        actual = load_params(Path(argv[3]).resolve(strict=True)).params_hash
        if actual != OLD_PARAMS_HASH:
            raise RuntimeError(f"old params hash is {actual}, not {OLD_PARAMS_HASH}")
        print(actual)
        return 0
    from sn120_v2.cli import main as cli_main
    return cli_main(argv[2:])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
