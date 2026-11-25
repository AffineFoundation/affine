#!/usr/bin/env python3
"""Stage c5252865 plus the reviewed physical validator compatibility overlay.

This reads an operator-supplied old package and kit files. It writes only a new
staging directory and an optional deterministic tarball. It never modifies the
input package, systemd, a live checkout, a journal, a store or chain state.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

OLD_COMMIT = "c525286580c84d795b3ca7cbdf2302e96fce31ba"
OLD_GPU_DIGEST = "5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47"
OVERLAY = ("canonical", "types", "params", "state", "controller", "dedup_compat", "journal")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file is missing: {path}")
    return path.read_bytes()


def pinned_json(path: Path) -> dict:
    value = json.loads(regular_file(path))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value


def checked_manifest(kit: Path) -> dict[str, str]:
    manifest = pinned_json(kit / "manifest.json")
    files = manifest.get("files")
    if manifest.get("old_commit") != OLD_COMMIT or manifest.get("old_gpu_digest") != OLD_GPU_DIGEST:
        raise ValueError("compatibility manifest does not pin c5252865 and its old GPU source")
    if not isinstance(files, dict):
        raise ValueError("compatibility manifest has no files object")
    for name, expected in files.items():
        path = kit / name
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError(f"unsafe compatibility path: {name}")
        if not isinstance(expected, str) or sha256(path) != expected:
            raise ValueError(f"compatibility hash mismatch: {name}")
    needed = {f"rollback-compat/{name}.py" for name in (*OVERLAY, "__init__")}
    if set(files) != needed:
        raise ValueError(f"compatibility manifest must cover exactly {sorted(needed)}")
    return files


def old_files(package: Path, manifest: dict[str, str]) -> dict[str, Path]:
    if package.is_symlink() or not package.is_dir():
        raise ValueError(f"old package is not a regular directory: {package}")
    for path in package.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"old package contains a symlink: {path}")
    found = {
        path.relative_to(package).as_posix(): path
        for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    expected = set(manifest)
    if set(found) != expected:
        raise ValueError(json.dumps({
            "error": "old package is not the complete pinned c5252865 source tree",
            "missing": sorted(expected - set(found)), "unexpected": sorted(set(found) - expected),
        }, sort_keys=True))
    changed = [name for name, path in found.items() if sha256(path) != manifest[name]]
    if changed:
        raise ValueError(f"old package differs from pinned c5252865: {changed}")
    return found


def gpu_digest(files: dict[str, Path]) -> str:
    hashes = {
        PurePosixPath(name).name: sha256(path)
        for name, path in files.items()
        if PurePosixPath(name).parent == PurePosixPath("gpu") and PurePosixPath(name).suffix == ".py"
    }
    raw = json.dumps(hashes, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def add_tarball(staging: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise ValueError(f"tarball must not already exist: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError(f"tarball parent must be an existing regular directory: {destination.parent}")
    paths = sorted(staging.rglob("*"), key=lambda path: path.relative_to(staging).as_posix())

    def normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.mode = 0o755 if info.isdir() else 0o644
        return info

    with destination.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in paths:
                archive.add(path, arcname=path.relative_to(staging).as_posix(), recursive=False, filter=normalized)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_package", type=Path, help="complete c5252865 v2/sn120_v2 source directory")
    parser.add_argument("staging", type=Path, help="new output directory; rsync its sn120_v2/ subdirectory")
    parser.add_argument("--kit", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--tar", dest="tarball", type=Path, help="optional new deterministic tar.gz path")
    args = parser.parse_args(argv)
    kit = args.kit.resolve(strict=True)
    if kit.is_symlink() or not kit.is_dir():
        raise ValueError(f"kit must be a regular directory: {kit}")
    compatibility = checked_manifest(kit)
    old_pin = pinned_json(kit / "validator-old-c5252865-manifest.json")
    if old_pin.get("old_commit") != OLD_COMMIT or old_pin.get("old_gpu_digest") != OLD_GPU_DIGEST:
        raise ValueError("old-file manifest does not pin c5252865 and its GPU source")
    old_hashes = old_pin.get("files")
    if not isinstance(old_hashes, dict) or not old_hashes:
        raise ValueError("old-file manifest has no complete files object")
    source = args.old_package.resolve(strict=True)
    files = old_files(source, old_hashes)
    actual_gpu = gpu_digest(files)
    if actual_gpu != OLD_GPU_DIGEST:
        raise ValueError(f"old GPU source digest is {actual_gpu}, not {OLD_GPU_DIGEST}")
    # Keep the pinned old package __init__: the runtime loader also leaves it
    # unchanged. Only the seven named modules are physically overlaid.
    staging = args.staging.absolute()
    if os.path.lexists(staging):
        raise ValueError(f"staging must be a new path: {staging}")
    if not staging.parent.is_dir() or staging.parent.is_symlink():
        raise ValueError(f"staging parent must be an existing regular directory: {staging.parent}")
    target = staging / "sn120_v2"
    target.mkdir(parents=True)
    for name, path in sorted(files.items()):
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    for name in OVERLAY:
        shutil.copyfile(kit / f"rollback-compat/{name}.py", target / f"{name}.py")
    final_hashes = {
        path.relative_to(target).as_posix(): sha256(path)
        for path in sorted(target.rglob("*")) if path.is_file()
    }
    expected_hashes = dict(old_hashes) | {
        f"{name}.py": compatibility[f"rollback-compat/{name}.py"] for name in OVERLAY
    }
    if final_hashes != expected_hashes:
        raise AssertionError("staged files differ from the complete old+overlay pin")
    report = {
        "schema": 1, "old_commit": OLD_COMMIT, "old_gpu_digest": OLD_GPU_DIGEST,
        "physical_overlay": sorted(f"{name}.py" for name in OVERLAY),
        "package": "sn120_v2", "source_files": len(files), "staged_files": len(final_hashes),
        "files": final_hashes,
    }
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    (staging / "restore-validator-manifest.json").write_bytes(report_bytes)
    if args.tarball:
        add_tarball(staging, args.tarball.absolute())
    print(f"RESTORE VALIDATOR PASS: complete c5252865 plus seven physical compatibility files; {len(final_hashes)} source files staged")
    print(json.dumps({
        "old_commit": OLD_COMMIT, "old_gpu_digest": OLD_GPU_DIGEST, "staged_files": len(final_hashes),
        "rsync_source": str(target) + "/", "manifest": str(staging / "restore-validator-manifest.json"),
        "manifest_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "tarball": str(args.tarball.absolute()) if args.tarball else None,
        "tarball_sha256": sha256(args.tarball.absolute()) if args.tarball else None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"RESTORE VALIDATOR REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
