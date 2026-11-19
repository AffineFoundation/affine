"""Prepare the exact frozen Generator king on the secret-free scorer pod.

Run as ``/root/venv/bin/python -m sn120_v2.generator.ensure_head VERSION``.
Only public head/checkpoint records are fetched. Presigned download URLs are
secrets and are never printed or included in errors. No scorer starts until all
base-file hashes, the atomic base swap and all eight servers have been verified.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from ..canonical import file_sha256, is_sha256, manifest_digest
from ..errors import ConfigError, InfraError, V2Error, VerifyError

PUBLIC_ROOT = "https://affine.io"
LINKS_URL = PUBLIC_ROOT + "/public/read_links.json"
HEAD_URL = PUBLIC_ROOT + "/public/head.json"
BASE = Path("/root/dl/base")
SERVE = "/root/gen/src/sn120_v2/generator/serve.sh"
VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
DOWNLOAD_TIMEOUT_S = 120
PREPARE_TIMEOUT_S = 900


def _read_json_url(url: str, *, maximum: int = 4 * 1024 * 1024) -> Any:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "sn120-generator-head/1"})
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed public records
            raw = response.read(maximum + 1)
        if len(raw) > maximum:
            raise ValueError("document too large")
        return json.loads(raw)
    except (OSError, ValueError) as exc:
        raise InfraError(f"cannot read a public head record: {type(exc).__name__}") from None


def _version(document: Any) -> str | None:
    if not isinstance(document, Mapping):
        return None
    for key in ("version_id", "head_version_id", "head_version", "version", "id"):
        value = document.get(key)
        if isinstance(value, str) and VERSION.fullmatch(value):
            return value
    for key in ("head", "checkpoint"):
        if isinstance(document.get(key), Mapping):
            value = _version(document[key])
            if value:
                return value
    return None


def _files_from_manifest(document: Mapping[str, Any]) -> dict[str, str]:
    files = document.get("base_files")
    if not isinstance(files, Mapping) and isinstance(document.get("manifest"), Mapping):
        files = document["manifest"].get("base_files")
    if not isinstance(files, Mapping) or not files:
        return {}
    return dict(files) if all(isinstance(name, str) and is_sha256(sha) for name, sha in files.items()) else {}


def _entry_files(entry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = entry.get("files")
    if isinstance(raw, Mapping):
        rows = [dict(value, name=name) if isinstance(value, Mapping) else {"name": name, "sha256": value} for name, value in raw.items()]
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []
    result = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise VerifyError("a base download entry has no filename")
        name, sha = row["name"], row.get("sha256", row.get("sha"))
        path = PurePosixPath(name)
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts) or "\\" in name:
            raise VerifyError("a base download entry has an unsafe filename")
        if not is_sha256(sha) or not isinstance(row.get("url"), str):
            raise VerifyError("a base download entry lacks its SHA-256 or download capability")
        size = row.get("size", row.get("bytes"))
        if type(size) is not int or size < 0:
            raise VerifyError("a base download entry lacks its byte size")
        result[name] = {"sha256": sha, "size": size, "url": row["url"]}
    if not result:
        raise VerifyError("the requested head has no downloadable base files")
    return result


def select_base(version: str, head: Mapping[str, Any], links: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Select the current published base or a matching checkpoint, never LoRAs."""
    if not isinstance(version, str) or not VERSION.fullmatch(version) or version in (".", ".."):
        raise ConfigError("the requested head version is not a safe identifier")
    current = _version(links) == version or _version(head) == version
    if current:
        base = links.get("base")
        if not isinstance(base, Mapping):
            raise VerifyError("the current published head has no complete folded base in read_links")
        entry = _entry_files(base)
        hashes = {name: row["sha256"] for name, row in entry.items()}
        expected = _files_from_manifest(head)
        if _version(head) == version and expected and hashes != expected:
            raise VerifyError("the published base links do not match head.json base_files")
        if expected and hashes != expected and base.get("id") != manifest_digest(expected):
            raise VerifyError("the current base links do not match the published head")
        return entry, hashes
    checkpoints = links.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        raise VerifyError("read_links checkpoints are invalid")
    for point in checkpoints:
        if isinstance(point, Mapping) and point.get("version_id") == version:
            if not isinstance(point.get("base"), Mapping):
                raise VerifyError("the matching checkpoint has no folded base")
            entry = _entry_files(point["base"])
            return entry, {name: row["sha256"] for name, row in entry.items()}
    raise VerifyError("requested head is neither the current published head nor a read_links checkpoint")


def download_base(entry: Mapping[str, Mapping[str, Any]], target: Path, *, opener: Callable[..., Any] = urllib.request.urlopen) -> None:
    """Hash and size every downloaded file before the target may be renamed."""
    target.mkdir(parents=True, exist_ok=False)
    for name, row in sorted(entry.items()):
        path = target.joinpath(*PurePosixPath(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.part")
        digest = hashlib.sha256()
        count = 0
        try:
            request = urllib.request.Request(row["url"], headers={"User-Agent": "sn120-generator-head/1"})
            with opener(request, timeout=DOWNLOAD_TIMEOUT_S) as response, temporary.open("xb") as output:
                while chunk := response.read(4 * 1024 * 1024):
                    count += len(chunk)
                    if count > row["size"]:
                        raise VerifyError(f"base file {name} exceeds its published size")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if count != row["size"] or digest.hexdigest() != row["sha256"]:
                raise VerifyError(f"base file {name} failed its published size or SHA-256 check")
            os.replace(temporary, path)
        except (OSError, ValueError) as exc:
            if isinstance(exc, V2Error):
                raise
            raise InfraError(f"base file {name} download failed: {type(exc).__name__}") from None
        finally:
            temporary.unlink(missing_ok=True)


def installed_version(base: Path | None = None) -> str:
    base = BASE if base is None else base

    try:
        value = (base / ".head_version").read_text().strip()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise VerifyError("cannot read the installed Generator head version") from exc
    if not VERSION.fullmatch(value):
        raise VerifyError("the installed Generator head marker is invalid")
    return value


def require_installed(version: str, base: Path | None = None) -> None:
    """The scorer's final local guard: wrong or missing model means no scoring."""
    if installed_version(base) != version:
        raise VerifyError("installed base .head_version does not match the scoring job")


def _replace_directory(source: Path, base: Path) -> None:
    """Swap complete directories, preserving the previous base for failure recovery."""
    base.parent.mkdir(parents=True, exist_ok=True)
    old = base.parent / f".{base.name}.old.{os.getpid()}"
    if old.exists():
        raise VerifyError("an old base swap needs operator inspection")
    try:
        if base.exists() or base.is_symlink():
            os.replace(base, old)
        try:
            os.replace(source, base)
        except BaseException:
            if old.exists() and not base.exists():
                os.replace(old, base)
            raise
    finally:
        if old.exists():
            shutil.rmtree(old)


def _serve(command: Callable[..., Any], action: str) -> None:
    result = command(["bash", SERVE, action], cwd="/", capture_output=True, text=True, timeout=PREPARE_TIMEOUT_S, check=False)
    if result.returncode != 0:
        raise InfraError(f"the eight-server {action} failed with exit code {result.returncode}; inspect pod server logs")


def restart_servers(*, command: Callable[..., Any] = subprocess.run) -> None:
    _serve(command, "stop")
    # serve.sh starts all eight and waits for /health and /v1/models on each.
    _serve(command, "start")


def ensure_head(
    version: str,
    *,
    base: Path = BASE,
    read_record: Callable[[str], Any] = _read_json_url,
    fetch_files: Callable[..., None] = download_base,
    restart: Callable[..., None] = restart_servers,
) -> dict[str, Any]:
    if installed_version(base) == version:
        return {"status": "already_prepared", "version": version}
    # Refuse an unknown version before downloading or stopping healthy servers.
    head, links = read_record(HEAD_URL), read_record(LINKS_URL)
    if not isinstance(head, Mapping) or not isinstance(links, Mapping):
        raise VerifyError("public head records must be JSON objects")
    entry, hashes = select_base(version, head, links)
    parent = base.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{base.name}.{version}.download.", dir=parent))
    prepared = parent / f"{base.name}.{version}"
    try:
        if prepared.exists():
            raise VerifyError("a versioned base destination already exists; inspect it before preparing the head")
        fetch_files(entry, staging)
        actual = {path.relative_to(staging).as_posix(): file_sha256(path) for path in staging.rglob("*") if path.is_file()}
        if actual != hashes:
            raise VerifyError("the complete staged base does not match the selected checkpoint files")
        (staging / ".head_version").write_text(version + "\n")
        _replace_directory(staging, prepared)
        _replace_directory(prepared, base)
        require_installed(version, base)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    try:
        restart()
    except BaseException:
        # No false readiness: keep the verified marker, but never hide a failed server.
        raise InfraError("the requested base was installed but the eight servers did not become ready")
    require_installed(version, base)
    return {"status": "prepared", "version": version, "files": len(hashes)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.generator.ensure_head")
    parser.add_argument("version", help="exact frozen job head_version")
    parser.add_argument("--base", type=Path, default=BASE, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        report = ensure_head(args.version, base=args.base)
        print(f"Generator base ready for requested head {report['version']}")
        return 0
    except V2Error as exc:
        # Never print an exception from urllib: it can include a presigned URL.
        print(f"Generator head preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
