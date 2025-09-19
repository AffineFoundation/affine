from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List


def ensure_parent(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def write_bytes(path: Path, data: bytes) -> None:
	ensure_parent(path)
	path.write_bytes(data)


def write_text(path: Path, data: str) -> None:
	ensure_parent(path)
	path.write_text(data, encoding="utf-8")


def read_bytes(path: Path) -> bytes:
	return path.read_bytes()


def read_text(path: Path) -> str:
	return path.read_text(encoding="utf-8")


def atomic_write_text(path: Path, data: str) -> None:
	"""Write text atomically by writing to a temp file and renaming.

	Prevents partial/corrupted state files that could strand watchdogs/containers.
	"""
	ensure_parent(path)
	tmp = path.with_suffix(path.suffix + ".tmp")
	tmp.write_text(data, encoding="utf-8")
	os.replace(tmp, path)


def list_dir(path: Path) -> List[str]:
	if not path.exists():
		return []
	return sorted([p.name for p in path.iterdir()])


