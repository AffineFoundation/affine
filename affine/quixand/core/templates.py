from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from ..config import Config


INDEX = Config.templates_dir() / "index.json"


class Templates:
	@staticmethod
	def build(path: str, name: Optional[str] = None) -> str:
		p = Path(path)
		if not p.exists():
			raise FileNotFoundError(path)
		dockerfile = p / "e2b.Dockerfile"
		if not dockerfile.exists():
			dockerfile = p / "Dockerfile"
		if not dockerfile.exists():
			raise FileNotFoundError("No e2b.Dockerfile or Dockerfile found")
		digest = _hash_dir(p)
		img_name = f"qs/{name or p.name}:{digest[:12]}"
		# Use runtime from Config
		cfg = Config()
		runtime = cfg.runtime or "docker"
		cmd = [runtime, "build", "-f", str(dockerfile), "-t", img_name, str(p)]
		subprocess.check_call(cmd)
		idx = _load_index()
		idx[name or p.name] = {"image": img_name, "digest": digest}
		INDEX.write_text(json.dumps(idx, indent=2))
		return img_name

	@staticmethod
	def ls() -> dict:
		return _load_index()

	@staticmethod
	def rm(name: str) -> None:
		idx = _load_index()
		idx.pop(name, None)
		INDEX.write_text(json.dumps(idx, indent=2))


def _hash_dir(path: Path) -> str:
	h = hashlib.sha256()
	for root, _, files in os.walk(path):
		for f in sorted(files):
			pp = Path(root) / f
			if pp.name.startswith(".git"):
				continue
			h.update(pp.read_bytes())
	return h.hexdigest()


def _load_index() -> dict:
	if not INDEX.exists():
		return {}
	try:
		return json.loads(INDEX.read_text())
	except Exception:
		return {}


