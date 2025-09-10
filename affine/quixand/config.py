from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_RUNTIME = "docker"  # or "podman"
STATE_DIR = Path(os.getenv("HOME", "~")).expanduser() / ".quixand"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
TEMPLATES_DIR = STATE_DIR / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


class Resources(BaseModel):
	cpu_limit: Optional[float] = Field(default=None, description="CPU limit in cores, e.g., 1.5")
	mem_limit: Optional[str] = Field(default=None, description="Memory limit string, e.g., 4g")
	pids_limit: Optional[int] = Field(default=None, description="Max PIDs")
	network: Optional[str] = Field(default="bridge", description="none|bridge|host")


class Config(BaseModel):
	adapter: Optional[str] = Field(default=os.getenv("QS_ADAPTER", "local-docker"))
	timeout: int = Field(default=int(os.getenv("QS_TIMEOUT_DEFAULT", DEFAULT_TIMEOUT_SECONDS)))
	image: str = Field(default=os.getenv("QS_IMAGE", DEFAULT_IMAGE))
	runtime: str = Field(default=os.getenv("QS_RUNTIME", DEFAULT_RUNTIME))
	root: Path = Field(default=Path(os.getenv("QS_ROOT", str(STATE_DIR))))
	metadata: Dict[str, Any] = Field(default_factory=lambda: _parse_json_env(os.getenv("QS_METADATA")))
	env: Dict[str, str] = Field(default_factory=dict)
	workdir: Optional[str] = Field(default="/workspace")

	class Config:
		arbitrary_types_allowed = True

	@staticmethod
	def state_file() -> Path:
		return STATE_FILE

	@staticmethod
	def templates_dir() -> Path:
		return TEMPLATES_DIR


def _parse_json_env(value: Optional[str]) -> Dict[str, Any]:
	if not value:
		return {}
	try:
		return json.loads(value)
	except Exception:
		return {}


