from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal, Optional


SandboxState = Literal["creating", "running", "stopping", "stopped", "error"]


@dataclass
class SandboxStatus:
	state: SandboxState
	created_at: datetime
	last_active_at: datetime
	timeout_at: Optional[datetime]
	metadata: dict


@dataclass
class CommandResult:
	text: str
	stdout: bytes
	stderr: bytes
	exit_code: int
	duration_s: float | None = None


@dataclass
class Execution:
	text: str
	images: list[str] | None
	files: list[str] | None
	stderr: str
	ok: bool


@dataclass
class FileInfo:
	path: str
	size: int
	is_dir: bool
	modified_at: datetime | None


# Adapter opaque handles
class SandboxHandle:  # pragma: no cover - marker type
	pass


class PTYHandle:  # pragma: no cover - marker type
	pass


