from __future__ import annotations

from typing import Optional

from .sandbox import Sandbox
from ..types import CommandResult


def run(sbx: Sandbox, cmd: list[str] | str, timeout: Optional[int] = None, env: Optional[dict] = None) -> CommandResult:
	return sbx.run(cmd, timeout=timeout, env=env)


