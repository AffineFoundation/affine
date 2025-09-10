from __future__ import annotations

from .sandbox import Sandbox
from ..types import Execution


def run_code(sbx: Sandbox, code: str) -> Execution:
	return sbx.run_code(code)


