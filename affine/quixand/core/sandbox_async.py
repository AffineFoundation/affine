from __future__ import annotations

import anyio
from dataclasses import dataclass
from typing import Optional

from ..adapters.base import Adapter
from ..core.sandbox import Sandbox
from ..types import CommandResult, Execution, SandboxStatus


class AsyncSandbox:
	def __init__(self, inner: Sandbox):
		self._inner = inner
		self.id = inner.id
		self.files = inner.files

	@classmethod
	async def create(cls, *args, **kwargs) -> "AsyncSandbox":
		# Run sync constructor in a thread to avoid blocking
		def _make():
			return Sandbox(*args, **kwargs)
		inner = await anyio.to_thread.run_sync(_make)
		return cls(inner)

	async def status(self) -> SandboxStatus:
		return await anyio.to_thread.run_sync(self._inner.status)

	async def refresh_timeout(self, seconds: int) -> None:
		await anyio.to_thread.run_sync(self._inner.refresh_timeout, seconds)

	async def shutdown(self) -> None:
		await anyio.to_thread.run_sync(self._inner.shutdown)

	async def run(self, cmd: list[str] | str, timeout: Optional[int] = None, env: Optional[dict] = None) -> CommandResult:
		return await anyio.to_thread.run_sync(self._inner.run, cmd, timeout, env)

	async def run_code(self, code: str) -> Execution:
		return await anyio.to_thread.run_sync(self._inner.run_code, code)

	async def install_pkg(self, spec: str) -> CommandResult:
		return await anyio.to_thread.run_sync(self._inner.install_pkg, spec)


