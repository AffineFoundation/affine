from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ..adapters.base import Adapter, Resources as AdapterResources, SandboxConfig
from ..adapters.local_docker import LocalDockerAdapter
from ..config import Config, Resources
from ..types import CommandResult, Execution, FileInfo, SandboxStatus
from .proxy import ProxyFacade


def _resolve_adapter(adapter: str | Adapter | None, cfg: Config) -> Adapter:
	# Accept explicit adapter instance
	if adapter is not None and not isinstance(adapter, str):
		return adapter  # type: ignore[return-value]
	# Name-based selection (default local-docker)
	name = adapter or cfg.adapter or "local-docker"
	if name == "local-docker":
		return LocalDockerAdapter(cfg)
	# Future: add registry for other adapters
	return LocalDockerAdapter(cfg)


@dataclass
class FilesFacade:
	_sb: "Sandbox"

	def write(self, path: str, data: bytes | str, mode: str = "text") -> None:
		text = mode != "binary"
		b = data.encode("utf-8") if isinstance(data, str) else data
		self._sb._adapter.fs_write(self._sb._handle, path, b, text)

	def read(self, path: str, mode: str = "text") -> str | bytes:
		text = mode != "binary"
		return self._sb._adapter.fs_read(self._sb._handle, path, text)

	def ls(self, path: str = ".") -> list[FileInfo]:
		return self._sb._adapter.fs_ls(self._sb._handle, path)

	def mkdir(self, path: str, parents: bool = False) -> None:
		self._sb._adapter.fs_mkdir(self._sb._handle, path, parents)

	def rm(self, path: str, recursive: bool = False) -> None:
		self._sb._adapter.fs_rm(self._sb._handle, path, recursive)

	def mv(self, src: str, dst: str) -> None:
		self._sb._adapter.fs_mv(self._sb._handle, src, dst)

	def glob(self, pattern: str) -> list[str]:
		# Use shell expansion via adapter run
		res = self._sb.run(["sh", "-lc", f"printf '%s\n' {pattern}"])
		return [line for line in res.text.splitlines() if line]

	def put(self, local: str, remote: str) -> None:
		self._sb._adapter.fs_put(self._sb._handle, local, remote)

	def get(self, remote: str, local: str) -> None:
		self._sb._adapter.fs_get(self._sb._handle, remote, local)


class Sandbox:
	def __init__(self, template: Optional[str] = None, timeout: int = 300, metadata: Optional[dict] = None, env: Optional[dict] = None, workdir: Optional[str] = None, adapter: str | Adapter | None = None, resources: Optional[Resources] = None, volumes: Optional[list] = None, command: Optional[list[str]] = None, entrypoint: Optional[list[str]] = None):
		self._cfg = Config()
		image = template or self._cfg.image
		self._adapter = _resolve_adapter(adapter or self._cfg.adapter, self._cfg)
		adapt_res = AdapterResources(
			cpu_limit=resources.cpu_limit if resources else None,
			mem_limit=resources.mem_limit if resources else None,
			pids_limit=resources.pids_limit if resources else None,
			network=resources.network if resources else "bridge",
		)
		sbx_cfg = SandboxConfig(image=image, timeout=timeout, env=env, workdir=workdir or self._cfg.workdir, metadata=metadata, resources=adapt_res, volumes=volumes, command=command, entrypoint=entrypoint)
		self._handle = self._adapter.create(sbx_cfg)
		self.files = FilesFacade(self)
		self.proxy = ProxyFacade(self)
		self.id = self._handle.id
		self._closed = False

	def status(self) -> SandboxStatus:
		return self._adapter.status(self._handle)

	def refresh_timeout(self, seconds: int) -> None:
		self._adapter.refresh_timeout(self._handle, seconds)

	def shutdown(self) -> None:
		if self._closed:
			return
		try:
			self._adapter.shutdown(self._handle)
		except Exception:
			pass
		self._closed = True

	def run(self, cmd: list[str] | str, timeout: Optional[int] = None, env: Optional[dict] = None) -> CommandResult:
		return self._adapter.run(self._handle, cmd, env, timeout)

	def pty(self, cmd: str = "/bin/bash"):
		# placeholder context manager
		class _CM:
			def __init__(self, outer: "Sandbox", c: str):
				self.outer = outer
				self.c = c
				self.pty = None
			def __enter__(self):
				self.pty = outer._adapter.pty_start(outer._handle, self.c, None)  # type: ignore[name-defined]
				return self
			def __exit__(self, exc_type, exc, tb):
				outer._adapter.pty_close(self.pty)  # type: ignore[name-defined]
			def send(self, data: str | bytes):
				b = data.encode("utf-8") if isinstance(data, str) else data
				outer._adapter.pty_send(self.pty, b)  # type: ignore[name-defined]
			def stream(self) -> Iterable[bytes]:
				yield from outer._adapter.pty_stream(self.pty)  # type: ignore[name-defined]
		outer = self
		return _CM(self, cmd)

	def run_code(self, code: str) -> Execution:
		return self._adapter.run_code(self._handle, code)

	def install_pkg(self, spec: str) -> CommandResult:
		return self._adapter.install_pkg(self._handle, spec)

	def expose(self, port: int, host_port: Optional[int] = None, protocol: str = "tcp"):
		return self._adapter.expose(self._handle, port, host_port, protocol)

	# Context manager support to ensure cleanup
	def __enter__(self) -> "Sandbox":
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.shutdown()

	def __del__(self):
		# Best-effort process exit cleanup if user forgets to call shutdown
		try:
			self.shutdown()
		except Exception:
			pass


