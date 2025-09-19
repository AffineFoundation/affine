from __future__ import annotations

import contextlib
import atexit
import signal
import weakref
from queue import LifoQueue, Empty
from threading import Lock
from typing import Iterator, Optional

from ..config import Config
from .sandbox import Sandbox

# Track active playgrounds for best-effort global cleanup
_ACTIVE_PLAYGROUNDS: "weakref.WeakSet[Playground]"  # type: ignore[name-defined]
_ACTIVE_PLAYGROUNDS = weakref.WeakSet()
_HANDLERS_INSTALLED = False
_HANDLERS_LOCK = Lock()


def _global_cleanup() -> None:
	# Best-effort: close all active playgrounds
	for ply in list(_ACTIVE_PLAYGROUNDS):
		try:
			ply._cleanup()
		except Exception:
			pass


def _install_handlers_once() -> None:
	global _HANDLERS_INSTALLED
	with _HANDLERS_LOCK:
		if _HANDLERS_INSTALLED:
			return
		atexit.register(_global_cleanup)
		# Chain signal handlers for graceful shutdown
		try:
			_prev_int = signal.getsignal(signal.SIGINT)
			_prev_term = signal.getsignal(signal.SIGTERM)
			def _wrap(prev):
				def _h(signum, frame):
					try:
						_global_cleanup()
					except Exception:
						pass
					try:
						if callable(prev):
							prev(signum, frame)  # type: ignore[misc]
					except Exception:
						pass
					if signum == signal.SIGTERM:
						raise SystemExit(143)
				return _h
			signal.signal(signal.SIGINT, _wrap(_prev_int))
			signal.signal(signal.SIGTERM, _wrap(_prev_term))
		except Exception:
			# Signals may be unsupported in some environments
			pass
		_HANDLERS_INSTALLED = True


class Playground:
	"""Pre-warms and manages a pool of reusable sandboxes.

	Example:
		import quixand as qs
		config = qs.Config(timeout=600, image="python:3.11-slim")
		ply = qs.Playground(n=5, config=config)
		with ply:
			sbx = ply.create()
			# use sbx

	When the playground context exits, all pre-warmed sandboxes are shutdown.
	"""

	def __init__(self, n: int, config: Optional[Config] = None):
		if n <= 0:
			raise ValueError("n must be > 0")
		self._n = n
		self._cfg = config or Config()
		self._pool: LifoQueue[Sandbox] = LifoQueue(maxsize=n)
		self._all: list[Sandbox] = []
		self._lock = Lock()
		self._entered = False
		self._closed = False
		_ACTIVE_PLAYGROUNDS.add(self)
		_install_handlers_once()

	def __enter__(self) -> "Playground":
		# Prewarm sandboxes
		with self._lock:
			if self._entered:
				return self
			for _ in range(self._n):
				sbx = Sandbox(
					template=self._cfg.image,
					timeout=self._cfg.timeout,
					metadata=self._cfg.metadata,
					env=self._cfg.env,
					workdir=self._cfg.workdir,
					adapter=self._cfg.adapter,
					resources=None,
				)
				self._all.append(sbx)
				self._pool.put(sbx)
			self._entered = True
		return self

	def prewarm(self) -> None:
		"""Create and pool all N sandboxes upfront for instant create() calls."""
		self.__enter__()

	def __exit__(self, exc_type, exc, tb) -> None:
		self._cleanup()

	def _cleanup(self) -> None:
		# Ensure every sandbox is torn down
		with self._lock:
			if self._closed:
				return
			for sbx in self._all:
				with contextlib.suppress(Exception):
					sbx.shutdown()
			self._all.clear()
			# Drain pool to unblock potential waiters
			while True:
				try:
					self._pool.get_nowait()
				except Empty:
					break
			self._entered = False
			self._closed = True
			with contextlib.suppress(Exception):
				_ACTIVE_PLAYGROUNDS.discard(self)

	def close(self) -> None:
		self._cleanup()

	def __del__(self):
		# Best-effort cleanup when GC runs
		with contextlib.suppress(Exception):
			self._cleanup()

	def create(self) -> Sandbox:
		"""Get a sandbox from the pool, creating one if pool is empty.

		Returned sandbox belongs to the caller until they are done with it.
		The caller may call `sandbox.shutdown()` early; the playground will
		otherwise shut it down on context exit.
		"""
		# Lazy prewarm on first create if not entered explicitly
		if not self._entered and not self._closed:
			self.__enter__()
		try:
			sbx = self._pool.get_nowait()
			return sbx
		except Empty:
			# Fall back to on-demand creation; also track for teardown
			sbx = Sandbox(
				template=self._cfg.image,
				timeout=self._cfg.timeout,
				metadata=self._cfg.metadata,
				env=self._cfg.env,
				workdir=self._cfg.workdir,
				adapter=self._cfg.adapter,
				resources=None,
			)
			with self._lock:
				self._all.append(sbx)
			return sbx

	def release(self, sbx: Sandbox) -> None:
		"""Return a sandbox back to the pool for reuse.

		If the pool is already full or the sandbox has been shut down,
		it will not be re-enqueued and the playground will ignore it.
		"""
		with self._lock:
			if sbx not in self._all:
				# Unknown to this playground; ignore
				return
			try:
				# Only put back if sandbox appears alive
				self._pool.put_nowait(sbx)
			except Exception:
				pass

	def acquire(self) -> Iterator[Sandbox]:
		"""Context-managed acquire that auto-releases back to the pool.

		Usage:
			with ply.acquire() as sbx:
				sbx.run(["echo", "hi"]) 
		"""
		class _CM:
			def __init__(self, outer: "Playground"):
				self._outer = outer
				self._sbx: Optional[Sandbox] = None

			def __enter__(self) -> Sandbox:
				self._sbx = outer.create()  # type: ignore[name-defined]
				return self._sbx

			def __exit__(self, exc_type, exc, tb) -> None:
				if self._sbx is not None:
					outer.release(self._sbx)  # type: ignore[name-defined]

		outer = self
		return _CM(self)


class Play(Playground):
	pass


