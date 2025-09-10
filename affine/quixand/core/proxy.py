from __future__ import annotations

import json
import shlex
import time
from typing import Any, Optional

from ..errors import QSProxyError


class ProxyFacade:
	def __init__(self, sbx: "Sandbox"):
		# Deferred import to avoid circular
		self._sb = sbx

	def _make_request(self, *, method: str, url: str, payload: dict | None, timeout: Optional[int]) -> tuple[int, str]:
		"""Make HTTP request inside container using curl."""
		
		# For GET requests without payload, use simpler curl command
		if method.upper() == "GET" and not payload:
			status_marker = "QS_PROXY_STATUS__:"
			marker = shlex.quote(status_marker + "%{http_code}")
			cmd = (
				"curl -sS -X GET "
				+ (f"--max-time {int(timeout)} " if timeout else "")
				+ shlex.quote(url)
				+ " -w '\\n" + status_marker + "%{http_code}'"
			)
		else:
			# For other methods or when payload is needed
			json_str = json.dumps(payload or {})
			json_quoted = shlex.quote(json_str)
			status_marker = "QS_PROXY_STATUS__:"
			marker = shlex.quote(status_marker + "%{http_code}")
			
			cmd = (
				"echo "
				+ json_quoted
				+ " | "
				+ "curl -sS -X "
				+ shlex.quote(method.upper())
				+ " -H 'Content-Type: application/json' "
				+ (f"--max-time {int(timeout)} " if timeout else "")
				+ "-d @- "
				+ shlex.quote(url)
				+ " -w '\\n" + status_marker + "%{http_code}'"
			)
		
		res = self._sb.run(cmd, timeout=timeout)
		text = res.text
		idx = text.rfind(status_marker)
		if idx == -1:
			raise QSProxyError("Proxy call failed: could not parse HTTP status from response")
		body = text[:idx].rstrip("\n")
		status_str = text[idx + len(status_marker):].strip()
		try:
			status = int(status_str)
		except ValueError:
			raise QSProxyError(f"Proxy call failed: invalid HTTP status '{status_str}'")
		return status, body

	def health(self, *, port: int = 8000, timeout: int = 30) -> None:
		"""Check service health by polling /health endpoint."""
		deadline = time.time() + timeout
		url = f"http://localhost:{port}/health"
		
		while time.time() < deadline:
			try:
				status, body = self._make_request(method="GET", url=url, payload=None, timeout=5)
				if status == 200:
					return
			except Exception:
				pass
			time.sleep(1)
		raise QSProxyError(f"Service not ready on {url} within {timeout}s")

	def run(
		self,
		*,
		port: int = 8000,
		path: str = "/run",
		method: str = "POST",
		payload: Optional[dict] = None,
		timeout: int = 60,
		ensure_ready: bool = True,
		fallback_paths: tuple[str, ...] = ("/env/run",),
		**kwargs: Any,
	) -> Any:
		"""Call an HTTP endpoint inside the container and return parsed JSON.

		Defaults to POSTing to http://localhost:{port}/run with kwargs as JSON.
		Raises QSProxyError if the endpoint is not defined or returns non-2xx.
		"""
		if ensure_ready:
			self.health(port=port, timeout=min(timeout, 30))

		data = payload if payload is not None else dict(kwargs)
		url = f"http://localhost:{port}{path}"

		status, body = self._make_request(method=method, url=url, payload=data, timeout=timeout)
		if status == 404 and fallback_paths:
			for fp in fallback_paths:
				alt_url = f"http://localhost:{port}{fp}"
				status, body = self._make_request(method=method, url=alt_url, payload=data, timeout=timeout)
				if status != 404:
					break

		if status < 200 or status >= 300:
			raise QSProxyError(f"Proxy call failed with HTTP {status}: {body[:200]}")

		# Try JSON decode, else return raw text
		try:
			return json.loads(body) if body else None
		except Exception:
			return body
