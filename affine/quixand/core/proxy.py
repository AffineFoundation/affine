from __future__ import annotations

import json
import shlex
import time
from typing import Any, Optional
from urllib.parse import urlencode

from ..errors import QSProxyError


class ProxyFacade:
	def __init__(self, sbx: "Sandbox"):
		# Deferred import to avoid circular
		self._sb = sbx
		self._method_cache = {}

	def _make_request(self, *, method: str, url: str, payload: dict | None, timeout: Optional[int], options: str = "") -> tuple[int, str]:
		"""Make HTTP request inside container using curl."""
		if method.upper() == "GET":
			if payload:
				query_string = urlencode(payload)
				url = f"{url}{'&' if '?' in url else '?'}{query_string}"
			status_marker = "QS_PROXY_STATUS__:"
			marker = shlex.quote(status_marker + "%{http_code}")
			cmd = (
				"curl -sS -X GET "
				+ (f" {options} " if options else "")
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
				+ (f" {options} " if options else "")
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

	def _detect_method(self, url: str, port: int, payload: dict | None) -> str:
		"""Detect the correct HTTP method for an endpoint using OPTIONS."""
		if url in self._method_cache:
			return self._method_cache[url]
		
		_, body = self._make_request(
			method="OPTIONS",
			url=url,
			payload=None,
			timeout=5,
			options="-i"
		)
		
		allow_header = None
		for line in body.lower().splitlines():
			if "allow:" in line:
				allow_header = line.upper()
				break
		assert allow_header, f"not support api: /{url.split('/')[-1]}"
		method = None
		if allow_header:
			if "POST" in allow_header:
				method = "POST"
			elif "GET" in allow_header:
				method = "GET"
			elif "PUT" in allow_header:
				method = "PUT"
		assert method, f"not support {allow_header}"
		
		self._method_cache[url] = method
		return method

	def _health(self, *, port: int = 8000, timeout: int = 30) -> None:
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

	def __getattr__(self, name: str):
		"""Dynamic method creation for API endpoints."""
		def dynamic_api_call(**kwargs):
			port = kwargs.pop("_port", 8000)
			timeout = kwargs.pop("_timeout", 60)
			ensure_ready = kwargs.pop("_ensure_ready", True)
			
			if ensure_ready:
				self._health(port=port, timeout=min(timeout, 30))
			url = f"http://localhost:{port}/{name}"
			method = self._detect_method(url, port, kwargs)
			status, body = self._make_request(method=method, url=url, payload=kwargs, timeout=timeout)
			
			if status < 200 or status >= 300:
				raise QSProxyError(f"Proxy call to {name} failed with HTTP {status}: {body[:200]}")

			try:
				return json.loads(body) if body else None
			except Exception:
				return body
		
		return dynamic_api_call