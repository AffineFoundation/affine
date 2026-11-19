"""Loopback-only, cancellable HTTP transport for the eight local vLLM servers.

No remote inference URL is accepted, including through proxies or redirects.
Disconnecting an active reply lets vLLM abort generation at the job deadline.
"""

from __future__ import annotations

import ipaddress
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..errors import ConfigError, InfraError
from ..gpu.evaluator import HttpError


class _LocalOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise InfraError("Generator inference refused an HTTP redirect")


@dataclass(frozen=True, slots=True)
class Endpoint:
    port: int

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def loopback_url(value: str, *, path: str = "") -> str:
    """Validate a literal loopback base; never resolve or follow other hosts."""
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
        address = ipaddress.ip_address(parsed.hostname) if parsed.hostname else None
    except ValueError as exc:
        raise ConfigError("Generator inference URL is invalid") from exc
    if (
        parsed.scheme != "http"
        or address is None
        or not address.is_loopback
        or address.is_link_local
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or port is None
        or not 1 <= port <= 65535
    ):
        raise ConfigError("Generator inference must use a literal http://127.0.0.1:<port> or http://[::1]:<port> endpoint")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def default_urls(count: int = 8) -> tuple[str, ...]:
    if count != 8:
        raise ConfigError("Generator production scoring requires exactly eight local vLLM servers")
    return tuple(f"http://127.0.0.1:{8100 + gpu}" for gpu in range(count))


class LocalHttp:
    """Small thread-safe HTTP client with explicit cancellation of active reads."""

    def __init__(self, *, opener: Any = None) -> None:
        # ProxyHandler({}) is intentional: even a configured environment proxy
        # must not receive pod-local prompts or the derived rollout seeds.
        self._opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), _LocalOnly())
        self._condition = threading.Condition()
        self._active: set[Any] = set()
        self._cancelled = False

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            replies = tuple(self._active)
        for reply in replies:
            try:
                reply.close()
            except OSError:
                pass

    def reset(self) -> None:
        with self._condition:
            if self._active:
                raise InfraError("cannot reset Generator HTTP while requests are active")
            self._cancelled = False

    def __call__(self, url: str, body: dict[str, Any] | None, timeout_s: float) -> dict[str, Any]:
        split = urllib.parse.urlsplit(url)
        loopback_url(f"{split.scheme}://{split.netloc}")
        if split.path not in ("/health", "/v1/models", "/v1/chat/completions") or split.query or split.fragment:
            raise ConfigError("Generator HTTP path is not a local vLLM health, model or chat endpoint")
        data = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        reply = None
        with self._condition:
            if self._cancelled:
                raise OSError("Generator HTTP request cancelled")
        try:
            reply = self._opener.open(request, timeout=max(0.001, timeout_s))
            with self._condition:
                if self._cancelled:
                    reply.close()
                    raise OSError("Generator HTTP request cancelled")
                self._active.add(reply)
            raw = reply.read()
        except urllib.error.HTTPError as exc:
            raise HttpError(exc.code) from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, OSError):
                raise reason
            raise OSError(type(reason).__name__) from exc
        finally:
            if reply is not None:
                with self._condition:
                    cancelled = self._cancelled
                    self._active.discard(reply)
                try:
                    reply.close()
                except (OSError, AttributeError):
                    # urllib/http.client may have detached its socket already
                    # during cancellation. It is never a completed rollout.
                    if not cancelled:
                        raise
            if self._cancelled and "raw" not in locals():
                raise OSError("Generator HTTP request cancelled")
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise InfraError(f"Generator local vLLM returned invalid JSON: {type(exc).__name__}") from exc
        if not isinstance(parsed, dict):
            raise InfraError("Generator local vLLM returned a non-object response")
        return parsed

    def check(self, base_url: str, timeout_s: float = 30.0) -> None:
        url = loopback_url(base_url)
        self(url + "/health", None, timeout_s)
        models = self(url + "/v1/models", None, timeout_s).get("data")
        if not isinstance(models, list) or not any(isinstance(card, dict) and card.get("id") == "null" for card in models):
            raise InfraError(f"Generator vLLM at {url} does not expose served-model-name null")


class BudgetDeadline(Exception):
    """No further inference may be dispatched; partial work is checkpointed."""
