from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import requests


@dataclass
class ChutesResponse:
    text: str
    request_id: Optional[str]
    latency_ms: int
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None


class ChutesClient:
    """Thin HTTP client for interacting with the Chutes inference API."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        default_timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        self._base_url = base_url
        self._timeout = default_timeout
        self._session = session or requests.Session()
        self._api_key = api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    def invoke(
        self,
        uid: int,
        payload: Mapping[str, Any],
        *,
        timeout: Optional[float] = None,
    ) -> ChutesResponse:
        url = f"{self._base_url}/invoke/{uid}"
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        start = time.perf_counter()
        response = self._session.post(url, json=dict(payload), headers=headers, timeout=timeout or self._timeout)
        response.raise_for_status()
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        text = data.get("text") or data.get("response") or response.text
        return ChutesResponse(
            text=text,
            request_id=data.get("request_id"),
            latency_ms=elapsed_ms,
            tokens_in=data.get("tokens_in"),
            tokens_out=data.get("tokens_out"),
        )

    def close(self) -> None:
        self._session.close()


__all__ = ["ChutesClient", "ChutesResponse"]
