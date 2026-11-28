"""Check this checkout's Params against the live public mining contract."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

from .params import Params

MINING_JSON_URL = "https://affine.io/mining.json"
USER_AGENT = "sn120-v2-replicate/1"
TIMEOUT_S = 30


class ParamsFetchError(RuntimeError):
    """The live contract could not be fetched or decoded."""


def fetch_mining_json(url: str = MINING_JSON_URL, timeout_s: float = TIMEOUT_S) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            document = json.load(response)
        if not isinstance(document, dict) or not isinstance(document.get("params"), dict):
            raise ValueError("mining.json has no params object")
        if not isinstance(document.get("params_hash"), str) or not document["params_hash"]:
            raise ValueError("mining.json has no params_hash")
        return document
    except (OSError, ValueError) as exc:
        raise ParamsFetchError(f"could not fetch {url}: {exc}") from exc


def recomputed_params_hash(document: dict[str, Any]) -> str:
    return Params.from_public_dict(document["params"]).params_hash


def main() -> int:
    try:
        document = fetch_mining_json()
    except ParamsFetchError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    published = document["params_hash"]
    recomputed = recomputed_params_hash(document)
    print(f"published_params_hash={published}")
    print(f"recomputed_params_hash={recomputed}")
    if recomputed != published:
        print("params_hash mismatch", file=sys.stderr)
        return 1
    print("params_hash matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
