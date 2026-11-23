"""The store settings under their plain names, for a credentials file that was not written for v2.

``R2Store.from_env`` reads only ``SN120_R2_*``, and that does not change: a
command picks up a plain name only when it calls ``store_env`` itself. Today
that is the public pull (``v2/site/pull.py``), which only reads. The upload
link job takes the two token names of ``PLAIN_NAMES`` from its own credentials
file (``links.store_token``). No other command looks at a plain name, so a
process with a write token cannot get its store from one by accident.

This module imports nothing outside the standard library and ``errors``.
"""

from __future__ import annotations

import re
from typing import Mapping
from urllib.parse import urlsplit

from .errors import ConfigError

# The name that R2Store.from_env reads, and the plain name of the same setting.
PLAIN_NAMES = {
    "SN120_R2_ENDPOINT": "R2_ENDPOINT",
    "SN120_R2_BUCKET": "R2_BUCKET",
    "SN120_R2_ACCESS_KEY_ID": "R2_ACCESS_KEY_ID",
    "SN120_R2_SECRET_ACCESS_KEY": "R2_SECRET_ACCESS_KEY",
}
ACCOUNT_ID = "R2_ACCOUNT_ID"  # used only when no endpoint is set
PREFIX = "SN120_R2_PREFIX"  # has no plain name: a prefix is always a choice for v2
ACCOUNT_ENDPOINT = "https://{account}.r2.cloudflarestorage.com"


def endpoint_url(value: str, name: str, bucket: str = "") -> str:
    """An endpoint as ``scheme://host[:port]``. A bare host gets ``https://``. ``name`` is the variable, for the error."""
    text = value.strip()
    try:
        parts = urlsplit(text if "://" in text else f"https://{text}")
        port = parts.port
    except ValueError:
        parts = port = None
    path = parts.path.strip("/") if parts else ""
    if (not parts or parts.scheme not in ("https", "http") or not parts.hostname or parts.username or parts.query
            or parts.fragment or (path and path != bucket)):  # "<endpoint>/<bucket>" is a common form: the bucket is dropped
        raise ConfigError(f"{name} is not a store endpoint: it must be a host name or an http(s) URL with no path (the value is not shown)")
    return f"{parts.scheme}://{parts.hostname}{f':{port}' if port else ''}"


def store_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The ``SN120_R2_*`` settings for ``R2Store.from_env``. A set ``SN120_R2_*`` name wins over its plain name.

    A missing setting is ``ConfigError``. The message names the variables and never holds a value.
    """
    def value(name: str) -> str:
        return environ.get(name, "").strip()

    found, source = {}, {}
    for name, plain in PLAIN_NAMES.items():
        source[name] = name if value(name) else plain
        found[name] = value(name) or value(plain)
    if not found["SN120_R2_ENDPOINT"] and value(ACCOUNT_ID):
        if not re.fullmatch(r"[A-Za-z0-9]+", value(ACCOUNT_ID)):
            raise ConfigError(f"{ACCOUNT_ID} is not an account id (the value is not shown)")
        found["SN120_R2_ENDPOINT"], source["SN120_R2_ENDPOINT"] = ACCOUNT_ENDPOINT.format(account=value(ACCOUNT_ID)), ACCOUNT_ID
    for name, plain in PLAIN_NAMES.items():
        if not found[name]:
            other = f"{plain} or {ACCOUNT_ID}" if name == "SN120_R2_ENDPOINT" else plain
            raise ConfigError(f"the store setting {name} is not set, and {other} is not set either; refusing to guess")
    found["SN120_R2_ENDPOINT"] = endpoint_url(found["SN120_R2_ENDPOINT"], source["SN120_R2_ENDPOINT"], found["SN120_R2_BUCKET"])
    return found | ({PREFIX: environ[PREFIX]} if environ.get(PREFIX) else {})
