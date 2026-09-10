"""The king seat: where the current SN120 king is served for datagen.

The controller on the validator box (ops/king-datagen/kingctl.py) rents a
box for every new king, serves it, and writes one small file on each
datagen pod:

    KING_BASE_URL=http://<ip>:<port>/v1
    KING_MODEL=king
    KING_KEY=<bearer>
    KING_DIGEST=<sha256 model_digest>
    KING_REIGN=<reign number>

The `king_*` policies in policies.toml route through these vars
(Endpoint.model_env / base_url_env), so the supervisor re-reads the file
every cycle and needs no restart when the crown moves. No file = the king
policies are unavailable and the scheduler skips them (fail-open to the
teacher policies, exactly like a missing provider key).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("rollouts.king")

KING_ENV_PATH = Path(os.environ.get("ROLLOUTS_KING_ENV",
                                    "/root/rollouts/.king_env"))
KING_VARS = ("KING_BASE_URL", "KING_MODEL", "KING_KEY", "KING_DIGEST",
             "KING_REIGN")


def read_king_env(path: Path = KING_ENV_PATH) -> dict[str, str]:
    """KEY=value lines -> dict (only KING_VARS; `export ` prefix tolerated).
    Missing or unreadable file -> {}."""
    out: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        if k in KING_VARS:
            out[k] = v.strip().strip('"').strip("'")
    return out


def refresh_king_env(env: dict, path: Path = KING_ENV_PATH) -> bool:
    """Sync the KING_* vars in `env` (in place) with the file. Returns True
    when the king changed since the previous call (for the log)."""
    new = read_king_env(path)
    if not new.get("KING_BASE_URL") or not new.get("KING_KEY"):
        new = {}
    old = {k: env[k] for k in KING_VARS if k in env}
    if new == old:
        return False
    for k in KING_VARS:
        env.pop(k, None)
    env.update(new)
    if new:
        log.info("king seat: reign %s digest %s at %s", new.get("KING_REIGN"),
                 (new.get("KING_DIGEST") or "")[:12], new.get("KING_BASE_URL"))
    else:
        log.info("king seat: no endpoint (%s missing or incomplete); "
                 "king policies idle", path)
    return True
