"""uvicorn entry: python -m affine.dash"""

from __future__ import annotations

import logging
import os

import uvicorn

from ..config import load_config
from .app import create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    dash = cfg.dashboard
    host = str(dash.get("api_host") or "127.0.0.1")
    port = int(dash.get("api_port") or 8787)
    # Allow env override without editing toml (ops / smoke).
    host = os.environ.get("AFFINE_DASH_HOST", host)
    port = int(os.environ.get("AFFINE_DASH_PORT", port))
    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
