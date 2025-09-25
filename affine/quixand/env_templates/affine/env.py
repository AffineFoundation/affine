#!/usr/bin/env python3

import os
import importlib


def _load_app():
    env = (os.environ.get("ENV_NAME") or "").lower()
    modname = {"abd": "abd", "ded": "ded", "sat": "sat"}.get(env)
    if not modname:
        raise SystemExit(f"Unknown or missing ENV_NAME: {env!r}. Expected one of: abd, ded, sat")
    m = importlib.import_module(modname)
    from common import inject_health_endpoint, inject_evaluator_endpoint
    inject_health_endpoint(m.app)
    inject_evaluator_endpoint(m.app, env)
    return m.app


app = _load_app()


