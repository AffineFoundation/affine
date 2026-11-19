"""Explicit Generator control-plane configuration, with no host defaults."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ConfigError


@dataclass(frozen=True, slots=True)
class Config:
    site_dir: Path
    work_dir: Path
    pool_dir: Path
    network: str = "finney"
    public_url: str = "https://affine.io"
    credentials_file: str = ""
    secret_file: Path = Path("generator-seed-secret")
    ssh: dict[str, Any] | None = None
    head_url: str = "https://affine.io/public/head.json"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        try:
            raw = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read Generator config {path}: {type(exc).__name__}") from exc
        if not isinstance(raw, dict):
            raise ConfigError("Generator config must be a JSON object")
        fields = {"site_dir", "work_dir", "pool_dir", "network", "public_url", "credentials_file", "secret_file", "ssh", "head_url"}
        unknown = set(raw) - fields
        if unknown:
            raise ConfigError(f"unknown Generator config fields: {', '.join(sorted(unknown))}")
        missing = {key for key in ("site_dir", "work_dir", "pool_dir") if not isinstance(raw.get(key), str) or not raw[key]}
        if missing:
            raise ConfigError(f"Generator config needs: {', '.join(sorted(missing))}")
        for key in ("network", "public_url", "credentials_file", "head_url"):
            if not isinstance(raw.get(key, getattr(cls, key, "")), str):
                raise ConfigError(f"Generator config {key} must be a string")
        ssh = raw.get("ssh")
        if ssh is not None and not isinstance(ssh, dict):
            raise ConfigError("Generator config ssh must be an object")
        return cls(
            Path(raw["site_dir"]).expanduser(), Path(raw["work_dir"]).expanduser(), Path(raw["pool_dir"]).expanduser(),
            raw.get("network", "finney"), raw.get("public_url", "https://affine.io"), raw.get("credentials_file", ""),
            Path(raw.get("secret_file", "generator-seed-secret")).expanduser(), ssh,
            raw.get("head_url", "https://affine.io/public/head.json"),
        )


def add_config(parser) -> None:
    parser.add_argument("--config", required=True, help="JSON file with site_dir, work_dir and pool_dir (never contains R2 tokens)")
