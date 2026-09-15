"""Publication of curriculum snapshots to data.affine.io/curriculum/** and
the one-line private Discord note.

Layout (bucket `affine-data`, the same R2 the fold publishes to):
  curriculum/ledger/<ledger_sha>.rows.parquet | .rollup.parquet | .json   immutable
  curriculum/ledger/latest.json                                          pointer
  curriculum/weights/<weights_sha>/{rule,groups,recurrence,deficit_by_source,
      counterfactual,criterion}.json + weights.parquet + diff.md          immutable
  curriculum/<for_epoch>/  the same files, at the epoch the next fold will
      publish (rewritten if that fold is skipped and the job runs again)
  curriculum/latest.json   {rule_version, mode, ledger_sha256, weights_sha256,
      manifest_sha256, for_epoch, against_epoch, computed_at, knobs, paths}
  curriculum/teacher_probe/probes.jsonl.gz   teacher-probe side table

Credentials: DATA_R2_ACCESS_KEY_ID / DATA_R2_SECRET_ACCESS_KEY (+ optional
DATA_R2_ENDPOINT) and DISCORD_BOT_TOKEN_ARBOS_BITTENSOR — read through
common.env_value, never printed.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROBES_PATH, env_value, log, sha256_bytes  # noqa: E402

from affine.config import load_config  # noqa: E402
from affine.corpus.publish import CorpusPublisher  # noqa: E402

PRIVATE_DISCORD_CHANNEL = "1510910974498967613"
NO_CACHE = "no-cache"
SNAPSHOT_FILES = ("rule.json", "weights.parquet", "groups.json", "recurrence.json",
                  "deficit_by_source.json", "counterfactual.json", "criterion.json", "diff.md",
                  "ledger.json", "top10_cards.md", "teacher_probe.jsonl.gz", "latest.json", "fold_vector.json")
CONTENT_TYPES = {".json": "application/json", ".parquet": "application/vnd.apache.parquet",
                 ".md": "text/markdown; charset=utf-8", ".gz": "application/gzip"}


def make_publisher(prefix: str = "") -> CorpusPublisher:
    cfg = load_config()
    ak, sk = env_value("DATA_R2_ACCESS_KEY_ID"), env_value("DATA_R2_SECRET_ACCESS_KEY")
    if not (ak and sk):
        raise SystemExit("DATA_R2_ACCESS_KEY_ID / DATA_R2_SECRET_ACCESS_KEY missing")
    return CorpusPublisher(bucket=cfg.data_r2["bucket"],
                           endpoint=env_value("DATA_R2_ENDPOINT") or cfg.secrets.data_r2_endpoint,
                           access_key_id=ak, secret_access_key=sk,
                           key_prefix=f"{prefix.strip('/')}/" if prefix.strip("/") else "", log=log)


def put_file(pub: CorpusPublisher, local: Path, key: str, *, immutable: bool) -> None:
    body = local.read_bytes()
    if immutable and pub.exists(key):
        remote = pub.get(key)
        if sha256_bytes(remote) == sha256_bytes(body):
            log(f"skip {key} (already published, same sha)")
            return
        raise RuntimeError(f"{key} exists with different bytes -- refusing to overwrite an immutable object")
    ctype = CONTENT_TYPES.get(local.suffix, "application/octet-stream")
    if immutable:
        pub.put(key, body, ctype)
    else:
        pub.put(key, body, ctype, cache_control=NO_CACHE)


def publish_snapshot(pub: CorpusPublisher, *, snapshot_dir: Path, ledger_dir: Path, ledger_sha: str,
                     weights_sha: str, for_epoch: int, latest_body: bytes) -> list[str]:
    keys: list[str] = []
    for suffix in ("rows.parquet", "rollup.parquet", "json"):
        local = ledger_dir / f"{ledger_sha}.{suffix}"
        key = f"curriculum/ledger/{ledger_sha}.{suffix}"
        put_file(pub, local, key, immutable=True)
        keys.append(key)
    put_file(pub, ledger_dir / "latest.json", "curriculum/ledger/latest.json", immutable=False)
    for name in SNAPSHOT_FILES:
        local = snapshot_dir / name
        if not local.is_file():
            continue
        put_file(pub, local, f"curriculum/weights/{weights_sha}/{name}", immutable=True)
        put_file(pub, local, f"curriculum/{for_epoch}/{name}", immutable=False)
        keys.append(f"curriculum/{for_epoch}/{name}")
    pub.put("curriculum/latest.json", latest_body, "application/json", cache_control=NO_CACHE)
    if PROBES_PATH.is_file():
        pub.put("curriculum/teacher_probe/probes.jsonl.gz", gzip.compress(PROBES_PATH.read_bytes()),
                "application/gzip", cache_control=NO_CACHE)
    return keys


def discord_line(text: str, channel: str = PRIVATE_DISCORD_CHANNEL) -> bool:
    token = env_value("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")
    if not token:
        log("discord: no bot token; line not posted")
        return False
    r = httpx.post(f"https://discord.com/api/v10/channels/{channel}/messages",
                   headers={"Authorization": f"Bot {token}"}, json={"content": text[:1900]}, timeout=30)
    if r.status_code >= 300:
        log(f"discord: HTTP {r.status_code}")
        return False
    return True
