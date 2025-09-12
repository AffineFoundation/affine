import os
import sys
import uuid
import asyncio
from typing import Dict, Any

from dotenv import load_dotenv
import importlib.util
from pathlib import Path


async def _amain() -> None:
    # Import database module directly to avoid package-level side effects (numpy/pandas deps)
    pkg_root = Path(__file__).resolve().parent.parent
    db_path = pkg_root / "affine" / "database.py"
    spec = importlib.util.spec_from_file_location("affine_database", str(db_path))
    db = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(db)  # type: ignore[attr-defined]

    # Validate required env
    seed = os.getenv("HIPPIUS_SEED_PHRASE")
    if not seed:
        raise RuntimeError("Missing HIPPIUS_SEED_PHRASE in environment/.env.local")
    bucket = db.FOLDER  # resolved from HIPPIUS_BUCKET
    endpoint = db.HIPPIUS_ENDPOINT
    region = db.HIPPIUS_REGION

    # Unique test ids to avoid collisions and ensure isolation
    env_name = f"affine-smoke-{uuid.uuid4().hex[:8]}"
    miner = f"smoke-miner-{uuid.uuid4().hex[:8]}"
    env_version = None  # keep None to exercise default path

    root = db._stream_root(miner, env_name, env_version)
    shard_key = db._shard_key(root, 1)

    # Two minimal rollout-like records that match the reader filters
    def rec(score: float, revision: str = "r1") -> Dict[str, Any]:
        return {
            "miner": {"hotkey": miner, "revision": revision},
            "evaluation": {"score": float(score)},
            "response": {"success": True},
            # Denormalized fields used by the reader for fast filtering
            "_dn": {
                "env_name": env_name,
                "env_version": env_version,
                "revision": revision,
                "score": float(score),
                "success": True,
            },
        }

    doc1 = rec(0.9)
    doc2 = rec(0.8)

    payload = db._dumps(doc1) + b"\n" + db._dumps(doc2) + b"\n"

    # 1) Put a shard directly. Reader can discover shard-00000001 via fallback without meta/catalog.
    async with db.get_client_ctx() as c:
        # Ensure bucket exists/accessible
        try:
            await c.head_bucket(Bucket=bucket)
        except Exception:
            try:
                await c.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
            except Exception:
                # If creation is not allowed or already exists, continue to object ops
                pass

        await c.put_object(Bucket=bucket, Key=shard_key, Body=payload, ContentType="application/json")

    # 2) Aggregate via public APIs to validate read path end-to-end
    pairs = [(miner, None)]  # allow any revision for this miner
    success = await db.aggregate_success_by_env(env_name=env_name, pairs=pairs, env_version=env_version)
    scores = await db.aggregate_scores_by_env(env_name=env_name, pairs=pairs, env_version=env_version)

    # Basic assertions
    if miner not in success:
        raise AssertionError(f"aggregate_success_by_env returned no entry for miner={miner}: {success}")
    if miner not in scores:
        raise AssertionError(f"aggregate_scores_by_env returned no entry for miner={miner}: {scores}")

    n_success = float(success[miner].get("n_success", 0.0))
    sum_success = float(success[miner].get("sum_score", 0.0))
    n_total = float(scores[miner].get("n_total", 0.0))
    sum_score = float(scores[miner].get("sum_score", 0.0))

    # Expect exactly 2 entries with sums 0.9 + 0.8 = 1.7
    if n_success != 2.0 or n_total != 2.0:
        raise AssertionError(f"Expected 2 records, got n_success={n_success}, n_total={n_total}")
    if abs(sum_success - 1.7) > 1e-6 or abs(sum_score - 1.7) > 1e-6:
        raise AssertionError(f"Expected score sum 1.7, got success_sum={sum_success}, total_sum={sum_score}")

    print({
        "endpoint": endpoint,
        "region": region,
        "bucket": bucket,
        "root": root,
        "shard_key": shard_key,
        "success": success[miner],
        "scores": scores[miner],
    })

    # 3) Best-effort cleanup (optional)
    try:
        async with db.get_client_ctx() as c:
            await c.delete_object(Bucket=bucket, Key=shard_key)
    except Exception:
        # Non-fatal if cleanup fails
        pass


def main() -> None:
    # Load .env.local explicitly so HIPPIUS_* are available
    load_dotenv(".env.local")
    try:
        asyncio.run(_amain())
    except Exception as e:
        print(f"Affine Hippius S3 smoke test FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
