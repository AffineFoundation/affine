#!/usr/bin/env python3
import os
import io
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple

from dotenv import load_dotenv
load_dotenv()

import affine as af

logger = logging.getLogger("runner_mock_s3_test")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(_handler)

# ----------------------------- Fake S3 client -------------------------------- #
class _Body:
    def __init__(self, data: bytes):
        self._bio = io.BytesIO(data)
    async def read(self) -> bytes:
        return self._bio.getvalue()
    def __aiter__(self):
        data = self._bio.getvalue()
        lines = data.splitlines(True)
        async def gen():
            for l in lines:
                yield l
        return gen()

class FakeS3:
    def __init__(self):
        self.buckets: Dict[str, Dict[str, bytes]] = {}
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        return False
    async def head_bucket(self, Bucket: str):
        if Bucket not in self.buckets:
            raise _client_error(404)
    async def create_bucket(self, Bucket: str, CreateBucketConfiguration: Optional[dict] = None):
        self.buckets.setdefault(Bucket, {})
    async def head_object(self, Bucket: str, Key: str):
        if Key not in self.buckets.setdefault(Bucket, {}):
            raise _client_error(404)
        data = self.buckets[Bucket][Key]
        return {"ContentLength": len(data), "ETag": f"etag-{len(data)}", "LastModified": _NowLike()}
    async def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: Optional[str] = None):
        self.buckets.setdefault(Bucket, {})[Key] = Body if isinstance(Body, (bytes, bytearray)) else bytes(Body)
        return {"ETag": f"etag-{len(self.buckets[Bucket][Key])}"}
    async def get_object(self, Bucket: str, Key: str):
        if Key not in self.buckets.setdefault(Bucket, {}):
            raise _client_error(404)
        return {"Body": _Body(self.buckets[Bucket][Key])}

class _NowLike:
    def isoformat(self) -> str:
        return "2025-01-01T00:00:00+00:00"

def _client_error(code: int):
    from botocore.exceptions import ClientError
    return ClientError({"Error": {"Code": str(code)}, "ResponseMetadata": {"HTTPStatusCode": code}}, "op")

# ----------------------------- Fake Subtensor -------------------------------- #
class FakeMeta:
    def __init__(self, hotkeys: List[str]):
        self.hotkeys = hotkeys

class FakeSubtensor:
    def __init__(self, hotkeys: List[str]):
        self._meta = FakeMeta(hotkeys)
    async def metagraph(self, _netuid: int):
        return self._meta
    async def get_current_block(self):
        return 0
    async def wait_for_block(self):
        return None

async def _fake_get_subtensor():
    return FakeSubtensor(["5FakeHotKeyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"])

# ----------------------------- Test helpers ---------------------------------- #
async def make_results(env_name: str, miner_hk: str, n: int) -> List[af.Result]:
    from affine.envs import ENVS as _ENVS
    env = _ENVS[env_name]()
    out: List[af.Result] = []
    for i in range(n):
        chal = af.Challenge(env=env, prompt=f"mock {i}")
        resp = af.Response(response="ok", latency_seconds=0.01, attempts=1, model="test/model", error=None, success=True)
        ev = af.Evaluation(env=env, score=1.0, extra={})
        miner = af.Miner(uid=i, hotkey=miner_hk, model="test/model", revision="r1")
        out.append(af.Result(miner=miner, challenge=chal, response=resp, evaluation=ev))
    return out

async def main():
    # Monkeypatch storage to FakeS3
    from affine import database as db
    fake = FakeS3()
    db.get_client_ctx = lambda: fake  # type: ignore
    os.environ["HIPPIUS_BUCKET"] = os.getenv("HIPPIUS_BUCKET", "affine-test")

    # Prepare fake network
    miner_hk = "5FakeHotKeyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    af.get_subtensor = _fake_get_subtensor  # type: ignore
    af.get_miners = lambda meta=None: asyncio.sleep(0, result={0: af.Miner(uid=0, hotkey=miner_hk, model="test/model", revision="r1", block=0)})  # type: ignore

    # Generate and sink results (no-lease path)
    import bittensor as bt
    wallet = bt.wallet(name=os.getenv("BT_WALLET_COLD", "default"), hotkey=os.getenv("BT_WALLET_HOT", "default"))
    env_name = os.getenv("AFFINE_TEST_ENV", "SAT")
    N = int(os.getenv("AFFINE_TEST_COUNT", "8"))
    results = await make_results(env_name, miner_hk, N)
    os.environ["AFFINE_NO_LEASE"] = "1"
    await af.sink(wallet=wallet, results=results)

    # Verify _load_catalog_and_active discovers shard-00000001 via fallback
    rt = db._stream_root(miner_hk, env_name, getattr(af.ENVS[env_name], "__version__", None))
    async with db.get_client_ctx() as c:
        known, active = await db._load_catalog_and_active(c, root=rt)
    assert known == [1], f"expected fallback known [1], got {known}"

    # Verify aggregate sees them
    pairs = [(miner_hk, None)]
    agg = await af.aggregate_scores_by_env(env_name=env_name, pairs=pairs, env_version=getattr(af.ENVS[env_name], "__version__", None))
    n_total = int((agg.get(miner_hk) or {}).get("n_total", 0))
    assert n_total >= N, f"expected at least {N}, got {n_total}"

    print(json.dumps({"ok": True, "n_uploaded": N, "known": known, "active": active, "n_total": n_total}, separators=(",", ":")))

if __name__ == "__main__":
    asyncio.run(main())
