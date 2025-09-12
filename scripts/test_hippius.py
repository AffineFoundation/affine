import os
import sys
import uuid
import asyncio
import base64
from dotenv import load_dotenv
from aiobotocore.session import get_session
from botocore.config import Config

# Load environment variables from .env file
load_dotenv()


def _get_env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return v


async def _amain() -> None:
    endpoint = os.getenv("HIPPIUS_ENDPOINT", "https://s3.hippius.com")
    region = os.getenv("HIPPIUS_REGION", "decentralized")
    seed = _get_env("HIPPIUS_SEED_PHRASE", "")
    bucket = os.getenv("HIPPIUS_BUCKET", os.getenv("AFFINE_BUCKET", "affine"))
    access_key = base64.b64encode(seed.encode("utf-8")).decode("utf-8")
    secret_key = seed

    async with get_session().create_client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(max_pool_connections=64),
    ) as c:
        # Ensure bucket exists
        try:
            await c.head_bucket(Bucket=bucket)
        except Exception:
            try:
                await c.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
            except Exception:
                # If creation fails because it exists or perms, continue to object ops
                pass

        key = f"affine/test/{uuid.uuid4().hex}.txt"
        payload = b"hello-hippius"

        await c.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="text/plain")
        obj = await c.get_object(Bucket=bucket, Key=key)
        data = await obj["Body"].read()
        if data != payload:
            raise RuntimeError("Downloaded content mismatch")
        head = await c.head_object(Bucket=bucket, Key=key)
        etag = head.get("ETag")
        size = int(head.get("ContentLength", 0))
        print({"endpoint": endpoint, "bucket": bucket, "key": key, "etag": etag, "size": size})


def main() -> None:
    try:
        asyncio.run(_amain())
    except Exception as e:
        print(f"Hippius test failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


