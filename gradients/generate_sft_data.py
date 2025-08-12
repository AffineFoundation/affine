#!/usr/bin/env python3
"""
Upload Affine data to S3
Usage: python affine_upload.py [--min-score SCORE] [--tail BLOCKS] [--max-results N]
"""

import sys
import os
import json
import asyncio
import argparse
from datetime import datetime
from typing import AsyncGenerator
import tempfile

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("Error: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

try:
    import affine
except ImportError:
    print("Error: Could not import affine module")
    sys.exit(1)

load_dotenv()

S3_ENDPOINT = os.getenv('S3_ENDPOINT')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_REGION = os.getenv('S3_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

if not all([S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION, S3_BUCKET_NAME]):
    print("Error: Missing S3 configuration in .env file")
    print("Required variables: S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION, S3_BUCKET_NAME")
    sys.exit(1)


async def stream_affine_data(
    min_score: float = 0.0,
    tail: int = 100,
    max_results: int | None = None
) -> AsyncGenerator[dict, None]:
    """Stream filtered affine data"""
    collected = 0
    
    async for res in affine.dataset(tail=tail):
        if res.evaluation.score >= min_score:
            yield {
                "prompt": res.challenge.prompt,
                "model": res.miner.model if res.miner.model else "unknown",
                "response": res.response.response if res.response.response else "",
                "environment": res.challenge.env.name,
                "score": res.evaluation.score,
                "miner_uid": res.miner.uid,
                "response_success": res.response.success,
                "response_latency": res.response.latency_seconds,
                "challenge_id": getattr(res.challenge, 'challenge_id', None)
            }
            collected += 1
            if max_results and collected >= max_results:
                break


async def collect_and_save(
    min_score: float = 0.0,
    tail: int = 100,
    max_results: int | None = None,
    show_progress: bool = True
) -> tuple[str, int, dict]:
    """Stream data directly to file and return filename, count, and stats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"affine_data_{timestamp}.json"
    
    entries = []
    stats = {"min": float('inf'), "max": float('-inf'), "sum": 0}
    count = 0
    
    if show_progress:
        print(f"Collecting affine data...")
        print(f"  Min score filter: {min_score}")
        print(f"  Processing last {tail} blocks")
        if max_results:
            print(f"  Max results: {max_results}")
    
    async for entry in stream_affine_data(min_score, tail, max_results):
        entries.append(entry)
        score = entry['score']
        stats["min"] = min(stats["min"], score)
        stats["max"] = max(stats["max"], score)
        stats["sum"] += score
        count += 1
        
        if show_progress and count % 50 == 0:
            print(f"  Collected {count} entries...")
    
    with open(filename, 'w') as f:
        json.dump(entries, f, indent=2)
    
    stats["avg"] = stats["sum"] / count if count > 0 else 0
    
    if show_progress:
        print(f"\nCollection complete: {count} entries")
    
    return filename, count, stats


def upload_to_s3(filename: str, dry_run: bool = False, presign_hours: int = 168) -> str | None:
    """Upload file to S3"""
    s3_key = f"affine_data/{os.path.basename(filename)}"
    file_size = os.path.getsize(filename) / 1024
    
    print(f"\nLocal file: {filename} ({file_size:.2f} KB)")
    
    if dry_run:
        print("Dry run - skipping upload")
        return None
    
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION
    )
    
    try:
        print(f"Uploading to s3://{S3_BUCKET_NAME}/{s3_key}")
        s3_client.upload_file(
            filename,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={'ContentType': 'application/json'}
        )
        
        print("Upload successful!")
        print(f"\nS3 URL: s3://{S3_BUCKET_NAME}/{s3_key}")
        
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=presign_hours * 3600
        )
        
        if presign_hours >= 24:
            days = presign_hours // 24
            print(f"\nPresigned URL ({days} day{'s' if days > 1 else ''}):")
        else:
            print(f"\nPresigned URL ({presign_hours} hour{'s' if presign_hours > 1 else ''}):")
        print(presigned_url)
        return presigned_url
        
    except ClientError as e:
        print(f"Upload error: {e}")
        return None


async def main():
    parser = argparse.ArgumentParser(description="Upload Affine data to S3")
    parser.add_argument("--min-score", type=float, default=0.0, help="Min score filter")
    parser.add_argument("--tail", type=int, default=100, help="Recent blocks to process")
    parser.add_argument("--max-results", type=int, help="Max results to collect")
    parser.add_argument("--dry-run", action="store_true", help="Skip S3 upload")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--presign-hours", type=int, default=168, help="Presigned URL expiration in hours (default: 168 = 1 week)")
    parser.add_argument("--delete-local", action="store_true", help="Delete local JSON after successful upload")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("=" * 60)
        print(f"Affine → S3 ({S3_BUCKET_NAME})")
        print("=" * 60)
    
    filename, count, stats = await collect_and_save(
        min_score=args.min_score,
        tail=args.tail,
        max_results=args.max_results,
        show_progress=not args.quiet
    )
    
    if count == 0:
        print("No data collected")
        return
    
    url = upload_to_s3(filename, dry_run=args.dry_run, presign_hours=args.presign_hours)
    
    if url and args.delete_local and not args.dry_run:
        os.remove(filename)
        print(f"\nDeleted local file: {filename}")
    
    if not args.quiet:
        print("\n" + "=" * 60)
        print(f"Total: {count} entries")
        print(f"Scores: avg={stats['avg']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())