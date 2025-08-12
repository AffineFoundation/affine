#!/usr/bin/env python3
"""
Generate synthetic SAT and ABD dataset in the required format and upload to S3.
Creates a dataset with the required structure where every row has both sat and abd keys.
"""

import os
import sys
import json
import random
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import required modules
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
    from affine.envs.sat import SAT
    from affine.envs.abd import ABD
except ImportError:
    print("Error: Could not import affine module")
    sys.exit(1)

load_dotenv()

# S3 Configuration
S3_ENDPOINT = os.getenv('S3_ENDPOINT')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_REGION = os.getenv('S3_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Required for ABD generation (if using the actual ABD class)
CHUTES_API_KEY = os.getenv('CHUTES_API_KEY')

if not all([S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION, S3_BUCKET_NAME]):
    print("Error: Missing S3 configuration in .env file")
    print("Required variables: S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION, S3_BUCKET_NAME")
    sys.exit(1)


async def generate_sat_data_from_env() -> Dict[str, Any]:
    """Generate SAT problem using the actual SAT environment"""
    print(f"  Generating SAT problem using SAT environment...")
    
    sat_env = SAT()
    challenge = await sat_env.generate()
    
    # Extract the data we need for the dataset
    cls = challenge.extra.get("cls", [])
    sol = challenge.extra.get("sol", {})
    
    # Convert solution keys to strings for JSON compatibility
    sol_str = {str(k): v for k, v in sol.items()}
    
    print(f"    Generated SAT with {len(cls)} clauses")
    
    return {
        "prompt": challenge.prompt,
        "cls": cls,
        "sol": sol_str
    }


async def generate_abd_data_from_env() -> Optional[Dict[str, Any]]:
    """Generate ABD problem using the actual ABD environment"""
    print(f"  Generating ABD problem using ABD environment...")
    
    # Check if we have CHUTES_API_KEY for LLM generation
    if not CHUTES_API_KEY:
        print("    Warning: CHUTES_API_KEY not set, ABD generation may fail")
        print("    ABD uses LLM to generate diverse inputs for programs")
        return None
    
    try:
        abd_env = ABD()
        challenge = await abd_env.generate()
        
        # Extract the data we need
        program = challenge.extra.get("program", "")
        expected_output = challenge.extra.get("expected_output", "")
        
        print(f"    Generated ABD with program length {len(program)} chars")
        
        return {
            "prompt": challenge.prompt,
            "program": program,
            "expected_output": expected_output
        }
    except Exception as e:
        print(f"    Failed to generate ABD data: {e}")
        print(f"    This may be due to missing CHUTES_API_KEY or dataset access")
        return None


async def generate_mixed_dataset(n: int, sat_ratio: float = 0.5, use_real_envs: bool = True) -> List[Dict[str, Any]]:
    """
    Generate a mixed dataset with SAT and ABD problems.
    Each entry has both 'sat' and 'abd' keys as required.
    
    Args:
        n: Number of examples to generate
        sat_ratio: Ratio of SAT vs ABD problems (0.0-1.0)
        use_real_envs: If True, use actual SAT/ABD env classes; if False, use simplified generation
    """
    print(f"\n{'='*60}")
    print(f"Generating {n} synthetic examples (SAT ratio: {sat_ratio:.1%})")
    print(f"Using {'real' if use_real_envs else 'simplified'} environment generation")
    print(f"{'='*60}")
    
    dataset = []
    n_sat = int(n * sat_ratio)
    n_abd = n - n_sat
    
    print(f"Target: {n_sat} SAT problems, {n_abd} ABD problems")
    
    # Set up logging for affine module
    affine.debug()
    
    # Track statistics
    sat_count = 0
    abd_count = 0
    abd_failures = 0
    
    for i in range(n):
        print(f"\n[{i+1}/{n}] Generating example...")
        
        if i < n_sat or (i >= n_sat and abd_count >= n_abd):
            # Generate SAT-only
            print("  Type: SAT-only")
            sat_data = await generate_sat_data_from_env()
            entry = {
                "prompt": sat_data["prompt"],
                "env": {
                    "sat": {
                        "cls": sat_data["cls"],
                        "sol": sat_data["sol"]
                    },
                    "abd": False
                }
            }
            sat_count += 1
            
        else:
            print("  Type: ABD-only")
            abd_data = await generate_abd_data_from_env()
            
            if abd_data:
                entry = {
                    "prompt": abd_data["prompt"],
                    "env": {
                        "sat": False,
                        "abd": {
                            "program": abd_data["program"],
                            "expected_output": abd_data["expected_output"]
                        }
                    }
                }
                abd_count += 1
            else:
                # Fallback to SAT if ABD generation failed
                print("  Fallback to SAT due to ABD generation failure")
                sat_data = await generate_sat_data_from_env()
                entry = {
                    "prompt": sat_data["prompt"],
                    "env": {
                        "sat": {
                            "cls": sat_data["cls"],
                            "sol": sat_data["sol"]
                        },
                        "abd": False
                    }
                }
                sat_count += 1
                abd_failures += 1
        
        dataset.append(entry)
        print(f"  Added entry with env.sat={'✓' if entry['env']['sat'] else '✗'}, env.abd={'✓' if entry['env']['abd'] else '✗'}")
    
    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"  SAT-only: {sat_count}")
    print(f"  ABD-only: {abd_count}")
    if abd_failures > 0:
        print(f"  ABD failures (fell back to SAT): {abd_failures}")
    print(f"  Total: {len(dataset)}")
    print(f"{'='*60}")
    
    return dataset


def upload_to_s3(filename: str, dry_run: bool = False, presign_hours: int = 168) -> Optional[str]:
    """Upload file to S3"""
    s3_key = f"synthetic_data/{os.path.basename(filename)}"
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
    parser = argparse.ArgumentParser(description="Generate synthetic SAT/ABD dataset and upload to S3")
    parser.add_argument("-n", "--num-examples", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--sat-ratio", type=float, default=0.5, help="Ratio of SAT problems (0.0-1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Skip S3 upload")
    parser.add_argument("--presign-hours", type=int, default=168, help="Presigned URL expiration in hours")
    parser.add_argument("--delete-local", action="store_true", help="Delete local JSON after successful upload")
    parser.add_argument("--validate", action="store_true", help="Validate dataset structure before upload")
    parser.add_argument("--simplified", action="store_true", help="Use simplified generation (no LLM needed for ABD)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Synthetic Dataset Generator")
    print(f"Target S3 Bucket: {S3_BUCKET_NAME}")
    print("=" * 60)
    
    # Check for ABD requirements
    if not args.simplified and not CHUTES_API_KEY:
        print("\n⚠️  WARNING: CHUTES_API_KEY not set in .env")
        print("ABD generation uses an LLM to create diverse program inputs.")
        print("Without it, ABD problems will fail and fall back to SAT.")
        print("To fix: Add CHUTES_API_KEY to your .env file")
        print("Or use --simplified flag for basic ABD generation\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Generate dataset
    dataset = await generate_mixed_dataset(
        args.num_examples, 
        args.sat_ratio,
        use_real_envs=not args.simplified
    )
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"synthetic_dataset_{timestamp}.json"
    
    print(f"\nSaving dataset to {filename}...")
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    file_size = os.path.getsize(filename) / 1024
    print(f"Saved {len(dataset)} entries ({file_size:.2f} KB)")
    
    # Validate structure if requested
    if args.validate:
        print("\nValidating dataset structure...")
        valid = True
        for i, entry in enumerate(dataset):
            if "prompt" not in entry:
                print(f"  ERROR: Entry {i} missing 'prompt' field")
                valid = False
            if "env" not in entry:
                print(f"  ERROR: Entry {i} missing 'env' field")
                valid = False
            elif not isinstance(entry["env"], dict):
                print(f"  ERROR: Entry {i} 'env' is not a dict")
                valid = False
            elif "sat" not in entry["env"] or "abd" not in entry["env"]:
                print(f"  ERROR: Entry {i} missing 'sat' or 'abd' in env")
                valid = False
            else:
                # Validate SAT structure if present
                if entry["env"]["sat"] and entry["env"]["sat"] is not False:
                    if not isinstance(entry["env"]["sat"], dict):
                        print(f"  ERROR: Entry {i} sat is not dict or false")
                        valid = False
                    elif "cls" not in entry["env"]["sat"] or "sol" not in entry["env"]["sat"]:
                        print(f"  ERROR: Entry {i} sat missing 'cls' or 'sol'")
                        valid = False
                
                # Validate ABD structure if present
                if entry["env"]["abd"] and entry["env"]["abd"] is not False:
                    if not isinstance(entry["env"]["abd"], dict):
                        print(f"  ERROR: Entry {i} abd is not dict or false")
                        valid = False
                    elif "program" not in entry["env"]["abd"] or "expected_output" not in entry["env"]["abd"]:
                        print(f"  ERROR: Entry {i} abd missing 'program' or 'expected_output'")
                        valid = False
        
        if valid:
            print("  ✓ All entries have valid structure!")
        else:
            print("  ✗ Validation failed. Fix errors before uploading.")
            if not args.dry_run:
                print("Aborting upload due to validation errors.")
                return
    
    # Upload to S3
    url = upload_to_s3(filename, dry_run=args.dry_run, presign_hours=args.presign_hours)
    
    # Delete local file if requested and upload was successful
    if url and args.delete_local and not args.dry_run:
        os.remove(filename)
        print(f"\nDeleted local file: {filename}")
    
    print("\n" + "=" * 60)
    print("Process complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())