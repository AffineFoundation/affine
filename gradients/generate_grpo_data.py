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
import tempfile
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

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
    from affine.utils.executor import ProgramExecutor
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

if not all([S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION, S3_BUCKET_NAME]):
    print("Error: Missing S3 configuration in .env file")
    print("Required variables: S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION, S3_BUCKET_NAME")
    sys.exit(1)


# Simple Python programs for ABD generation
SIMPLE_PROGRAMS = [
    {
        "program": "n = int(input())\nprint(n * 2)",
        "inputs": ["5", "10", "3", "7", "100"],
    },
    {
        "program": "a = int(input())\nb = int(input())\nprint(a + b)",
        "inputs": ["3\n4", "10\n20", "5\n5", "100\n200"],
    },
    {
        "program": "s = input()\nprint(s.upper())",
        "inputs": ["hello", "world", "test", "affine"],
    },
    {
        "program": "n = int(input())\nprint(n ** 2)",
        "inputs": ["4", "5", "10", "7"],
    },
    {
        "program": "x = int(input())\ny = int(input())\nprint(x - y)",
        "inputs": ["10\n3", "20\n5", "100\n25"],
    },
    {
        "program": "n = int(input())\nresult = 1\nfor i in range(1, n+1):\n    result *= i\nprint(result)",
        "inputs": ["5", "4", "3", "6"],
    },
    {
        "program": "s = input()\nprint(len(s))",
        "inputs": ["hello", "testing", "a", "longstring"],
    },
    {
        "program": "n = int(input())\nif n % 2 == 0:\n    print('even')\nelse:\n    print('odd')",
        "inputs": ["4", "7", "10", "13"],
    },
    {
        "program": "n = int(input())\nfor i in range(n):\n    print(i)",
        "inputs": ["3", "5", "2"],
    },
    {
        "program": "a = input()\nb = input()\nprint(a + b)",
        "inputs": ["hello\nworld", "foo\nbar", "test\n123"],
    }
]


async def generate_sat_data(n: int = 7, k: int = 5, m: int = None) -> Dict[str, Any]:
    """Generate SAT problem data with solution"""
    print(f"  Generating SAT problem (n={n}, k={k})...")
    
    m = m or int(4.26 * n)
    
    # Generate random solution
    sol = {i: random.choice([True, False]) for i in range(1, n+1)}
    
    # Generate clauses that are satisfied by the solution
    cls = []
    for _ in range(m):
        vs = random.sample(list(sol), k)
        sv = random.choice(vs)  # Force at least one satisfied literal
        clause = []
        for v in vs:
            if v == sv:
                # This literal must satisfy the clause
                lit = v if sol[v] else -v
            else:
                # Random literal
                lit = v if random.choice([True, False]) else -v
            clause.append(lit)
        cls.append(clause)
    
    # Create the formula string
    formula = " ∧ ".join("(" + " ∨ ".join(f"{'¬' if l<0 else ''}x{abs(l)}" for l in c) + ")" for c in cls)
    
    prompt = (
        f"Find a satisfying assignment for the following {k}-SAT formula over variables x1..x{n}:\n"
        f"{formula}\n"
        "Provide your answer as comma-separated assignments like `x1=True, x2=False, ...`, "
        "or respond `UNSAT` if it has no solution."
    )
    
    print(f"    Generated SAT with {len(cls)} clauses")
    
    return {
        "prompt": prompt,
        "cls": cls,
        "sol": {str(k): v for k, v in sol.items()}  # Convert keys to strings for JSON
    }


def execute_program(program: str, input_data: str) -> Tuple[str, str]:
    """Execute a Python program with given input"""
    print(f"    Executing program with input: {input_data[:50]}...")
    
    executor = ProgramExecutor(timeout=5)
    output, error = executor.execute(program, input_data)
    
    if error:
        print(f"    Warning: Program execution error: {error}")
    else:
        print(f"    Program output: {output.strip()[:50]}...")
    
    return output.strip(), error


async def generate_abd_data(program_idx: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Generate ABD (program abduction) data"""
    print(f"  Generating ABD problem...")
    
    if program_idx is None:
        prog_data = random.choice(SIMPLE_PROGRAMS)
    else:
        prog_data = SIMPLE_PROGRAMS[program_idx % len(SIMPLE_PROGRAMS)]
    
    program = prog_data["program"]
    input_choice = random.choice(prog_data["inputs"])
    
    # Execute program to get output
    output, error = execute_program(program, input_choice)
    
    if error or not output:
        print(f"    Failed to generate ABD data: {error}")
        return None
    
    prompt = f"""You are a programming expert. Given a Python program and its expected output, you need to determine the exact input that would produce this output.

Program:
```python
{program}
```

Expected Output:
```
{output}
```

Task: Analyze the program to understand what input format it expects from stdin, then provide the input data that would produce the expected output.

You can provide any explanations, analysis, or reasoning you want. However, you MUST include the input data within <INPUT> </INPUT> tags.

Format the input data like this:
<INPUT>
[input data here - each line on a separate line as the program expects]
</INPUT>

I will extract only the content between these tags.

Requirements for the input data within the tags:
1. Each line of input should be on a separate line
2. Use the exact format the program expects  
3. Provide the raw input values that should be fed to stdin
4. Do not include any prefixes or extra formatting within the INPUT tags

Please analyze the program and provide the required input:"""

    print(f"    Generated ABD with program length {len(program)} chars")
    
    return {
        "prompt": prompt,
        "program": program,
        "expected_output": output,
        "solution_input": input_choice  # Store the actual input for reference
    }


async def generate_mixed_dataset(n: int, sat_ratio: float = 0.5) -> List[Dict[str, Any]]:
    """
    Generate a mixed dataset with SAT and ABD problems.
    Each entry has both 'sat' and 'abd' keys as required.
    """
    print(f"\n{'='*60}")
    print(f"Generating {n} synthetic examples (SAT ratio: {sat_ratio:.1%})")
    print(f"{'='*60}")
    
    dataset = []
    n_sat = int(n * sat_ratio)
    n_abd = n - n_sat
    
    print(f"Target: {n_sat} SAT problems, {n_abd} ABD problems")
    
    # Track statistics
    sat_count = 0
    abd_count = 0
    both_count = 0
    
    for i in range(n):
        print(f"\n[{i+1}/{n}] Generating example...")
        
        # Decide what type of example to generate
        # 10% chance of generating both SAT and ABD for the same entry
        both_chance = random.random() < 0.1
        
        if both_chance:
            # Generate both SAT and ABD data
            print("  Type: BOTH (SAT + ABD)")
            sat_data = await generate_sat_data()
            abd_data = await generate_abd_data()
            
            if abd_data:
                entry = {
                    "prompt": f"Hybrid challenge:\n\n1. {sat_data['prompt']}\n\n2. {abd_data['prompt']}",
                    "env": {
                        "sat": {
                            "cls": sat_data["cls"],
                            "sol": sat_data["sol"]
                        },
                        "abd": {
                            "program": abd_data["program"],
                            "expected_output": abd_data["expected_output"]
                        }
                    }
                }
                both_count += 1
            else:
                # Fallback to SAT-only if ABD generation failed
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
                
        elif i < n_sat or (i >= n_sat and abd_count >= n_abd):
            # Generate SAT-only
            print("  Type: SAT-only")
            sat_data = await generate_sat_data()
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
            # Generate ABD-only
            print("  Type: ABD-only")
            abd_data = await generate_abd_data(program_idx=i)
            
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
                sat_data = await generate_sat_data()
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
        
        dataset.append(entry)
        print(f"  Added entry with env.sat={'✓' if entry['env']['sat'] else '✗'}, env.abd={'✓' if entry['env']['abd'] else '✗'}")
    
    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"  SAT-only: {sat_count}")
    print(f"  ABD-only: {abd_count}")
    print(f"  Both: {both_count}")
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
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Synthetic Dataset Generator")
    print(f"Target S3 Bucket: {S3_BUCKET_NAME}")
    print("=" * 60)
    
    # Generate dataset
    dataset = await generate_mixed_dataset(args.num_examples, args.sat_ratio)
    
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
    # Set up logging for affine module
    affine.debug()  # Enable debug logging
    asyncio.run(main())