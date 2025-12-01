#!/usr/bin/env python3
"""
Universal evaluation script for affine environments

Supports two evaluation modes:
1. Using Chutes service via --uid
2. Using local model via --base-url

Usage:
    ./evaluate_local_model.py --env ABD-V2 --uid 7
    ./evaluate_local_model.py --env ABD-V2 --model your-model --base-url http://172.17.0.1:30000/v1 --samples 10
    ./evaluate_local_model.py --env ALFWORLD --samples 10 --model deepseek-ai/DeepSeek-V3 --base-url https://llm.chutes.ai/v1 --output ./eval.json
"""
import asyncio
import argparse
import sys
import os
import json
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supported environment names (will be mapped to actual classes after argparse)
ENVIRONMENT_NAMES = ['SAT', 'ABD', 'ABD-V2', 'DED', 'DED-V2', 'ALFWORLD', 'WEBSHOP', 'BABYAI', 'SCIWORLD', 'TEXTCRAFT']


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate models on affine environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate using Chutes service (requires CHUTES_API_KEY)
  %(prog)s --env ABD-V2 --uid 7
  
  # Evaluate using local model
  %(prog)s --env ABD-V2 --model your-model --base-url http://172.17.0.1:30000/v1
  
  # Evaluate with specific task and seed
  %(prog)s --env ALFWORLD --task-id 2 --seed 42 --model deepseek-ai/DeepSeek-V3 --base-url https://llm.chutes.ai/v1 --samples 5

  # Evaluate across a range of task IDs (one sample per task)
  %(prog)s --env ALFWORLD --task-id-range 0 9 --model your-model --base-url http://172.17.0.1:30000/v1
  %(prog)s --env ABD --task-id-range 0 4 --uid 7 --output results.json

Supported environments: """ + ', '.join(ENVIRONMENT_NAMES)
    )

    parser.add_argument('--env', required=True, type=str,
                        help='Environment name (case-insensitive)')
    
    # Mode selection: either uid or (model + base-url)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--uid', type=int,
                           help='Miner UID (use Chutes service)')
    mode_group.add_argument('--model',
                           help='Model name (use with --base-url)')
    
    parser.add_argument('--base-url',
                       help='Model service URL (required with --model)')
    parser.add_argument('--task-id', type=int,
                       help='Task ID for environment')
    parser.add_argument('--seed', type=int,
                       help='Random seed for evaluation')
    parser.add_argument('--samples', type=int, default=1,
                       help='Number of evaluation samples (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (default: 0.0)')
    parser.add_argument('--output', '-o',
                       help='Output file path for JSON results (optional)')

    args = parser.parse_args()

    # Normalize environment name to uppercase for case-insensitive matching
    args.env = args.env.upper()
    
    # Validate environment name
    if args.env not in ENVIRONMENT_NAMES:
        parser.error(f'invalid environment: {args.env!r} (choose from {", ".join(ENVIRONMENT_NAMES)})')

    # Validation
    if args.model and not args.base_url:
        parser.error('--base-url is required when using --model')

    return args


async def evaluate_with_uid(env_instance, uid: int, task_id: Optional[int], seed: Optional[int], samples: int, temperature: float, af):
    """Evaluate using Chutes service via miner UID"""
    print(f"\nFetching miner info for UID {uid}...")
    miner = await af.miners(uid)
    if not miner:
        raise ValueError(f"Unable to get miner info for UID {uid}")

    print(f"Miner found: {miner.get(uid).model}")

    total_score = 0.0
    total_time = 0.0
    results = []

    for i in range(samples):
        print(f"\rProgress: {i+1}/{samples}", end="", flush=True)

        eval_kwargs = {'temperature': temperature}
        if task_id is not None:
            eval_kwargs['task_id'] = task_id
        if seed is not None:
            eval_kwargs['seed'] = seed
        
        result = await env_instance.evaluate(miner, **eval_kwargs)

        # Result is a dict with uid as key
        eval_result = result[uid]
        total_score += eval_result.score
        total_time += eval_result.latency_seconds
        
        results.append(result[uid].model_dump())

    print()  # New line after progress
    return total_score, total_time, results


async def evaluate_with_model(env_instance, model: str, base_url: str, task_id: Optional[int], seed: Optional[int], samples: int, temperature: float):
    """Evaluate using direct model endpoint"""
    print(f"\nUsing model: {model}")
    print(f"Service URL: {base_url}")

    total_score = 0.0
    total_time = 0.0
    results = []

    for i in range(samples):
        print(f"\rProgress: {i+1}/{samples}", end="", flush=True)

        eval_kwargs = {
            'model': model,
            'base_url': base_url,
            'temperature': temperature
        }
        if task_id is not None:
            eval_kwargs['task_id'] = task_id
        if seed is not None:
            eval_kwargs['seed'] = seed
        
        result = await env_instance.evaluate(**eval_kwargs)

        total_score += result.score
        total_time += result.latency_seconds

        results.append(result.model_dump())

    print()  # New line after progress
    return total_score, total_time, results


async def evaluate_with_model_range(env_instance, model: str, base_url: str, task_id_start: int, task_id_end: int, temperature: float):
    """Evaluate using direct model endpoint across a range of task IDs (one sample per task)"""
    print(f"\nUsing model: {model}")
    print(f"Service URL: {base_url}")

    total_score = 0.0
    total_time = 0.0
    all_results = []
    task_results = {}

    task_count = task_id_end - task_id_start + 1
    eval_counter = 0

    for task_id in range(task_id_start, task_id_end + 1):
        eval_counter += 1
        print(f"\rProgress: {eval_counter}/{task_count} (Task {task_id})", end="", flush=True)

        eval_kwargs = {
            'model': model,
            'base_url': base_url,
            'temperature': temperature,
            'task_id': task_id
        }

        result = await env_instance.evaluate(**eval_kwargs)

        total_score += result.score
        total_time += result.latency_seconds

        sample_result = result.model_dump()
        sample_result['task_id'] = task_id
        all_results.append(sample_result)

        task_results[task_id] = {
            'score': result.score,
            'success': result.success,
            'time': result.latency_seconds,
            'error': result.error
        }

    print()  # New line after progress
    return total_score, total_time, all_results, task_results


async def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Import affine AFTER argparse to avoid bittensor hijacking
    import affine as af
    af.trace()
    
    # Map environment names to actual classes
    ENVIRONMENTS = {
        'sat': af.SAT,
        'abd': af.ABD,
        'abd-v2': af.ABD_V2,
        'ded': af.DED,
        'ded-v2': af.DED_V2,
        'alfworld': af.ALFWORLD,
        'webshop': af.WEBSHOP,
        'babyai': af.BABYAI,
        'sciworld': af.SCIWORLD,
        'textcraft': af.TEXTCRAFT,
    }
    
    # Check API key for Chutes service
    if args.uid and not os.getenv("CHUTES_API_KEY"):
        print("\n❌ CHUTES_API_KEY environment variable not set")
        print("   Please set: export CHUTES_API_KEY='your-key'")
        print("   Or create .env file with: CHUTES_API_KEY=your-key")
        sys.exit(1)
    
    # Set fake API key for local model testing (required by Docker env)
    if args.model and not os.getenv("CHUTES_API_KEY"):
        os.environ["CHUTES_API_KEY"] = "fake-test-key-for-local-testing"
        print("⚠️  CHUTES_API_KEY not set, using temporary test key")
    
    print("=" * 60)
    print("Evaluation Configuration:")
    print(f"  Environment: {args.env}")
    if args.uid:
        print(f"  Mode: Chutes service (UID: {args.uid})")
    else:
        print(f"  Mode: Direct model")
        print(f"  Model: {args.model}")
        print(f"  Base URL: {args.base_url}")
    if args.task_id is not None:
        print(f"  Task ID: {args.task_id}")
    if args.seed is not None:
        print(f"  Seed: {args.seed}")
    print(f"  Samples: {args.samples}")
    print(f"  Temperature: {args.temperature}")
    print("=" * 60)
    
    try:
        # Create environment instance
        print(f"\nLoading {args.env} environment...")
        env_class = ENVIRONMENTS[args.env.lower()]
        env_instance = env_class()
        print("✓ Environment loaded")
        
        # Run evaluation
        print(f"\nStarting evaluation ({args.samples} sample(s))...")
        
        # Use af.miners for UID mode
        if args.uid:
            total_score, total_time, results = await evaluate_with_uid(
                env_instance, args.uid, args.task_id, args.seed, args.samples, args.temperature, af
            )
        else:
            total_score, total_time, results = await evaluate_with_model(
                env_instance, args.model, args.base_url, args.task_id, args.seed, args.samples, args.temperature
            )
        
        # Prepare summary data
        if args.task_id_range is not None:
            # Range mode: total_samples = task_count (one sample per task)
            task_count = args.task_id_range[1] - args.task_id_range[0] + 1
            total_samples = task_count
        else:
            # Single mode: total_samples = samples
            total_samples = args.samples

        summary = {
            'environment': args.env,
            'mode': 'chutes' if args.uid else 'direct',
            'samples': args.samples,
            'total_score': total_score,
            'average_score': total_score / total_samples,
            'total_time': total_time,
            'average_time': total_time / total_samples,
            'results': results
        }

        # Add mode-specific info
        if args.uid:
            summary['uid'] = args.uid
        else:
            summary['model'] = args.model
            summary['base_url'] = args.base_url

        if args.task_id is not None:
            summary['task_id'] = args.task_id
        if args.seed is not None:
            summary['seed'] = args.seed
        
        summary['temperature'] = args.temperature
        
        # Save to JSON file if specified
        if args.output:
            output_path = args.output
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f"\n✓ Results saved to: {output_path}")
            except Exception as e:
                print(f"\n✗ Failed to save results: {e}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Evaluation Summary:")
        print(f"  Environment: {args.env}")

        if args.task_id_range is not None:
            task_count = args.task_id_range[1] - args.task_id_range[0] + 1
            print(f"  Task Range: {args.task_id_range[0]} - {args.task_id_range[1]} ({task_count} tasks)")
            print(f"  Total Evaluations: {task_count}")
        else:
            total_samples = args.samples
            print(f"  Total Samples: {args.samples}")

        print(f"  Total Score: {total_score:.4f}")
        print(f"  Average Score: {total_score/total_samples:.4f}")
        print(f"  Total Time: {total_time:.2f} seconds")
        print(f"  Average Time: {total_time/total_samples:.2f} seconds/sample")

        # Show task-by-task results for range mode
        if args.task_id_range is not None and task_results:
            print(f"\nResults by Task ID:")
            for task_id in sorted(task_results.keys()):
                task_info = task_results[task_id]
                status = "✓" if task_info.get('success', False) else "✗"
                score = task_info.get('score', 0.0)
                time = task_info.get('time', 0.0)
                print(f"  [{status}] Task {task_id}: score={score:.4f}, time={time:.2f}s")
                if task_info.get('error'):
                    print(f"      Error: {task_info['error']}")

        # Show individual results for single task mode with multiple samples
        elif args.samples > 1 and not args.task_id_range:
            print(f"\nDetailed Results:")
            for idx, r in enumerate(results):
                status = "✓" if r.get('success', False) else "✗"
                score = r.get('score', 0.0)
                latency = r.get('latency_seconds', r.get('latency', 0.0))
                print(f"  [{status}] Sample {idx}: score={score:.4f}, time={latency:.2f}s")
                if r.get('error'):
                    print(f"      Error: {r['error']}")

        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())