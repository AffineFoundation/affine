#!/usr/bin/env python3
"""
Universal evaluation script for affine environments

Supports two evaluation modes:
1. Using Chutes service via --uid
2. Using local model via --base-url

Usage:
    ./evaluate_local_model.py --env ABD --uid 7
    ./evaluate_local_model.py --env ABD --model your-model --base-url http://172.17.0.1:30000/v1 --samples 10
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
ENVIRONMENT_NAMES = ['SAT', 'ABD', 'DED', 'ALFWORLD', 'WEBSHOP', 'BABYAI', 'SCIWORLD', 'TEXTCRAFT']

# AgentGym environments (require task_id)
AGENTGYM_ENVS = {'ALFWORLD', 'WEBSHOP', 'BABYAI', 'SCIWORLD', 'TEXTCRAFT'}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate models on affine environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate using Chutes service (requires CHUTES_API_KEY)
  %(prog)s --env ABD --uid 7
  
  # Evaluate using local model
  %(prog)s --env ABD --model your-model --base-url http://172.17.0.1:30000/v1
  
  # Evaluate AgentGym environment with specific task
  %(prog)s --env ALFWORLD --task-id 2 --model deepseek-ai/DeepSeek-V3 --base-url https://llm.chutes.ai/v1 --samples 5

Supported environments: """ + ', '.join(ENVIRONMENT_NAMES)
    )

    parser.add_argument('--env', required=True, choices=ENVIRONMENT_NAMES,
                        help='Environment name')
    
    # Mode selection: either uid or (model + base-url)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--uid', type=int,
                           help='Miner UID (use Chutes service)')
    mode_group.add_argument('--model',
                           help='Model name (use with --base-url)')
    
    parser.add_argument('--base-url',
                       help='Model service URL (required with --model)')
    parser.add_argument('--task-id', type=int,
                       help='Task ID for AgentGym environments')
    parser.add_argument('--samples', type=int, default=1,
                       help='Number of evaluation samples (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (default: 0.7)')
    parser.add_argument('--output', '-o',
                       help='Output file path for JSON results (optional)')

    args = parser.parse_args()

    # Validation
    if args.model and not args.base_url:
        parser.error('--base-url is required when using --model')
    
    if args.task_id is not None and args.env not in AGENTGYM_ENVS:
        parser.error(f'--task-id is only applicable for AgentGym environments: {", ".join(AGENTGYM_ENVS)}')

    return args


async def evaluate_with_uid(env_instance, uid: int, task_id: Optional[int], samples: int, temperature: float, af):
    """Evaluate using Chutes service via miner UID"""
    print(f"\nFetching miner info for UID {uid}...")
    miner = await af.miners(uid)
    if not miner:
        raise ValueError(f"Unable to get miner info for UID {uid}")

    miner_info = miner.get(uid)
    print(f"Miner found:")
    print(f"  UID: {uid}")
    print(f"  Model: {miner_info.model}")
    if hasattr(miner_info, 'chute') and miner_info.chute:
        print(f"  Chute: {miner_info.chute}")
    if hasattr(miner_info, 'slug') and miner_info.slug:
        print(f"  Slug: {miner_info.slug}")
    if hasattr(miner_info, 'hotkey') and miner_info.hotkey:
        print(f"  Hotkey: {miner_info.hotkey[:16]}...")
    if hasattr(miner_info, 'revision') and miner_info.revision:
        print(f"  Revision: {miner_info.revision}")
    if hasattr(miner_info, 'block') and miner_info.block:
        print(f"  Block: {miner_info.block}")

    results = []
    # Use semaphore to limit concurrency to 3
    semaphore = asyncio.Semaphore(16)

    async def run_single_evaluation(i):
        """Run a single evaluation with semaphore control"""
        async with semaphore:
            print(f"\n[Sample {i+1}/{samples}] Running (UID: {uid})...", flush=True)

            eval_kwargs = {'temperature': temperature}
            if task_id is not None:
                eval_kwargs['task_id'] = 29

            try:
                result = await env_instance.evaluate(miner, **eval_kwargs)

                # Result is a dict with uid as key
                eval_result = result[uid]

                result_data = {
                    'index': i,
                    'uid': uid,
                    'score': eval_result.score,
                    'success': eval_result.success,
                    'latency': eval_result.latency_seconds,
                    'error': eval_result.error
                }

                # Stream output for this sample
                status = "✓" if eval_result.success else "✗"
                print(f"[{status}] Sample {i+1} (UID: {uid}): score={eval_result.score:.4f}, time={eval_result.latency_seconds:.2f}s", flush=True)
                if eval_result.error:
                    print(f"    Error: {eval_result.error}", flush=True)
                if not eval_result.success:
                    # Print more detailed error info for failed tests
                    print(f"    Success: False", flush=True)
                    if hasattr(eval_result, 'traceback') and eval_result.traceback:
                        print(f"    Traceback: {eval_result.traceback}", flush=True)
                    if hasattr(eval_result, 'output') and eval_result.output:
                        print(f"    Output: {eval_result.output[:200]}...", flush=True)

                return result_data
            except Exception as e:
                print(f"[✗] Sample {i+1} (UID: {uid}): FAILED", flush=True)
                print(f"    Exception: {type(e).__name__}: {str(e)}", flush=True)
                import traceback
                print(f"    Traceback:", flush=True)
                traceback.print_exc()

                result_data = {
                    'index': i,
                    'uid': uid,
                    'score': 0.0,
                    'success': False,
                    'latency': 0.0,
                    'error': f"{type(e).__name__}: {str(e)}"
                }
                return result_data

    # Run all evaluations concurrently with max 3 concurrent tasks
    tasks = [run_single_evaluation(i) for i in range(samples)]
    results = await asyncio.gather(*tasks)

    # Calculate totals
    total_score = sum(r['score'] for r in results)
    total_time = sum(r['latency'] for r in results)

    return total_score, total_time, results


async def evaluate_with_model(env_instance, model: str, base_url: str, task_id: Optional[int], samples: int, temperature: float):
    """Evaluate using direct model endpoint"""
    print(f"\nUsing model: {model}")
    print(f"Service URL: {base_url}")

    results = []
    # Use semaphore to limit concurrency to 3
    semaphore = asyncio.Semaphore(3)

    async def run_single_evaluation(i):
        """Run a single evaluation with semaphore control"""
        async with semaphore:
            print(f"\n[Sample {i+1}/{samples}] Running...", flush=True)

            eval_kwargs = {
                'model': model,
                'base_url': base_url,
                'temperature': temperature
            }
            if task_id is not None:
                eval_kwargs['task_id'] = task_id

            try:
                result = await env_instance.evaluate(**eval_kwargs)

                result_data = result.model_dump()
                result_data['index'] = i

                # Stream output for this sample
                status = "✓" if result.success else "✗"
                print(f"[{status}] Sample {i+1}: score={result.score:.4f}, time={result.latency_seconds:.2f}s", flush=True)
                if result.error:
                    print(f"    Error: {result.error}", flush=True)
                if not result.success:
                    # Print more detailed error info for failed tests
                    print(f"    Success: False", flush=True)
                    if hasattr(result, 'traceback') and result.traceback:
                        print(f"    Traceback: {result.traceback}", flush=True)
                    if hasattr(result, 'output') and result.output:
                        print(f"    Output: {result.output[:200]}...", flush=True)

                return result_data
            except Exception as e:
                print(f"[✗] Sample {i+1}: FAILED", flush=True)
                print(f"    Exception: {type(e).__name__}: {str(e)}", flush=True)
                import traceback
                print(f"    Traceback:", flush=True)
                traceback.print_exc()

                result_data = {
                    'index': i,
                    'score': 0.0,
                    'success': False,
                    'latency_seconds': 0.0,
                    'error': f"{type(e).__name__}: {str(e)}"
                }
                return result_data

    # Run all evaluations concurrently with max 3 concurrent tasks
    tasks = [run_single_evaluation(i) for i in range(samples)]
    results = await asyncio.gather(*tasks)

    # Calculate totals
    total_score = sum(r['score'] for r in results)
    total_time = sum(r.get('latency_seconds', 0.0) for r in results)

    return total_score, total_time, results


async def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Import affine AFTER argparse to avoid bittensor hijacking
    import affine as af
    af.trace()
    
    # Map environment names to actual classes
    ENVIRONMENTS = {
        'SAT': af.SAT,
        'ABD': af.ABD,
        'DED': af.DED,
        'ALFWORLD': af.ALFWORLD,
        'WEBSHOP': af.WEBSHOP,
        'BABYAI': af.BABYAI,
        'SCIWORLD': af.SCIWORLD,
        'TEXTCRAFT': af.TEXTCRAFT,
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
    print(f"  Samples: {args.samples}")
    print(f"  Temperature: {args.temperature}")
    print("=" * 60)
    
    try:
        # Create environment instance
        print(f"\nLoading {args.env} environment...")
        env_class = ENVIRONMENTS[args.env]
        env_instance = env_class()
        print("✓ Environment loaded")
        
        # Run evaluation
        print(f"\nStarting evaluation ({args.samples} sample(s))...")
        
        # Use af.miners for UID mode
        if args.uid:
            total_score, total_time, results = await evaluate_with_uid(
                env_instance, args.uid, args.task_id, args.samples, args.temperature, af
            )
        else:
            total_score, total_time, results = await evaluate_with_model(
                env_instance, args.model, args.base_url, args.task_id, args.samples, args.temperature
            )
        
        # Prepare summary data
        summary = {
            'environment': args.env,
            'mode': 'chutes' if args.uid else 'direct',
            'samples': args.samples,
            'total_score': total_score,
            'average_score': total_score / args.samples,
            'total_time': total_time,
            'average_time': total_time / args.samples,
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
        print(f"  Total Samples: {args.samples}")
        print(f"  Total Score: {total_score:.4f}")
        print(f"  Average Score: {total_score/args.samples:.4f}")
        print(f"  Total Time: {total_time:.2f} seconds")
        print(f"  Average Time: {total_time/args.samples:.2f} seconds/sample")
        
        # Show all results recap
        if args.samples > 1:
            print(f"\nAll Results Recap:")
            for idx, r in enumerate(results):
                status = "✓" if r.get('success', False) else "✗"
                score = r.get('score', 0.0)
                latency = r.get('latency_seconds', r.get('latency', 0.0))
                print(f"  [{status}] Sample {idx+1}: score={score:.4f}, time={latency:.2f}s")
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