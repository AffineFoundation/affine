#!/usr/bin/env python3
"""
Universal test script for evaluating local sglang models on affine environments
Based on examples/sdk2.py

Usage:
    ./evaluate_local_model.py [model_name] [environment] [num_samples]

Examples:
    ./evaluate_local_model.py your-model BABYAI 500
    ./evaluate_local_model.py your-model WEBSHOP 100
    ./evaluate_local_model.py your-model ABD 50 --base-url http://172.17.0.1:30001/v1
"""
import asyncio
import affine as af
import sys
import os
import argparse

# Check and set CHUTES_API_KEY if not set
if not os.getenv("CHUTES_API_KEY"):
    os.environ["CHUTES_API_KEY"] = "fake-test-key-for-local-testing"
    print("⚠️  CHUTES_API_KEY not set, using temporary test key")

# Enable affine tracing
af.trace()


# Supported environments
SUPPORTED_ENVS = {
    'SAT': af.SAT,
    'ABD': af.ABD,
    'DED': af.DED,
    'ALFWORLD': af.ALFWORLD,
    'WEBSHOP': af.WEBSHOP,
    'BABYAI': af.BABYAI,
    'SCIWORLD': af.SCIWORLD,
    'TEXTCRAFT': af.TEXTCRAFT,
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test local sglang models on affine environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s your-model BABYAI 500
  %(prog)s your-model WEBSHOP 100
  %(prog)s your-model ABD 50 --base-url http://172.17.0.1:30001/v1

Supported environments: """ + ', '.join(SUPPORTED_ENVS.keys())
    )

    parser.add_argument('model',
                        help='Model name (required)')
    parser.add_argument('env',
                        choices=SUPPORTED_ENVS.keys(),
                        help='Environment name (required)')
    parser.add_argument('num_samples', type=int,
                        help='Number of test samples (required)')
    parser.add_argument('--base-url', default='http://172.17.0.1:30000/v1',
                        help='Model service URL (default: http://172.17.0.1:30000/v1)')

    return parser.parse_args()


async def main():
    """
    Test model using local sglang service
    """
    # Parse command line arguments
    args = parse_args()

    model = args.model
    env_name = args.env
    num_samples = args.num_samples
    base_url = args.base_url

    print(f"=" * 60)
    print(f"Test Configuration:")
    print(f"  Model: {model}")
    print(f"  Environment: {env_name}")
    print(f"  Num Samples: {num_samples}")
    print(f"  Service URL: {base_url}")
    print(f"  Note: Using 172.17.0.1 to access host sglang service")
    print(f"=" * 60)

    # Create Miner object for local testing (fixed uid and hotkey)
    miner = af.Miner(
        uid=0,
        hotkey="local-test",
        model=model,
        slug="localhost:30000"
    )

    # Test specified environment
    print(f"\nTesting {env_name} environment...")
    print("Loading Docker image, please wait...")
    try:
        # Dynamically create environment instance
        env_class = SUPPORTED_ENVS[env_name]
        env_instance = env_class()
        print("✓ Environment loaded, starting evaluation...")
        print(f"Evaluating {num_samples} times (1 sample per iteration), this may take a while...")

        # Accumulate statistics
        total_score = 0.0
        total_time = 0.0
        all_results = []

        for i in range(num_samples):
            print(f"\rProgress: {i+1}/{num_samples}", end="", flush=True)

            evaluation = await env_instance.evaluate(
                miner=miner,
                model=model,
                base_url=base_url
            )

            # Accumulate results
            total_score += evaluation.score
            total_time += evaluation.latency_seconds

            all_results.append({
                'index': i,
                'score': evaluation.score,
                'success': evaluation.success,
                'latency': evaluation.latency_seconds,
                'extra': evaluation.extra
            })

        print(f"\n\n✓ {env_name} Evaluation Summary:")
        print(f"  Model: {model}")
        print(f"  Environment: {env_name}")
        print(f"  Total Tests: {num_samples}")
        print(f"  Total Score: {total_score:.4f}")
        print(f"  Average Score: {total_score/num_samples:.4f}")
        print(f"  Total Time: {total_time:.2f} seconds")
        print(f"  Average Time: {total_time/num_samples:.2f} seconds/sample")

    except Exception as e:
        print(f"\n✗ {env_name} test failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "=" * 60)
    print("Test completed!")
    print(f"=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nTest failed: {e}")
        sys.exit(1)
