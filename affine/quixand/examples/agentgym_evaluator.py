#!/usr/bin/env python3

import os
import json
import time
import logging
import argparse
from affine import quixand as qs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not os.getenv("CHUTES_API_KEY"):
    logger.warning("CHUTES_API_KEY is not set. Proxy endpoints may not work correctly.")
    exit(0)

def test_evaluator_endpoint(sandbox):
    evaluation_request = {
        "model": "deepseek-ai/DeepSeek-R1",
        "ids": [0], # testcase ids
        "max_round": 5,
        # "base_url": "https://llm.chutes.ai/v1",
        # "data_len": 200,
        # "timeout": 300
        # "max_tokens": 2048,
        # "temperature": 0.7,
        # "top_p": 0.95,
    }

    print("Evaluation request:")
    print(json.dumps(evaluation_request, indent=2))
    print()
    
    start_time = time.time()
    try:
        response = sandbox.proxy.evaluator(**evaluation_request, _timeout=600)
        elapsed = time.time() - start_time

        print(f"Evaluation completed in {elapsed:.2f} seconds\n")
        
        # Display results
        print("=== Evaluation Results ===")
        print(f"Task: {response['task_name']}")
        print(f"Average Score: {response['total_score']:.3f}")
        print(f"Success Rate: {response['success_rate']:.3f}")
        print(f"Number Evaluated: {response['num_evaluated']}")
        print(f"Time Taken: {response['time_taken']:.2f}s")
        
        print("\n=== Detailed Results ===")
        for detail in response['details']:
            print(f"  ID {detail['id']}")
            print(f"  Reward {detail['reward']}")
            print(f"  Success {detail['success']}")
            print(f"  experiences {str(detail['experiences'])}")

        return response

    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return None


def main():
    AVAILABLE_ENVS = [
        "webshop",
        "alfworld",
        "babyai",
        "sciworld",
        # "webarena", # not support
        "textcraft",
        "sqlgym",
        "maze",
        "wordle",
    ]
    
    parser = argparse.ArgumentParser(description='Run AgentGym evaluator for specified environment')
    parser.add_argument(
        '--env',
        type=str,
        default="alfworld",
        choices=AVAILABLE_ENVS,
        help=f'Environment name to evaluate. Available options: {", ".join(AVAILABLE_ENVS)}'
    )
    
    args = parser.parse_args()
    env_name = args.env
    
    print(f"Building Docker image for {env_name} environment...")
    sandbox = qs.get_sandbox(f"agentgym:{env_name}")
    print(f"Container ID: {sandbox.container_id[:12]}\n")

    try:
        result = test_evaluator_endpoint(sandbox)
        
        if result:
            print("\n=== Test Summary ===")
            print("✓ Evaluator endpoint is working correctly!")
            print(f"Successfully evaluated {result['num_evaluated']} examples")
        else:
            print("\n=== Test Summary ===")
            print("✗ Evaluator endpoint test failed")
            print("Please check the error messages above")
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nCleaning up...")
        sandbox.shutdown()
        print("Done!")


if __name__ == "__main__":
    main()