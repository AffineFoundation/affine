#!/usr/bin/env python3

from affine import quixand as qs
import os


def main():
    print("=== Basic get_sandbox Example ===\n")
    print("Using the global get_sandbox function - no manager needed!\n")

    # Get a shared sandbox (creates new instance)
    print("1. Creating a shared sandbox...")
    sandbox1 = qs.get_sandbox("python:3.11-slim", shared=True)
    print(f"   Sandbox Container ID: {sandbox1.container_id[:12]}")

    # Execute some commands
    result = sandbox1.run(["python", "-c", "print('Hello from sandbox 1!')"])
    print(f"   Output: {result.text.strip()}")

    # Write and read a file
    sandbox1.files.write("test.py", "print('Test file executed')")
    result = sandbox1.run(["python", "test.py"])
    print(f"   File execution: {result.text.strip()}")

    print("\n2. Creating another non-shared sandbox...")
    sandbox2 = qs.get_sandbox(
        "agentgym:alfworld",
        shared=False,
    )
    print(f"   Sandbox Container ID: {sandbox2.container_id[:12]}")

    response = sandbox2.proxy.evaluator(
        model="deepseek-ai/DeepSeek-R1", ids=[0], max_round=5, _timeout=600
    )
    print(f"   response: {response}")

    print("\n3. Global manager statistics:")
    print(f"current stats: ", qs.get_manager_stats())

    del sandbox2
    print("\n4. After release sandbox2:")
    print(f"   Verify that the sandbox will be automatically released without the need for shutdown")
    print(f"   Active sandboxes: {qs.get_manager_stats()}")

if __name__ == "__main__":
    main()
