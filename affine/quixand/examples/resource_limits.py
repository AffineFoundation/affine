#!/usr/bin/env python3
"""
Resource Limits Example
Demonstrates how to set and test resource limits for sandboxes
"""

from affine.quixand import Sandbox
from affine.quixand.config import Resources
import time

def main():
    """Demonstrate resource limiting capabilities"""
    print("=== Resource Limits Example ===\n")
    
    # Example 1: Memory limit
    print("1. Testing memory limits...")
    limits = Resources(mem_limit="50m")  # 50MB limit
    
    with Sandbox(resources=limits, timeout=30) as sandbox:
        print(f"Sandbox created with 50MB memory limit")
        
        # Try to allocate small amount (should succeed)
        result = sandbox.run("python3 -c 'data = bytearray(10 * 1024 * 1024); print(\"Allocated 10MB\")'")
        if result.exit_code == 0:
            print(f"[OK] {result.text.strip()}")
        
        # Try to allocate too much (should fail)
        result = sandbox.run("python3 -c 'data = bytearray(100 * 1024 * 1024); print(\"Allocated 100MB\")'")
        if result.exit_code != 0:
            print("[OK] Large allocation failed as expected")
    
    # Example 2: CPU limits
    print("\n2. Testing CPU limits...")
    limits = Resources(cpu_limit=0.5)  # 50% CPU (0.5 cores)
    
    with Sandbox(resources=limits, timeout=30) as sandbox:
        print(f"Sandbox created with 50% CPU limit")
        
        # CPU intensive task
        code = """
import time
start = time.time()
# CPU intensive loop
for _ in range(10**7):
    _ = sum(range(100))
elapsed = time.time() - start
print(f"Task completed in {elapsed:.2f}s")
"""
        result = sandbox.run(f"python3 -c '{code}'")
        print(f"Result: {result.text.strip()}")
    
    # Example 3: PID limits
    print("\n3. Testing PID limits...")
    limits = Resources(pids_limit=10)
    
    with Sandbox(resources=limits, timeout=30) as sandbox:
        print(f"Sandbox created with max 10 PIDs")
        
        # Try to create many processes
        code = """
import os
for i in range(5):
    if os.fork() == 0:
        # Child process
        import time
        time.sleep(1)
        exit(0)
print("Created 5 child processes")
"""
        result = sandbox.run(f"python3 -c '{code}'")
        if result.exit_code == 0:
            print(f"[OK] {result.text.strip()}")

    print("\n=== Example completed successfully ===")
    return 0

if __name__ == "__main__":
    exit(main())