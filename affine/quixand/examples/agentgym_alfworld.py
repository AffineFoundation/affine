#!/usr/bin/env python3

import random
from affine import quixand as qs

def main():
    print("=== AgentGym AlfWorld Example ===\n")
    
    print("Building Docker image with AlfWorld environment...")
    image = qs.Templates.agentgym("alfworld")
    print(f"Image built: {image}\n")
    
    print("Creating sandbox container...")
    sandbox = qs.Sandbox(template=image)
    print(f"Container ID: {sandbox.container_id[:8]}\n")
    
    try:
        print("Creating AlfWorld environment instance...")
        response = sandbox.proxy.create()
        env_id = response.get("id", 0)
        print(f"Environment instance created with ID: {env_id}\n")
        
        print("Starting a new game...")
        reset_response = sandbox.proxy.reset(
            id=env_id,
            game=0,  # Game ID
            world_type="Text"  # Text-based world
        )
        print(f"start :\n{reset_response}\n")
        step_response = sandbox.proxy.step(
            id=env_id,
            action="go to dresser 1"
        )
        print(f"step 1:\n{step_response}\n")

        step_response = sandbox.proxy.step(
            id=env_id,
            action="take alarmclock 2 from dresser 1"
        )
        print(f"step 2:\n{step_response}\n")

        step_response = sandbox.proxy.step(
            id=env_id,
            action="examine alarmclock 2"
        )
        print(f"step 3:\n{step_response}\n")

        step_response = sandbox.proxy.step(
            id=env_id,
            action="use desklamp 1"
        )
        print(f"step 4:\n{step_response}\n")

    except Exception as e:
        print(f"Error occurred: {e}")
        
    finally:
        print("\nShutting down container...")
        sandbox.shutdown()
        print("Done!")


if __name__ == "__main__":
    main()
