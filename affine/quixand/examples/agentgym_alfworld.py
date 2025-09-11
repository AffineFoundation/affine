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
            game=5,  # Game ID
            world_type="Text"  # Text-based world
        )
        available_actions = reset_response.get("available_actions", [])
        print(f"reset_response:\n{reset_response}\n")
        print("-" * 50 + "\n")
        
        step_count = 0
        max_steps = 30
        done = False

        while not done and step_count < max_steps:
            if not available_actions:
                print("No available actions. Game might be stuck.")
                break
            
            action = random.choice(available_actions)
            print(f"Step {step_count + 1}: Taking action - '{action}'")
            
            step_response = sandbox.proxy.step(
                id=env_id,
                action=action
            )
            
            observation = step_response.get("observation", "")
            done = step_response.get("done", False)
            reward = step_response.get("reward", 0)
            available_actions = step_response.get("available_actions", [])
            
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
            if observation:
                obs_preview = observation[:150] + "..." if len(observation) > 150 else observation
                print(f"  Observation: {obs_preview}")
            print()
            
            step_count += 1

            if done:
                print(f"Task completed successfully in {step_count} steps!")
                break
        
        if not done:
            print(f"Reached maximum steps ({max_steps}) without completing the task.")
        
        print("\nGetting final environment details...")
        detail_response = sandbox.proxy.detail(id=env_id)
        
        if detail_response:
            print("Final environment state:")
            for key, value in detail_response.items():
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error occurred: {e}")
        
    finally:
        print("\nShutting down container...")
        sandbox.shutdown()
        print("Done!")


if __name__ == "__main__":
    main()
