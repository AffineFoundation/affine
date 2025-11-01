import asyncio
import affine as af
from dotenv import load_dotenv

af.trace()

load_dotenv()


async def main():
    # Get miner info for UID = 160
    # NOTE: HF_USER and HF_TOKEN .env value is required for this command.
    miner = await af.miners(160)
    assert miner, "Unable to obtain miner, please check if registered"

    # Generate and evaluate a DED challenge
    # All environment logic is now encapsulated in Docker images via affinetes
    ded_env = await af.DED()
    evaluation = await ded_env.evaluate(miner)
    print("=" * 50)
    print("Environment:", ded_env.env_name)
    print("Score:", evaluation.score)
    print("Details:", evaluation.extra)

    # Generate and evaluate an ALFWORLD challenge
    # For AgentGym tasks, you can specify task IDs
    alfworld_env = await af.ALFWORLD()
    # evaluation = await alfworld_env.evaluate(miner, task_id=[0,1,2])  # Multiple tasks
    # evaluation = await alfworld_env.evaluate(miner, task_id=10)        # Single task
    evaluation = await alfworld_env.evaluate(miner)  # Random task
    print("=" * 50)
    print("Environment:", alfworld_env.env_name)
    print("Score:", evaluation.score)
    print("Details:", evaluation.extra)

    # List all available environments
    print("=" * 50)
    print("\nAll Available Environments:")
    envs = af.tasks.list_available_environments()
    for env_type, env_names in envs.items():
        print(f"\n{env_type}:")
        for name in env_names:
            print(f"  - {name}")


if __name__ == "__main__":
    asyncio.run(main())