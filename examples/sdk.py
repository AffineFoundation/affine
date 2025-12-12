import asyncio
import affine as af
from dotenv import load_dotenv
import sys
import os
import json

af.trace()

load_dotenv()


async def main():
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("\n   ❌ CHUTES_API_KEY environment variable not set")
        print("   Please set: export CHUTES_API_KEY='your-key'")
        print("   Or create .env file with: CHUTES_API_KEY=your-key")
        sys.exit(1)

    # Get miner info for UID = 160
    # NOTE: HF_USER and HF_TOKEN .env value is required for this command.
    uid = 243
    miner = await af.miners(uid)
    assert miner, "Unable to obtain miner, please check if registered"

    # Generate and evaluate a DED challenge
    # All environment logic is now encapsulated in Docker images via affinetes
    ded_env = af.DED()
    evaluation = await ded_env.evaluate(miner, task_id=20100)
    print("=" * 50)
    print(json.dumps(evaluation[uid].dict(), indent=2, ensure_ascii=False))

    # Generate and evaluate an ABD challenge
    abd_env = af.ABD()
    evaluation = await abd_env.evaluate(miner, task_id=20200)
    print("=" * 50)
    print(json.dumps(evaluation[uid].dict(), indent=2, ensure_ascii=False))

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