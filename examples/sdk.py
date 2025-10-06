import asyncio
import affine as af
from dotenv import load_dotenv

af.trace()

load_dotenv()


async def main():
    # Get all miner info or only for UID = 5
    # NOTE: HF_USER and HF_TOKEN .env value is required for this command.
    # miners = await af.miners()
    miner = await af.miners(160)
    assert miner, "Unable to obtain miner, please check if registered"

    # Generate a DED challenge
    chal = await af.DED()  # support SAT ABD DED ALFWORLD BABYAI SCIWORLD WEBSHOP

    # Query the model directly.
    # NOTE: A CHUTES_API_KEY .env value is required for this command.
    evaluation = await chal.evaluate(miner)
    print("=" * 50)
    print("chal:", chal.env_name)
    print(evaluation)

    chal_alfworld = await af.ALFWORLD()
    # Generate a ALFWORLD challenge, For agentgym type tasks, the task ID can be specified, otherwise it will be randomly assigned
    # evaluation = await chal_alfworld.evaluate(miner, task_id=[0,1,2])
    # evaluation = await chal_alfworld.evaluate(miner, task_id=10)
    evaluation = await chal_alfworld.evaluate(miner)
    print("=" * 50)
    print("chal:", chal_alfworld.env_name)
    print(evaluation)

    print("=" * 50)
    print("\nAll Available Environments:")
    envs = af.tasks.list_available_environments()
    for env_type, env_names in envs.items():
        print(f"\n{env_type}:")
        for name in env_names:
            print(f"  - {name}")


if __name__ == "__main__":
    asyncio.run(main())
