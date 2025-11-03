import asyncio
import affine as af
from dotenv import load_dotenv
import os
import sys

af.trace()

load_dotenv()


async def main():
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("\n   ❌ CHUTES_API_KEY environment variable not set")
        print("   Please set: export CHUTES_API_KEY='your-key'")
        print("   Or create .env file with: CHUTES_API_KEY=your-key")
        sys.exit(1)

    ded_env = af.ALFWORLD()
    evaluation = await ded_env.evaluate(
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        task_id=[2]
    )
    print(f"\nALFWORLD Evaluation Result:")
    print(f"{evaluation}")

    abd_env = af.ABD()
    evaluation_abd = await abd_env.evaluate(
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        num_samples=2
    )
    print(f"\nABD Evaluation Result:")
    print(evaluation_abd)


if __name__ == "__main__":
    asyncio.run(main())