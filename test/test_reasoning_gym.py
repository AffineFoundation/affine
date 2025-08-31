import asyncio

from affine.envs.reasoning_gym import ReasoningGym
import affine as af


async def test_reasoning_env():
    print("Testing ReasoningGym environment...")
    env = ReasoningGym(
        dataset_names=[
            "aiw",
            "arc_agi",
            "advanced_geometry",
            "basic_arithmetic",
            "boxnet",
            "countdown",
        ],
        num_samples=1,
    )

    challenge = await env.generate()
    print(f"Dataset used: {challenge.extra['dataset_name']}")
    print(f"Challenge prompt:\n{challenge.prompt}\n")

    mock_response = af.Response(
        response="42",
        latency_seconds=0.1,
        attempts=1,
        model="test",
        error=None,
        success=True,
    )
    evaluation = await env.evaluate(challenge, mock_response)
    print(f"Score: {evaluation.score}")
    print(f"Expected answer: {evaluation.extra.get('expected')}")


async def main():
    await test_reasoning_env()
    print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
