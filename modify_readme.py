with open("README.md", "r") as f:
    content = f.read()

# Find the end of the SDK section and add the Affinetes Usage section
sdk_section_end = '''if __name__ == "__main__":
    asyncio.run(main())
```'''

affinetes_section = '''

# Affinetes Usage
Affinetes provides container orchestration for RL environments with Docker support. Here's how to use it:

```python
import affinetes as af_env
import asyncio

async def main():
    # Load environment from Docker image
    env = af_env.load_env(
        image="bignickeye/agentgym:sciworld-v2",
        env_vars={"CHUTES_API_KEY": "your-api-key"}
    )
    
    # Execute methods
    result = await env.evaluate(
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        task_id=10
    )
    
    print(f"Score: {result['score']}")
    
    # Cleanup
    await env.cleanup()

asyncio.run(main())
```'''

# Replace the SDK section end with the SDK section + Affinetes section
new_content = content.replace(sdk_section_end, sdk_section_end + affinetes_section)

with open("README.md", "w") as f:
    f.write(new_content)

print("Successfully added Affinetes Usage section to README.md")