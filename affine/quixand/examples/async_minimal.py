import asyncio
from affine import quixand as qs


async def main():
	sbx = await qs.AsyncSandbox.create(template="python:3.11-slim")
	res = await sbx.run(["python", "-c", "print(42)"])
	print(res.text)
	await sbx.shutdown()


if __name__ == "__main__":
	asyncio.run(main())


