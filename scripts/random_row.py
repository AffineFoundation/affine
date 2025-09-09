from __future__ import annotations
import affine as af

import asyncio

async def main():

    print (await af.select_rows(limit = 1))


if __name__ == "__main__":
    asyncio.run(main())


