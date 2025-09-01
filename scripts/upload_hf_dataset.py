#!/usr/bin/env python3
from __future__ import annotations
import os
import asyncio
import click
import datasets as hf_ds
from typing import List, Dict

# Reuse affine's DB engine and table definitions
import affine as af
from affine.database import _get_engine as _get_engine, _sm as _sm, dataset_rows as dataset_rows, BATCH_SIZE as DEFAULT_BATCH
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError
import asyncpg


async def _upload_dataset(dataset_name: str, config: str, split: str, batch_size: int) -> int:
    await _get_engine()
    sm = _sm()

    ds = hf_ds.load_dataset(dataset_name, name=None if config == "default" else config, split=split)
    total_rows = len(ds)

    async with sm() as session:
        def _make_stmt(rows: List[Dict]):
            values = [
                {
                    "dataset_name": dataset_name,
                    "config": config,
                    "split": split,
                    "row_index": r["__row_index__"],
                    "data": {k: v for k, v in r.items() if k != "__row_index__"},
                }
                for r in rows
            ]
            stmt = pg_insert(dataset_rows).values(values)
            stmt = stmt.on_conflict_do_nothing(index_elements=[
                dataset_rows.c.dataset_name,
                dataset_rows.c.config,
                dataset_rows.c.split,
                dataset_rows.c.row_index,
            ])
            return stmt

        async def _execute_batch_with_retries(rows: List[Dict], *, max_retries: int = 5) -> None:
            delay = 0.5
            attempt = 0
            while True:
                try:
                    await session.execute(_make_stmt(rows))
                    await session.commit()
                    return
                except (DBAPIError, asyncpg.ConnectionDoesNotExistError) as e:
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                    is_invalidated = isinstance(e, DBAPIError) and getattr(e, "connection_invalidated", False)
                    msg = str(getattr(e, "orig", e)).lower()
                    is_disconnect = isinstance(e, asyncpg.ConnectionDoesNotExistError) or "connection was closed" in msg
                    retriable = is_invalidated or is_disconnect
                    attempt += 1
                    if not retriable or attempt >= max_retries:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 5.0)

        batch: List[Dict] = []
        idx = 0
        for row in ds:  # type: ignore
            payload = dict(row)
            payload["__row_index__"] = idx
            batch.append(payload)
            idx += 1
            if len(batch) >= batch_size:
                await _execute_batch_with_retries(batch)
                batch.clear()
            # lightweight progress
            af.logger.info(f"Progress: {idx}/{total_rows} rows uploaded")

        if batch:
            await _execute_batch_with_retries(batch)
            af.logger.info(f"Progress: {idx}/{total_rows} rows uploaded")

    return idx


@click.command()
@click.argument("dataset_name", type=str)
@click.option("--df_config", "config", type=str, default="default", help="Dataset config (HF)")
@click.option("--split", "split", type=str, default="train", help="Dataset split (HF)")
@click.option("--batch-size", "batch_size", type=int, default=int(os.getenv("BATCH_SIZE", DEFAULT_BATCH)), show_default=True)
def main(dataset_name: str, config: str, split: str, batch_size: int) -> None:
    """Upload a Hugging Face dataset's rows into Postgres `dataset_rows`.

    Example:
      scripts/upload_hf_dataset.py satpalsr/rl-python --config default --split train
    """
    af.setup_logging(1)

    async def _run():
        uploaded = await _upload_dataset(dataset_name, config, split, batch_size)
        af.logger.info(f"Uploaded {uploaded} rows for {dataset_name} [{config}/{split}] to dataset_rows")

    asyncio.run(_run())


if __name__ == "__main__":
    main()


