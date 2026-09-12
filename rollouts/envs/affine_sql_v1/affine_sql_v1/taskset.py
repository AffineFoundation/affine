"""affine-sql-v1: text-to-SQL over Spider train, the database in a sqlite sandbox.

Data engineering was an empty category of D (inventory §4). Spider
(`xlangai/spider`, train split, 7,000 questions over 146 databases,
CC-BY-SA-4.0) is the standard text-to-SQL train set; its sqlite databases
ship as `spider_data.zip` on `HAL-9001/spider-databases`. The dev split
(1,034) stays out of D as a held-out scorecard.

Shape (any shell harness): the task database is copied into the container
at `/workspace/<db_id>.sqlite`; the prompt carries the schema (`CREATE
TABLE` statements read from the database) and the question; the agent may
explore with the `sqlite3` CLI (installed at setup; the image is a plain
`python:3.12-slim`); its FINAL reply must contain exactly one ```sql block.
Reward `correct` (fold key): gold and predicted SQL run on a fresh host-side
copy of the database and the result sets are compared - as ordered lists
when the gold query has an ORDER BY, as multisets otherwise; a predicted
query that errors or times out scores 0.

Task identity: `name = f"spider-{db_id}-{sha256(db_id\\nquestion\\nquery)[:10]}"`
(rollouts/catalog.py `_spider_meta` computes the same string).
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import verifiers.v1 as vf
from datasets import load_dataset
from huggingface_hub import hf_hub_download

DATASET = "xlangai/spider"
SPLIT = "train"
DB_REPO = "HAL-9001/spider-databases"
DB_ZIP = "spider_data.zip"
IMAGE = "python:3.12-slim"
WORKDIR = "/workspace"
QUERY_TIMEOUT_S = 30
MAX_ROWS = 100_000

SYSTEM = (
    "You are a data engineer answering questions with SQL on a SQLite "
    "database. The database file is in the sandbox; inspect the schema and "
    "sample rows with the `sqlite3` command line tool (e.g. `sqlite3 "
    f"{WORKDIR}/DB.sqlite '.tables'`) and run candidate queries to check "
    "them. Your final reply must contain exactly one ```sql code block with "
    "the single SELECT query that answers the question; it is graded by "
    "running it and comparing the result set with the reference answer."
)
SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SETUP = (
    "command -v sqlite3 >/dev/null 2>&1 || "
    "(apt-get update -qq >/dev/null && apt-get install -y -qq --no-install-recommends "
    "sqlite3 >/dev/null 2>&1 && rm -rf /var/lib/apt/lists/*)"
)


def task_name(db_id: str, question: str, query: str) -> str:
    digest = hashlib.sha256(f"{db_id}\n{question}\n{query}".encode("utf-8")).hexdigest()
    return f"spider-{db_id}-{digest[:10]}"


def spider_db_root() -> Path:
    """Extract spider_data.zip once (~1 GB unpacked) next to the HF cache."""
    root = Path(os.environ.get("SPIDER_DB_ROOT", "~/.cache/affine/spider")).expanduser()
    marker = root / ".extracted"
    if marker.exists():
        return root
    zip_path = hf_hub_download(DB_REPO, DB_ZIP, repo_type="dataset")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if "/database/" in m and m.endswith(".sqlite")]
        zf.extractall(root, members=members)
    marker.write_text("ok")
    return root


def find_db(root: Path, db_id: str) -> Path | None:
    hits = list(root.glob(f"*/database/{db_id}/{db_id}.sqlite"))
    return hits[0] if hits else None


def schema_of(db_path: Path) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    finally:
        con.close()
    return "\n\n".join(r[0].strip() + ";" for r in rows)


def last_sql_block(text: str) -> str | None:
    blocks = SQL_BLOCK_RE.findall(text or "")
    return blocks[-1].strip().rstrip(";").strip() if blocks else None


def run_query(db_path: Path, sql: str) -> list[tuple]:
    """Run one read-only query on a scratch copy of the database."""
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "db.sqlite"
        scratch.write_bytes(db_path.read_bytes())
        con = sqlite3.connect(str(scratch), timeout=QUERY_TIMEOUT_S)
        deadline_steps = [0]

        def progress() -> int:
            deadline_steps[0] += 1
            return 1 if deadline_steps[0] > 20_000 else 0   # ~30 s of VM steps

        con.set_progress_handler(progress, 100_000)
        try:
            cur = con.execute(sql)
            return [tuple(_norm(v) for v in row) for row in cur.fetchmany(MAX_ROWS)]
        finally:
            con.close()


def _norm(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, bytes):
        return value.hex()
    return value


def results_match(gold: list[tuple], pred: list[tuple], ordered: bool) -> bool:
    if len(gold) != len(pred):
        return False
    if ordered:
        return gold == pred
    return sorted(map(repr, gold)) == sorted(map(repr, pred))


class SqlData(vf.TaskData):
    db_id: str
    question: str
    gold_sql: str
    db_path: str
    ordered: bool


class SqlTask(vf.Task[SqlData]):
    NEEDS_CONTAINER = True

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.run(["mkdir", "-p", WORKDIR], {})
        await runtime.write(f"{WORKDIR}/{self.data.db_id}.sqlite",
                            Path(self.data.db_path).read_bytes())
        result = await runtime.run(["sh", "-c", SETUP], {})
        if result.exit_code != 0:
            raise RuntimeError(f"sqlite3 install failed: {(result.stderr or '')[-300:]}")

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        pred = last_sql_block(trace.last_reply or "")
        trace.info["predicted_sql"] = pred
        if not pred or not pred.lower().lstrip("( ").startswith(("select", "with")):
            return 0.0
        db_path = Path(self.data.db_path)
        try:
            gold_rows = run_query(db_path, self.data.gold_sql)
            pred_rows = run_query(db_path, pred)
        except sqlite3.Error as exc:
            trace.info["sql_error"] = str(exc)[:300]
            return 0.0
        return 1.0 if results_match(gold_rows, pred_rows, self.data.ordered) else 0.0

    async def validate(self, runtime: vf.Runtime) -> bool:
        try:
            run_query(Path(self.data.db_path), self.data.gold_sql)
        except sqlite3.Error:
            return False
        return True


class SqlConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole train split)."""
    docker_image: str = IMAGE


class SqlTaskset(vf.Taskset[SqlTask, SqlConfig]):
    def load(self) -> list[SqlTask]:
        want = set(self.config.tasks)
        root = spider_db_root()
        rows = load_dataset(DATASET, split=SPLIT)
        schemas: dict[str, str] = {}
        tasks: list[SqlTask] = []
        for i, row in enumerate(rows):
            db_id, question, query = row["db_id"], row["question"], row["query"]
            name = task_name(db_id, question, query)
            if want and name not in want:
                continue
            db_path = find_db(root, db_id)
            if db_path is None:
                continue
            if db_id not in schemas:
                schemas[db_id] = schema_of(db_path)
            prompt = (
                f"Database: `{WORKDIR}/{db_id}.sqlite` (SQLite)\n\nSchema:\n"
                f"```sql\n{schemas[db_id]}\n```\n\nQuestion: {question}\n\n"
                "Write one SQLite query that answers the question. Explore the "
                "data first if the schema leaves room for doubt, then give the "
                "final query in a single ```sql block."
            )
            tasks.append(SqlTask(
                SqlData(
                    idx=i,
                    name=name,
                    system_prompt=SYSTEM,
                    prompt=prompt,
                    image=self.config.docker_image,
                    workdir=WORKDIR,
                    resources=vf.TaskResources(cpu=1, memory=2, disk=5),
                    db_id=db_id,
                    question=question,
                    gold_sql=query,
                    db_path=str(db_path),
                    ordered="order by" in query.lower(),
                ),
                self.config.task,
            ))
        return tasks
