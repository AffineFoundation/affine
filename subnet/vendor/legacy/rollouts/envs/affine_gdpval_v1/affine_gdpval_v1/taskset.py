"""affine-gdpval-v1: OpenAI's GDPval tasks under Artificial Analysis' Stirrup harness.

220 real professional tasks (44 occupations, 9 sectors) with reference files
(spreadsheets, PDFs, decks) and a rubric per task (`openai/gdpval`). Artificial
Analysis runs exactly these as GDPval-AA v2 (10 % of the Intelligence Index)
with its Stirrup harness: one `code_exec` tool, `finish(paths)`, 250 turns,
deliverable FILES judged pairwise.

WHAT THIS ENV IS FOR: the harness probe and the king/teacher control on the
AA agent shape -- does the model emit valid tool calls under Stirrup, does it
call `finish`, what does it submit. It is NOT for D: the tasks are the
benchmark's own. sources.toml registers the source under
[decontamination.affine_gdpval] with `require_gen_marker = true`, so the fold
drops every rollout of it (`decontam_no_gen_marker`). The D-bound env is the
synthesized-variant successor (new occupations / files, `[GEN:...]` uids).

Grade here is protocol-level (the datagen pods have no judge endpoint):
`solved` = the agent called `finish` with at least one submitted file that
exists and is non-empty in the output dir. Metrics: `abandoned`,
`deliverable_type_match` (a submitted file shares an extension with the gold
deliverables), `turns`, `submitted_files`. The rubric travels on the task
data (`rubric_json`) for a judge-graded reward later.

Runtime: Stirrup's local sandbox. Reference files are staged by `setup` under
`/workspace/reference_files/` (the harness uploads that directory into the
sandbox as `<sandbox>/reference_files/...`; the prompt's `{{WORKDIR}}` token
becomes the sandbox path). The prompt follows AA's GDPval-AA task prompt with
the runtime section rewritten for our container.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

import httpx
import verifiers.v1 as vf
from datasets import load_dataset
from filelock import FileLock

DATASET_NAME = "openai/gdpval"
DATASET_REVISION = "11e7900cdcac61bc4daf59e65feb238acda98fbf"
SPLIT = "train"
IMAGE = "python:3.12-slim"

REFERENCE_DIR = "/workspace/reference_files"
OUTPUT_DIR = "/workspace/output"
RESULT_PATH = f"{OUTPUT_DIR}/.stirrup_result.json"
CACHE_DIR = Path(os.environ.get("AFFINE_GDPVAL_CACHE", Path.home() / ".cache" / "affine_gdpval"))

# Artificial Analysis' GDPval-AA task-submission system prompt, verbatim
# (artificialanalysis.ai/methodology/intelligence-benchmarking, 2026-09).
SYSTEM_PROMPT = (
    "You are an AI agent completing a standalone professional task. Your job is to use the "
    "provided tools to produce the requested deliverables within 250 steps, then submit your "
    "work.\n\n"
    "When you are done, call the `finish` tool as your final step with:\n"
    "1. A brief summary of what you accomplished.\n"
    "2. Absolute paths to every deliverable file.\n\n"
    "If you have genuinely concluded that the task cannot be completed because required inputs "
    "are missing, a hard dependency is unavailable, or the request is incoherent, call the "
    "`abandon_task_finish` tool with a brief reason instead. Do not use it to escape "
    "difficulty.\n\n"
    "You cannot interact with the user during the task. Make reasonable assumptions when "
    "needed and record them in your finish summary."
)

# AA's task-submission prompt with the runtime paragraph rewritten for our
# sandbox (a python:3.12 container, packages via uv, sandbox cwd).
PROMPT_TEMPLATE = """## Runtime

You are running in an isolated Linux sandbox. Use the `code_exec` tool to read, create, and modify files. Your working directory is `{workdir}`; write every file you create under it.

Every command runs independently: no working directory, environment variable, or other shell state carries over from one call to the next. Prefer absolute paths for both files and commands, and do not navigate with `cd` across calls — a `cd` in one command is gone by the next. When a step genuinely needs a different directory, chain it into the same command (e.g. `cd {workdir}/work && python build.py`).

Python 3.12 is installed. Install any package you need with `uv pip install --system <package>` (python-docx, python-pptx, openpyxl, pdfplumber, reportlab, pandas, matplotlib are all available that way). Commands are terminated after 10 minutes. Keep them bounded, persist intermediate results to disk, and split long jobs into smaller steps.
{reference_section}
## Completing Your Work

In order to complete the task you must use the `finish` tool to submit your work. If you do not use the `finish` tool you will fail this task!

As a last resort if you really cannot make any meaningful progress, use `abandon_task_finish` with a brief reason instead of submitting files.

**Required in your finish call:**
1. A brief summary of what you accomplished
2. A list of **ABSOLUTE file paths** for the required output files (Do not submit folders).

## Task

Here is the task you need to complete:

{task}

Please begin working on the task now."""

REFERENCE_SECTION = """
## Reference Files Location

The reference files for the task are available in your environment's file system.

Here are their paths:

{paths}
"""


def _reference_name(path: str) -> str:
    # 'reference_files/<hash>/Population v2.xlsx' -> 'Population v2.xlsx'
    return PurePosixPath(unquote(path)).name


def _cached_download(url: str) -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / hashlib.sha256(url.encode()).hexdigest()
    with FileLock(str(cached) + ".lock"):
        if cached.exists():
            return cached.read_bytes()
        headers = {}
        if os.environ.get("HF_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
        resp = httpx.get(url, timeout=600, follow_redirects=True, headers=headers)
        resp.raise_for_status()
        tmp = cached.with_suffix(".tmp")
        tmp.write_bytes(resp.content)
        tmp.rename(cached)
        return resp.content


class GDPvalData(vf.TaskData):
    task_id: str
    sector: str
    occupation: str
    task_text: str
    """The GDPval task description (the prompt above wraps it)."""
    reference_files: list[tuple[str, str]]
    """(file name inside the sandbox's reference_files dir, download url)."""
    deliverable_extensions: list[str]
    """Extensions of the gold deliverables (lower-case, with the dot)."""
    rubric_json: str


class GDPvalTask(vf.Task[GDPvalData]):
    NEEDS_CONTAINER = True

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.run(["mkdir", "-p", REFERENCE_DIR, OUTPUT_DIR], {})
        for name, url in self.data.reference_files:
            await runtime.write(f"{REFERENCE_DIR}/{name}", _cached_download(url))

    async def _result(self, runtime: vf.Runtime) -> dict:
        try:
            raw = await runtime.read(RESULT_PATH)
            return json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            return {"finished": False, "tool": None, "paths": [], "turns": 0}

    async def _submitted_files(self, runtime: vf.Runtime) -> list[str]:
        res = await runtime.run(
            ["sh", "-c", f"find {OUTPUT_DIR} -type f -size +0 ! -name '.stirrup_result.json' 2>/dev/null"], {})
        return [line.strip() for line in (res.stdout or "").splitlines() if line.strip()]

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        result = await self._result(runtime)
        files = await self._submitted_files(runtime)
        exts = {PurePosixPath(f).suffix.lower() for f in files}
        trace.info["stirrup"] = result
        trace.info["submitted_files"] = files
        trace.info["finished"] = bool(result.get("finished")) and result.get("tool") == "finish"
        trace.info["abandoned"] = result.get("tool") == "abandon_task_finish"
        trace.info["type_match"] = bool(exts & set(self.data.deliverable_extensions))

    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        return float(trace.info["finished"] and bool(trace.info["submitted_files"]))

    @vf.metric
    async def abandoned(self, trace: vf.Trace) -> float:
        return float(trace.info["abandoned"])

    @vf.metric
    async def deliverable_type_match(self, trace: vf.Trace) -> float:
        return float(trace.info["type_match"])

    @vf.metric
    async def submitted_files(self, trace: vf.Trace) -> float:
        return float(len(trace.info["submitted_files"]))

    @vf.metric
    async def turns(self, trace: vf.Trace) -> float:
        return float(trace.info["stirrup"].get("turns") or 0)


class GDPvalConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """GDPval `task_id`s to load (empty = all 220)."""


def build_prompt(task_text: str, reference_names: list[str]) -> str:
    section = ""
    if reference_names:
        paths = "\n".join(f"- {{{{WORKDIR}}}}/reference_files/{n}" for n in reference_names)
        section = REFERENCE_SECTION.format(paths=paths)
    return PROMPT_TEMPLATE.format(workdir="{{WORKDIR}}", reference_section=section, task=task_text)


class GDPvalTaskset(vf.Taskset[GDPvalTask, GDPvalConfig]):
    def load(self) -> list[GDPvalTask]:
        want = set(self.config.tasks)
        rows = load_dataset(DATASET_NAME, split=SPLIT, revision=DATASET_REVISION)
        tasks: list[GDPvalTask] = []
        for i, row in enumerate(rows):
            tid = str(row["task_id"])
            if want and tid not in want:
                continue
            refs = [(_reference_name(p), u)
                    for p, u in zip(row.get("reference_files") or [], row.get("reference_file_urls") or [])]
            exts = sorted({PurePosixPath(unquote(p)).suffix.lower()
                           for p in (row.get("deliverable_files") or []) if PurePosixPath(p).suffix})
            tasks.append(GDPvalTask(
                GDPvalData(
                    idx=i,
                    name=tid,
                    image=IMAGE,
                    system_prompt=SYSTEM_PROMPT,
                    prompt=build_prompt(row["prompt"], [n for n, _ in refs]),
                    task_id=tid,
                    sector=str(row.get("sector") or ""),
                    occupation=str(row.get("occupation") or ""),
                    task_text=row["prompt"],
                    reference_files=refs,
                    deliverable_extensions=exts,
                    rubric_json=str(row.get("rubric_json") or ""),
                ),
                self.config.task,
            ))
        if want and not tasks:
            raise ValueError(f"no GDPval task matched {sorted(want)[:5]}...")
        return tasks
