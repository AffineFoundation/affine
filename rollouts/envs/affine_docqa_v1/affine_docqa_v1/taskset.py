"""affine-docqa-v1: one question over a bundle of real public documents (60-95k tokens), `text` dialect.

aa-gap-fill-plan §4.1 (Jacob "go", 2026-09-22 15:24 UTC). AA-LCR (5 % of the
Index) puts ~100k tokens of company reports, government consultations and
papers in the context and asks one question whose answer combines several
documents; an equality checker grades a short answer. D had retrieval
(MRCR) and aggregation (Oolong) over synthetic text, no multi-document
reasoning over real documents.

Tasks are TEACHER-GENERATED and verified (`generate.py`): questions need
>= 2 documents (a removal test proves it), the teacher re-answers them
blind with the documents shuffled. Documents: SEC EDGAR filings, GOV.UK
consultation outcomes (OGL v3), Europe PMC CC-BY articles (`fetch.py`).
Uids carry `[GEN:e<epoch>]` (fold decontamination). Bundles are published
to data.affine.io (`publish.py`); the taskset fetches them on first use.

Shape = AA's: `null` harness, the whole bundle inline in the user turn, one
visible reply ending in `Answer: <value>`; `solved` = normalised exact
match (numbers with 1 % tolerance, dates by ISO day). Prefixes are 60-95k
tokens — under the slicer's 110k cap and the 131k serving window.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import verifiers.v1 as vf

from affine_gen_v1.store import GenTaskStore

SOURCE = "affine_docqa"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_EPOCH = 1

SYSTEM = (
    "You answer one question from a set of long documents given in full. Read what is needed, reason "
    "carefully, then finish your reply with a single final line of the form `Answer: <value>` — a number "
    "with its unit, a date, a name, or at most eight words. If the documents do not contain the information, "
    "the final line is `Answer: not answerable`."
)

ANSWER_RE = re.compile(r"Answer:\s*(.+?)\s*$", re.I | re.M)
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}


def render_documents(docs: list[dict]) -> str:
    parts = []
    for d in docs:
        parts.append(f"<document id=\"{d['id']}\" title=\"{d['title']}\" source=\"{d['source']}\" date=\"{d.get('date', '')}\">\n"
                     f"{d['text']}\n</document>")
    return "\n\n".join(parts)


def build_prompt(docs: list[dict], question: str) -> str:
    return (f"{render_documents(docs)}\n\n## Question\n\n{question}\n\n"
            "Use the documents above only. End with one line `Answer: <value>`.")


def answer_of(reply: str) -> str | None:
    m = ANSWER_RE.findall(reply or "")
    return m[-1].strip().rstrip(".") if m else None


def _norm(s: str) -> str:
    s = s.lower().strip().strip("\"'`*")
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return re.sub(r"[^a-z0-9%.\- ]", "", s).strip()


def _num(s: str) -> float | None:
    m = NUM_RE.search(s.replace(" ", ""))
    if not m:
        return None
    try:
        v = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    low = s.lower()
    for word, mult in (("billion", 1e9), ("bn", 1e9), ("million", 1e6), ("mn", 1e6), ("thousand", 1e3), ("k", 1e3)):
        if re.search(rf"\d\s*{word}\b", low):
            v *= mult; break
    return v


def _date(s: str) -> date | None:
    s = s.strip()
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s) or re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", s)
    if m:
        g = m.groups()
        day, mon, year = (g[0], g[1], g[2]) if g[0].isdigit() else (g[1], g[0], g[2])
        if mon.lower() in MONTHS:
            try:
                return date(int(year), MONTHS[mon.lower()], int(day))
            except ValueError:
                return None
    return None


def answers_match(pred: str | None, gold: str, answer_type: str = "short_text", rel_tol: float = 0.01) -> bool:
    if pred is None:
        return False
    if _norm(pred) == _norm(gold):
        return True
    if answer_type == "date" or (_date(gold) is not None and _date(pred) is not None):
        return _date(pred) is not None and _date(pred) == _date(gold)
    if answer_type == "number" or (_num(gold) is not None and _num(pred) is not None and answer_type != "entity"):
        a, b = _num(pred), _num(gold)
        return a is not None and b is not None and abs(a - b) <= rel_tol * max(1.0, abs(b))
    return _norm(gold) in _norm(pred) and len(_norm(gold)) >= 4


class DocQAData(vf.TaskData):
    uid: str
    theme: str
    answer: str
    answer_type: str
    bundle: str
    n_docs: int
    tokens: int


class DocQATask(vf.Task[DocQAData]):
    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        return float(answers_match(answer_of(trace.last_reply), self.data.answer, self.data.answer_type))

    @vf.metric
    async def answered(self, trace: vf.Trace) -> float:
        return float(answer_of(trace.last_reply) is not None)


class DocQAConfig(vf.TasksetConfig):
    tasks: list[str] = []
    epoch: int = DEFAULT_EPOCH


def list_catalog(epoch: int = DEFAULT_EPOCH) -> list[dict]:
    store = GenTaskStore(SOURCE, PACKAGE_DIR, epoch)
    return [{"uid": r["uid"], "domain": r["theme"].split("/")[0], "topic": r["theme"], "tokens": r.get("tokens", 0)}
            for r in store.tasks()]


class DocQATaskset(vf.Taskset[DocQATask, DocQAConfig]):
    def load(self) -> list[DocQATask]:
        want = set(self.config.tasks)
        store = GenTaskStore(SOURCE, PACKAGE_DIR, self.config.epoch)
        tasks: list[DocQATask] = []
        for i, rec in enumerate(store.tasks()):
            if want and rec["uid"] not in want:
                continue
            bundle = store.read_json(f"bundles/{rec['bundle']}.json.gz")
            tasks.append(DocQATask(
                DocQAData(
                    idx=i, name=rec["uid"], system_prompt=SYSTEM, prompt=build_prompt(bundle["docs"], rec["question"]),
                    uid=rec["uid"], theme=rec["theme"], answer=rec["answer"], answer_type=rec.get("answer_type", "short_text"),
                    bundle=rec["bundle"], n_docs=int(rec.get("n_docs") or len(bundle["docs"])), tokens=int(rec.get("tokens") or 0),
                ),
                self.config.task,
            ))
        if want and not tasks:
            raise ValueError(f"no docqa task matched {sorted(want)[:5]}...")
        return tasks
