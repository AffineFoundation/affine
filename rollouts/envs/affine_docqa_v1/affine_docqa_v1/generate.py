#!/usr/bin/env python3
"""Build document bundles and generate + verify affine_docqa questions.

aa-gap-fill-plan §4.1. Runs on the datagen box (ENGY + network). Steps:

  1. fetch  — public documents (fetch.py): EDGAR 10-Ks per company, GOV.UK
              consultation outcomes, Europe PMC CC-BY articles per query.
  2. bundle — 3-6 documents of one theme (same company across years; one
              organisation's consultations; one query's papers), trimmed
              so the bundle is 60k-95k tokens (chars / 3.6) — inside our
              131k serving window with room for the reply and under the
              slicer's 110k-token prefix cap.
  3. ask    — the teacher reads the bundle and writes 3 questions whose
              short exact answer needs >= 2 documents (a difference, a
              ratio, a "doc A says X, what does doc B say").
  4. verify — (a) blind re-answer 2x with the documents in a different
              order: both must match the gold; (b) remove one required
              document: the answer must NOT match (or be "not answerable")
              — proves the multi-document dependency. Keep only tasks that
              pass both.

Budget: --budget-usd (pilot 80 ≈ 300 bundles; full run 350, approved). The
input side dominates (~100k tokens × 4 passes per bundle). Output goes to
data/e<epoch>/ (tasks.jsonl.gz, bundles/<id>.json.gz, spend.json); the
package .gitignores data/ — run publish.py to push it to data.affine.io.

  python -m affine_docqa_v1.generate --epoch 1 --bundles 300 --budget-usd 80
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

from affine_gen_v1.store import GenTaskStore, gen_uid
from affine_gen_v1.teacher import BudgetExceeded, Spend, TeacherClient

from affine_docqa_v1 import fetch
from affine_docqa_v1.taskset import (PACKAGE_DIR, SOURCE, SYSTEM, answer_of, answers_match, build_prompt,
                                     render_documents)

CHARS_PER_TOKEN = 3.6
BUNDLE_MIN_TOKENS, BUNDLE_MAX_TOKENS = 60_000, 95_000
PMC_QUERIES = ["CRISPR base editing efficiency", "malaria vaccine efficacy trial", "gut microbiome metabolic syndrome",
               "lithium ion battery degradation", "wildfire smoke health outcomes", "machine learning protein structure",
               "antibiotic resistance surveillance", "coral reef bleaching temperature", "type 2 diabetes GLP-1",
               "air pollution cognitive decline", "soil carbon sequestration agriculture", "long COVID prevalence"]

ASK_SYSTEM = (
    "You write long-context reasoning questions over a set of documents, for evaluating language models. "
    "Reply with ONE JSON object: {\"questions\": [{\"question\": str, \"answer\": str, \"answer_type\": "
    "\"number\"|\"date\"|\"entity\"|\"short_text\", \"docs_needed\": [document ids], \"derivation\": str}]} with "
    "exactly 3 questions. Each answer must be SHORT and EXACT (a number with unit, a date, a name, or <= 8 words), "
    "must require information from at least TWO different documents (a difference, a ratio, a comparison, a "
    "'document A states X; what does document B state for the same item'), and must not be answerable from any "
    "single document alone. No opinions, no summaries. Use the document ids given in the <document> tags."
)


def trim_bundle(docs: list[dict], max_tokens: int) -> list[dict]:
    budget = int(max_tokens * CHARS_PER_TOKEN)
    per = budget // max(1, len(docs))
    out = []
    for d in docs:
        t = d["text"]
        out.append({**d, "text": t if len(t) <= per else t[:per].rsplit("\n", 1)[0]})
    return out


def bundle_tokens(docs: list[dict]) -> int:
    return int(sum(len(d["text"]) for d in docs) / CHARS_PER_TOKEN)


def make_bundles(client, n_bundles: int, rng: random.Random) -> list[dict]:
    bundles: list[dict] = []
    companies = list(fetch.EDGAR_CIKS.items()); rng.shuffle(companies)
    for company, cik in companies:
        if len(bundles) >= n_bundles * 0.5:
            break
        try:
            docs = fetch.edgar_filings(client, company, cik, limit=3)
        except Exception as e:  # noqa: BLE001
            print(f"edgar {company}: {e}", file=sys.stderr); continue
        if len(docs) >= 2:
            bundles.append({"theme": f"edgar/{company}", "docs": trim_bundle(docs, BUNDLE_MAX_TOKENS)})
    start = 0
    while len(bundles) < n_bundles * 0.8:
        try:
            docs = fetch.govuk_consultations(client, count=40, start=start)
        except Exception as e:  # noqa: BLE001
            print(f"govuk: {e}", file=sys.stderr); break
        if not docs:
            break
        by_org: dict[str, list[dict]] = {}
        for d in docs:
            by_org.setdefault(d.get("organisation") or "gov", []).append(d)
        for org, ds in by_org.items():
            for i in range(0, len(ds) - 1, 3):
                chunk = ds[i:i + 4]
                if len(chunk) >= 2:
                    bundles.append({"theme": f"govuk/{org}", "docs": trim_bundle(chunk, BUNDLE_MAX_TOKENS)})
        start += 40
    for q in PMC_QUERIES:
        if len(bundles) >= n_bundles:
            break
        try:
            docs = fetch.europepmc_articles(client, q, page_size=12)
        except Exception as e:  # noqa: BLE001
            print(f"pmc {q}: {e}", file=sys.stderr); continue
        for i in range(0, len(docs) - 2, 4):
            chunk = docs[i:i + 4]
            if len(chunk) >= 3:
                bundles.append({"theme": f"pmc/{q}", "docs": trim_bundle(chunk, BUNDLE_MAX_TOKENS)})
    bundles = [b for b in bundles if bundle_tokens(b["docs"]) >= BUNDLE_MIN_TOKENS or len(b["docs"]) >= 3]
    rng.shuffle(bundles)
    return bundles[:n_bundles]


def parse_questions(text: str) -> list[dict]:
    text = text.strip()
    i, j = text.find("{"), text.rfind("}")
    try:
        obj = json.loads(text[i:j + 1])
    except Exception:  # noqa: BLE001
        return []
    qs = obj.get("questions") if isinstance(obj, dict) else None
    return [q for q in (qs or []) if isinstance(q, dict) and q.get("question") and q.get("answer")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--bundles", type=int, default=300)
    ap.add_argument("--budget-usd", type=float, default=80.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    store = GenTaskStore(SOURCE, PACKAGE_DIR, a.epoch)
    spend = Spend(a.budget_usd)
    teacher = TeacherClient(spend)

    with fetch.client() as c:
        bundles = make_bundles(c, a.bundles, rng)
    print(f"{len(bundles)} bundles", file=sys.stderr)
    (store.local_dir / "bundles").mkdir(parents=True, exist_ok=True)
    kept, rejects = [], []
    try:
        for b in bundles:
            docs = b["docs"]
            bid = gen_uid("bundle", a.epoch, "|".join(d["url"] for d in docs)).split("[")[0]
            with gzip.open(store.local_dir / "bundles" / f"{bid}.json.gz", "wt", encoding="utf-8") as f:
                json.dump({"theme": b["theme"], "docs": docs}, f, ensure_ascii=False)
            rendered = render_documents(docs)
            raw = teacher.complete("ask", ASK_SYSTEM, rendered + "\n\nWrite the 3 questions as the JSON object.",
                                   max_tokens=3000, temperature=0.7)
            for q in parse_questions(raw):
                needed = [str(x) for x in (q.get("docs_needed") or [])]
                if len(needed) < 2:
                    rejects.append({"bundle": bid, "why": "single_doc"}); continue
                gold = q["answer"]
                # (a) blind re-answer, shuffled order, twice
                ok = 0
                for _ in range(2):
                    order = docs[:]; rng.shuffle(order)
                    reply = teacher.complete("verify", SYSTEM, build_prompt(order, q["question"]), max_tokens=4000, temperature=0.6)
                    ok += int(answers_match(answer_of(reply), gold, q.get("answer_type", "short_text")))
                if ok < 2:
                    rejects.append({"bundle": bid, "why": f"blind_{ok}_of_2"}); continue
                # (b) removal: drop one required document -> must fail
                drop = rng.choice(needed)
                reduced = [d for d in docs if d["id"] != drop]
                if len(reduced) == len(docs):
                    rejects.append({"bundle": bid, "why": "bad_doc_id"}); continue
                reply = teacher.complete("removal", SYSTEM, build_prompt(reduced, q["question"]), max_tokens=4000, temperature=0.6)
                if answers_match(answer_of(reply), gold, q.get("answer_type", "short_text")):
                    rejects.append({"bundle": bid, "why": "single_doc_suffices"}); continue
                kept.append({"uid": gen_uid("docqa", a.epoch, bid + q["question"]), "bundle": bid, "theme": b["theme"],
                             "question": q["question"], "answer": gold, "answer_type": q.get("answer_type", "short_text"),
                             "docs_needed": needed, "n_docs": len(docs), "tokens": bundle_tokens(docs)})
            print(f"kept {len(kept)} after {bid} ({b['theme']}) spend USD {spend.usd:.2f}", file=sys.stderr)
    except BudgetExceeded as e:
        print(f"STOP: {e}", file=sys.stderr)
    finally:
        store.write_tasks(kept)
        with gzip.open(store.local_dir / "rejects.jsonl.gz", "wt") as f:
            for r in rejects:
                f.write(json.dumps(r) + "\n")
        spend.write(store.local_dir / "spend.json")
        print(json.dumps({"bundles": len(bundles), "kept": len(kept), "rejected": len(rejects), "spend": spend.to_dict()}, indent=1))


if __name__ == "__main__":
    main()
