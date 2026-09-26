#!/usr/bin/env python3
"""Build document bundles and generate + verify affine_docqa questions.

aa-gap-fill-plan §4.1. Runs on the datagen box (ENGY + network). Steps:

  1. fetch  — public documents (fetch.py): EDGAR filings of ~400 filers
              (10-K / 20-F / 10-Q / DEF 14A), GOV.UK publications of seven
              types with their HTML attachments, US Federal Register rules
              and proposed rules, Europe PMC CC-BY articles for ~60 queries.
  2. bundle — 2-5 documents of one theme (a filer across years or forms;
              filers of one SIC code in one year; one organisation's
              publications; one docket's proposed + final rule; one query's
              papers), trimmed so the bundle is 60k-95k tokens (chars /
              3.6) — inside our 131k serving window with room for the reply
              and under the slicer's 110k-token prefix cap. Each source is
              pulled to its SOURCE_QUOTA first, then the total is topped up
              from whatever still has supply (2026-09-23: the e1/e2 lists
              capped the run at 75 bundles; the target is >= 600).
              `--plan-only` builds the bundles without ENGY and prints the
              counts, so the supply can be checked before spending.
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
Bundles are asked/verified in --workers threads; every kept question is
appended the moment it passes (a SIGTERM loses nothing) and reruns skip
uids already in the file.

  python -m affine_docqa_v1.generate --epoch 1 --bundles 300 --budget-usd 80
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import random
import sys
import threading
from pathlib import Path

from affine_gen_v1.store import GenTaskStore, gen_uid
from affine_gen_v1.teacher import BudgetExceeded, Spend, TeacherClient

from affine_docqa_v1 import fetch
from affine_docqa_v1.taskset import (PACKAGE_DIR, SOURCE, SYSTEM, answer_of, answers_match, build_prompt,
                                     render_documents)

CHARS_PER_TOKEN = 3.6
BUNDLE_MIN_TOKENS, BUNDLE_MAX_TOKENS = 60_000, 95_000
# Share of the bundle target each source is asked for first; whatever a source
# cannot supply is topped up from the others (EDGAR has the deepest well).
SOURCE_QUOTA = {"edgar": 0.45, "govuk": 0.25, "pmc": 0.20, "fedreg": 0.10}
PMC_QUERIES = [
    # biomedicine
    "CRISPR base editing efficiency", "malaria vaccine efficacy trial", "gut microbiome metabolic syndrome",
    "antibiotic resistance surveillance", "type 2 diabetes GLP-1", "long COVID prevalence",
    "tuberculosis treatment outcomes cohort", "HIV pre-exposure prophylaxis adherence", "sepsis biomarkers mortality",
    "Alzheimer disease amyloid trial", "stroke thrombectomy outcomes", "childhood obesity intervention",
    "influenza vaccine effectiveness season", "dengue incidence climate", "cervical cancer screening HPV",
    "depression digital intervention randomized", "sleep duration cardiovascular risk", "vitamin D supplementation trial",
    "asthma exacerbation air quality", "opioid prescribing trends", "breast cancer immunotherapy response",
    "kidney disease progression cohort", "maternal mortality low income countries", "hearing loss prevalence adults",
    # environment / earth
    "wildfire smoke health outcomes", "coral reef bleaching temperature", "soil carbon sequestration agriculture",
    "air pollution cognitive decline", "microplastics freshwater ecosystems", "urban heat island mortality",
    "glacier mass balance remote sensing", "biodiversity loss land use change", "ocean acidification shellfish",
    "drought crop yield modelling", "flood risk climate adaptation cities", "pollinator decline pesticide",
    # energy / materials / engineering
    "lithium ion battery degradation", "perovskite solar cell stability", "hydrogen electrolysis catalyst",
    "wind turbine fatigue monitoring", "concrete carbon footprint reduction", "grid scale energy storage economics",
    "additive manufacturing metal defects", "electric vehicle charging demand",
    # computing / data
    "machine learning protein structure", "deep learning medical image segmentation", "federated learning privacy healthcare",
    "large language models clinical notes", "reinforcement learning robotics manipulation", "graph neural networks drug discovery",
    "wearable sensors activity recognition", "misinformation social media spread",
    # social / economics / policy
    "minimum wage employment effects", "universal basic income pilot", "remote work productivity pandemic",
    "school closures learning loss", "microfinance poverty outcomes", "carbon pricing emissions evidence",
    "housing affordability policy evaluation", "vaccine hesitancy determinants survey",
]

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


def _bundle(theme: str, docs: list[dict]) -> dict | None:
    """A bundle needs >= 2 distinct documents and, after trimming, either
    BUNDLE_MIN_TOKENS or >= 3 documents (short docs still make a multi-doc
    question)."""
    seen, uniq = set(), []
    for d in docs:
        if d and d["url"] not in seen:
            seen.add(d["url"]); uniq.append(d)
    if len(uniq) < 2:
        return None
    docs = trim_bundle(uniq, BUNDLE_MAX_TOKENS)
    if bundle_tokens(docs) < BUNDLE_MIN_TOKENS and len(docs) < 3:
        return None
    return {"theme": theme, "docs": docs}


def _chunks(items: list, size: int, min_size: int = 2) -> list[list]:
    out = [items[i:i + size] for i in range(0, len(items), size)]
    if len(out) > 1 and len(out[-1]) < min_size:
        out[-2].extend(out.pop())
    return [c for c in out if len(c) >= min_size]


def _company_name(sec_name, fallback: str) -> str:
    """SEC submissions carry the filer name in capitals ("APPLE INC")."""
    name = str(sec_name or fallback)
    return name.title() if name.isupper() else name


def edgar_bundles(client, rng: random.Random, n_companies: int):
    """Per filer: (a) the 3 newest 10-K/20-F, (b) 4 newest 10-Q, (c) the
    older 10-Ks in threes, (d) 10-K + DEF 14A + 10-Q of one year; then
    cross-filer bundles: newest annual report of 2-3 filers sharing a SIC
    code (same industry, same year window)."""
    companies = fetch.edgar_companies(client, top=n_companies)
    rng.shuffle(companies)
    by_sic: dict[str, list[tuple[str, dict]]] = {}
    for _name, cik in companies:
        try:
            sub = fetch.edgar_submissions(client, cik)
        except Exception as e:  # noqa: BLE001
            print(f"edgar {cik}: {e}", file=sys.stderr); continue
        company = _company_name(sub.get("name"), _name)
        idx = fetch.edgar_filing_index(sub, cik)
        annual = [f for f in idx if f["form"] in ("10-K", "20-F")]
        quarterly = [f for f in idx if f["form"] == "10-Q"]
        proxies = [f for f in idx if f["form"] == "DEF 14A"]
        if annual:
            by_sic.setdefault(str(sub.get("sic") or ""), []).append((company, annual[0]))
        plans = []
        if len(annual) >= 2:
            plans.append(("annual", annual[:3]))
        if len(quarterly) >= 3:
            plans.append(("quarterly", quarterly[:4]))
        for i, older in enumerate(_chunks(annual[3:9], 3, 2)):
            plans.append((f"annual-older{i + 1}", older))
        if annual and proxies and quarterly:
            plans.append(("annual+proxy+quarter", [annual[0], proxies[0], quarterly[0]]))
        for tag, filings in plans:
            try:
                docs = [fetch.edgar_fetch(client, company, f) for f in filings]
            except Exception as e:  # noqa: BLE001
                print(f"edgar {company} {tag}: {e}", file=sys.stderr); continue
            b = _bundle(f"edgar/{company}/{tag}", [d for d in docs if d])
            if b:
                yield b
    for sic, rows in by_sic.items():
        if not sic or len(rows) < 2:
            continue
        rng.shuffle(rows)
        for group in _chunks(rows, 3, 2):
            try:
                docs = [fetch.edgar_fetch(client, c, f) for c, f in group]
            except Exception as e:  # noqa: BLE001
                print(f"edgar sic {sic}: {e}", file=sys.stderr); continue
            b = _bundle(f"edgar/sic{sic}/" + "+".join(c for c, _ in group), [d for d in docs if d])
            if b:
                yield b


def govuk_bundles(client, rng: random.Random, pages_per_type: int = 8, page: int = 100):
    """Documents of every publication type (body + HTML attachments), grouped
    by organisation into bundles of 2-5; each type is paged newest-first.
    Measured 2026-09-23: 94 of 210 newest items yield a >= 5k-char document
    (the rest are PDF-only), p50 14k-178k chars by type."""
    types = list(fetch.GOVUK_DOC_TYPES)
    rng.shuffle(types)
    for doc_type in types:
        for p in range(pages_per_type):
            try:
                items = fetch.govuk_search(client, doc_type, count=page, start=p * page)
            except Exception as e:  # noqa: BLE001
                print(f"govuk {doc_type} p{p}: {e}", file=sys.stderr); break
            if not items:
                break
            by_org: dict[str, list[dict]] = {}
            for item in items:
                try:
                    d = fetch.govuk_document(client, item)
                except Exception as e:  # noqa: BLE001
                    print(f"govuk {item.get('link')}: {e}", file=sys.stderr); continue
                if d:
                    by_org.setdefault(d.get("organisation") or "gov", []).append(d)
            for org, ds in by_org.items():
                for chunk in _chunks(ds, 4, 2):
                    b = _bundle(f"govuk/{doc_type}/{org}", chunk)
                    if b:
                        yield b


def fedreg_bundles(client, rng: random.Random, pages_per_agency: int = 2):
    """Rules and proposed rules of one agency; documents sharing a docket
    (proposed + final rule) are bundled together first, the rest by agency."""
    agencies = list(fetch.FEDREG_AGENCIES)
    rng.shuffle(agencies)
    for agency in agencies:
        metas = []
        for doc_type in fetch.FEDREG_TYPES:
            for p in range(1, pages_per_agency + 1):
                try:
                    metas.extend(fetch.fedreg_documents(client, agency, doc_type, page=p))
                except Exception as e:  # noqa: BLE001
                    print(f"fedreg {agency} {doc_type} p{p}: {e}", file=sys.stderr); break
        by_docket: dict[str, list[dict]] = {}
        loose = []
        for m in metas:
            dockets = [d for d in (m.get("docket_ids") or []) if d]
            (by_docket.setdefault(dockets[0], []) if dockets else loose).append(m)
        groups = [ms for ms in by_docket.values() if len(ms) >= 2]
        for ms in by_docket.values():
            if len(ms) < 2:
                loose.extend(ms)
        rng.shuffle(loose)
        groups.extend(_chunks(loose, 3, 2))
        for ms in groups:
            try:
                docs = [fetch.fedreg_fetch(client, m) for m in ms[:4]]
            except Exception as e:  # noqa: BLE001
                print(f"fedreg {agency}: {e}", file=sys.stderr); continue
            tag = ms[0]["docket_ids"][0] if ms[0].get("docket_ids") else "misc"
            b = _bundle(f"fedreg/{agency}/{tag}", [d for d in docs if d])
            if b:
                yield b


def pmc_bundles(client, rng: random.Random, pages_per_query: int = 2):
    queries = list(PMC_QUERIES)
    rng.shuffle(queries)
    for q in queries:
        for p in range(1, pages_per_query + 1):
            try:
                docs = fetch.europepmc_articles(client, q, page_size=25, page=p)
            except Exception as e:  # noqa: BLE001
                print(f"pmc {q} p{p}: {e}", file=sys.stderr); break
            if not docs:
                break
            for chunk in _chunks(docs, 4, 3):
                b = _bundle(f"pmc/{q}", chunk)
                if b:
                    yield b


def make_bundles(client, n_bundles: int, rng: random.Random, n_companies: int = 400,
                 sources: tuple[str, ...] = tuple(SOURCE_QUOTA)) -> list[dict]:
    """Pull each source up to its quota, then top the total up from the
    sources that still have supply. Bundles are distinct by document set;
    fetching is lazy, so nothing beyond `n_bundles` is downloaded."""
    gens = {"edgar": edgar_bundles(client, rng, n_companies), "govuk": govuk_bundles(client, rng),
            "fedreg": fedreg_bundles(client, rng), "pmc": pmc_bundles(client, rng)}
    gens = {k: v for k, v in gens.items() if k in sources}
    quota = {k: int(round(SOURCE_QUOTA[k] * n_bundles)) for k in gens}
    got = {k: 0 for k in gens}
    seen_sets: set[frozenset] = set()
    bundles: list[dict] = []
    exhausted: set[str] = set()

    def pull(source: str) -> bool:
        try:
            b = next(gens[source])
        except StopIteration:
            exhausted.add(source); return False
        key = frozenset(d["url"] for d in b["docs"])
        if key in seen_sets:
            return True
        seen_sets.add(key); bundles.append(b); got[source] += 1
        return True

    for source in gens:
        while got[source] < quota[source] and source not in exhausted and len(bundles) < n_bundles:
            pull(source)
    while len(bundles) < n_bundles and len(exhausted) < len(gens):
        for source in [s for s in gens if s not in exhausted]:
            if len(bundles) >= n_bundles:
                break
            pull(source)
    print("bundles by source: " + ", ".join(f"{k} {v}" for k, v in got.items())
          + (f"; exhausted: {sorted(exhausted)}" if exhausted else ""), file=sys.stderr)
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
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--edgar-companies", type=int, default=400, help="filers taken from the SEC ticker table (market-cap order)")
    ap.add_argument("--sources", default=",".join(SOURCE_QUOTA), help="comma list of edgar,govuk,pmc,fedreg")
    ap.add_argument("--plan-only", action="store_true",
                    help="fetch + bundle only (no ENGY): print counts by source and the token histogram, then exit")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    store = GenTaskStore(SOURCE, PACKAGE_DIR, a.epoch)

    with fetch.client() as c:
        bundles = make_bundles(c, a.bundles, rng, n_companies=a.edgar_companies,
                               sources=tuple(s for s in a.sources.split(",") if s))
    print(f"{len(bundles)} bundles, {a.workers} workers", file=sys.stderr)
    if a.plan_only:
        toks = sorted(bundle_tokens(b["docs"]) for b in bundles)
        by_src: dict[str, int] = {}
        for b in bundles:
            by_src[b["theme"].split("/")[0]] = by_src.get(b["theme"].split("/")[0], 0) + 1
        print(json.dumps({"bundles": len(bundles), "by_source": by_src,
                          "tokens_p10_p50_p90": [toks[len(toks) // 10], toks[len(toks) // 2], toks[9 * len(toks) // 10]] if toks else [],
                          "n_docs_mean": round(sum(len(b["docs"]) for b in bundles) / max(1, len(bundles)), 2)}, indent=1))
        return
    spend = Spend(a.budget_usd)
    teacher = TeacherClient(spend)
    (store.local_dir / "bundles").mkdir(parents=True, exist_ok=True)
    seen = store.existing_uids()
    # A restarted run (same --epoch) must not re-question a bundle that already
    # holds kept questions: the teacher would write near-duplicates under new uids.
    done_bundles = {r.get("bundle") for r in store.tasks()} if seen else set()
    lock = threading.Lock()
    state = {"kept": 0, "rejects": {}, "stop": False}

    def reject(bid: str, why: str) -> None:
        with lock:
            state["rejects"][why] = state["rejects"].get(why, 0) + 1
        store.append_reject({"bundle": bid, "why": why})

    def bundle(b: dict) -> None:
        if state["stop"]:
            return
        docs = b["docs"]
        bid = gen_uid("bundle", a.epoch, "|".join(d["url"] for d in docs)).split("[")[0]
        if bid in done_bundles:
            return
        brng = random.Random(int.from_bytes(hashlib.sha256(f"{a.seed}|{bid}".encode()).digest()[:4], "big"))
        try:
            with gzip.open(store.local_dir / "bundles" / f"{bid}.json.gz", "wt", encoding="utf-8") as f:
                json.dump({"theme": b["theme"], "docs": docs}, f, ensure_ascii=False)
            rendered = render_documents(docs)
            raw = teacher.complete("ask", ASK_SYSTEM, rendered + "\n\nWrite the 3 questions as the JSON object.",
                                   max_tokens=3000, temperature=0.7)
            for q in parse_questions(raw):
                needed = [str(x) for x in (q.get("docs_needed") or [])]
                if len(needed) < 2:
                    reject(bid, "single_doc"); continue
                gold = q["answer"]
                ok = 0
                for _ in range(2):
                    order = docs[:]; brng.shuffle(order)
                    reply = teacher.complete("verify", SYSTEM, build_prompt(order, q["question"]), max_tokens=4000, temperature=0.6)
                    ok += int(answers_match(answer_of(reply), gold, q.get("answer_type", "short_text")))
                if ok < 2:
                    reject(bid, f"blind_{ok}_of_2"); continue
                drop = brng.choice(needed)
                reduced = [d for d in docs if d["id"] != drop]
                if len(reduced) == len(docs):
                    reject(bid, "bad_doc_id"); continue
                reply = teacher.complete("removal", SYSTEM, build_prompt(reduced, q["question"]), max_tokens=4000, temperature=0.6)
                if answers_match(answer_of(reply), gold, q.get("answer_type", "short_text")):
                    reject(bid, "single_doc_suffices"); continue
                rec = {"uid": gen_uid("docqa", a.epoch, bid + q["question"]), "bundle": bid, "theme": b["theme"],
                       "question": q["question"], "answer": gold, "answer_type": q.get("answer_type", "short_text"),
                       "docs_needed": needed, "n_docs": len(docs), "tokens": bundle_tokens(docs)}
                with lock:
                    if rec["uid"] in seen:
                        continue
                    seen.add(rec["uid"])
                    store.append_task(rec)
                    state["kept"] += 1
                    spend.write(store.local_dir / "spend.json")
                    n = state["kept"]
                print(f"kept {n} ({b['theme']}) spend USD {spend.usd:.2f}", file=sys.stderr)
        except BudgetExceeded as e:
            state["stop"] = True
            print(f"STOP: {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - one bad bundle must not kill the run
            reject(bid, f"error_{type(e).__name__}")

    try:
        with cf.ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(bundle, bundles))
    finally:
        spend.write(store.local_dir / "spend.json")
        print(json.dumps({"bundles": len(bundles), "kept_this_run": state["kept"], "kept_total": len(seen),
                          "rejected": sum(state["rejects"].values()), "rejects_by_reason": state["rejects"],
                          "spend": spend.to_dict()}, indent=1))


if __name__ == "__main__":
    main()
