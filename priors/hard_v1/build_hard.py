"""Build the PRIVATE hard-v1 candidate set (CrystalMath-rewritten + INTELLECT-3-RL math) with the build_dapo.py pipeline:
pinned raw -> source strip -> exact normalized dedupe (conflicting answers dropped) -> gradeable-answer filter (live grader
self-test) -> decontam vs live pool + all of DAPO + benchmarks (exact normalized, 13-gram >=3 hits with boilerplate ignored,
short-text containment, fuzzy 0.75 same answer / 0.85 any) -> cross-source (Crystal wins) + within-source near-dup.
usage: build_hard.py RAW_DIR OUT_DIR    (outputs: candidates.jsonl, removed.jsonl, build_report.json; never publish)"""
import hashlib, json, re, sys, time, os
from collections import Counter, defaultdict
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dapo_ref as B
import grader

SRC = {
 "crystal": dict(hf_repo="ycchen/Crystal-Math-Preview", config="rewritten", file="rewritten/train.parquet",
                 revision="03fc77140fc0fe9f9f9fb32b5514ad3183428628",
                 sha256="fb44c4d7fd40b630807b54098f46c9dd0239e0cc4b570636d06195e11eb204a1",
                 license="per-row (card: license other/mixed): Apache-2.0, MIT, CC0-1.0, CC-BY-4.0, CC-BY/CC-BY-SA mixed, CC-BY-NC-4.0 (ODA-Math), CC-BY-NC-SA-4.0 (MathArena)"),
 "i3math": dict(hf_repo="PrimeIntellect/INTELLECT-3-RL", config="math", file="math/train-00000-of-00001.parquet",
                revision="a9eb9183224cec63fe9a71e7a3a80cff14868624",
                sha256="fce6fa97e44fab1129754409448152e0f2b2c36267b6480def654713e94e06b1",
                license="NONE DECLARED (no license field in the dataset card or repo at this revision) - legal status unclear"),
}
REFS = {
 "evals.jsonl (2026-11-01 build: aime2022-2026, aimo_aime22_24, amc23, math500, MATH test 5000, OlympiadBench, Omni-MATH, affine_math)": "pinned in fetch_evals.py",
 "gsm8k": "openai/gsm8k main test+train @740312add88f781978c0658806c59bc2815b9866",
 "olymmath": "RUC-AIBOX/OlymMATH EN/ZH EASY/HARD @2c6532ea2cf929ac1c421532af5951553eaee727",
 "amc_aimo": "AI-MO/aimo-validation-amc (AMC12 2022-2023) @69d78a4a2c840e82d69af6bc742bda09005f6316",
 "arxivmath": "MathArena/arxivmath @f0a9cf83259855c31b37168eca646ec6f0e2598c (= bench pin)",
 "dapo_pool": "live dapo env (11,813) sha256 18d6951e...",
 "dapo_other": "rest of DAPO-Math-17k @65877096 (normalized-unique, not in the live pool)",
}
CRYSTAL_DROP_SOURCES = {"DAPO-17K", "Omni-MATH", "IMO-AnswerBench", "MathArena", "OlymMATH", "BeyondAIME", "AMO-Bench"}
DISTRACT = ["0", "1", "2", "-7", "x", "\\frac{1}{2}"]
WORDY = re.compile(r"(?<![\\A-Za-z])[A-Za-z]{3,}")


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def gradeable(ans):
    a = str(ans).strip()
    if not a or len(a) > 120: return "answer_empty_or_long"
    if WORDY.search(re.sub(r"\\(text|mathrm|operatorname)\{[^{}]*\}", "", a)) and not re.fullmatch(r"[-+]?\d+", a): return "answer_has_words"
    try:
        if grader.grade_text(a, "\\boxed{%s}" % a, "")[0] != "positive": return "answer_not_self_equal"
        for d in DISTRACT:
            if B.norm_answer(d) == B.norm_answer(a): continue
            if grader.grade_text(a, "\\boxed{%s}" % d, "")[0] == "positive": return "answer_grader_degenerate"
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt): raise
        return "answer_grader_error"
    return None


def load_refs(raw):
    E = [json.loads(l) for l in open("/workspace/priors/raw/evals.jsonl")]
    def add(src, i, p, a): E.append(dict(source=src, id="%s:%s" % (src, i), problem=str(p), answer=str(a)))
    for f in ("gsm8k_test", "gsm8k_train"):
        for i, r in enumerate(pd.read_parquet(f"{raw}/{f}.parquet").itertuples()): add(f.replace("_", ""), i, r.question, r.answer.split("####")[-1].strip())
    for f in ("EN-EASY", "EN-HARD", "ZH-EASY", "ZH-HARD"):
        for i, l in enumerate(open(f"{raw}/olymmath_{f}.jsonl")):
            r = json.loads(l); add("olymmath", f"{f}-{i}", r["problem"], r["answer"])
    for r in pd.read_parquet(f"{raw}/amc_aimo.parquet").itertuples(): add("amc_aimo", r.id, r.problem, r.answer)
    for i, r in enumerate(pd.read_parquet(f"{raw}/arxivmath.parquet").itertuples()): add("arxivmath", i, r.problem, r.answer)
    pool = {B.norm_text(json.loads(l)["problem"]) for l in open("/workspace/priors/out/dapo_pool.jsonl")}
    df = pd.read_parquet("/workspace/priors/raw/data/dapo-math-17k.parquet"); seen = set()
    for r in df.itertuples(index=False):
        c = list(r.prompt)[0]["content"]; p = c[len(B.PREFIX):-len(B.SUFFIX)].strip(); k = B.norm_text(p)
        if k in seen: continue
        seen.add(k); add("dapo_pool" if k in pool else "dapo_other", r.extra_info["index"], p, r.reward_model["ground_truth"])
    return E


class Index:
    def __init__(s, E, boiler):
        s.E = E; s.boiler = boiler; s.exact = defaultdict(list); s.ng = defaultdict(set); s.nge = []; s.q = defaultdict(set); s.short = []
        for e in E: s.add(e)
    def add(s, e):
        j = len(s.nge) if hasattr(s, "_n") else None
        e.setdefault("n", B.norm_text(e["problem"])); e.setdefault("t", B.toks(e["problem"])); e.setdefault("a", B.norm_answer(e["answer"]))
        j = s.__dict__.setdefault("_n", 0); s._n += 1
        if j >= len(s.E): s.E.append(e)
        s.exact[e["n"]].append(e["id"])
        g = B.ngrams(e["t"], B.NGRAM, True) - s.boiler; s.nge.append(len(g))
        for h in g: s.ng[h].add(j)
        for h in B.ngrams(e["t"], 4): s.q[h].add(j)
        if len(e["t"]) < B.NGRAM + 2 and len(e["n"]) >= B.CONTAIN_MIN_CHARS: s.short.append(j)
    def match(s, text, ans):
        n = B.norm_text(text); t = B.toks(text)
        if n in s.exact: return ("exact_normalized", s.exact[n][0], 1.0)
        hits = Counter()
        for h in B.ngrams(t, B.NGRAM, True):
            for j in s.ng.get(h, ()): hits[j] += 1
        for j, c in hits.most_common(5):
            if c >= min(B.NGRAM_HITS, s.nge[j]) and s.nge[j] > 0: return ("ngram13_overlap", s.E[j]["id"], c)
        for j in s.short:
            if s.E[j]["n"] in n: return ("contains_short_eval_text", s.E[j]["id"], len(s.E[j]["n"]))
        cand = Counter()
        for h in B.ngrams(t, 4):
            sj = s.q.get(h, ())
            if len(sj) > 50: continue
            for j in sj: cand[j] += 1
        a = B.norm_answer(ans)
        for j, c in cand.most_common(25):
            if c < 3: break
            sm = B.sim(n, s.E[j]["n"])
            if s.E[j]["a"] == a and sm >= B.FUZZ_SAME_ANS: return ("fuzzy_same_answer", s.E[j]["id"], round(sm, 3))
            if sm >= B.FUZZ_ANY: return ("fuzzy_high_similarity", s.E[j]["id"], round(sm, 3))
        return None


def main(raw, out):
    t0 = time.time(); rep = dict(sources={}, stages={}, removed_by_reason={}, contamination={})
    removed = []
    for k, v in SRC.items():
        assert sha(f"{raw}/{'crystal_rewritten' if k == 'crystal' else 'i3_math'}.parquet") == v["sha256"], k
        rep["sources"][k] = v
    rep["decontam_references"] = REFS
    # ---- load
    cr = pd.read_parquet(f"{raw}/crystal_rewritten.parquet"); i3 = pd.read_parquet(f"{raw}/i3_math.parquet")
    items = {"crystal": [], "i3math": []}
    st = {"crystal": dict(raw=len(cr)), "i3math": dict(raw=len(i3))}
    for r in cr.itertuples(index=False):
        if r.source in CRYSTAL_DROP_SOURCES:
            removed.append(dict(src="crystal", src_id=r.id, reason="source_is_benchmark_or_dapo", matched=r.source)); continue
        items["crystal"].append(dict(src="crystal", src_id=r.id, problem=r.problem.strip(), answer=str(int(r.answer)), license=r.license,
                                     crystal_source=r.source, original_problem=r.original_problem if isinstance(r.original_problem, str) else ""))
    st["crystal"]["after_source_strip"] = len(items["crystal"])
    st["i3math"]["after_source_strip"] = len(i3)
    for i, r in enumerate(i3.itertuples(index=False)):
        items["i3math"].append(dict(src="i3math", src_id="i3math:%d" % i, problem=str(r.question).strip(), answer=str(r.answer).strip(),
                                    license="none-declared", ref_p_qwen3_4b_thinking=float(r[2])))
    # ---- exact normalized dedupe within source
    for k in items:
        by = {}; conf = set()
        for x in items[k]:
            n = B.norm_text(x["problem"])
            if n in by:
                if B.norm_answer(by[n]["answer"]) != B.norm_answer(x["answer"]): conf.add(n)
                removed.append(dict(src=k, src_id=x["src_id"], reason="dup_within_source", matched=by[n]["src_id"])); continue
            by[n] = x
        for n in conf:
            x = by.pop(n); removed.append(dict(src=k, src_id=x["src_id"], reason="dedupe_conflicting_answers"))
        items[k] = list(by.values()); st[k]["after_exact_dedupe"] = len(items[k])
    # ---- gradeable
    for k in items:
        keep = []
        for x in items[k]:
            why = gradeable(x["answer"])
            if why: removed.append(dict(src=k, src_id=x["src_id"], reason=why, answer=x["answer"][:80]))
            else: keep.append(x)
        items[k] = keep; st[k]["after_gradeable"] = len(keep)
    # ---- references + boilerplate
    E = load_refs(raw); rep["decontam_reference_counts"] = dict(Counter(e["source"] for e in E))
    for e in E: e["n"] = B.norm_text(e["problem"]); e["t"] = B.toks(e["problem"]); e["a"] = B.norm_answer(e["answer"])
    dfng = Counter()
    for k in items:
        for x in items[k]: dfng.update(B.ngrams(B.toks(x["problem"]), B.NGRAM, True))
    for t in {e["n"]: e["t"] for e in E}.values(): dfng.update(B.ngrams(t, B.NGRAM, True))
    boiler = {h for h, c in dfng.items() if c > B.BOILER_DF}; rep["boilerplate_13grams_ignored"] = len(boiler)
    idx = Index(list(E), boiler)
    # ---- decontam: crystal first (problem AND original problem), then i3 vs refs + kept crystal; kept items join the index
    final = []
    for k in ("crystal", "i3math"):
        keep = []
        for x in items[k]:
            why = idx.match(x["problem"], x["answer"])
            if not why and x.get("original_problem"): why = idx.match(x["original_problem"], "__none__")
            if why:
                src = why[1].split(":")[0]
                reason = "cross_source_dup" if src == "crystal" else ("dup_within_source_near" if src == k else why[0])
                removed.append(dict(src=k, src_id=x["src_id"], reason=reason, method=why[0], matched=why[1], score=why[2], answer=x["answer"][:80]))
                continue
            keep.append(x); idx.add(dict(source=k, id="%s:%s" % (k, x["src_id"]), problem=x["problem"], answer=x["answer"]))
        st[k]["after_decontam_and_dedupe"] = len(keep); final += keep
    # ---- write
    final.sort(key=lambda x: (x["src"], x["src_id"]))
    with open(f"{out}/candidates.jsonl", "w") as f:
        for i, x in enumerate(final):
            x.pop("original_problem", None)
            f.write(json.dumps(dict(id="hard:%d" % i, idx=i, env="hard_v1", uid=hashlib.sha256(B.norm_text(x["problem"]).encode()).hexdigest()[:16], **x),
                               sort_keys=True, ensure_ascii=False) + "\n")
    with open(f"{out}/removed.jsonl", "w") as f:
        for r in removed: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep["stages"] = st
    rep["removed_by_reason"] = {k: dict(Counter(r["reason"] for r in removed if r["src"] == k)) for k in items}
    rep["contamination"] = {k: dict(Counter(r["matched"].split(":")[0] for r in removed if r["src"] == k and r["reason"] not in
                                            ("dup_within_source", "dedupe_conflicting_answers") and "matched" in r and not r["reason"].startswith("answer")))
                            for k in items}
    rep["contamination_by_method"] = {k: dict(Counter(r.get("method", r["reason"]) for r in removed if r["src"] == k and "matched" in r and r["reason"] != "dup_within_source")) for k in items}
    rep["candidates"] = len(final); rep["candidates_sha256"] = sha(f"{out}/candidates.jsonl")
    rep["params"] = dict(ngram=B.NGRAM, ngram_min_words=B.MIN_WORDS, ngram_hits=B.NGRAM_HITS, boilerplate_df=B.BOILER_DF,
                         fuzz_same_answer=B.FUZZ_SAME_ANS, fuzz_any=B.FUZZ_ANY, contain_min_chars=B.CONTAIN_MIN_CHARS,
                         crystal_dropped_sources=sorted(CRYSTAL_DROP_SOURCES), grader="allan-lenient-v1-786cdb9 self-test + distractors %s" % DISTRACT)
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(f"{out}/build_report.json", "w"), indent=1); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
