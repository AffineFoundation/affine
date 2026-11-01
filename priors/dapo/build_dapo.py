"""Build the DAPO-Math-17K pool for the SN120 prior-table sampler and the prior-v1 gate pool (dapo env).
raw parquet (pinned rev) -> extract problem/answer -> exact dedupe -> answer filter -> decontaminate -> pinned JSONL.
Outputs: OUT/dapo_pool.jsonl, OUT/dapo_removed.jsonl, OUT/dapo_build_report.json (the public gate pool
miner/data/gate-pool-v1.jsonl carries dapo_pool.jsonl's id/problem/answer/src_index/uid, see priors/gate/build_public_pool.py)
usage: python build_dapo.py RAW_PARQUET EVALS_JSONL OUT_DIR"""
import hashlib, json, re, sys, unicodedata, time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import pandas as pd

HF_REPO = "BytedTsinghua-SIA/DAPO-Math-17k"; HF_REV = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
PARQUET_SHA256 = "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941"
PREFIX = ("Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer "
          "(without quotes) where $Answer is the answer to the problem.\n\n")
SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
NGRAM = 13; BOILER_DF = 4; MIN_WORDS = 7; NGRAM_HITS = 3           # 13-gram in > BOILER_DF docs (DAPO+refs) = boilerplate
FUZZ_SAME_ANS = 0.75; FUZZ_ANY = 0.85; CONTAIN_MIN_CHARS = 40


def norm_text(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\\(?:dfrac|tfrac)", r"\\frac", s)
    s = re.sub(r"\\(?:left|right|displaystyle|,|;|!|quad|qquad)", " ", s)
    s = re.sub(r"\$+|\\\(|\\\)|\\\[|\\\]", " ", s)
    s = re.sub(r"[^a-z0-9\\{}^_+\-*/=<>().,]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s


def toks(s):
    """word tokens (letters / digit runs); latex punctuation and sub/superscript marks are dropped, so notation-only runs
    ("a_1, a_2, a_3, ldots") collapse instead of forming 13-grams."""
    return re.findall(r"[a-z]+|\d+", norm_text(s))


def wordy(gram):
    """a 13-gram counts only if it carries >= MIN_WORDS real words (alphabetic, >= 3 letters): blocks pure-notation matches."""
    return sum(1 for w in gram if len(w) >= 3 and w.isalpha()) >= MIN_WORDS


def ngrams(t, n, need_words=False):
    return {hash(" ".join(t[i:i + n])) for i in range(len(t) - n + 1) if not need_words or wordy(t[i:i + n])}


def norm_answer(a):
    a = str(a).strip().strip("$").replace(" ", "").replace("\\!", "").replace("{,}", "")
    a = re.sub(r"^\\boxed\{(.*)\}$", r"\1", a); a = re.sub(r"\\text\{([^{}]*)\}", r"\1", a)
    m = re.fullmatch(r"([-+]?\d+)(\.0+)?", a.replace(",", ""))
    if m: return str(int(m.group(1)))
    return a


def sim(a, b):
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def main(raw, evals_path, out):
    t0 = time.time()
    assert hashlib.sha256(open(raw, "rb").read()).hexdigest() == PARQUET_SHA256, "parquet sha256 != pinned"
    df = pd.read_parquet(raw); rep = dict(hf_repo=HF_REPO, hf_revision=HF_REV, parquet_sha256=PARQUET_SHA256, raw_rows=len(df))
    removed = []
    # 1) extract + exact dedupe (normalized problem text; identical answer required, conflicting answers dropped)
    by_norm = {}; bad_template = 0; conflicts = set()
    for r in df.itertuples(index=False):
        msgs = list(r.prompt); c = msgs[0]["content"]
        if len(msgs) != 1 or not c.startswith(PREFIX) or not c.endswith(SUFFIX): bad_template += 1; continue
        prob = c[len(PREFIX):-len(SUFFIX)].strip(); ans = str(r.reward_model["ground_truth"]).strip()
        k = norm_text(prob)
        if k in by_norm:
            if by_norm[k]["answer"] != ans: conflicts.add(k)
            by_norm[k]["dup_rows"] += 1; continue
        by_norm[k] = dict(problem=prob, answer=ans, src_index=r.extra_info["index"], dup_rows=1)
    rep["unique_src_index"] = int(df["extra_info"].map(lambda x: x["index"]).nunique())
    rep["bad_template_rows"] = bad_template; rep["deduped"] = len(by_norm)
    for k in conflicts:
        x = by_norm.pop(k); removed.append(dict(src_index=x["src_index"], reason="dedupe_conflicting_answers", answer=x["answer"]))
    rep["dedupe_conflicting_answer_groups"] = len(conflicts)
    items = list(by_norm.values())
    # 2) answer filter: clean integer answers only (the grader parses them exactly; see test_grader_dapo.py)
    keep = []
    for x in items:
        if re.fullmatch(r"-?\d{1,12}", x["answer"]): keep.append(x)
        else: removed.append(dict(src_index=x["src_index"], reason="answer_not_clean_integer", answer=x["answer"]))
    rep["after_answer_filter"] = len(keep); items = keep
    # 3) decontaminate
    E = [json.loads(l) for l in open(evals_path)]
    rep["decontam_reference_counts"] = dict(Counter(e["source"] for e in E))
    for e in E: e["n"] = norm_text(e["problem"]); e["t"] = toks(e["problem"]); e["a"] = norm_answer(e["answer"])
    exact = defaultdict(list)
    for e in E: exact[e["n"]].append(e["id"])
    D_toks = [toks(x["problem"]) for x in items]
    # boilerplate = 13-grams frequent across DAPO + every reference set (unique normalized docs), e.g.
    # "can be written as frac m n where m and n are relatively prime positive integers find"
    df_ng = Counter()
    for t in D_toks: df_ng.update(ngrams(t, NGRAM, True))
    for k in {e["n"]: e["t"] for e in E}.values(): df_ng.update(ngrams(k, NGRAM, True))
    boiler = {h for h, c in df_ng.items() if c > BOILER_DF}
    rep["boilerplate_13grams_ignored"] = len(boiler)
    ng_idx = defaultdict(set); n_grams_e = {}
    for j, e in enumerate(E):
        g = ngrams(e["t"], NGRAM, True) - boiler; n_grams_e[j] = len(g)
        for h in g: ng_idx[h].add(j)
    # candidate index for fuzzy: rare word 4-grams
    q_idx = defaultdict(set)
    for j, e in enumerate(E):
        for h in ngrams(e["t"], 4): q_idx[h].add(j)
    q_big = {h for h, s in q_idx.items() if len(s) > 50}
    short = [j for j, e in enumerate(E) if len(e["t"]) < NGRAM + 2 and len(e["n"]) >= CONTAIN_MIN_CHARS]
    keep = []
    for x, t in zip(items, D_toks):
        n = norm_text(x["problem"]); why = None
        if n in exact: why = ("exact_normalized", exact[n][0], 1.0)
        if not why:
            hits = Counter()
            for h in ngrams(t, NGRAM, True):
                for j in ng_idx.get(h, ()): hits[j] += 1
            # flag when >= min(NGRAM_HITS, eval's own gram count) distinct non-boilerplate 13-grams are shared
            for j, c in hits.most_common(5):
                if c >= min(NGRAM_HITS, n_grams_e[j]): why = ("ngram13_overlap", E[j]["id"], c); break
        if not why:
            for j in short:
                if E[j]["n"] in n: why = ("contains_short_eval_text", E[j]["id"], len(E[j]["n"])); break
        if not why:
            cand = Counter()
            for h in ngrams(t, 4) - q_big:
                for j in q_idx.get(h, ()): cand[j] += 1
            a = norm_answer(x["answer"])
            for j, c in cand.most_common(25):
                if c < 3: break
                s = sim(n, E[j]["n"])
                if E[j]["a"] == a and s >= FUZZ_SAME_ANS: why = ("fuzzy_same_answer", E[j]["id"], round(s, 3)); break
                if s >= FUZZ_ANY: why = ("fuzzy_high_similarity", E[j]["id"], round(s, 3)); break
        if why:
            removed.append(dict(src_index=x["src_index"], reason=why[0], matched=why[1], score=why[2], answer=x["answer"],
                                problem_head=x["problem"][:160]))
        else: keep.append(x)
    rep["after_decontam"] = len(keep)
    # 4) stable ids + pinned JSONL (sorted by source uuid -> stable order; id = dapo:<position>, uid = sha256 of normalized text)
    keep.sort(key=lambda x: x["src_index"])
    rows = []
    for i, x in enumerate(keep):
        rows.append(dict(id=f"dapo:{i}", idx=i, env="dapo", src_index=x["src_index"],
                         uid=hashlib.sha256(norm_text(x["problem"]).encode()).hexdigest()[:16],
                         problem=x["problem"], answer=x["answer"], dup_rows=x["dup_rows"]))
    data = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode()
    open(f"{out}/dapo_pool.jsonl", "wb").write(data)
    with open(f"{out}/dapo_removed.jsonl", "w") as f:
        for r in removed: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep.update(final=len(rows), pool_sha256=hashlib.sha256(data).hexdigest(), removed_by_reason=dict(Counter(r["reason"] for r in removed)),
               decontam_removed_by_source=dict(Counter(r["matched"].split(":")[0] for r in removed if "matched" in r)),
               params=dict(ngram=NGRAM, ngram_min_words=MIN_WORDS, ngram_hits=NGRAM_HITS, boilerplate_df=BOILER_DF, fuzz_same_answer=FUZZ_SAME_ANS, fuzz_any=FUZZ_ANY,
                           contain_min_chars=CONTAIN_MIN_CHARS), seconds=round(time.time() - t0, 1))
    json.dump(rep, open(f"{out}/dapo_build_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
