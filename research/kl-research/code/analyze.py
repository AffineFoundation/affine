# Run on box: python analyze.py  (reads /workspace/kl-research/sync/{runs,evals,pool_stats})
import json, glob, os, re, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/workspace/kl-research"; S = f"{B}/sync"; C = f"{B}/charts"; os.makedirs(C, exist_ok=True)
stats = [json.load(open(f)) for f in glob.glob(f"{S}/pool_stats/shard*.stats.json")]
TEACHER_TPS = sum(s["ntok"] for s in stats) / sum(s["sec"] for s in stats) if stats else 5500.0  # teacher sampled tokens / GPU-sec
runs = {}
for f in sorted(glob.glob(f"{S}/runs/*/log.jsonl")):
    name = f.split("/")[-2]; R, H = [], []
    for l in open(f):
        d = json.loads(l)
        if d["kind"] == "round": R.append(d)
        elif d["kind"] == "heldout": H.append(d)
    runs[name] = (R, H)
evals = {}
for f in glob.glob(f"{S}/evals/*.json"):
    if f.endswith(".samples.json"): continue
    n = os.path.basename(f)[:-5]
    try: d = json.load(open(f))
    except: continue
    if "__r" in n:
        arm, r = n.split("__r"); evals.setdefault(arm, {})[int(r)] = d
    else: evals[n] = d
def gpu_h(R, upto):
    """GPU-hours to reach round `upto`: student train + miner scoring + teacher sampling of all candidates drawn."""
    if upto == 0: return 0.0
    d = R[upto - 1]
    return (d["train_sec"] + d["score_sec"] + d["cand_tokens"] / TEACHER_TPS) / 3600
def tok_at(R, r): return 0 if r == 0 else R[r - 1]["tokens"]
PH1 = ["fixed", "random", "sel", "selraw", "full", "adv"]
def arms(prefixes): return [a for a in prefixes if a in runs]
def bench_avg(d): return np.mean([d[k]["acc"] for k in ["gsm8k", "math", "mmlu", "humaneval"]])
def plot_heldout(names, tag):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a in names:
        R, H = runs[a]
        x = [h["tokens"] / 1e6 for h in H]; y = [h["topkkl"] for h in H]
        xg = [gpu_h(R, h["round"]) for h in H]
        ax[0].plot(x, y, label=a); ax[1].plot(xg, y, label=a)
    ax[0].set_xlabel("training tokens (M)"); ax[1].set_xlabel("GPU-hours (train + miner scoring + teacher sampling)")
    for x_ in ax: x_.set_ylabel("held-out top-20 KL(teacher||student), nats/token"); x_.legend(); x_.grid(alpha=.3)
    fig.suptitle(f"{tag}: held-out KL to teacher (11 domains, 880 prompts)"); fig.tight_layout()
    p = f"{C}/{tag}_heldout_kl.png"; fig.savefig(p, dpi=110); plt.close(fig); return p
def plot_bench(names, tag):
    keys = ["gsm8k", "math", "mmlu", "humaneval", "avg"]
    fig, ax = plt.subplots(2, 5, figsize=(24, 9))
    for a in names:
        if a not in evals: continue
        R, H = runs[a]; E = evals[a]; rs = sorted(r for r in E if r <= len(R))
        for j, k in enumerate(keys):
            y = [bench_avg(E[r]) if k == "avg" else E[r][k]["acc"] for r in rs]
            ax[0, j].plot([tok_at(R, r) / 1e6 for r in rs], y, marker="o", label=a)
            ax[1, j].plot([gpu_h(R, r) for r in rs], y, marker="o", label=a)
    for j, k in enumerate(keys):
        for i in range(2):
            if "teacher" in evals:
                t = bench_avg(evals["teacher"]) if k == "avg" else evals["teacher"][k]["acc"]
                ax[i, j].axhline(t, color="k", ls="--", lw=1, label="teacher 8B")
            ax[i, j].set_title(k); ax[i, j].grid(alpha=.3)
        ax[0, j].set_xlabel("training tokens (M)"); ax[1, j].set_xlabel("GPU-hours")
    ax[0, 0].legend(fontsize=8); fig.suptitle(f"{tag}: benchmarks (greedy, non-thinking)"); fig.tight_layout()
    p = f"{C}/{tag}_benchmarks.png"; fig.savefig(p, dpi=100); plt.close(fig); return p
def plot_style(names, tag):
    keys = ["len", "bold", "header", "bullet", "latex"]
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.5))
    for a in names:
        if a not in evals: continue
        R, H = runs[a]; E = evals[a]; rs = sorted(r for r in E if r <= len(R))
        for j, k in enumerate(keys):
            ax[j].plot([tok_at(R, r) / 1e6 for r in rs], [E[r]["style"][k] for r in rs], marker="o", label=a)
    for j, k in enumerate(keys):
        if "teacher" in evals: ax[j].axhline(evals["teacher"]["style"][k], color="k", ls="--", lw=1, label="teacher")
        ax[j].set_title(f"style: {k}"); ax[j].set_xlabel("training tokens (M)"); ax[j].grid(alpha=.3)
    ax[0].legend(fontsize=8); fig.tight_layout(); p = f"{C}/{tag}_style.png"; fig.savefig(p, dpi=100); plt.close(fig); return p
def plot_scores(names, tag):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a in names:
        R, _ = runs[a]
        w = 5; y = np.convolve([d["score"]["main"] for d in R], np.ones(w) / w, mode="valid")
        y2 = np.convolve([d["score"]["gap"] for d in R], np.ones(w) / w, mode="valid")
        ax[0].plot(range(w, len(R) + 1), y, label=a); ax[1].plot(range(w, len(R) + 1), y2, label=a)
    ax[0].set_title("pre-training batch score (main: capped gap - 0.5*H_T, z-check)"); ax[1].set_title("pre-training batch mean gap (logpT - logpS)")
    for x in ax: x.set_xlabel("round"); x.legend(fontsize=8); x.grid(alpha=.3)
    fig.tight_layout(); p = f"{C}/{tag}_batch_scores.png"; fig.savefig(p, dpi=110); plt.close(fig); return p
def plot_adv():
    if "adv" not in runs: return None
    R, _ = runs["adv"]
    srcs = sorted({s for d in R for s in d["srcstat"]})
    fig, ax = plt.subplots(1, 3, figsize=(22, 6))
    for k, a in zip(["uncapped", "main"], ax[:2]):
        if k == "uncapped": k = "gap"
        m = {s: np.mean([d["srcstat"][s][k] for d in R[:20] if s in d["srcstat"]]) for s in srcs}
        order = sorted(srcs, key=lambda s: -m[s])
        a.barh(order[::-1], [m[s] for s in order[::-1]], color=["tab:red" if s.startswith("adv") else "tab:blue" for s in order[::-1]])
        a.set_title(f"per-source mean candidate score, rounds 1-20: {'naive mean gap' if k=='gap' else 'main score'}")
    for s in srcs:
        y = [d["bandit"].get(s, np.nan) for d in R]
        ax[2].plot(y, label=s, ls="--" if s.startswith("adv") else "-")
    ax[2].set_title("adversarial miner: bandit value per source (naive gap)"); ax[2].legend(fontsize=7, ncol=2); ax[2].set_xlabel("round")
    fig.tight_layout(); p = f"{C}/adversarial_sources.png"; fig.savefig(p, dpi=100); plt.close(fig); return p
def plot_src(names, tag):
    fig, axs = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5), squeeze=False)
    for a, ax in zip(names, axs[0]):
        R, _ = runs[a]; srcs = sorted({s for d in R for s in d["src"]})
        bins = np.array_split(np.arange(len(R)), 10)
        bottom = np.zeros(len(bins))
        for s in srcs:
            y = np.array([np.sum([R[i]["src"].get(s, 0) for i in b]) / max(1, np.sum([R[i]["nsamp"] for i in b])) for b in bins])
            ax.bar(range(len(bins)), y, bottom=bottom, label=s); bottom += y
        ax.set_title(f"{a}: selected-source mix over training"); ax.set_xlabel("training decile")
    axs[0][0].legend(fontsize=7); fig.tight_layout(); p = f"{C}/{tag}_source_mix.png"; fig.savefig(p, dpi=100); plt.close(fig); return p
def table(names):
    rows = []
    base = evals.get("base")
    for a in names:
        R, H = runs[a]; E = evals.get(a, {})
        last = max(E) if E else None
        h0, hl = H[0], H[-1]
        row = dict(arm=a, rounds=len(R), tokens_M=R[-1]["tokens"] / 1e6 if R else 0, gpu_h=gpu_h(R, len(R)),
                   kl0=h0["topkkl"], kl_end=hl["topkkl"], gap_end=hl["gap"], score_mean=np.mean([d["score"]["main"] for d in R]),
                   mean_len=np.mean([d["mean_len"] for d in R]))
        if last is not None:
            e = E[last]; row.update(eval_round=last, gsm8k=e["gsm8k"]["acc"], math=e["math"]["acc"], mmlu=e["mmlu"]["acc"], humaneval=e["humaneval"]["acc"], avg=bench_avg(e), style_len=e["style"]["len"])
        rows.append(row)
    return rows
if __name__ == "__main__":
    out = {"teacher_tps": TEACHER_TPS}
    ph1 = arms(PH1); ph2 = sorted([a for a in runs if a.startswith("p2_")])
    paths = []
    if ph1:
        paths += [plot_heldout(ph1, "phase1"), plot_bench(ph1, "phase1"), plot_style(ph1, "phase1"), plot_scores(ph1, "phase1"), plot_src([a for a in ph1 if a in ("sel", "selraw", "full", "adv")], "phase1"), plot_adv()]
        out["phase1"] = table(ph1)
    if ph2:
        cmp = ph2 + [a for a in ["sel", "random"] if a in runs]
        paths += [plot_heldout(cmp, "phase2"), plot_bench(cmp, "phase2"), plot_style(cmp, "phase2"), plot_scores(cmp, "phase2")]
        out["phase2"] = table(cmp)
    p3b = [a for a in ["b_fixed", "b_random", "b_sel"] if a in runs]
    p3s = [a for a in ["random", "sel", "random_s1", "sel_s1", "s_divcap", "s_mc_sel", "s_mc_rand"] if a in runs]
    if p3b:
        paths += [plot_heldout(p3b, "phase3_basestudent"), plot_bench(p3b, "phase3_basestudent"), plot_style(p3b, "phase3_basestudent")]
        out["phase3_base"] = table(p3b)
    if p3s:
        paths += [plot_heldout(p3s, "phase3_variants"), plot_bench(p3s, "phase3_variants")]
        out["phase3_variants"] = table(p3s)
    for k in ["base", "teacher", "basemodel"]:
        if k in evals: out[k] = {t: evals[k][t]["acc"] for t in ["gsm8k", "math", "mmlu", "humaneval"]} | {"avg": bench_avg(evals[k]), "style": evals[k]["style"]}
    out["charts"] = [p for p in paths if p]
    json.dump(out, open(f"{B}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))

def matched(names, toks=(5e6, 10e6, 15e6), gpuh=(0.5, 1.0, 2.0)):
    out = {}
    for a in names:
        R, H = runs[a]
        def at_tok(T):
            c = [h for h in H if h["tokens"] <= T]; return round(c[-1]["topkkl"], 4) if c and H[-1]["tokens"] >= T * 0.97 else None
        def at_gh(G):
            c = [h for h in H if gpu_h(R, h["round"]) <= G]; return round(c[-1]["topkkl"], 4) if c and gpu_h(R, H[-1]["round"]) >= G * 0.97 else None
        E = evals.get(a, {})
        bm = {r: round(bench_avg(E[r]), 4) for r in sorted(E) if r <= len(R)}
        bm_tok = {round(tok_at(R, r) / 1e6, 1): v for r, v in bm.items()}
        out[a] = dict(kl_at_tokM={int(t / 1e6): at_tok(t) for t in toks}, kl_at_gpuh={g: at_gh(g) for g in gpuh},
                      bench_avg_by_tokM=bm_tok, gpuh_per_Mtok=round(gpu_h(R, len(R)) / (R[-1]["tokens"] / 1e6), 4))
    return out
if __name__ == "__main__":
    allarms = [a for a in runs]
    m = matched(allarms)
    s = json.load(open(f"{B}/summary.json")); s["matched"] = m
    json.dump(s, open(f"{B}/summary.json", "w"), indent=1, default=float)
    for a, v in m.items(): print(a, v)
