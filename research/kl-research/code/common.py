import pickle, glob, numpy as np, torch, torch.nn.functional as F, math
from torch.utils.checkpoint import checkpoint

def load_pool(path="/root/kl/data/pool"):
    recs = []
    for f in sorted(glob.glob(f"{path}/shard*.pkl")):
        recs += pickle.load(open(f, "rb"))
    for r in recs:
        p = np.exp(r["tl"].astype(np.float32)); s = p.sum(1); res = np.clip(1 - s, 1e-8, 1)
        lp_top = r["tl"].astype(np.float32)
        H = -(p * lp_top).sum(1) - res * np.log(res)
        # variance of -log p under teacher (approx from top-20 + residual bucket)
        m2 = (p * lp_top ** 2).sum(1) + res * np.log(res) ** 2
        r["H"] = H.astype(np.float32); r["V"] = np.clip(m2 - H ** 2, 1e-4, None).astype(np.float32)
        r["n"] = len(r["rids"])
    return recs

def teacher_stats(r, mask=None):
    m = np.ones(r["n"], bool) if mask is None else mask
    if m.sum() == 0: m = np.ones(r["n"], bool)
    nll = -r["lp"][m]; H = r["H"][m]; V = r["V"][m]
    z = (nll - H).sum() / math.sqrt(V.sum() + 1e-6)
    return dict(ent=float(H.mean()), cons=float((nll - H).mean()), z=float(z))

def score_sample(lpS, r, mask=None, beta=0.5, cap=2.0, zmax=4.0, variant="main"):
    m = np.ones(r["n"], bool) if mask is None else mask
    if m.sum() == 0: m = np.ones(r["n"], bool)
    g = (r["lp"] - lpS)[m]
    gap = float(g.mean()); ts = teacher_stats(r, m)
    raw = min(cap, max(0.0, gap))
    ent_sub = min(cap, max(0.0, gap - beta * ts["ent"]))
    main = ent_sub if abs(ts["z"]) <= zmax else 0.0
    nocap_raw = gap
    return dict(gap=gap, raw=raw, entsub=ent_sub, main=main, uncapped=nocap_raw, **ts)

def batch_score(scores, ns, key="main", L=512):
    w = np.minimum(np.array(ns), L).astype(np.float64)
    return float((w * np.array([s[key] for s in scores])).sum() / w.sum())

def make_microbatches(recs, max_tok=16384, max_len=4096 + 1536):
    order = sorted(range(len(recs)), key=lambda i: len(recs[i]["pids"]) + recs[i]["n"])
    mbs, cur, curmax = [], [], 0
    for i in order:
        L = min(len(recs[i]["pids"]) + recs[i]["n"], max_len)
        nm = max(curmax, L)
        if cur and nm * (len(cur) + 1) > max_tok:
            mbs.append(cur); cur, curmax = [], 0; nm = L
        cur.append(i); curmax = nm
    if cur: mbs.append(cur)
    return mbs

def collate(recs, idxs, device, max_len=4096 + 1536):
    seqs, resp_pos = [], []
    for i in idxs:
        r = recs[i]; s = np.concatenate([r["pids"], r["rids"]])[:max_len]
        seqs.append(s); resp_pos.append((len(r["pids"]), len(s)))
    L = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long); att = torch.zeros(len(seqs), L, dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.from_numpy(s.astype(np.int64)); att[b, :len(s)] = 1
    return ids.to(device), att.to(device), resp_pos

def _chunk_logp(h, W, tgt, topi):
    with torch.autocast("cuda", enabled=False):
        logits = (h.float() @ W.float().t())
        lsm = torch.log_softmax(logits, -1)
    return lsm.gather(1, tgt[:, None])[:, 0], lsm.gather(1, topi)

def student_forward(model, recs, idxs, device, need_top=False, grad=False, chunk=4096):
    """returns hidden-based per-token logp of sampled tokens (and at teacher top-20 ids) for response positions."""
    ids, att, rp = collate(recs, idxs, device)
    out = model.model(input_ids=ids, attention_mask=att)
    hs = out.last_hidden_state
    H, T, TI, lens = [], [], [], []
    for b, (a, e) in enumerate(rp):
        n = e - a
        H.append(hs[b, a - 1:e - 1]); T.append(ids[b, a:e]); lens.append(n)
        r = recs[idxs[b]]
        TI.append(torch.from_numpy(r["ti"][:n].astype(np.int64)).to(device))
    H = torch.cat(H); T = torch.cat(T); TI = torch.cat(TI)
    W = model.lm_head.weight
    lps, tops = [], []
    for c in range(0, H.shape[0], chunk):
        if grad:
            a_, b_ = checkpoint(_chunk_logp, H[c:c + chunk], W, T[c:c + chunk], TI[c:c + chunk], use_reentrant=False)
        else:
            a_, b_ = _chunk_logp(H[c:c + chunk], W, T[c:c + chunk], TI[c:c + chunk])
        lps.append(a_); tops.append(b_)
    return torch.cat(lps), torch.cat(tops), lens

@torch.no_grad()
def student_logps(model, recs, idxs, device, max_tok=32768):
    """per-sample numpy logp of sampled tokens + top-k KL per token"""
    model.eval(); res = {}
    sub = [recs[i] for i in idxs]
    for mb in make_microbatches(sub, max_tok):
        gi = [idxs[j] for j in mb]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp, top, lens = student_forward(model, recs, gi, device, need_top=True)
        lp = lp.float().cpu().numpy(); top = top.float().cpu().numpy(); o = 0
        for i, n in zip(gi, lens):
            r = recs[i]; tl = r["tl"][:n].astype(np.float32); pt = np.exp(tl); pt = pt / pt.sum(1, keepdims=True)
            ltn = np.log(pt + 1e-12)
            kl = (pt * (ltn - top[o:o + n])).sum(1)
            res[i] = (lp[o:o + n], kl, np.exp(top[o:o + n, 0]))  # student prob of teacher-top1 token
            o += n
    return res
