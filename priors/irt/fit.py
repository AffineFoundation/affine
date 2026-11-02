import json, os, sys, math, time, collections
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("IRT_THREADS", "4"))); torch.manual_seed(0)
sys.path.insert(0, "/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code")
import baseline_v1 as BV, prior_gate as PG
PARAM = sys.argv[1] if len(sys.argv) > 1 else "ab"
SIG_LA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
SIG_LOC = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0   # fixed prior sd of b (theta units) or c (logits)   # fixed prior sd of log a (joint MAP collapses it to 0 if free)     # "ab" = a(theta-b) as specified; "si" = a*theta + c sensitivity check
# ---------------- data
ck, tk, N, S = [], [], [], []
for l in open(os.environ.get("IRT_WORK", "/tmp/irt") + "/cells.tsv"):
    c, t, n, s = l.rstrip("\n").split("\t"); ck.append(c); tk.append(t); N.append(int(n)); S.append(int(s))
SET = sys.argv[4] if len(sys.argv) > 4 else "chain"    # all | chain (drop offline Allan arms) | recent (checkpoints with dapo data)
if SET != "all":
    dap = set(c for c, t in zip(ck, tk) if t.startswith("dapo:"))
    ok = [(not c.startswith("x")) and (SET == "chain" or c in dap) for c in ck]
    ck, tk, N, S = [[v for v, k in zip(L, ok) if k] for L in (ck, tk, N, S)]

pv = PG._merge({}); pool = PG.load_pool(pv)
rows = BV.load_rows(pv["table"], pool["ids"])          # live prior table, read-only (mode=ro)
tasks = [r["task_id"] for r in rows]; ti = {t: i for i, t in enumerate(tasks)}
env = np.array([0 if r["env"] == "affine_math" else 1 for r in rows])
cks = sorted(set(ck)); ci = {c: i for i, c in enumerate(cks)}
m_idx = torch.tensor([ci[c] for c in ck]); i_idx = torch.tensor([ti[t] for t in tk])
n_t = torch.tensor(N, dtype=torch.float64); s_t = torch.tensor(S, dtype=torch.float64)
M, I = len(cks), len(tasks); E = torch.tensor(env)
print("data: ckpts %d tasks %d cells %d rollouts %d" % (M, I, len(N), sum(N)))
# ---------------- params
dt = torch.float64
z = torch.zeros(M, dtype=dt, requires_grad=True)
la = torch.full((I,), math.log(0.5), dtype=dt, requires_grad=True)
# init location from pooled pass rate
pooled = np.zeros(I); cnt = np.zeros(I)
np.add.at(pooled, i_idx.numpy(), np.array(S)); np.add.at(cnt, i_idx.numpy(), np.array(N))
p0 = (pooled + 0.5) / (cnt + 1.0); lg0 = np.log(p0 / (1 - p0))
loc = torch.tensor(-lg0 / 0.5 if PARAM == "ab" else lg0, dtype=dt).requires_grad_(True)   # b (ab) or c (si)
hyp = torch.zeros(8, dtype=dt, requires_grad=True)    # mu_la[2], logsd_la[2], mu_loc[2], logsd_loc[2]
with torch.no_grad(): hyp[0:2] = math.log(0.5); hyp[4:6] = float(np.mean(loc.numpy())); hyp[6:8] = math.log(float(np.std(loc.numpy())) + 1e-3)
def theta():
    return (z - z.mean()) / (z.std(unbiased=False) + 1e-9)   # scale fixed: checkpoint abilities standardised to mean 0, sd 1
def logits(th):
    a = la.exp()
    return a[i_idx] * (th[m_idx] - loc[i_idx]) if PARAM == "ab" else a[i_idx] * th[m_idx] + loc[i_idx]
def nlp():
    th = theta(); x = logits(th)
    ll = (s_t * torch.nn.functional.logsigmoid(x) + (n_t - s_t) * torch.nn.functional.logsigmoid(-x)).sum()
    mu_la, sd_la = hyp[0:2][E], SIG_LA; mu_l, sd_l = hyp[4:6][E], SIG_LOC
    lp = -0.5 * (z ** 2).sum()
    lp += (-0.5 * ((la - mu_la) / sd_la) ** 2).sum()          # log a_i ~ N(env mean, env sd)
    lp += (-0.5 * ((loc - mu_l) / sd_l) ** 2).sum()           # b_i ~ N(env mean, env sd)
    
    return -(ll + lp)
params = [z, la, loc, hyp]
t0 = time.time()
opt = torch.optim.Adam(params, lr=0.02)
for it in range(2000):
    opt.zero_grad(); L = nlp(); L.backward(); opt.step()
    if it % 300 == 0: print("adam", it, float(L.detach()), round(time.time() - t0, 1))
opt = torch.optim.LBFGS(params, lr=1, max_iter=500, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-10)
def closure():
    opt.zero_grad(); L = nlp(); L.backward(); return L
prev = None
for k in range(40):
    L = float(opt.step(closure)); print("lbfgs", k, L, round(time.time() - t0, 1), flush=True)
    if prev is not None and abs(prev - L) < 0.5: break
    prev = L
th = theta().detach().numpy(); a = la.exp().detach().numpy(); lc = loc.detach().numpy(); h = hyp.detach().numpy()
b = lc if PARAM == "ab" else -lc / a
print("hyper: mu_loga", h[0:2], "sd_loga(fixed)", SIG_LA, "mu_loc", h[4:6], "sd_loc(fixed)", SIG_LOC)
# per-checkpoint coverage + raw pass
cov = collections.Counter(ck); roll = collections.Counter()
for c, n in zip(ck, N): roll[c] += n
np.savez(os.environ.get("IRT_WORK", "/tmp/irt") + "/fit_%s_%g_%g_%s.npz" % (PARAM, SIG_LA, SIG_LOC, SET), theta=th, a=a, b=b, loc=lc, env=env, hyp=h,
         cks=np.array(cks), cov=np.array([cov[c] for c in cks]), roll=np.array([roll[c] for c in cks]),
         s_tab=np.array([r["s"] for r in rows]), n_tab=np.array([r["n"] for r in rows]),
         cnt=cnt, pooled=pooled, tasks=np.array(tasks))
print("done", round(time.time() - t0, 1), "s")
