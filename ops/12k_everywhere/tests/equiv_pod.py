"""RA/20261102-12k-everywhere equivalence + 12K acceptance test (pod CPU, runtime-exact venv). For real queued/trained miner files:
  1. lifecycle check: old ff_format (8192/12288) vs new (12288/16384 + legacy 8192 cap) -> identical tokens/stop/reward per rollout.
  2. trainer rows: v8.3.3l _layout vs v8.3.3m _layout with FF_OVERLONG=(6000,8192) (must be identical: reward, advantage, active, keep)
     and v8.3.3m default (10096,12288): rows that differ must all be rollouts > 6000 tokens (the moved overlong ramp).
  3. synthetic 12K group: outputs of 12287 (+EOS) and 12288 (capped) tokens: new check accepts, old check rejects; v8.3.3m rows +
     context_limit accept prompt+12288; v8.3.3l context check rejects.
usage: equiv_pod.py TOKENIZER_DIR CONFIG_SETTINGS_JSON FILE:HOTKEY ..."""
import json, os, sys, importlib.util, math
from pathlib import Path
W = Path("/workspace/affine-mimo-trainer-prod"); SRC = W/"source-final-v823u"
os.chdir(SRC); sys.path.insert(0, str(SRC))
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m); return m
R = Path("/workspace/rec/equiv")
F_old = load("ff_format_old", R/"ff_format.old.py"); F_new = load("ff_format_new", R/"ff_format.new.py")
sys.path.insert(0, str(W/"host-v833l"))
FT = load("ff_trainer", W/"host-v833l/ff_trainer.py")
BL = load("v8211_batched_l", W/"host-v833l/v8211_batched.py")
BM = load("v8211_batched_m", W/"host-v833m/v8211_batched.py")
from subnet.mimo_grpo.config import TrainerConfig
from subnet.mimo_grpo.data import digest
tok = F_new.tokenizer(sys.argv[1])
config = TrainerConfig(**json.loads(Path(sys.argv[2]).read_text()))
def groups_of(checked):
    out = []
    for i, g in enumerate(checked['groups']):
        tr = tuple(FT.RewardedTrajectory(nonce=n, tokens=tuple(r['tokens']), outcome=FT.label(r['stop'], r['reward']), stop=r['stop'], reward=float(r['reward']))
                   for n, r in enumerate(g['rollouts']))
        out.append(FT.FreeGroup(task_id=digest(dict(i=i, p=g['prompt'])), prompt=tuple(g['prompt_tokens']), trajectories=tr, admission_sha256=digest(dict(i=i))))
    return out
def rows(B, gs):
    return [(r['reward'], r['advantage'], r['active'], r['keep'], len(r['rollout'].tokens)) for r in B._layout(gs, config)]
res = dict(files=[])
for arg in sys.argv[3:]:
    path, hk = arg.rsplit(":", 1); raw = Path(path).read_bytes()
    a = F_old.validate(raw, hk, tok, 125, 1000, True); b = F_new.validate(raw, hk, tok, 125, 1000, True)
    same_check = json.dumps(a['groups']) == json.dumps(b['groups'])
    gs = groups_of(b)
    rl = rows(BL, gs)
    BM.FF_OVERLONG = (6000, 8192); rm_compat = rows(BM, gs)
    BM.FF_OVERLONG = (10096, 12288); rm = rows(BM, gs)
    diff = [x for x, y in zip(rl, rm) if x != y]
    lens = [x[4] for x in rl]
    res['files'].append(dict(file=Path(path).name[:12], rollouts=len(rl), groups=len(gs), max_len=max(lens), over6000=sum(l > 6000 for l in lens),
                             capped=sum(r['stop'] == 'max_tokens' for g in b['groups'] for r in g['rollouts']),
                             check_identical=same_check, rows_identical_compat=(rl == rm_compat), rows_differing_default=len(diff),
                             differing_all_over_6000=all(x[4] > 6000 for x in diff), differing_min_len=min([x[4] for x in diff], default=None)))
    print(json.dumps(res['files'][-1]), flush=True)
# synthetic 12K group from the first file's first prompt
path, hk = sys.argv[3].rsplit(":", 1); v = json.loads(Path(path).read_bytes())
prompt = v['groups'][0]['prompt']
piece = tok.decode(tok.encode(" the value of x is 42, so", add_special_tokens=False))
def text_of(n):
    ids = (tok.encode(piece, add_special_tokens=False) * (n // 4 + 10))[:n]
    t = tok.decode(ids)
    while len(tok.encode(t, add_special_tokens=False)) > n: t = t[:-1]
    while len(tok.encode(t, add_special_tokens=False)) < n: t += " x"
    return t
outs = {}
for n in (12287, 12288):
    t = text_of(n); outs[n] = (t, len(tok.encode(t, add_special_tokens=False)))
syn = dict(version='affine-batch-v2', hotkey=hk, train={}, groups=[dict(prompt=prompt, rollouts=[dict(output=outs[12287][0], reward=1.0), dict(output=outs[12288][0], reward=0.0), dict(output="The answer is 42.", reward=1.0)])])
raw = json.dumps(syn).encode()
try: F_old.validate(raw, hk, tok, 125, 1000, True); old_ok = True
except F_old.Invalid as e: old_ok = "rejected: " + str(e)[:120]
nb = F_new.validate(raw, hk, tok, 125, 1000, True)
g = groups_of(nb); rm = rows(BM, g)
class RT: max_context_tokens = 12288
ctx_new = all(len(x.prompt) + len(r.tokens) <= BM.context_limit(RT) for x in g for r in x.trajectories)
ctx_old = all(len(x.prompt) + len(r.tokens) <= RT.max_context_tokens for x in g for r in x.trajectories)
res['synthetic'] = dict(output_token_counts=[outs[12287][1], outs[12288][1]], stops=[r['stop'] for r in nb['groups'][0]['rollouts']],
                        lens_with_eos=[len(r['tokens']) for r in nb['groups'][0]['rollouts']], prompt_tokens=len(nb['groups'][0]['prompt_tokens']),
                        old_check=old_ok, new_check="accepted", v833m_rows=[(round(a, 6), round(b, 6), c, d, e) for a, b, c, d, e in rm],
                        v833m_context_ok=ctx_new, v833l_context_ok=ctx_old)
print(json.dumps(dict(synthetic=res['synthetic'])), flush=True)
Path("/workspace/rec/equiv/RESULT.json").write_text(json.dumps(res, indent=1) + "\n")
