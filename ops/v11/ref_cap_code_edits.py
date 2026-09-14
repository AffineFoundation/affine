"""wvk 17 staging — code side of the teacher reference cap (`[duel].ref_max_tokens`).

Adds a teacher-only sampling budget for the k reference rollouts. Absent /
None = the old shared cap (max_thought_tokens + max_action_tokens), so every
wvk <= 16 verdict replays unchanged. Miners' caps are untouched.

Edits (anchored, idempotent):
  affine/affine/config.py      DuelCfg.ref_max_tokens + parse/validate
  affine/evalsrv/dueling.py    ref_token_caps(); score_side samples refs
                               under it; duel_params stamps ref_max_tokens
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sub(path: Path, old: str, new: str, marker: str) -> None:
    s = path.read_text()
    if marker in s:
        print(f"{path.name}: already has {marker!r}")
        return
    if s.count(old) != 1:
        raise SystemExit(f"{path.name}: anchor not unique/missing: {old[:60]!r}")
    path.write_text(s.replace(old, new))
    print(f"{path.name}: patched ({marker})")


cfg = REPO / "affine/affine/config.py"
sub(cfg, '''    band_c: float = 2.0
    band_floor: float = 0.002
''', '''    band_c: float = 2.0
    band_floor: float = 0.002
    # Teacher-only sampling budget (thought + action tokens) for the k
    # reference rollouts (staged 2026-09-14, wvk 17). None = the miners'
    # max_thought_tokens + max_action_tokens, i.e. the pre-wvk-17 shared
    # cap. Miners' caps are not affected by this knob.
    ref_max_tokens: int | None = None
''', "ref_max_tokens: int | None")
sub(cfg, '''        max_thought_tokens=int(d["max_thought_tokens"]),
        max_action_tokens=int(d["max_action_tokens"]),
''', '''        max_thought_tokens=int(d["max_thought_tokens"]),
        max_action_tokens=int(d["max_action_tokens"]),
        ref_max_tokens=_ref_max_tokens(d),
''', "ref_max_tokens=_ref_max_tokens(d)")
s = cfg.read_text()
if "def _ref_max_tokens(" not in s:
    anchor = "\ndef _r2(r: dict) -> R2Cfg:\n"
    if s.count(anchor) != 1:
        raise SystemExit("config.py: _r2 anchor")
    s = s.replace(anchor, '''
def _ref_max_tokens(d: dict) -> int | None:
    v = d.get("ref_max_tokens")
    if v is None:
        return None
    v = int(v)
    shared = int(d["max_thought_tokens"]) + int(d["max_action_tokens"])
    if v < shared:
        raise ValueError(f"[duel] ref_max_tokens {v} must be >= max_thought_tokens + "
                         f"max_action_tokens = {shared} (the teacher may not get a "
                         f"smaller budget than the miners)")
    return v

''' + anchor)
    cfg.write_text(s)
    print("config.py: _ref_max_tokens added")

du = REPO / "affine/evalsrv/dueling.py"
sub(du, '''    def caps(action_kind: str | None) -> tuple[int, int]:
        return by_kind.get(action_kind or dialects.DEFAULT_KIND, dflt)
    return caps
''', '''    def caps(action_kind: str | None) -> tuple[int, int]:
        return by_kind.get(action_kind or dialects.DEFAULT_KIND, dflt)
    return caps


def ref_token_caps(duel_cfg: dict):
    """(max_thought, max_action) the TEACHER samples its k references under.

    `[duel].ref_max_tokens` (staged 2026-09-14, wvk 17) is a teacher-only
    total budget: a reference may run to ref_max_tokens (thought + action)
    where the miners stay at max_thought_tokens + max_action_tokens. The
    action share is the kind's max_action; the thought share takes the
    rest. Absent / None = exactly the miners' caps (pre-wvk-17 behaviour;
    wvk <= 16 verdicts replay unchanged). Never below the kind's own cap.
    Why: at the shared 1,792-token cap the teacher's own reference finished
    `length` on ~21% of deep turns (mean_refs 2.6–2.8 of 3, boxed 1.75), so
    those turns lost references or were dropped (refs < 2) — the miner was
    judged against fewer, truncated references exactly where the task is
    hardest.
    """
    caps = token_caps(duel_cfg)
    ref_total = duel_cfg.get("ref_max_tokens")
    ref_total = int(ref_total) if ref_total is not None else None

    def ref_caps(action_kind: str | None) -> tuple[int, int]:
        thought, action = caps(action_kind)
        if ref_total is None or ref_total <= thought + action:
            return thought, action
        return ref_total - action, action
    return ref_caps
''', "def ref_token_caps(")
sub(du, '''    caps = token_caps(duel_cfg)
    score_bank = bool(duel_cfg.get("score_bank", False))
''', '''    caps = token_caps(duel_cfg)
    ref_caps = ref_token_caps(duel_cfg)
    score_bank = bool(duel_cfg.get("score_bank", False))
''', "ref_caps = ref_token_caps(duel_cfg)")
sub(du, '''        action_kind = rec.get("action_kind")
        max_thought, max_action = caps(action_kind)
        async with turn_sem:
''', '''        action_kind = rec.get("action_kind")
        max_thought, max_action = caps(action_kind)
        ref_thought, ref_action = ref_caps(action_kind)
        async with turn_sem:
''', "ref_thought, ref_action = ref_caps(action_kind)")
sub(du, '''                refs.ensure_raw(
                    tid, teacher, prefix, n_teacher, temperature,
                    max_thought, max_action, action_kind),
                sample_miner_rollouts(
''', '''                refs.ensure_raw(
                    tid, teacher, prefix, n_teacher, temperature,
                    ref_thought, ref_action, action_kind),
                sample_miner_rollouts(
''', "ref_thought, ref_action, action_kind),")
sub(du, '''            "max_thought_tokens": int(duel_cfg["max_thought_tokens"]),
            "max_action_tokens": int(duel_cfg["max_action_tokens"]),
            "max_tokens_by_kind": {
''', '''            "max_thought_tokens": int(duel_cfg["max_thought_tokens"]),
            "max_action_tokens": int(duel_cfg["max_action_tokens"]),
            # Teacher-only reference budget (wvk 17); None = shared cap.
            "ref_max_tokens": (int(duel_cfg["ref_max_tokens"])
                               if duel_cfg.get("ref_max_tokens") is not None else None),
            "max_tokens_by_kind": {
''', '"ref_max_tokens": (int(duel_cfg["ref_max_tokens"])')
print("done")
