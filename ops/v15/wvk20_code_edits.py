"""wvk 20 staging — teacher-relative miner thought cap.

Knob `[duel].thought_cap_ratio` (default 0 = pre-wvk-20: the fixed
max_thought_tokens cap). When > 0, per turn:
    cap_T = max(max_thought_tokens, floor(thought_cap_ratio × L_T))
where L_T = the longest VALID reference thought on that turn (refs without a
parseable action are already excluded by sample_teacher_rollouts), counted
in TEACHER-tokenizer tokens (the teacher's own tokenizer, the model that
wrote the references). Both sides read the same RefCache entry, so cap_T is
identical for king and challenger on a turn. Action cap and the teacher's
ref_max_tokens are unchanged. Refs are now sampled BEFORE the miner sample
(the per-turn overlap is lost; turn_conc turns in flight keep the miner
engines busy).

Stamps: duel_params.thought_cap_rule = "max(fixed, <ratio>*L_T)",
duel_params.thought_cap_ratio, duel_params.thought_cap_tokenizer; per row
cap_tokens + ref_thought_tokens (L_T); per side n_turns_cap_raised /
mean_cap_tokens telemetry.

Edits (anchored, idempotent):
  evalsrv/dueling.py, affine/config.py
"""

from pathlib import Path

REPO = Path("/home/const/subnet120")


def sub(path: Path, old: str, new: str, marker: str) -> None:
    s = path.read_text()
    if marker in s:
        print(f"{path.name}: already has {marker!r}")
        return
    if s.count(old) != 1:
        raise SystemExit(f"{path.name}: anchor not unique/missing: {old[:70]!r}")
    path.write_text(s.replace(old, new))
    print(f"{path.name}: patched ({marker})")


du = REPO / "affine/evalsrv/dueling.py"
sub(du, '''    thought_echo = score_mode in ("min_rg", "min_rga")
    action_echo = score_mode == "min_rga"

    async def one(rec: dict) -> None:
        nonlocal done
        tid = turn_id(rec)
        prefix = rec["prefix"]
        # Per-turn action dialect; absent on pre-dialect corpus records,
        # which means bash (affine.dialects.DEFAULT_KIND).
        action_kind = rec.get("action_kind")
        max_thought, max_action = caps(action_kind)
        ref_thought, ref_action = ref_caps(action_kind)
        async with turn_sem:
            if abort_event is not None and abort_event.is_set():
                raise DuelAborted("superseded by a new duel request")
            # Miner only needs the prefix x. Teacher refs (z_C, y_C) are
            # independent. Running them in series left miner GPUs idle at
            # duel start (chal-00076: all 8 at 0% while 64 turns sat in
            # ensure_raw). Same calls, overlapped. No sticky_key on the
            # miner sample: n_miner=1 cannot reuse a prefix cache, and
            # hash-pinning left one copy idle.
            raw, miner_rollouts = await asyncio.gather(
                refs.ensure_raw(
                    tid, teacher, prefix, n_teacher, temperature,
                    ref_thought, ref_action, action_kind),
                sample_miner_rollouts(
                    miner, prefix, n_miner, temperature,
                    max_thought, max_action, action_kind=action_kind),
            )
            if not raw:
                done += 1
                return
''', '''    thought_echo = score_mode in ("min_rg", "min_rga")
    action_echo = score_mode == "min_rga"
    # wvk 20 (2026-09-16): teacher-relative miner thought cap. 0 = fixed cap.
    thought_cap_ratio = float(duel_cfg.get("thought_cap_ratio", 0.0))
    teacher_tok = (get_tokenizer(teacher.cfg.repo, teacher.cfg.revision)
                   if thought_cap_ratio > 0 else None)

    def teacher_cap(raw_refs: list[tuple[str, str]], fixed: int) -> tuple[int, int]:
        """(cap_T, L_T): L_T = longest valid reference thought in teacher
        tokens; cap_T = max(fixed, floor(ratio·L_T))."""
        lt = max((len(teacher_tok.encode(z or "", add_special_tokens=False))
                  for z, _ in raw_refs), default=0)
        return max(int(fixed), int(thought_cap_ratio * lt)), lt

    async def one(rec: dict) -> None:
        nonlocal done
        tid = turn_id(rec)
        prefix = rec["prefix"]
        # Per-turn action dialect; absent on pre-dialect corpus records,
        # which means bash (affine.dialects.DEFAULT_KIND).
        action_kind = rec.get("action_kind")
        max_thought, max_action = caps(action_kind)
        ref_thought, ref_action = ref_caps(action_kind)
        cap_tokens, ref_lt = max_thought, None
        async with turn_sem:
            if abort_event is not None and abort_event.is_set():
                raise DuelAborted("superseded by a new duel request")
            if thought_cap_ratio > 0:
                # wvk 20: the miner's thought cap depends on the teacher's
                # references, so they come first; both sides read the same
                # RefCache entry and therefore the same cap_T. The miner
                # engines stay busy on the other turns in flight.
                raw = await refs.ensure_raw(
                    tid, teacher, prefix, n_teacher, temperature,
                    ref_thought, ref_action, action_kind)
                if not raw:
                    done += 1
                    return
                cap_tokens, ref_lt = teacher_cap(raw, max_thought)
                miner_rollouts = await sample_miner_rollouts(
                    miner, prefix, n_miner, temperature,
                    cap_tokens, max_action, action_kind=action_kind)
            else:
                # Miner only needs the prefix x. Teacher refs (z_C, y_C) are
                # independent. Running them in series left miner GPUs idle at
                # duel start (chal-00076: all 8 at 0% while 64 turns sat in
                # ensure_raw). Same calls, overlapped. No sticky_key on the
                # miner sample: n_miner=1 cannot reuse a prefix cache, and
                # hash-pinning left one copy idle.
                raw, miner_rollouts = await asyncio.gather(
                    refs.ensure_raw(
                        tid, teacher, prefix, n_teacher, temperature,
                        ref_thought, ref_action, action_kind),
                    sample_miner_rollouts(
                        miner, prefix, n_miner, temperature,
                        max_thought, max_action, action_kind=action_kind),
                )
                if not raw:
                    done += 1
                    return
''', "def teacher_cap(")
sub(du, '''        t = await miner_terms(
            teacher, miner, prefix, ref, n_miner, temperature,
            max_thought, max_action,
''', '''        t = await miner_terms(
            teacher, miner, prefix, ref, n_miner, temperature,
            cap_tokens, max_action,
''', "cap_tokens, max_action,\n            score_bank")
sub(du, '''        t.update({"turn_id": tid, "miner": miner.cfg.name})
''', '''        t.update({"turn_id": tid, "miner": miner.cfg.name,
                  # wvk 20: the thought cap this side sampled under and the
                  # longest reference thought (teacher tokens) behind it.
                  "cap_tokens": cap_tokens, "ref_thought_tokens": ref_lt})
''', '"cap_tokens": cap_tokens')
# imports: get_tokenizer from .chat
s = du.read_text()
if "from .chat import get_tokenizer" not in s:
    old = "from . import amatch\n"
    if s.count(old) != 1:
        raise SystemExit("dueling.py: import anchor")
    s = s.replace(old, "from . import amatch\nfrom .chat import get_tokenizer\n")
    du.write_text(s)
    print("dueling.py: get_tokenizer imported")
# per-side telemetry
sub(du, '''    # A_match telemetry (2026-09-14): share of reference actions equal to
    # the side's action, and the same minus the refs' own agreement.
    out.update(amatch.summarize(rows))
    return out
''', '''    # A_match telemetry (2026-09-14): share of reference actions equal to
    # the side's action, and the same minus the refs' own agreement.
    out.update(amatch.summarize(rows))
    # wvk 20: teacher-relative thought cap telemetry.
    caps_ = [r.get("cap_tokens") for r in rows if isinstance(r.get("cap_tokens"), int)]
    fixed = min(caps_) if caps_ else None
    out["n_turns_cap_raised"] = (sum(1 for c in caps_ if c > fixed) if caps_ else 0)
    out["mean_cap_tokens"] = (sum(caps_) / len(caps_)) if caps_ else None
    out["max_cap_tokens"] = max(caps_) if caps_ else None
    return out
''', 'out["n_turns_cap_raised"]')
# duel_params stamp
sub(du, '''            # wvk 19: a first-slice pass needs a confirmation slice to crown.
            "confirmation_required": bool(duel_cfg.get("confirmation_required", False)),
''', '''            # wvk 19: a first-slice pass needs a confirmation slice to crown.
            "confirmation_required": bool(duel_cfg.get("confirmation_required", False)),
            # wvk 20: teacher-relative miner thought cap (0 = fixed cap).
            "thought_cap_ratio": float(duel_cfg.get("thought_cap_ratio", 0.0)),
            "thought_cap_rule": (f"max(fixed, {float(duel_cfg.get('thought_cap_ratio', 0.0)):g}*L_T)"
                                 if float(duel_cfg.get("thought_cap_ratio", 0.0)) > 0 else "fixed"),
            "thought_cap_tokenizer": (teacher[0].repo if isinstance(teacher, list) else teacher.repo)
                                     if float(duel_cfg.get("thought_cap_ratio", 0.0)) > 0 else None,
''', '"thought_cap_ratio": float(duel_cfg.get(')

cfg = REPO / "affine/affine/config.py"
sub(cfg, '''    confirmation_required: bool = False
''', '''    confirmation_required: bool = False
    # wvk 20 (2026-09-16): per turn the miner may think up to
    # max(max_thought_tokens, floor(thought_cap_ratio × L_T)) tokens, L_T = the
    # longest valid teacher reference thought on the turn (teacher tokens).
    # 0.0 = fixed cap (pre-wvk-20).
    thought_cap_ratio: float = 0.0
''', "thought_cap_ratio: float = 0.0")
sub(cfg, '''        confirmation_required=bool(d.get("confirmation_required", False)),
''', '''        confirmation_required=bool(d.get("confirmation_required", False)),
        thought_cap_ratio=float(d.get("thought_cap_ratio", 0.0)),
''', 'thought_cap_ratio=float(d.get(')
print("done")
