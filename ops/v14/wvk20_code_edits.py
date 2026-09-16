"""wvk 20 staging — additive A term: turn = min(R, G) + w·(A_match − pair).

Knob `[duel].a_term_w` (default 0.0 = pre-wvk-20: exactly min(R, G)). When
w > 0, every VALID turn adds w·(a_match − ref_pair) where a_match is the
share of the teacher's k reference actions equal to the side's action
after dialect normalisation and ref_pair the refs' own pairwise agreement
(evalsrv/amatch.py, live telemetry since wvk 18). The centring makes a
generic / mode-guessed action worth nothing on a turn where the teacher
agrees with itself anyway (pair high), and credits only agreement beyond
what the teacher's own self-agreement predicts. Forfeit turns get the
forfeit floor and no A credit; turns where A_match is undefined (`text`
kind, or refs without a normal form) add 0.

Edits (anchored, idempotent):
  affine/score.py        a_term(); side_turn_score / score_miner / duel take a_term_w
  evalsrv/dueling.py     score_duel / score_miner calls pass a_term_w; duel_params
                         stamps a_term_w; ranking_formula suffix when w > 0
  affine/config.py       DuelCfg.a_term_w
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


sc = REPO / "affine/affine/score.py"
sub(sc, '''def is_forfeit(row: dict) -> bool:
''', '''DEFAULT_A_TERM_W = 0.0
DEFAULT_A_TERM_EXCLUDE_KINDS: tuple[str, ...] = ("terminus_json",)
# v2 generic filter (2026-09-16): {kind: {"exact": set[str], "heads": set[str]}}
# and the excluded dialects; set by evalsrv from [duel].a_term_generic_file /
# a_term_exclude_kinds before scoring (configure_a_term). Empty = no filter.
A_TERM_GENERIC: dict[str, dict[str, set[str]]] = {}
A_TERM_EXCLUDE: tuple[str, ...] = DEFAULT_A_TERM_EXCLUDE_KINDS


def configure_a_term(exclude_kinds, generic_json: dict | None) -> None:
    """Install the v2 exclusions / generic list (from the published JSON:
    {"generic": {kind: [{"action":…}, …]}, "heads": {kind: [{"head":…}, …]}})."""
    global A_TERM_EXCLUDE, A_TERM_GENERIC
    A_TERM_EXCLUDE = tuple(exclude_kinds or ())
    g: dict[str, dict[str, set[str]]] = {}
    for kind, items in ((generic_json or {}).get("generic") or {}).items():
        g.setdefault(kind, {"exact": set(), "heads": set()})["exact"] = {i["action"] for i in items}
    for kind, items in ((generic_json or {}).get("heads") or {}).items():
        g.setdefault(kind, {"exact": set(), "heads": set()})["heads"] = {i["head"] for i in items}
    A_TERM_GENERIC = g


def a_term(row: dict, w: float = DEFAULT_A_TERM_W,
           exclude_kinds: tuple[str, ...] | None = None,
           generic: dict | None = None) -> float:
    """w·(A_match − pair) for one VALID turn row; 0 when the term is off,
    undefined (no normal form — `text` — or no valid refs), the turn's
    dialect is excluded (Terminus), or the side's normalised action / its
    command head is on the published generic list for that dialect.
    A_match / ref_pair / action_kind / a_norm / a_head are the per-turn
    fields evalsrv stamps (evalsrv/amatch.py). Never called on a forfeit."""
    if w <= 0:
        return 0.0
    a, p = row.get("a_match"), row.get("ref_pair")
    if not isinstance(a, (int, float)) or not isinstance(p, (int, float)):
        return 0.0
    kind = row.get("action_kind") or "bash"
    if kind in (exclude_kinds if exclude_kinds is not None else A_TERM_EXCLUDE):
        return 0.0
    g = (generic if generic is not None else A_TERM_GENERIC).get(kind) or {}
    if row.get("a_norm") and row["a_norm"] in g.get("exact", ()):
        return 0.0
    if row.get("a_head") and row["a_head"] in g.get("heads", ()):
        return 0.0
    return float(w) * (float(a) - float(p))


def is_forfeit(row: dict) -> bool:
''', "def a_term(")
sub(sc, '''                    forfeit_turn_score: float | None = DEFAULT_FORFEIT_TURN_SCORE,
                    action_norm_bytes: float | None = DEFAULT_ACTION_NORM_BYTES
                    ) -> float | None:
    """One side's score on one turn: the turn rule, or the forfeit floor.

    None when the side forfeited and the contract has no floor (legacy:
    the turn is dropped)."""
    if is_forfeit(row):
        return forfeit_turn_score
    return turn_score(row["pairs"], tau, score_mode, band_c, band_floor,
                      action_norm_bytes)
''', '''                    forfeit_turn_score: float | None = DEFAULT_FORFEIT_TURN_SCORE,
                    action_norm_bytes: float | None = DEFAULT_ACTION_NORM_BYTES,
                    a_term_w: float = DEFAULT_A_TERM_W
                    ) -> float | None:
    """One side's score on one turn: the turn rule (+ the wvk-20 A term when
    a_term_w > 0), or the forfeit floor.

    None when the side forfeited and the contract has no floor (legacy:
    the turn is dropped). A forfeit never earns A credit."""
    if is_forfeit(row):
        return forfeit_turn_score
    return turn_score(row["pairs"], tau, score_mode, band_c, band_floor,
                      action_norm_bytes) + a_term(row, a_term_w)
''', "a_term_w: float = DEFAULT_A_TERM_W\n                    ) -> float | None:")
sub(sc, '''                forfeit_turn_score: float | None = DEFAULT_FORFEIT_TURN_SCORE,
                action_norm_bytes: float | None = DEFAULT_ACTION_NORM_BYTES
                ) -> MinerScore:
    """Score one miner: mean per-turn score + telemetry.
''', '''                forfeit_turn_score: float | None = DEFAULT_FORFEIT_TURN_SCORE,
                action_norm_bytes: float | None = DEFAULT_ACTION_NORM_BYTES,
                a_term_w: float = DEFAULT_A_TERM_W
                ) -> MinerScore:
    """Score one miner: mean per-turn score + telemetry.
''', "a_term_w: float = DEFAULT_A_TERM_W\n                ) -> MinerScore:")
s = sc.read_text()
old = '''    turn_scores = [turn_score(r["pairs"], tau, score_mode, band_c, band_floor,
'''
if s.count(old) != 1:
    raise SystemExit("score.py: turn_scores anchor")
# find the full statement to replace (through the closing paren)
i = s.index(old)
j = s.index("]\n", i) + 2
stmt = s[i:j]
if "a_term(r, a_term_w)" not in stmt:
    new_stmt = stmt.rstrip("\n").rstrip("]") if False else None
    # rebuild: keep the comprehension but add the A term
    inner = stmt[len("    turn_scores = ["):stmt.rindex("]")]
    # inner is like: turn_score(r["pairs"], ...) for r in valid   (possibly multi-line)
    head, _, tail = inner.rpartition(" for r in ")
    new_stmt = "    turn_scores = [" + head + " + a_term(r, a_term_w) for r in " + tail + "]\n"
    s = s[:i] + new_stmt + s[j:]
    sc.write_text(s)
    print("score.py: patched (turn_scores + a_term)")
else:
    print("score.py: turn_scores already has a_term")
sub(sc, '''         action_norm_bytes: float | None = DEFAULT_ACTION_NORM_BYTES,
         min_z: float = DEFAULT_MIN_Z
         ) -> DuelResult:
''', '''         action_norm_bytes: float | None = DEFAULT_ACTION_NORM_BYTES,
         min_z: float = DEFAULT_MIN_Z,
         a_term_w: float = DEFAULT_A_TERM_W
         ) -> DuelResult:
''', "min_z: float = DEFAULT_MIN_Z,\n         a_term_w")
sub(sc, '''        rc = side_turn_score(c_by[tid], tau, score_mode, band_c, band_floor,
                             forfeit_turn_score, action_norm_bytes)
        rk = side_turn_score(k_by[tid], tau, score_mode, band_c, band_floor,
                             forfeit_turn_score, action_norm_bytes)
''', '''        rc = side_turn_score(c_by[tid], tau, score_mode, band_c, band_floor,
                             forfeit_turn_score, action_norm_bytes, a_term_w)
        rk = side_turn_score(k_by[tid], tau, score_mode, band_c, band_floor,
                             forfeit_turn_score, action_norm_bytes, a_term_w)
''', "forfeit_turn_score, action_norm_bytes, a_term_w)")
# the two score_miner calls inside duel() for cs/ks
s = sc.read_text()
i = s.index("def duel(")
j = s.index("\ndef ", i + 10)
body = s[i:j]
if "a_term_w=a_term_w)" not in body:
    body2 = body.replace("action_norm_bytes=action_norm_bytes)",
                         "action_norm_bytes=action_norm_bytes,\n                     a_term_w=a_term_w)")
    s = s[:i] + body2 + s[j:]
    sc.write_text(s)
    print("score.py: duel()'s score_miner calls pass a_term_w")

du = REPO / "affine/evalsrv/dueling.py"
s = du.read_text()
if "a_term_w" not in s:
    # score_duel call: add a_term_w after action_norm_bytes kw inside decide()
    old = '''            return score_duel(
                c_rows, k_rows,
                k_sigma=float(duel_cfg["k_sigma"]),
'''
    if s.count(old) != 1:
        raise SystemExit("dueling.py: score_duel anchor")
    s = s.replace(old, '''            return score_duel(
                c_rows, k_rows,
                a_term_w=a_term_w,
                k_sigma=float(duel_cfg["k_sigma"]),
''')
    # define a_term_w near band_c read
    old2 = '''        band_c = float(duel_cfg.get("band_c", 2.0))
        band_floor = float(duel_cfg.get("band_floor", 0.002))
'''
    if s.count(old2) != 1:
        raise SystemExit("dueling.py: band_c anchor")
    s = s.replace(old2, old2 + '''        # wvk 20: additive A term w·(A_match − pair); 0 = off (pre-wvk-20).
        a_term_w = float(duel_cfg.get("a_term_w", 0.0))
        if a_term_w > 0:
            gfile = str(duel_cfg.get("a_term_generic_file") or "")
            gjson = json.loads(Path(gfile).read_text()) if gfile else None
            score_module.configure_a_term(
                duel_cfg.get("a_term_exclude_kinds") or ["terminus_json"], gjson)
''')
    if "from affine import score as score_module" not in s:
        s = s.replace("from affine import dialects\n", "from affine import dialects\nfrom affine import score as score_module\n", 1)
    # stamp
    old3 = '''            # wvk 19: a first-slice pass needs a confirmation slice to crown.
            "confirmation_required": bool(duel_cfg.get("confirmation_required", False)),
'''
    if s.count(old3) != 1:
        raise SystemExit("dueling.py: stamp anchor")
    s = s.replace(old3, old3 + '''            # wvk 20: additive A term weight (0 = off).
            "a_term_w": a_term_w,
''')
    # ranking formula suffix: after the forfeit clause
    old4 = '''    if forfeit_turn_score is not None:
        ranking_formula += (
            f"; forfeit (no parseable action) scores {forfeit_turn_score:g}")
'''
    if s.count(old4) != 1:
        raise SystemExit("dueling.py: ranking_formula anchor")
    s = s.replace(old4, '''    if a_term_w > 0:
        ranking_formula += (
            f" + {a_term_w:g}·(A_match − pair) on valid turns, A_match = share of the k "
            f"reference actions equal to the side's action (dialect-normalised), "
            f"pair = the refs' own pairwise agreement; 0 where undefined (text)")
''' + old4)
    du.write_text(s)
    print("dueling.py: patched (a_term_w plumbing + stamp + formula)")
else:
    print("dueling.py: already has a_term_w")

# _miner_summary / _by_dialect score_miner calls need a_term_w so the published
# per-side 'reason' equals the mean of the scored turns.
s = du.read_text()
if "a_term_w=a_term_w" not in s.split("def _miner_summary", 1)[1].split("def _by_dialect", 1)[0]:
    s = s.replace('''def _miner_summary(rows: list[dict], tau: float | None,
                   score_mode: str = "reason",
                   band_c: float = 2.0, band_floor: float = 0.002,
                   forfeit_turn_score: float | None = None,
                   action_norm_bytes: float | None = None) -> dict:
    """Per-side summary: the score (reason) plus measured-not-scored telemetry."""
    s = score_miner(rows, bank_frac=_mean_bank(rows), tau=tau,
                    score_mode=score_mode, band_c=band_c,
                    band_floor=band_floor,
                    forfeit_turn_score=forfeit_turn_score,
                    action_norm_bytes=action_norm_bytes)
''', '''def _miner_summary(rows: list[dict], tau: float | None,
                   score_mode: str = "reason",
                   band_c: float = 2.0, band_floor: float = 0.002,
                   forfeit_turn_score: float | None = None,
                   action_norm_bytes: float | None = None,
                   a_term_w: float = 0.0) -> dict:
    """Per-side summary: the score (reason) plus measured-not-scored telemetry."""
    s = score_miner(rows, bank_frac=_mean_bank(rows), tau=tau,
                    score_mode=score_mode, band_c=band_c,
                    band_floor=band_floor,
                    forfeit_turn_score=forfeit_turn_score,
                    action_norm_bytes=action_norm_bytes,
                    a_term_w=a_term_w)
''')
    s = s.replace('''def _by_dialect(rows: list[dict], kind_by_tid: dict[str, str],
                tau: float | None, score_mode: str,
                band_c: float, band_floor: float,
                forfeit_turn_score: float | None = None,
                action_norm_bytes: float | None = None) -> dict[str, dict]:
''', '''def _by_dialect(rows: list[dict], kind_by_tid: dict[str, str],
                tau: float | None, score_mode: str,
                band_c: float, band_floor: float,
                forfeit_turn_score: float | None = None,
                action_norm_bytes: float | None = None,
                a_term_w: float = 0.0) -> dict[str, dict]:
''')
    s = s.replace('''        s = score_miner(grp, tau=tau, score_mode=score_mode,
                        band_c=band_c, band_floor=band_floor,
                        forfeit_turn_score=forfeit_turn_score,
                        action_norm_bytes=action_norm_bytes)
''', '''        s = score_miner(grp, tau=tau, score_mode=score_mode,
                        band_c=band_c, band_floor=band_floor,
                        forfeit_turn_score=forfeit_turn_score,
                        action_norm_bytes=action_norm_bytes,
                        a_term_w=a_term_w)
''')
    s = s.replace('''    king_sum = _miner_summary(king_rows, tau, score_mode, band_c, band_floor,
                              forfeit_turn_score, action_norm_bytes)
    chall_sum = _miner_summary(chall_rows, tau, score_mode, band_c, band_floor,
                               forfeit_turn_score, action_norm_bytes)
''', '''    king_sum = _miner_summary(king_rows, tau, score_mode, band_c, band_floor,
                              forfeit_turn_score, action_norm_bytes, a_term_w)
    chall_sum = _miner_summary(chall_rows, tau, score_mode, band_c, band_floor,
                               forfeit_turn_score, action_norm_bytes, a_term_w)
''')
    s = s.replace('''    king_sum["by_dialect"] = _by_dialect(
        king_rows, kind_by_tid, tau, score_mode, band_c, band_floor,
        forfeit_turn_score, action_norm_bytes)
    chall_sum["by_dialect"] = _by_dialect(
        chall_rows, kind_by_tid, tau, score_mode, band_c, band_floor,
        forfeit_turn_score, action_norm_bytes)
''', '''    king_sum["by_dialect"] = _by_dialect(
        king_rows, kind_by_tid, tau, score_mode, band_c, band_floor,
        forfeit_turn_score, action_norm_bytes, a_term_w)
    chall_sum["by_dialect"] = _by_dialect(
        chall_rows, kind_by_tid, tau, score_mode, band_c, band_floor,
        forfeit_turn_score, action_norm_bytes, a_term_w)
''')
    du.write_text(s)
    print("dueling.py: summaries take a_term_w")

# rows carry the side's normalised action + command head (v2 generic filter)
sub(du, '''        t["a_match"], t["ref_pair"] = amatch.turn_agreement(
            y_side, [r["y"] for r in ref], action_kind)
        rows.append(t)
''', '''        t["a_match"], t["ref_pair"] = amatch.turn_agreement(
            y_side, [r["y"] for r in ref], action_kind)
        # wvk 20 v2: the side's normalised action and its command head, so
        # the generic filter in score.a_term can look them up.
        t["action_kind"] = action_kind or dialects.DEFAULT_KIND
        t["a_norm"] = amatch.norm_action(y_side, action_kind) if y_side else None
        t["a_head"] = amatch.head_of(t["a_norm"], t["action_kind"]) if t["a_norm"] else None
        rows.append(t)
''', 't["a_head"] = amatch.head_of(')

am = REPO / "affine/evalsrv/amatch.py"
sub(am, '''def summarize(rows: list[dict]) -> dict:''', '''HEAD_TWO_WORD = ("git", "docker", "npm", "pip", "pip3", "cargo", "go", "make")
BASH_TOOL_NAMES = ("bash", "execute_bash", "run_command", "shell", "Bash")


def command_of(norm: str, kind: str) -> str | None:
    """The shell command inside a normalised action: the bash command itself,
    or the command argument of a single bash-like tool call; None otherwise."""
    if kind == "bash":
        return norm
    if kind == "tool_call":
        try:
            calls = json.loads(norm)
        except (json.JSONDecodeError, TypeError):
            return None
        if len(calls) == 1 and calls[0][0] in BASH_TOOL_NAMES:
            args = calls[0][1] or {}
            return args.get("command") or args.get("cmd") or next(iter(args.values()), None)
    return None


def head_of(norm: str | None, kind: str) -> str | None:
    """Command head for the generic filter (wvk 20 v2): the first word of a
    shell command (two words for git / docker / npm / pip / cargo / go /
    make, `python -m <mod>` for module runs; a leading `cd X &&` is
    skipped); for a non-shell tool call the tool name as `tool:<name>`."""
    if not norm:
        return None
    cmd = command_of(norm, kind)
    if cmd is None:
        if kind == "tool_call":
            try:
                calls = json.loads(norm)
            except (json.JSONDecodeError, TypeError):
                return None
            return f"tool:{calls[0][0]}" if calls else None
        return None
    toks = cmd.strip().split()
    if toks and toks[0] == "cd" and "&&" in toks:
        toks = toks[toks.index("&&") + 1:]
    if not toks:
        return None
    head = toks[0]
    if head in HEAD_TWO_WORD and len(toks) > 1:
        head += " " + toks[1]
    elif head in ("python", "python3") and len(toks) > 2 and toks[1] == "-m":
        head += " -m " + toks[2]
    return head


def summarize(rows: list[dict]) -> dict:''', "def head_of(")

cfg = REPO / "affine/affine/config.py"
sub(cfg, '''    confirmation_required: bool = False
''', '''    confirmation_required: bool = False
    # wvk 20 (staged 2026-09-16): additive A term, turn = min(R,G) +
    # a_term_w·(A_match − pair) on valid turns. 0.0 = off (pre-wvk-20).
    a_term_w: float = 0.0
    # v2: dialects that never earn A credit, and the published generic-action
    # list ({kind: {exact: [...], heads: [...]}}) that earns none either.
    a_term_exclude_kinds: tuple[str, ...] = ("terminus_json",)
    a_term_generic_file: str = ""
''', "a_term_w: float = 0.0")
sub(cfg, '''        confirmation_required=bool(d.get("confirmation_required", False)),
''', '''        confirmation_required=bool(d.get("confirmation_required", False)),
        a_term_w=float(d.get("a_term_w", 0.0)),
        a_term_exclude_kinds=tuple(str(k) for k in (d.get("a_term_exclude_kinds") or ["terminus_json"])),
        a_term_generic_file=str(d.get("a_term_generic_file") or ""),
''', 'a_term_w=float(d.get("a_term_w", 0.0)),')
print("done")
