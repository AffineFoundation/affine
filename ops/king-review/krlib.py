"""Shared pieces of the per-reign king review (ops/king-review).

The review reads the king seat's rollout traces "by hand, with an LLM":
sample failed king rollouts per (env, harness), ask a long-context judge
where each one went wrong, aggregate the failure patterns, and emit a
side-table of pivotal turns the fold can route into D (phase 2).

This module holds what every step shares:
  - TraceStore: the R2 trace manifest + chunk cache + a rollout index
  - Rollout / Turn: one envelope parsed into indexed turns (thought, action,
    observation) with deterministic labels (loop_onset / in_loop / escape,
    no_action, completion, token cap) -- the same definitions as the
    king-loop-labels measurement (internal/king-loops/label_loops.py) and
    affine/corpus/loops.py on the king_loop_onset branch
  - transcript rendering for the judge (explicit turn indices, truncated
    observations, collapsed repeat stretches)
  - env group lookup from rollouts/rollouts/sources.toml

Nothing here touches scoring, affine.toml or the fold.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tomllib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from affine import dialects
from affine.corpus.trace import message_text
from affine.corpus.view import rollout_outcome, run_tag
from datagen.slicer import make_traj_id

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES_TOML = REPO_ROOT / "rollouts" / "rollouts" / "sources.toml"
DEFAULT_TRACES_URL = "https://data.affine.io/traces/manifest.json"
DEFAULT_SNAPSHOT_URL = "https://affine.io/api/v1/snapshot"
DEFAULT_CACHE_DIR = Path(os.environ.get("KING_REVIEW_CACHE", "/tmp/king-review/traces"))

KING_PREFIX = "king_"
TEACHER_PREFIX = "teacher_"
OBS_HEAD = 200
WS_RE = re.compile(r"\s+")
NUM_RE = re.compile(r"\d+")
HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b")
PATH_RE = re.compile(r"(?:/[\w.\-@+]+){2,}/")
FENCE_RE = re.compile(r"```mswea_bash_command[ \t]*\n")
ERROR_RE = re.compile(
    r"traceback|error|exception|no such file|not found|command not found|"
    r"permission denied|<returncode>[1-9]|exit code [1-9]|exit status [1-9]|"
    r"failed|fatal:|cannot |syntax|wasted call|is not a terminal|timed out|"
    r"killed|segmentation fault", re.I)
NUDGE_RE = re.compile(
    r"exactly one|format error|please always provide|did not include|"
    r"no (?:bash )?code block|your response must|must contain|"
    r"could not parse|invalid json|malformed|wasted call|"
    r"refer to that earlier|unknown tool|tool .* not found", re.I)
EMPTY_OBS_RE = re.compile(
    r"^(?:<returncode>\d+</returncode>\s*)?<output>\s*</output>$|"
    r"^\(?(?:no output|empty)\)?$|^current terminal screen:\s*$", re.I)


# ----------------------------------------------------------------------------
# small helpers

def norm_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


def norm_action(s: str) -> str:
    s = s.lower()
    s = PATH_RE.sub("/", s)
    s = HEX_RE.sub("H", s)
    s = NUM_RE.sub("0", s)
    return norm_ws(s)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def king_digest_of(model: str) -> str:
    """`king/king-0ce59769300c` -> `king-0ce59769300c`; other models -> ""."""
    tail = (model or "").rsplit("/", 1)[-1]
    return tail if tail.startswith("king-") else ""


def digest12(king: str) -> str:
    """Accepts `king-<12>`, `<12>`, or a full sha256 -> `<12>`."""
    k = king.strip()
    if k.startswith("king-"):
        k = k[len("king-"):]
    return k[:12]


def resolve_current_king(snapshot_url: str = DEFAULT_SNAPSHOT_URL) -> dict:
    """The live king from the public dashboard API: {digest12, revision,
    reign_number, crowned_at}."""
    r = httpx.get(snapshot_url, timeout=30)
    r.raise_for_status()
    king = r.json()["king"]
    return {"digest12": king["revision"][:12], "revision": king["revision"],
            "reign_number": king.get("reign_number"),
            "crowned_at": king.get("crowned_at")}


def load_env_groups(path: Path = SOURCES_TOML) -> dict[str, str]:
    raw = tomllib.loads(path.read_text())
    return {name: cfg.get("group", "?") for name, cfg in raw.get("source", {}).items()}


def canon_args(args) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return args.strip()
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(args)


def tool_call_action(tool_calls: list[dict]) -> str:
    parts = []
    for c in tool_calls:
        fn = c.get("function") or c
        parts.append(f"{fn.get('name')}({canon_args(fn.get('arguments'))})")
    return "\n".join(parts)


def terminus_commands(action_json: str) -> str:
    try:
        obj = json.loads(action_json)
        cmds = obj.get("commands") or []
        keys = "\n".join(str(c.get("keystrokes", c)) if isinstance(c, dict)
                         else str(c) for c in cmds).strip()
        if keys:
            return keys
        return "<task_complete>" if obj.get("task_complete") else "<wait>"
    except (ValueError, AttributeError):
        return action_json


def obs_kind(obs: str | None) -> str:
    if obs is None:
        return "none"
    s = obs.strip()
    if not s or EMPTY_OBS_RE.match(norm_ws(s)):
        return "empty"
    head = s[:1500]
    if NUDGE_RE.search(head):
        return "nudge"
    if ERROR_RE.search(head):
        return "error"
    return "ok"


def node_plain(m: dict) -> str:
    text = message_text(m.get("content"))
    if m.get("tool_calls"):
        text += "\n" + tool_call_action(m["tool_calls"])
    return text


def reply_chains(nodes: list[dict]) -> list[list[int]]:
    """Root->node id chain for every sampled assistant node."""
    linear = "parent" not in nodes[-1]
    out = []
    for i, nd in enumerate(nodes):
        m = nd.get("message") or {}
        if m.get("role") != "assistant" or not nd.get("sampled"):
            continue
        if linear:
            out.append(list(range(i + 1)))
            continue
        chain, j = [], i
        while j is not None:
            chain.append(j)
            j = nodes[j].get("parent")
        chain.reverse()
        out.append(chain)
    return out


# ----------------------------------------------------------------------------
# trace store

@dataclass
class IndexRow:
    chunk: str
    line: int
    rollout_id: str
    source: str
    policy_id: str
    harness: str
    action_kind: str
    model: str
    king: str
    sid: str
    stop_condition: str
    outcome: str
    n_replies: int
    stored_at: str


def _index_chunk(path: str) -> list[dict]:
    rows = []
    name = os.path.basename(path)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                e = json.loads(line)
                t = e.get("trace") or {}
                pol = e.get("policy") or {}
                nodes = t.get("nodes") or []
                n_replies = sum(1 for nd in nodes if nd.get("sampled")
                                and (nd.get("message") or {}).get("role") == "assistant")
                rows.append(asdict(IndexRow(
                    chunk=name, line=i, rollout_id=e.get("rollout_id", ""),
                    source=e.get("source", ""), policy_id=pol.get("id", ""),
                    harness=pol.get("harness", ""),
                    action_kind=pol.get("action_kind") or dialects.DEFAULT_KIND,
                    model=pol.get("model", ""), king=king_digest_of(pol.get("model", "")),
                    sid=str((e.get("task") or {}).get("sid", "")),
                    stop_condition=t.get("stop_condition") or "",
                    outcome=rollout_outcome(t) if nodes else "errored",
                    n_replies=n_replies, stored_at=e.get("stored_at", ""))))
    except Exception as ex:  # a corrupt chunk must not sink the review
        rows.append({"chunk": name, "line": -1, "error": f"{type(ex).__name__}: {ex}"})
    return rows


class TraceStore:
    """Manifest -> local chunk cache -> rollout index (all cached on disk)."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR,
                 manifest_url: str = DEFAULT_TRACES_URL):
        self.cache_dir = Path(cache_dir)
        self.chunk_dir = self.cache_dir / "chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_url = manifest_url
        self.base_url = manifest_url.rsplit("/traces/", 1)[0]
        self.manifest: dict | None = None

    def sync(self, *, log=print, workers: int = 16) -> dict:
        r = httpx.get(self.manifest_url, timeout=60)
        r.raise_for_status()
        self.manifest = r.json()
        (self.cache_dir / "manifest.json").write_text(json.dumps(self.manifest))
        missing = [c for c in self.manifest["chunks"]
                   if not (self.chunk_dir / os.path.basename(c["key"])).exists()]
        log(f"traces: {len(self.manifest['chunks'])} chunks in manifest, "
            f"{len(missing)} to fetch")
        if missing:
            with httpx.Client(timeout=120) as client, ThreadPoolExecutor(workers) as pool:
                list(pool.map(lambda c: self._fetch(client, c), missing))
        return self.manifest

    def _fetch(self, client: httpx.Client, c: dict) -> None:
        dest = self.chunk_dir / os.path.basename(c["key"])
        for attempt in range(3):
            try:
                resp = client.get(f"{self.base_url}/{c['key']}")
                resp.raise_for_status()
                if hashlib.sha256(resp.content).hexdigest() != c["sha256"]:
                    raise ValueError(f"sha mismatch for {c['key']}")
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(resp.content)
                tmp.rename(dest)
                return
            except Exception:
                if attempt == 2:
                    raise

    def manifest_sha(self) -> str:
        if self.manifest is None:
            self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        return sha256_hex(json.dumps(sorted(c["key"] for c in self.manifest["chunks"])))

    def index(self, *, log=print, procs: int = 4) -> list[dict]:
        """Rollout index for the current manifest; cached per manifest and
        extended incrementally (only chunks not yet indexed are read)."""
        idx_path = self.cache_dir / "rollout_index.jsonl"
        rows: list[dict] = []
        done: set[str] = set()
        if idx_path.exists():
            for line in idx_path.open():
                r = json.loads(line)
                rows.append(r)
                done.add(r["chunk"])
        wanted = [os.path.basename(c["key"]) for c in
                  (self.manifest or json.loads((self.cache_dir / "manifest.json").read_text()))["chunks"]]
        todo = [str(self.chunk_dir / n) for n in wanted if n not in done
                and (self.chunk_dir / n).exists()]
        log(f"index: {len(rows)} rows cached, {len(todo)} chunks to index")
        if todo:
            with ProcessPoolExecutor(procs) as pool, idx_path.open("a") as out:
                for chunk_rows in pool.map(_index_chunk, todo, chunksize=8):
                    for r in chunk_rows:
                        out.write(json.dumps(r) + "\n")
                        rows.append(r)
        wanted_set = set(wanted)
        return [r for r in rows if "error" not in r and r["chunk"] in wanted_set]

    def load_envelope(self, chunk: str, line: int) -> dict:
        with gzip.open(self.chunk_dir / chunk, "rt", encoding="utf-8") as f:
            for i, raw in enumerate(f):
                if i == line:
                    return json.loads(raw)
        raise KeyError(f"{chunk}:{line}")


# ----------------------------------------------------------------------------
# one rollout, parsed into turns

@dataclass
class Turn:
    idx: int
    node_id: int
    reasoning: str
    visible: str
    action: str          # action string used for repeat detection ("" = none)
    action_raw: str      # the raw action span (fence / tool call / JSON)
    action_kind: str
    n_actions: int
    finish: str | None
    obs: str | None
    obs_kind: str
    prefix_chars: int
    # deterministic labels
    loop: str = "normal"           # normal | loop_onset | in_loop | escape
    repeats_turn: int | None = None
    repeat_exact: bool = False
    repeat_thought: bool = False
    no_action: bool = False
    is_last: bool = False
    completion: bool = False       # last reply of an agent_completed rollout
    token_cap: bool = False


@dataclass
class Rollout:
    rollout_id: str
    source: str
    env_group: str
    policy_id: str
    harness: str
    action_kind: str
    model: str
    king: str
    sid: str
    repo: str
    language: str
    stop_condition: str
    outcome: str
    task_prompt: str
    traj_id: str
    turns: list[Turn] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def loop_onsets(self) -> list[int]:
        return [t.idx for t in self.turns if t.loop == "loop_onset"]

    def det_labels(self, idx: int) -> list[str]:
        t = self.turns[idx]
        out = [t.loop]
        if t.no_action:
            out.append("no_action")
        if t.completion:
            out.append("completion")
        if t.token_cap:
            out.append("token_cap")
        if t.is_last:
            out.append("last_turn")
        return out


def extract_action(m: dict, kind: str, *, is_final: bool, agent_completed: bool
                   ) -> tuple[str, str, int, str]:
    content = FENCE_RE.sub("```bash\n", message_text(m.get("content")))
    if m.get("tool_calls"):
        act = tool_call_action(m["tool_calls"])
        return act, "tool_call", len(m["tool_calls"]), act
    acts = dialects.get(kind).actions(content)
    if len(acts) == 1:
        raw = acts[0]
        if kind == "terminus_json":
            return terminus_commands(raw), kind, 1, raw
        return raw, kind, 1, raw
    if (not acts and is_final and agent_completed
            and dialects.get(kind).ends_in_text and content.strip()):
        return content.strip(), "text", 1, content.strip()
    return "", kind, len(acts), ""


def task_prompt_of(envelope: dict) -> str:
    t = envelope.get("trace") or {}
    data = ((t.get("task") or {}).get("data") or {})
    for key in ("prompt", "problem_statement", "question", "instruction", "task"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v
    row = data.get("row") if isinstance(data.get("row"), dict) else {}
    for key in ("problem_statement", "prompt", "question", "problem", "instruction"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    # fall back to the first user message of the trace
    for nd in t.get("nodes") or []:
        m = nd.get("message") or {}
        if m.get("role") == "user":
            return message_text(m.get("content"))
    return ""


def parse_rollout(envelope: dict, env_groups: dict[str, str] | None = None) -> Rollout:
    t = envelope["trace"]
    pol = envelope.get("policy") or {}
    task = envelope.get("task") or {}
    kind = pol.get("action_kind") or dialects.DEFAULT_KIND
    nodes = t.get("nodes") or []
    agent_completed = t.get("stop_condition") == "agent_completed"
    finish = {c["node"]: c.get("finish_reason") for c in (t.get("calls") or [])
              if isinstance(c, dict) and "node" in c}
    ro = Rollout(
        rollout_id=envelope.get("rollout_id", ""), source=envelope.get("source", ""),
        env_group=(env_groups or {}).get(envelope.get("source", ""), "?"),
        policy_id=pol.get("id", ""), harness=pol.get("harness", ""), action_kind=kind,
        model=pol.get("model", ""), king=king_digest_of(pol.get("model", "")),
        sid=str(task.get("sid", "")), repo=task.get("repo") or "",
        language=task.get("language") or "",
        stop_condition=t.get("stop_condition") or "", outcome=rollout_outcome(t),
        task_prompt=task_prompt_of(envelope),
        traj_id=make_traj_id(str(task.get("sid", "")), run_tag(t)),
        errors=[str((e or {}).get("message", ""))[:300] for e in (t.get("errors") or [])])
    if not nodes:
        return ro
    chains = reply_chains(nodes)
    n_rep = len(chains)
    plain_cache: dict[int, str] = {}

    def plain(i: int) -> str:
        if i not in plain_cache:
            plain_cache[i] = node_plain(nodes[i]["message"])
        return plain_cache[i]

    obs: list[str | None] = []
    for k, chain in enumerate(chains):
        if k + 1 >= n_rep:
            obs.append(None)
            continue
        nxt = chains[k + 1]
        L = 0
        while L < len(chain) and L < len(nxt) and chain[L] == nxt[L]:
            L += 1
        parts = [plain(i) for i in nxt[L:-1]
                 if nodes[i]["message"].get("role") in ("user", "tool")]
        obs.append("\n".join(parts))

    norm_seen: dict[str, list[int]] = defaultdict(list)
    exact_seen: set[str] = set()
    thought_seen: set[str] = set()
    obs_heads: list[str | None] = []
    state = "normal"
    prev: tuple[str, str | None] | None = None
    for k, chain in enumerate(chains):
        node_id = chain[-1]
        m = nodes[node_id]["message"]
        is_final = k == n_rep - 1
        act, eff_kind, n_actions, raw = extract_action(
            m, kind, is_final=is_final, agent_completed=agent_completed)
        content = message_text(m.get("content"))
        reasoning = m.get("reasoning_content") or ""
        visible = content
        if raw and eff_kind != "text" and raw in content:
            visible = content[:content.index(raw)]
        o = obs[k]
        ohead = norm_ws(o)[:OBS_HEAD] if o is not None else None
        obs_heads.append(ohead)
        turn = Turn(
            idx=k, node_id=node_id, reasoning=reasoning, visible=visible.strip(),
            action=act, action_raw=raw, action_kind=eff_kind, n_actions=n_actions,
            finish=finish.get(node_id), obs=o, obs_kind=obs_kind(o),
            prefix_chars=sum(len(plain(i)) for i in chain[:-1]),
            no_action=not act, is_last=is_final,
            completion=is_final and agent_completed,
            token_cap=finish.get(node_id) == "length")
        if act:
            a_norm = norm_action(act)
            earlier = norm_seen.get(a_norm) or []
            repeats = earlier[-1] if earlier else None
            turn.repeat_exact = act.strip() in exact_seen
            same_obs = (repeats is not None and ohead is not None
                        and ohead == obs_heads[repeats])
            if repeats is not None and same_obs:
                turn.loop = "loop_onset" if state == "normal" else "in_loop"
                turn.repeats_turn = repeats
                state = "loop"
            elif state == "loop" and prev is not None:
                changed_act = a_norm != prev[0]
                changed_obs = ohead is not None and ohead != prev[1]
                if changed_act and changed_obs:
                    turn.loop = "escape"
                    state = "normal"
                elif changed_act and ohead is None:
                    turn.loop = "normal"
                else:
                    turn.loop = "in_loop"
                    turn.repeats_turn = repeats
            if reasoning:
                key = norm_ws(reasoning)
                turn.repeat_thought = key in thought_seen
                thought_seen.add(key)
            norm_seen[a_norm].append(k)
            exact_seen.add(act.strip())
            prev = (a_norm, ohead)
        ro.turns.append(turn)
    return ro


# ----------------------------------------------------------------------------
# transcript rendering for the judge

def clip(s: str, head: int, tail: int = 0) -> str:
    s = s or ""
    if len(s) <= head + tail:
        return s
    cut = len(s) - head - tail
    if tail:
        return f"{s[:head]}\n[... {cut} chars omitted ...]\n{s[-tail:]}"
    return f"{s[:head]}\n[... {cut} chars omitted ...]"


def short_action(t: Turn, limit: int = 600) -> str:
    if t.action_kind == "tool_call" or t.action_kind == "terminus_json":
        return clip(t.action, limit)
    return clip(t.action_raw or t.action, limit)


def render_transcript(ro: Rollout, *, max_chars: int, obs_head: int = 900,
                      obs_tail: int = 300, thought_chars: int = 1200) -> tuple[str, dict]:
    """Turn-indexed transcript. Consecutive turns that repeat the same
    normalized action AND get the same observation head are collapsed into
    one line that names the turn range, so a 60-turn loop costs one block.
    If the result is still over `max_chars`, the middle turns are elided
    (first and last thirds kept). Returns (text, stats)."""
    blocks: list[tuple[int, int, str]] = []  # (first_idx, last_idx, text)
    k = 0
    n = len(ro.turns)
    while k < n:
        t = ro.turns[k]
        j = k
        if t.action:
            a = norm_action(t.action)
            oh = norm_ws(t.obs or "")[:OBS_HEAD]
            while (j + 1 < n and ro.turns[j + 1].action
                   and norm_action(ro.turns[j + 1].action) == a
                   and norm_ws(ro.turns[j + 1].obs or "")[:OBS_HEAD] == oh
                   and ro.turns[j + 1].obs is not None):
                j += 1
        if j - k >= 2:
            head = render_turn(t, obs_head, obs_tail, thought_chars)
            body = (f"{head}\n[turns {k + 1}..{j} ({j - k} more turns): the agent "
                    f"repeated the SAME action and received the SAME observation each time]")
            blocks.append((k, j, body))
            k = j + 1
            continue
        blocks.append((k, k, render_turn(t, obs_head, obs_tail, thought_chars)))
        k += 1
    stats = {"n_turns": n, "n_blocks": len(blocks), "elided_middle": False}
    total = sum(len(b[2]) for b in blocks)
    if total > max_chars and len(blocks) > 6:
        keep_head = max(3, len(blocks) // 3)
        keep_tail = max(3, len(blocks) // 3)
        while keep_head + keep_tail < len(blocks):
            cur = sum(len(b[2]) for b in blocks[:keep_head]) + \
                sum(len(b[2]) for b in blocks[-keep_tail:])
            if cur <= max_chars:
                break
            if keep_head > 3:
                keep_head -= 1
            elif keep_tail > 3:
                keep_tail -= 1
            else:
                break
        if keep_head + keep_tail < len(blocks):
            lo = blocks[keep_head - 1][1] + 1
            hi = blocks[-keep_tail][0] - 1
            middle = (f"[turns {lo}..{hi} omitted for length -- {hi - lo + 1} turns; "
                      f"if you believe the pivot lies there, say so with lower confidence]")
            blocks = blocks[:keep_head] + [(lo, hi, middle)] + blocks[-keep_tail:]
            stats["elided_middle"] = True
            stats["elided_range"] = [lo, hi]
    text = "\n\n".join(b[2] for b in blocks)
    stats["chars"] = len(text)
    return text, stats


def render_turn(t: Turn, obs_head: int, obs_tail: int, thought_chars: int) -> str:
    lines = [f"### TURN {t.idx}"]
    if t.reasoning:
        lines.append("[thinking] " + clip(norm_ws(t.reasoning), thought_chars, 200))
    if t.visible and t.action_kind != "text":
        lines.append("[visible] " + clip(norm_ws(t.visible), 600, 100))
    if t.action:
        tag = "final reply" if t.action_kind == "text" else f"action ({t.action_kind})"
        lines.append(f"[{tag}] " + short_action(t))
    else:
        why = "hit the token cap" if t.token_cap else f"{t.n_actions} actions parsed"
        lines.append(f"[action] NONE -- no single parseable action ({why})")
    if t.obs is None:
        lines.append("[observation] (none -- last turn of the rollout)")
    else:
        lines.append(f"[observation:{t.obs_kind}] " + clip(t.obs.strip(), obs_head, obs_tail))
    return "\n".join(lines)


def render_teacher_summary(ro: Rollout, *, max_chars: int = 16000) -> str:
    """The teacher's rollout on the same task, compressed to its action
    sequence (one line per turn, short observation kind), plus its final
    reply. Enough to say what the teacher did differently; not a full
    transcript."""
    lines = [f"Teacher outcome: {ro.outcome} (stop: {ro.stop_condition}, {ro.n_turns} turns, "
             f"harness {ro.harness})"]
    for t in ro.turns:
        act = short_action(t, 220).replace("\n", " ")
        if not act:
            act = "NONE"
        lines.append(f"T{t.idx}: {act}  -> obs:{t.obs_kind}")
    if ro.turns and ro.turns[-1].action_kind == "text":
        lines.append("Teacher final reply: " + clip(norm_ws(ro.turns[-1].action), 1200))
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = clip(text, max_chars // 2, max_chars // 2)
    return text


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
