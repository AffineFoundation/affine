"""Slice resolved mini-swe-agent trajectories into corpus turn records.

One record per clean assistant turn, drop-in compatible with the production
duel corpus (affine/scripts/corpus_push.py contract; consumed by
evalsrv/dueling.py):

  traj_id         "{owner__repo}.{sha8}.pr_{N}__{runtag}" — repo stratum is
                  the segment before the first ".", the pr_N segment matches
                  the stratification phase regex
  turn_idx        assistant ordinal within the trajectory (int)
  prefix          chat messages [{role, content}] ending on a user message
  reference_turn  the assistant message (THOUGHT + one closed ```bash block)
  suffix_len / n_prefix_chars

plus tag fields the duel code ignores: instance_id, repo, model, phase
(early/mid/late thirds of the trajectory, for later oversampling),
action_kind, generated_at.

Standalone use on a directory of *.traj.json files:

  python -m datagen.slicer --traj-dir DIR --out turns.jsonl \
      [--resolved-ids ids.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("datagen.slicer")

# Same action pattern as the duel chat contract (evalsrv/chat.py BASH_RE).
BASH_RE = re.compile(r"```bash\n.*?\n```", re.DOTALL)
# Foreign mini v2 scaffold fence; a no-op on trajectories generated with the
# datagen agent config, which already speaks ```bash natively.
FOREIGN_FENCE_RE = re.compile(r"```mswea_bash_command[ \t]*\n")
TRAILING_NUM_RE = re.compile(r"-(\d+)$")
# Serving-window cap from affine/scripts/corpus_push.py.
MAX_PREFIX_CHARS = 120_000
# Reference-leakage predicate, byte-for-byte the ops/datagen_refresh.py
# prefilter (which drops such records downstream anyway — 4.6-5% of early
# shards): whole ```bash block, whitespace collapsed + lowercased, longer
# than 40 chars, substring of any normalized prefix message. Catches agents
# re-running a command verbatim and every submit turn (the submit command is
# quoted in the task instructions).
LEAK_MIN_CHARS = 40
WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return FOREIGN_FENCE_RE.sub("```bash\n", text)


def _norm_ws(text: str) -> str:
    return WS_RE.sub(" ", text).strip().lower()


def _message_text(msg: dict) -> str | None:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content
                 if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(parts) if parts else None
    return None


def make_traj_id(instance_id: str, run_tag: str) -> str:
    m = TRAILING_NUM_RE.search(instance_id)
    num = m.group(1) if m else "0"
    stem = instance_id[: m.start()] if m else instance_id
    sha8 = hashlib.sha256(instance_id.encode()).hexdigest()[:8]
    return f"{stem}.{sha8}.pr_{num}__{run_tag}"


def _phase(assistant_idx: int, n_assistant: int) -> str:
    frac = assistant_idx / max(1, n_assistant)
    if frac < 1 / 3:
        return "early"
    if frac < 2 / 3:
        return "mid"
    return "late"


def slice_messages(messages: list[dict], *, instance_id: str, repo: str,
                   model: str, run_tag: str, generated_at: str,
                   max_prefix_chars: int = MAX_PREFIX_CHARS) -> list[dict]:
    msgs: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            continue  # drop mini's terminal "exit" bookkeeping message
        text = _message_text(m)
        if text is None:
            continue
        msgs.append({"role": role, "content": _normalize(text)})

    n_assistant = sum(1 for m in msgs if m["role"] == "assistant")
    traj_id = make_traj_id(instance_id, run_tag)
    norm_contents = [_norm_ws(m["content"]) for m in msgs]
    records: list[dict] = []
    n_leaked = 0
    a_idx = 0
    for pos, msg in enumerate(msgs):
        if msg["role"] != "assistant":
            continue
        idx = a_idx
        a_idx += 1
        prefix = msgs[:pos]
        if not prefix or prefix[-1]["role"] != "user":
            continue
        ref = msg["content"]
        # Exactly one closed action block: format-error responses and
        # ambiguous multi-block replies are real history (they stay in later
        # prefixes) but not scorable references.
        blocks = BASH_RE.findall(ref)
        if len(blocks) != 1:
            continue
        # Reference leaked into the prefix: the turn would be dropped by the
        # corpus-refresh prefilter and leakage-gated at duel time. The turn
        # stays in later records' prefix history (it is real history).
        body = _norm_ws(blocks[0])
        if (len(body) > LEAK_MIN_CHARS
                and any(body in norm for norm in norm_contents[:pos])):
            n_leaked += 1
            continue
        n_prefix_chars = sum(len(p["content"]) for p in prefix)
        if n_prefix_chars > max_prefix_chars:
            break  # prefixes only grow from here
        records.append({
            "traj_id": traj_id,
            "turn_idx": idx,
            "prefix": [dict(p) for p in prefix],
            "reference_turn": ref,
            "suffix_len": len(ref),
            "n_prefix_chars": n_prefix_chars,
            "instance_id": instance_id,
            "repo": repo,
            "model": model,
            "phase": _phase(idx, n_assistant),
            "action_kind": "bash",
            "generated_at": generated_at,
        })
    if n_leaked:
        log.info("slicer: %s: %d leaked reference turn(s) skipped, %d kept",
                 instance_id, n_leaked, len(records))
    return records


def slice_traj_file(path: Path, model_label: str | None = None,
                    max_prefix_chars: int = MAX_PREFIX_CHARS) -> list[dict]:
    """model_label overrides the traj-recorded litellm model string with the
    provider-qualified tag (e.g. engy/glm-5.2) the loop knows for this run."""
    raw = path.read_bytes()
    data = json.loads(raw)
    info = data.get("info", {})
    instance_id = (data.get("instance_id") or info.get("instance_id")
                   or path.name.removesuffix(".traj.json"))
    model = (model_label
             or info.get("config", {}).get("model", {}).get("model_name")
             or "unknown")
    repo = instance_id.rsplit("-", 1)[0].replace("__", "/", 1)
    # Content-derived tag: deterministic, and unique even if the same
    # instance is ever rolled out twice (turn_id collisions would make the
    # duel ref cache score against the wrong reference).
    run_tag = hashlib.sha256(raw).hexdigest()[:8]
    generated_at = datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    return slice_messages(
        data.get("messages", []), instance_id=instance_id, repo=repo,
        model=model, run_tag=run_tag, generated_at=generated_at,
        max_prefix_chars=max_prefix_chars)


def _load_resolved_ids(path: Path) -> set[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("resolved_ids", [])
    return {str(i) for i in data}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Slice mini-swe-agent trajectories into corpus turns")
    ap.add_argument("--traj-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resolved-ids", type=Path, default=None,
                    help="JSON list (or {'resolved_ids': [...]}) of instance "
                         "ids to keep; default keeps every trajectory")
    ap.add_argument("--max-prefix-chars", type=int, default=MAX_PREFIX_CHARS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    keep = _load_resolved_ids(args.resolved_ids) if args.resolved_ids else None
    n_trajs = n_skipped = 0
    records: list[dict] = []
    for path in sorted(args.traj_dir.rglob("*.traj.json")):
        recs = slice_traj_file(path, max_prefix_chars=args.max_prefix_chars)
        iid = recs[0]["instance_id"] if recs else None
        if keep is not None and (iid is None or iid not in keep):
            n_skipped += 1
            continue
        n_trajs += 1
        records.extend(recs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({
        "trajectories": n_trajs, "skipped": n_skipped,
        "turns": len(records), "out": str(args.out),
    }))


if __name__ == "__main__":
    main()
