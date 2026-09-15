#!/usr/bin/env python
"""Teacher probe for king-group admission (improvement loop P4, 2026-09-14).

"Can the teacher answer here?" For a candidate turn the teacher is sampled
3x at the prefix under the duel's own settings (T = 0.8, 1,792 tokens =
max_thought 1024 + max_action 768, the turn's dialect parser). The fold
admits the turn only if >= 2 samples parse to an action and the parsed
actions are not all identical (identical refs = R == 0). Results are a
side-table the fold reads: one JSON line per turn_id in
`affine/state/teacher_probe/probes.jsonl`.

Two inputs:
  --pending  turns the fold held back for lack of a probe (the fold writes
             `affine/state/teacher_probe/pending.jsonl`: turn_id, group,
             kind, prefix messages);
  --published GROUP[,GROUP]  already-published turns of those stratum
             namespaces from the live index (prefix materialized from the
             public view chunks), for the one-off retire pass;
             `--per-stratum N` samples at most N turns per stratum (king_fail
             holds ~54 turns per stratum; the small groups ~1).

Teacher endpoint: Engy `qwen3.8-27b` (the same hosted copy of
Qwen/Qwen3.8-27B the recoverable pipeline uses; key ENGY_EVAL / ENGY_2), so
the probe never competes with the duel pods' teacher swarm. Cost: one call
per sample, ~6 s each, `--concurrency` in flight.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine import dialects  # noqa: E402
from affine.corpus.materialize import materialize_turn  # noqa: E402

STATE_DIR = REPO / "affine" / "state" / "teacher_probe"
PROBES_PATH = STATE_DIR / "probes.jsonl"
PENDING_PATH = STATE_DIR / "pending.jsonl"
PUBLIC_BASE = "https://data.affine.io"
MODEL = "qwen3.8-27b"
TEMPERATURE = 0.8
MAX_TOKENS = 1792
N_SAMPLES = 3
WS_RE = re.compile(r"\s+")


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def norm(s: str) -> str:
    return WS_RE.sub(" ", s).strip().lower()


def load_probed() -> set[str]:
    if not PROBES_PATH.exists():
        return set()
    return {json.loads(l)["turn_id"] for l in PROBES_PATH.read_text().split("\n") if l.strip()}


class Teacher:
    def __init__(self, key: str):
        self.h = httpx.Client(base_url="https://api.engy.ai/v1",
                              headers={"Authorization": f"Bearer {key}"}, timeout=300)

    def sample(self, messages: list[dict]) -> tuple[str, str | None]:
        for attempt in range(4):
            try:
                r = self.h.post("/chat/completions", json={
                    "model": MODEL, "messages": messages,
                    "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS})
                r.raise_for_status()
                ch = r.json()["choices"][0]
                return (ch["message"].get("content") or ""), ch.get("finish_reason")
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        return "", None


def probe_turn(teacher: Teacher, item: dict) -> dict:
    kind = item["kind"]
    d = dialects.get(kind)
    prefix = [{"role": m["role"], "content": m["content"]} for m in item["prefix"]]
    actions: list[str] = []
    kinds: list[str] = []
    finishes: list[str | None] = []
    texts: list[str] = []       # prose samples with no action in the dialect
    visible = 0
    for _ in range(N_SAMPLES):
        content, fin = teacher.sample(prefix)
        finishes.append(fin)
        visible += bool(content.strip())
        acts = d.actions(content) if d.mandate_ok(prefix) else []
        if len(acts) >= 1:
            actions.append(norm(acts[-1]))
            kinds.append(kind)
            continue
        if content.strip() and kind != "text":
            # Jacob's rule (2026-09-14): the teacher answers / says "done"
            # in prose where the king acted -> the fold may admit the turn
            # with kind `text`. `text` parses the whole visible reply, tool
            # XML included, so a stray tool call still counts as prose here.
            texts.append(norm(content))
        if dialects.get("tool_call").actions(content):
            kinds.append("tool_call")
        elif content.strip():
            kinds.append("text")
        else:
            kinds.append("none")
    return {
        "turn_id": item["turn_id"], "group": item.get("group"), "kind": kind,
        "n": N_SAMPLES, "n_valid": len(actions),
        "identical": len(actions) >= 2 and len(set(actions)) == 1,
        "n_distinct": len(set(actions)), "visible": visible,
        "text_valid": len(texts), "text_distinct": len(set(texts)),
        "finish": finishes, "sample_kinds": kinds,
        "prefix_sha": hashlib.sha256(json.dumps(prefix, sort_keys=True).encode()).hexdigest()[:16],
        "probed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def published_items(groups: list[str], per_stratum: int, seed: int = 7,
                    manifest_sha: str = "", only_ids: set[str] | None = None) -> list[dict]:
    """Published turns of the given stratum namespaces, prefixes from the
    view chunks. `manifest_sha` reads a historical manifest (rows retired
    since are still addressable through it); `only_ids` restricts."""
    h = httpx.Client(headers={"User-Agent": "affine-teacher-probe"}, follow_redirects=True, timeout=300)
    mkey = f"corpus/manifests/{manifest_sha}.json" if manifest_sha else "corpus/manifest.json"
    m = h.get(f"{PUBLIC_BASE}/{mkey}").json()
    t = pq.read_table(io.BytesIO(h.get(f"{PUBLIC_BASE}/{m['index']['key']}").content),
                      columns=["turn_id", "stratum", "chunk_key", "traj_line", "turn_idx", "action_kind"])
    rows = [dict(zip(t.column_names, r)) for r in zip(*(t.column(c).to_pylist() for c in t.column_names))]
    by_stratum: dict[str, list[dict]] = {}
    for r in rows:
        ns = str(r["stratum"]).split(":")[0]
        if ns in groups and (only_ids is None or r["turn_id"] in only_ids):
            by_stratum.setdefault(r["stratum"], []).append(r)
    rng = random.Random(seed)
    picked: list[dict] = []
    for s, rs in by_stratum.items():
        rng.shuffle(rs)
        picked.extend(rs[:per_stratum] if per_stratum > 0 else rs)
    log(f"published: {len(by_stratum)} strata in {groups}, {len(picked)} turns picked")
    cache: dict[str, list[str]] = {}
    items: list[dict] = []
    for r in picked:
        if r["chunk_key"] not in cache:
            cache[r["chunk_key"]] = gzip.decompress(
                h.get(f"{PUBLIC_BASE}/{r['chunk_key']}").content).decode().split("\n")
        rec = json.loads(cache[r["chunk_key"]][int(r["traj_line"])])
        meta = next(mm for mm in rec["turns"] if int(mm["turn_idx"]) == int(r["turn_idx"]))
        turn = materialize_turn(rec, meta)
        items.append({"turn_id": r["turn_id"], "group": str(r["stratum"]).split(":")[0],
                      "kind": r["action_kind"], "prefix": turn["prefix"]})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pending", action="store_true", help="probe the fold's pending.jsonl")
    ap.add_argument("--published", default="", help="comma list of stratum namespaces")
    ap.add_argument("--per-stratum", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-probe already probed turn ids")
    ap.add_argument("--groups", default="", help="comma list: restrict --pending items to these fold groups")
    ap.add_argument("--manifest-sha", default="", help="read --published rows from this historical manifest")
    ap.add_argument("--turn-ids", default="", help="file of turn ids to restrict --published to")
    ap.add_argument("--reprobe-text", action="store_true",
                    help="re-probe rows recorded before the text fields existed that have "
                         ">= 2 prose samples and < 2 valid actions (combine with --published/--manifest-sha)")
    args = ap.parse_args()
    key = os.environ.get("ENGY_EVAL") or os.environ.get("ENGY_2")
    if not key:
        sys.exit("ENGY_EVAL / ENGY_2 missing")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    if args.pending and PENDING_PATH.exists():
        seen: set[str] = set()
        for l in PENDING_PATH.read_text().split("\n"):
            if l.strip():
                it = json.loads(l)
                if it["turn_id"] not in seen:
                    seen.add(it["turn_id"])
                    items.append(it)
        if args.groups:
            want = {g.strip() for g in args.groups.split(",") if g.strip()}
            items = [it for it in items if it.get("group") in want]
        log(f"pending: {len(items)} turns")
    n_pending = len(items)
    only_ids: set[str] | None = None
    if args.turn_ids:
        only_ids = {l.strip() for l in Path(args.turn_ids).read_text().split("\n") if l.strip()}
    reprobe: set[str] = set()
    if args.reprobe_text and PROBES_PATH.exists():
        for l in PROBES_PATH.read_text().split("\n"):
            if l.strip():
                r = json.loads(l)
                if "text_distinct" not in r and r["n_valid"] < 2 and r.get("kind") != "text" \
                        and sum(k == "text" for k in r.get("sample_kinds") or []) >= 2:
                    reprobe.add(r["turn_id"])
        log(f"reprobe-text: {len(reprobe)} rows lack the text fields and have >= 2 prose samples")
        only_ids = reprobe if only_ids is None else (only_ids & reprobe)
    if args.published:
        items += published_items([g.strip() for g in args.published.split(",") if g.strip()],
                                 args.per_stratum, manifest_sha=args.manifest_sha, only_ids=only_ids)
    done = set() if (args.force or args.reprobe_text) else load_probed()
    if args.reprobe_text:
        items = [it for it in items if it["turn_id"] in reprobe]
    # The fold lists only turns it holds as unprobed (a row lacking the text
    # fields counts as unprobed), so pending items are never skipped.
    pending_ids = {it["turn_id"] for it in items[:n_pending]}
    items = [it for it in items if it["turn_id"] not in done or it["turn_id"] in pending_ids]
    if args.limit:
        items = items[:args.limit]
    log(f"probing {len(items)} turns x {N_SAMPLES} samples with concurrency {args.concurrency}")
    teacher = Teacher(key)
    n_ok = n_pass = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.concurrency) as ex, open(PROBES_PATH, "a") as out:
        futs = {ex.submit(probe_turn, teacher, it): it for it in items}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                row = fut.result()
            except Exception as e:  # keep going; the turn stays unprobed
                log(f"probe failed for {futs[fut]['turn_id']}: {type(e).__name__}: {e}")
                continue
            out.write(json.dumps(row) + "\n")
            out.flush()
            n_ok += 1
            n_pass += row["n_valid"] >= 2 and not row["identical"]
            if i % 50 == 0:
                log(f"{i}/{len(items)} probed, {n_pass}/{n_ok} pass, {time.time() - t0:.0f}s")
    log(f"done: {n_ok} probed, {n_pass} pass ({100 * n_pass / max(1, n_ok):.0f}%), "
        f"{time.time() - t0:.0f}s, {n_ok * N_SAMPLES} teacher samples")


if __name__ == "__main__":
    main()
