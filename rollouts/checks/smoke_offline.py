"""Offline smoke: fixtures -> envelopes -> store -> index -> view -> state
-> scheduler picks. No docker, no model APIs, no network.

  python rollouts/checks/smoke_offline.py
"""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

from rollouts.adapters.verifiers import envelopes_from_traces
from rollouts.index import RolloutIndex
from rollouts.panel import panel_keys
from rollouts.registry import load_registry
from rollouts.scheduler import Scheduler, UnifiedState
from rollouts.schema import PolicyStamp
from rollouts.store import TraceStore
from rollouts.views.duel_turns import derive_and_validate

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
STAMP = PolicyStamp(policy_id="glm_textbased", model="engy/glm-5.2",
                    harness="mini_swe_textbased", endpoint="engy")


def fake_meta(uid: str) -> dict:
    sid = uid.split("/")[-1].replace(".", "_").replace(":", "_")
    return {"uid": uid, "sid": f"{sid}-0", "repo": f"fixture/{sid}".lower(),
            "language": "python", "image": ""}


def main() -> None:
    registry = load_registry()
    panel = panel_keys()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = TraceStore(root / "traces")
        index = RolloutIndex(root / "traces")
        state = UnifiedState(root / "state.jsonl")
        env = {"ENGY": "x", "OPENROUTER": "x", "HF_TOKEN": "x"}
        sched = Scheduler(registry, state, env=env)

        source_cycle = ["swesmith", "terminal_lego", "swelego"]
        fixture_by_source = {
            "swesmith": "swesmith-20260811T095956Z.jsonl.gz",
            "terminal_lego": "terminal_lego-20260811T034038Z.jsonl.gz",
            "swelego": "swelego-20260810T215043Z.jsonl.gz",
        }
        for src_name in source_cycle:
            path = FIXTURES / fixture_by_source[src_name]
            episodes = [json.loads(line) for line in
                        gzip.open(path, "rt", encoding="utf-8")
                        if line.strip()]
            uids = {t["task"]["data"].get("name") or "unknown"
                    for ep in episodes for t in ep.get("traces", [])}
            meta = {uid: fake_meta(uid) for uid in uids}
            with tempfile.TemporaryDirectory() as rd:
                traces = Path(rd) / "traces.jsonl"
                with gzip.open(path, "rb") as fin, open(traces, "wb") as f:
                    f.write(fin.read())
                envelopes, unknown = envelopes_from_traces(
                    traces, source=src_name,
                    env_id=registry.sources[src_name].taskset_id,
                    meta_by_uid=meta, policy=STAMP)
            assert not unknown
            kept, drops = derive_and_validate(envelopes, panel)
            kept_by_rollout: dict[str, int] = {}
            sid_to_rid = {e["task"]["sid"]: e["rollout_id"]
                          for e in envelopes}
            for rec in kept:
                rid = sid_to_rid[rec["instance_id"]]
                kept_by_rollout[rid] = kept_by_rollout.get(rid, 0) + 1
            chunk = store.append_batch(envelopes, f"{src_name}-smoke")
            index.append(envelopes, chunk, kept_by_rollout)
            for e in envelopes:
                state.mark(src_name, e["task"]["uid"], "unresolved",
                           policy_id="glm_textbased",
                           n_turns=kept_by_rollout.get(e["rollout_id"], 0))
            print(f"{src_name}: {len(envelopes)} rollouts, "
                  f"{len(kept)} kept turns, drops={drops}")

        # Round-trip: store contents match what went in.
        stored = list(store.iter_envelopes())
        assert len(stored) == 13, len(stored)
        # Index aggregates match state aggregates.
        idx_kept = {k[0]: v for k, v in
                    RolloutIndex(root / "traces").kept_turns_by(
                        "source").items()}
        assert idx_kept == state.kept_by_source, (
            idx_kept, state.kept_by_source)
        print("index == state kept-turn totals:", idx_kept)

        # Scheduler: swesmith is now far ahead of target for its share;
        # the pick must be a source with zero kept turns.
        remaining = {name: 100 for name in registry.sources}
        pick = sched.pick_source(remaining)
        assert state.kept_by_source.get(pick, 0) == 0, pick
        policy = sched.pick_policy(pick)
        assert policy.id == "glm_textbased"
        print("scheduler pick:", pick, "policy:", policy.id)

        # Zero-yield cooldown kicks in after 3 dead batches.
        for _ in range(3):
            sched.record_batch_yield("nl2repobench", 0)
        elig = sched.eligible(remaining)
        assert "nl2repobench" not in elig
        print("cooldown works; eligible now:", sorted(elig))

        # Restart-safety: state reloads to the same aggregates.
        state2 = UnifiedState(root / "state.jsonl")
        assert state2.kept_by_source == state.kept_by_source
        assert state2.kept_by_policy == state.kept_by_policy
        print("state restart round-trip ok")
    print(json.dumps({"ok": True}))


if __name__ == "__main__":
    main()
