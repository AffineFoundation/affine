"""Regression gate: the duel_turns@v3 view must reproduce both legacy
pipelines on real archived fixtures.

  verifiers path   vendor_primelane.convert.convert_run + validate_records
                   vs adapters.verifiers + views.duel_turns
                   -> byte-equal records (vendor's generated_at is injected
                      into the new path; it is a wall-clock stamp, not
                      content)
  mini_swe path    datagen.slicer.slice_traj_file
                   vs adapters.mini_swe + views.duel_turns.derive_turns
                   -> byte-equal modulo the two additive tag fields the
                      unified pipeline stamps on every record
                      (source, language)

  PYTHONPATH=rollouts python rollouts/checks/regression_duel_turns.py
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vendor_primelane import convert as vendor_convert  # noqa: E402

from datagen.slicer import slice_traj_file  # noqa: E402

from rollouts.adapters.mini_swe import traj_to_trace  # noqa: E402
from rollouts.adapters.verifiers import envelopes_from_traces  # noqa: E402
from rollouts.panel import panel_keys  # noqa: E402
from rollouts.schema import PolicyStamp, make_envelope  # noqa: E402
from rollouts.views.duel_turns import (  # noqa: E402
    derive_turns,
    validate_records,
)

MODEL_LABEL = "engy/glm-5.2"
STAMP = PolicyStamp(policy_id="regression", model=MODEL_LABEL,
                    harness="mini_swe_textbased", endpoint="engy")


def fake_meta(uid: str) -> dict:
    """Catalog identity for fixture tasks (the real catalogs live on the
    pod). Both paths receive the same meta, so equality still exercises
    the full slicing + validation machinery."""
    sid = uid.split("/")[-1].replace(".", "_").replace(":", "_")
    return {"uid": uid, "sid": f"{sid}-0", "repo": f"fixture/{sid}".lower(),
            "language": "python", "image": ""}


def check_verifiers_fixture(path: Path) -> tuple[int, int]:
    episodes = [json.loads(line)
                for line in gzip.open(path, "rt", encoding="utf-8")
                if line.strip()]
    uids = {t["task"]["data"].get("name") or "unknown"
            for ep in episodes for t in ep.get("traces", [])}
    meta_by_uid = {uid: fake_meta(uid) for uid in uids}
    panel = panel_keys()

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        with gzip.open(path, "rb") as fin, \
                open(run_dir / "traces.jsonl", "wb") as fout:
            fout.write(fin.read())
        old_records, _ = vendor_convert.convert_run(
            run_dir, "fixture_src", MODEL_LABEL, meta_by_uid, *panel)
    old_kept, old_drops = vendor_convert.validate_records(
        old_records, *panel)

    # Vendor stamps one wall-clock generated_at per convert_run call.
    generated_at = old_records[0]["generated_at"] if old_records else None
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        with gzip.open(path, "rb") as fin, \
                open(run_dir / "traces.jsonl", "wb") as fout:
            fout.write(fin.read())
        envelopes, unknown = envelopes_from_traces(
            run_dir / "traces.jsonl", source="fixture_src",
            env_id="fixture-env", meta_by_uid=meta_by_uid, policy=STAMP)
        assert not unknown, f"unknown uids: {unknown}"
    new_records = []
    for env in envelopes:
        new_records.extend(
            derive_turns(env, panel=panel, generated_at=generated_at))
    new_kept, new_drops = validate_records(new_records, panel)

    old_dump = [json.dumps(r, sort_keys=True) for r in old_kept]
    new_dump = [json.dumps(r, sort_keys=True) for r in new_kept]
    assert old_drops == new_drops, (old_drops, new_drops)
    assert old_dump == new_dump, (
        f"{path.name}: record mismatch "
        f"(old={len(old_dump)} new={len(new_dump)})")
    return len(old_kept), len(envelopes)


def check_mini_swe_fixture(path: Path) -> tuple[int, int]:
    raw = gzip.open(path, "rb").read()
    with tempfile.TemporaryDirectory() as td:
        traj = Path(td) / path.name.removesuffix(".gz")
        traj.write_bytes(raw)
        old_records = slice_traj_file(traj, model_label=MODEL_LABEL)
        generated_at = old_records[0]["generated_at"] if old_records else None

    data = json.loads(raw)
    iid = data.get("instance_id") or "unknown"
    repo = iid.rsplit("-", 1)[0].replace("__", "/", 1)
    trace = traj_to_trace(raw, instance_id=iid, model_label=MODEL_LABEL,
                          endpoint="engy", resolved=None,
                          generated_at=generated_at)
    task = {"uid": iid, "sid": iid, "repo": repo, "language": "python",
            "image": ""}
    env = make_envelope(source="swerebench_main", env_id="mini_swe",
                        task=task, policy=STAMP, trace=trace)
    new_records = derive_turns(env, panel=panel_keys())

    stripped = [{k: v for k, v in r.items()
                 if k not in ("source", "language")} for r in new_records]
    old_dump = [json.dumps(r, sort_keys=True) for r in old_records]
    new_dump = [json.dumps(r, sort_keys=True) for r in stripped]
    assert old_dump == new_dump, (
        f"{path.name}: record mismatch "
        f"(old={len(old_dump)} new={len(new_dump)})")
    return len(old_records), 1


def main() -> None:
    results = {}
    for path in sorted(FIXTURES.glob("*.jsonl.gz")):
        kept, n_rollouts = check_verifiers_fixture(path)
        results[path.name] = {"rollouts": n_rollouts, "kept_turns": kept,
                              "match": True}
    for path in sorted(FIXTURES.glob("*.traj.json.gz")):
        kept, n_rollouts = check_mini_swe_fixture(path)
        results[path.name] = {"rollouts": n_rollouts, "kept_turns": kept,
                              "match": True}
    print(json.dumps({"ok": True, "fixtures": results}, indent=1))


if __name__ == "__main__":
    main()
