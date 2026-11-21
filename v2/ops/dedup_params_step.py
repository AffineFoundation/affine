#!/usr/bin/env python3
"""Write a full set-params file from a state directory's pinned journal Params.

This reads only journal.jsonl, writes only the requested output file, and never
opens a controller, store, chain, network connection or production service.
Every pinned serialized field, including king-x2 and the recipe, is retained
except for the reviewed release overrides below. The seven dedup fields, crown
threshold, merged-evaluation switch and this checkout's GPU code digest are set
explicitly so omitted legacy fields cannot reset or select stale worker code.

Use between rounds, after reviewing the full file and printed expected diff:

  python -m sn120_v2 set-params --state-dir ... --stack sn120_v2.assembly:stack --params <file>

The expected hashes/diff describe that set-params operation. They are only valid
for the journal read here; rerun if the pinned Params change before applying.
The diff compares effective values, so an explicit field whose value was already
its legacy default does not appear in the diff (all seven fields are still set).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

# Resolve this checkout under python -I or with a different editable install,
# without replacing an already-loaded package (and losing its submodule attrs).
_package_dir = Path(__file__).resolve().parents[1] / "sn120_v2"
_package = sys.modules.get("sn120_v2")
if _package is None:
    _spec = importlib.util.spec_from_file_location(
        "sn120_v2", _package_dir / "__init__.py", submodule_search_locations=[str(_package_dir)]
    )
    assert _spec is not None and _spec.loader is not None
    _package = importlib.util.module_from_spec(_spec)
    sys.modules["sn120_v2"] = _package
    _spec.loader.exec_module(_package)

from sn120_v2.canonical import canonical_json, file_sha256, manifest_digest
from sn120_v2.cli import changed_params
from sn120_v2.errors import ConfigError
from sn120_v2.journal import JOURNAL_FILE
from sn120_v2.params import Params
from sn120_v2.state import load

DEDUP_FIELDS = {
    "train_seed_scope": "round",
    "dedup_enabled": True,
    "dedup_rollout_overlap": 0.5,
    "dedup_cos_shared_seed": 0.135,
    "dedup_cos_own_seed": 0.002,
    "min_signal_rollouts": 1,
    "min_step_norm": 0.0,
    "z_bar": 1.5,
    "eval_merged": True,
}


def prepare(state_dir: str | Path) -> tuple[Params, Params, dict[str, object]]:
    """Return pinned and proposed Params, and the exact full override object."""
    pinned = load(Path(state_dir) / JOURNAL_FILE).params
    if pinned is None:
        raise ConfigError(f"the journal in {state_dir} has no pinned Params")
    full = pinned.to_dict() | DEDUP_FIELDS
    gpu = Path(__file__).resolve().parents[1] / "sn120_v2/gpu"
    digest = manifest_digest({path.name: file_sha256(path) for path in sorted(gpu.glob("*.py"))})
    full["recipe"] = dict(full["recipe"], code_digest=digest)
    # set-params parses new overrides with load_params/from_jsonable, not
    # Params.from_dict. This preserves all serialized fields without importing
    # the pinned record's _legacy_fields omission bookkeeping.
    proposed = Params.from_json(canonical_json(full))
    return pinned, proposed, full


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, help="directory containing journal.jsonl; read only")
    parser.add_argument("--output", required=True, help="full parameter JSON to write; must not already exist")
    args = parser.parse_args(argv)
    pinned, proposed, full = prepare(args.state_dir)
    output = Path(args.output)
    # Never overwrite a previously reviewed operator file. The journal and all
    # other state files are untouched; only this new artifact is created.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(full, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "journal": str(Path(args.state_dir) / JOURNAL_FILE),
        "output": str(output),
        "params_hash_before": pinned.params_hash,
        "params_hash": proposed.params_hash,
        "changed": changed_params(pinned, proposed),
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
