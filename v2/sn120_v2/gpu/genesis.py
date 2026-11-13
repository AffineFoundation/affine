"""Build the genesis file sets on a node and print their identities.

    python -m sn120_v2.gpu.genesis --out <dir>

It downloads the pinned checkpoint files of the launch model to
``<dir>/base``, checks that they hash to the pinned base hash, and writes the
fp32 master to ``<dir>/master``: every floating-point tensor of the base in
float32, in weight files of the same names. bf16 to fp32 loses nothing. The
last line of the output is one JSON object; its ``master_digest`` is the
value for ``--master-digest`` of the cutover import.

The command writes local files only. It uploads nothing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

from huggingface_hub import snapshot_download

from ..canonical import manifest_digest
from ..errors import VerifyError
from ..params import GENESIS_BASE_HASH, GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION
from .folder import upcast
from .resident import hash_tree

# The base is these files plus every ``*.safetensors`` file of the revision.
# This is the file set that the pinned base hash names; other files of the
# repository (README, image processor configs) are not part of the head.
CHECKPOINT_FILES = (
    *("chat_template.jinja", "config.json", "generation_config.json", "merges.txt"),
    *("model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"),
)


def build(out: Path, repo: str, revision: str, base_hash: str) -> dict[str, str]:
    base, master = out / "base", out / "master"
    if not base.is_dir():
        snapshot_download(repo, revision=revision, local_dir=base, allow_patterns=[*CHECKPOINT_FILES, "*.safetensors"])
        shutil.rmtree(base / ".cache", ignore_errors=True)  # download bookkeeping is not a model file
    found = manifest_digest(hash_tree(base))
    if found != base_hash:
        raise VerifyError(f"the files in {base} hash to {found}, not to the pinned base hash {base_hash}")
    shutil.rmtree(master, ignore_errors=True)
    digest = upcast(base, master)
    return {"base_hash": found, "base_dir": str(base), "master_digest": digest, "master_dir": str(master)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the genesis bf16 base and fp32 master; print their identities.")
    parser.add_argument("--out", type=Path, required=True, help="directory for base/ and master/ (about 57 GB)")
    parser.add_argument("--repo", default=GENESIS_MODEL_REPO)
    parser.add_argument("--revision", default=GENESIS_MODEL_REVISION)
    parser.add_argument("--base-hash", default=GENESIS_BASE_HASH, help="the manifest digest the base files must have")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out, args.repo, args.revision, args.base_hash)))


if __name__ == "__main__":
    main()
