"""Build the genesis file sets on a node, write their manifests and stage them in the store.

    python -m sn120_v2.gpu.genesis --out <dir> [--upload]

It downloads the pinned checkpoint files of the launch model to
``<dir>/base``, checks that they hash to the pinned base hash, and writes the
fp32 master to ``<dir>/master``: every floating-point tensor of the base in
float32, in weight files of at most 4 GiB each. bf16 to fp32 loses nothing.

It writes ``<dir>/genesis_files.json``: ``{"base_files": {name: sha256},
"master_files": {name: sha256}}``, the input of ``bootstrap``. The last line
of the output is one JSON object; its ``master_digest`` is the fp32 master
digest that the cutover import needs.

With ``--upload`` it stages every file in the store named by the
``SN120_R2_*`` variables, at the public keys of the base and the master. Then
it reads every stored file back and compares its sha256 with the local file.
A difference removes that object and stops the command. Without ``--upload``
it writes local files only.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

from huggingface_hub import snapshot_download

from .. import layout
from ..canonical import manifest_digest
from ..errors import VerifyError
from ..params import GENESIS_BASE_HASH, GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION
from ..r2 import R2Store
from ..seams import ObjectStore
from .folder import upcast
from .resident import hash_tree

# The base is these files plus every ``*.safetensors`` file of the revision.
# This is the file set that the pinned base hash names; other files of the
# repository (README, image processor configs) are not part of the head.
CHECKPOINT_FILES = (
    *("chat_template.jinja", "config.json", "generation_config.json", "merges.txt"),
    *("model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"),
)


FILES_JSON = "genesis_files.json"


def build(out: Path, repo: str, revision: str, base_hash: str) -> dict[str, str]:
    base, master = out / "base", out / "master"
    if not base.is_dir():
        snapshot_download(repo, revision=revision, local_dir=base, allow_patterns=[*CHECKPOINT_FILES, "*.safetensors"])
        shutil.rmtree(base / ".cache", ignore_errors=True)  # download bookkeeping is not a model file
    base_files = hash_tree(base)
    found = manifest_digest(base_files)
    if found != base_hash:
        raise VerifyError(f"the files in {base} hash to {found}, not to the pinned base hash {base_hash}")
    shutil.rmtree(master, ignore_errors=True)
    digest = upcast(base, master)
    files = {"base_files": base_files, "master_files": hash_tree(master)}
    if manifest_digest(files["master_files"]) != digest:
        raise VerifyError(f"the files in {master} changed while they were hashed")
    (out / FILES_JSON).write_text(json.dumps(files, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"base_hash": found, "base_dir": str(base), "master_digest": digest, "master_dir": str(master), "genesis_files": str(out / FILES_JSON)}


def stage(store: ObjectStore, out: Path, result: dict[str, str], workers: int = 4) -> int:
    """Upload every base and master file to its public key, then check the stored bytes. Returns the file count."""
    files = json.loads((out / FILES_JSON).read_text(encoding="utf-8"))
    jobs = [(layout.public_base_file(result["base_hash"], name), out / "base" / name, sha) for name, sha in files["base_files"].items()]
    jobs += [(layout.public_master_file(result["master_digest"], name), out / "master" / name, sha) for name, sha in files["master_files"].items()]

    def one(job: tuple[str, Path, str]) -> None:
        key, path, sha = job
        # A file of the right size from an earlier run is not sent again; its bytes are still checked below.
        if not store.exists(key) or store.size(key) != path.stat().st_size:
            store.put_file(key, path)
        if store.sha256(key) != sha:
            store.delete(key)
            raise VerifyError(f"the stored copy of {key} differs from the local file; it was removed")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, jobs))
    return len(jobs)


def main(argv: Sequence[str] | None = None, open_store: Callable[[], ObjectStore] = R2Store.from_env) -> None:
    parser = argparse.ArgumentParser(description="Build the genesis bf16 base and fp32 master, write genesis_files.json, and stage the files with --upload.")
    parser.add_argument("--out", type=Path, required=True, help="directory for base/, master/ and genesis_files.json (about 57 GB)")
    parser.add_argument("--repo", default=GENESIS_MODEL_REPO)
    parser.add_argument("--revision", default=GENESIS_MODEL_REVISION)
    parser.add_argument("--base-hash", default=GENESIS_BASE_HASH, help="the manifest digest the base files must have")
    parser.add_argument("--upload", action="store_true", help="stage the files in the store of the SN120_R2_* variables and check them there")
    args = parser.parse_args(argv)
    store = open_store() if args.upload else None  # a wrong store setting stops the command before the download
    result = build(args.out, args.repo, args.revision, args.base_hash)
    result["uploaded_files"] = stage(store, args.out, result) if store is not None else 0
    print(json.dumps(result))


if __name__ == "__main__":
    main()
