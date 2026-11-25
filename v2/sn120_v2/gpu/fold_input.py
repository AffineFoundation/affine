"""A future fold's serving input, without changing a sealed parent head.

Old MiMo heads omitted processor configs that vLLM needs. Their manifest and
verified cache stay byte-for-byte sealed. For a fold of the pinned MiMo
lineage only, this module hard-links (or copies) the parent into a separate
cache and adds missing small metadata from the pinned genesis revision.

The same prepared directory must feed merged-candidate evaluation and the
install fold: adding files only during publication would change base_hash
after the candidate was tested. Once published, a child carries every file,
so the fallback is needed only for an older parent.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from functools import cache
from pathlib import Path
from typing import Callable

from ..canonical import file_sha256
from ..errors import InfraError, VerifyError
from ..params import GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION, GenesisSpec
from ..types import HeadVersion
from .resident import hash_tree

# Carry every root-level small JSON/text metadata file of genesis, including
# processor configs, tokenizers, chat templates and special_tokens_map.json.
# No weight file is fetched: allow_patterns is applied to repository file names.
GENESIS_PATTERNS = ("*.json", "*.jinja", "chat_template", "merges.txt", "vocab.txt")
METADATA_BYTES = 64 * 1024**2

FoldInput = Callable[[Path, HeadVersion], Path]
Download = Callable[[str, str, Path], None]


def unchanged(base_dir: Path, head: HeadVersion) -> Path:
    """The default for simulations and local replication: no network, no copy."""
    return base_dir


def download_genesis(repo: str, revision: str, out: Path) -> None:
    """Fetch only root-level small model metadata, never weight shards."""
    from huggingface_hub import snapshot_download

    if not out.is_dir() or not any(out.iterdir()):
        snapshot_download(repo, revision=revision, local_dir=out, allow_patterns=list(GENESIS_PATTERNS))
    shutil.rmtree(out / ".cache", ignore_errors=True)


class FoldInputs:
    """Prepare old MiMo parents for training, serving, candidate folds and publication."""

    def __init__(self, work_dir: Path, genesis: GenesisSpec, download: Download = download_genesis) -> None:
        self.root = Path(work_dir) / "fold-inputs"
        self.genesis_dir = Path(work_dir) / "genesis-metadata" / genesis.model_revision
        self.genesis, self.download = genesis, download
        self._lock = threading.RLock()

    @cache
    def _metadata(self) -> dict[str, Path]:
        self.genesis_dir.mkdir(parents=True, exist_ok=True)
        self.download(self.genesis.model_repo, self.genesis.model_revision, self.genesis_dir)
        files = {}
        for path in sorted(self.genesis_dir.iterdir()):
            if (path.name == ".cache" or not path.is_file() or path.suffix == ".safetensors"
                    or path.name.endswith(".safetensors.index.json")):
                continue
            if path.stat().st_size > METADATA_BYTES:
                raise InfraError(f"genesis metadata {path.name} is larger than {METADATA_BYTES} bytes")
            files[path.name] = path
        return files

    def __call__(self, base_dir: Path, head: HeadVersion) -> Path:
        if (self.genesis.model_repo, self.genesis.model_revision) != (GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION):
            return base_dir
        names = {path.name for path in base_dir.iterdir() if path.is_file()}
        with self._lock:
            # Do not stop at the common vLLM files: a small tokenizer or
            # processor file present only at genesis must also reach the child.
            added = {name: path for name, path in self._metadata().items() if name not in names}
            if not added:
                return base_dir
            target = self.root / head.base_hash
            marker = self.root / f"{head.base_hash}.json"
            source = hash_tree(base_dir)
            extra = {name: file_sha256(path) for name, path in added.items()}
            expected = {"base_hash": head.base_hash, "parent_files": source, "genesis_files": extra,
                        "repo": self.genesis.model_repo, "revision": self.genesis.model_revision}
            if target.is_dir():
                if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != expected:
                    raise VerifyError(f"the prepared fold input {target} does not match its sealed parent")
                if hash_tree(target) != source | extra:
                    raise VerifyError(f"the prepared fold input {target} has missing, changed or extra files")
                return target
            if target.exists() or marker.exists():
                raise VerifyError(f"the prepared fold input {target} is incomplete; remove it and run again")
            stage = self.root / f"{head.base_hash}.part"
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir(parents=True)
            try:
                for name in sorted(source):
                    (stage / name).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(base_dir / name, stage / name)
                    except OSError:
                        shutil.copy2(base_dir / name, stage / name)
                for name, path in added.items():
                    shutil.copy2(path, stage / name)
                if any(file_sha256(stage / name) != sha for name, sha in (source | extra).items()):
                    raise VerifyError("a file changed while the fold input was prepared")
                self.root.mkdir(parents=True, exist_ok=True)
                stage.rename(target)
                temporary = marker.with_suffix(".part")
                temporary.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
                temporary.replace(marker)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
            return target
