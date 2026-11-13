"""The ``Folder`` seam: merge the head's LoRAs into a new fp32 master and a new bf16 base.

The fold starts from the verified fp32 master, never from the bf16 base:
rounding to bf16 happens one time, at the end, so it cannot add up over
folds. The master and the base go to two directories and two prefixes. The
arithmetic runs on the CPU, one weight file at a time.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file

from .. import layout
from ..canonical import manifest_digest
from ..errors import ContractError, VerifyError
from ..params import Params
from ..seams import ObjectStore
from ..types import FoldArtifact, HeadVersion
from .plan import BoxConfig, check_pins
from .resident import Resident, hash_tree, upload_tree

_PREFIX, _A, _B = "base_model.model.", ".lora_A.weight", ".lora_B.weight"
_META = {"format": "pt"}


def read_lora(path: Path, rank: int) -> dict[str, torch.Tensor]:
    """``{weight name: scale * B @ A}`` in fp32 for one LoRA directory. Any surprise in the file fails closed."""
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    tensors = load_file(str(path / "adapter_model.safetensors"))
    if config.get("r") != rank:
        raise VerifyError("the LoRA rank differs from the rank recorded in the head")
    deltas = {}
    for key in sorted(tensors):
        if not key.startswith(_PREFIX) or not key.endswith((_A, _B)):
            raise VerifyError(f"unexpected LoRA tensor {key}")
        if key.endswith(_A):
            a, b = tensors[key].float(), tensors.get(key[: -len(_A)] + _B)
            if b is None or a.shape[0] != rank or b.shape[1:] != (rank,):
                raise VerifyError(f"the LoRA pair of {key} is missing or has a wrong shape")
            deltas[key[len(_PREFIX) : -len(_A)] + ".weight"] = (config["lora_alpha"] / rank) * (b.float() @ a)
    if 2 * len(deltas) != len(tensors):
        raise VerifyError("a LoRA B matrix has no A matrix")
    return deltas


def fold_files(master_dir: Path, base_dir: Path, loras: Sequence[dict[str, torch.Tensor]], out_master: Path, out_base: Path) -> None:
    """Write the new master (fp32) and the new base (each tensor in the dtype the old base has)."""
    out_master.mkdir(parents=True)
    out_base.mkdir(parents=True)
    waiting = {name for deltas in loras for name in deltas}
    shards = sorted(master_dir.glob("*.safetensors"))
    if not shards or sorted(p.name for p in base_dir.glob("*.safetensors")) != [p.name for p in shards]:
        raise VerifyError("the fp32 master and the base do not have the same weight files")
    for shard in shards:
        master, base = load_file(str(shard)), load_file(str(base_dir / shard.name))
        if sorted(master) != sorted(base):
            raise VerifyError(f"{shard.name}: the master and the base hold different tensors")
        for name, weight in master.items():
            if weight.is_floating_point() and weight.dtype != torch.float32:
                raise VerifyError(f"{name} is {weight.dtype} in the master; a master is fp32 and is never upcast here")
            for deltas in loras:
                if name in deltas:
                    if deltas[name].shape != weight.shape:
                        raise VerifyError(f"the LoRA update of {name} has a wrong shape")
                    weight = weight + deltas[name]
            if weight.is_floating_point() and not bool(torch.isfinite(weight).all()):
                raise VerifyError(f"{name} is not finite after the fold")
            master[name], base[name] = weight, weight.to(base[name].dtype)
            waiting.discard(name)
        save_file(master, str(out_master / shard.name), metadata=_META)
        save_file(base, str(out_base / shard.name), metadata=_META)
    if waiting:
        raise VerifyError(f"{len(waiting)} LoRA tensors match no weight of the head, for example {sorted(waiting)[0]}")
    for file in base_dir.iterdir():  # config and tokenizer files do not change
        if file.is_file() and file.suffix != ".safetensors":
            shutil.copy2(file, out_base / file.name)


def upcast(base_dir: Path, out_dir: Path) -> str:
    """Build the first fp32 master from a base. Returns its digest. Only genesis may use this."""
    out_dir.mkdir(parents=True)
    for shard in sorted(base_dir.glob("*.safetensors")):
        tensors = {name: t.float() if t.is_floating_point() else t for name, t in load_file(str(shard)).items()}
        save_file(tensors, str(out_dir / shard.name), metadata=_META)
    return manifest_digest(hash_tree(out_dir))


class LoraFolder:
    def __init__(self, store: ObjectStore, config: BoxConfig):
        self.store, self.config = store, config
        self.resident = Resident(store, config.work_dir)

    def fold(self, head: HeadVersion, params: Params, out_prefix: str) -> FoldArtifact:
        check_pins(self.config.image_digest, params)
        if not head.loras:
            raise ContractError("the head has no LoRA to fold")
        hashes = [ref.lora_hash for ref in head.loras]
        self.resident.prune({head.base_hash, head.base_fp32_master_digest, *hashes})
        try:
            master_dir = self.resident.tree(layout.public_master_prefix(head.base_fp32_master_digest), head.base_fp32_master_digest)
        except KeyError as exc:
            raise VerifyError("a file of the fp32 master is missing") from exc
        base_dir = self.resident.tree(layout.public_base_prefix(head.base_hash), head.base_hash)
        loras = [read_lora(self.resident.tree(layout.public_lora_prefix(ref.lora_hash), ref.lora_hash), ref.rank) for ref in head.loras]
        out = Path(self.config.work_dir) / "fold" / head.version_id
        shutil.rmtree(out, ignore_errors=True)
        try:
            fold_files(master_dir, base_dir, loras, out / "master", out / "base")
            base_files, _ = upload_tree(self.store, out / "base", f"{out_prefix}/base")
            master_files, _ = upload_tree(self.store, out / "master", f"{out_prefix}/master")
            artifact = FoldArtifact(
                parent_version_id=head.version_id,
                base_hash=manifest_digest(base_files),
                base_files=base_files,
                base_fp32_master_digest=manifest_digest(master_files),
                master_files=master_files,
                prefix=out_prefix,
                image_digest=self.config.image_digest,
            )
            self.resident.adopt(out / "base", artifact.base_hash)  # the next round needs no download
            self.resident.adopt(out / "master", artifact.base_fp32_master_digest)
            return artifact
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    print(upcast(Path(sys.argv[1]), Path(sys.argv[2])))
