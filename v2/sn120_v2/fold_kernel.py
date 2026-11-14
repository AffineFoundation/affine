"""Replay one fold of the head: the same bytes on every machine.

A fold adds the update of each LoRA (``scale * B @ A``) to the fp32 master of
the head, and rounds the sum one time to the dtype of the base. This module
does that work a second time, away from the GPU box, and gives the file hashes
of the result. A reader compares them with the hashes in the public head
record. Equal hashes: the reader holds the head, bit for bit.

Only one step of a fold can give different bits on two machines: the matrix
product ``B @ A``. The other steps (one fp32 multiply by the scale, one fp32
add, one rounding to bf16) have one correct result in IEEE 754 arithmetic.
Each step of the head chain names the kernel that made its product:

``fixed-order-v1``
    For each element ``(i, j)`` of the product, in fp32::

        acc = +0.0
        for k = 0, 1, ..., rank - 1:   acc = fma(B[i, k], A[k, j], acc)
        update = fl32(scale) * acc

    ``fma`` is the fused multiply-add: the exact product plus ``acc``, rounded
    one time to the nearest fp32, ties to the even last bit. The scale is
    ``lora_alpha / rank`` rounded to fp32, and it is applied one time, after the
    sum. No library decides the order, so the bits are the same on every CPU
    and on a GPU. The code does each step in fp64 and rounds the fp64 sum to
    odd; the rounding to fp32 that follows then gives exactly the fused result.
``torch-matmul-v1``
    ``scale * (B @ A)`` by torch on the CPU, as ``gpu/folder.py`` does it. The
    matrix library of the torch build and the type of the CPU decide the bits.

The files are written one tensor at a time, so a replay needs little memory.
The bytes of each file are those that ``safetensors`` writes for the same
tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

import torch
from safetensors.torch import load_file

from .canonical import manifest_digest
from .errors import VerifyError

FIXED_ORDER, TORCH_MATMUL = "fixed-order-v1", "torch-matmul-v1"
KERNELS = (FIXED_ORDER, TORCH_MATMUL)
PART_BYTES = 4 * 1024**3  # the same limit as gpu/folder.py: one master part is one upload
_PREFIX, _A, _B = "base_model.model.", ".lora_A.weight", ".lora_B.weight"
_META = {"format": "pt"}
_FLOATS = {"F32": (torch.float32, 4), "BF16": (torch.bfloat16, 2), "F16": (torch.float16, 2)}
_BITS = {2: torch.int16, 4: torch.int32}
_BLOCK = 64 * 1024**2  # bytes of fp32 that one read of an unchanged tensor holds
# torch runs an operation on fewer elements than this in the thread that calls it.
# So the threads of this module do not fight with the threads of torch.
_CPU_CHUNK = 32768
_GPU_CHUNK = 1 << 24

# The five shapes (rows, columns) of the weights that a LoRA of the launch model changes.
SHAPES = {"gate_up": (12288, 4096), "down": (4096, 12288), "q": (8192, 4096), "kv": (1024, 4096), "o": (4096, 4096)}
# sha256 (first 16 characters) of the fixed-order-v1 update for ``mark_pair(shape, rank)``: (shape, rank, scale).
# A change of one bit in one element changes a mark.
MARKS = {
    ("gate_up", 64, 2.0): "51e48a5b1841ad44",
    ("down", 64, 2.0): "a4d6a3fa3b6a2e2e",
    ("q", 64, 2.0): "8f5395d927415d72",
    ("kv", 64, 2.0): "79393a1df5d101c3",
    ("o", 64, 2.0): "40b6362633e71b1f",
    ("gate_up", 16, 2.0): "907a0e1422e4dbe5",
    ("down", 16, 2.0): "9be08a1cc058ecc4",
    ("q", 16, 2.0): "255e2663918f34d7",
    ("kv", 16, 2.0): "c9c9e3b56d7488c4",
    ("o", 16, 2.0): "606d77c2145bddaf",
    ("kv", 16, 37.7 / 16): "7d76aaf6cc09de57",
    ("o", 64, 0.3): "3d72ac72183d1e64",
}


# ---- the product ----


def _fused_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``acc = fma(b[:, k], a[k, :], acc)`` for each rank in order. ``a`` and ``b`` are fp64 copies of fp32 values."""
    rows, columns, device = b.shape[0], a.shape[1], a.device
    acc = torch.zeros(rows, columns, dtype=torch.float32, device=device)
    product, before, total, part, lost = (torch.empty(rows, columns, dtype=torch.float64, device=device) for _ in range(5))
    bits, lost_bits = total.view(torch.int64), lost.view(torch.int64)
    for k in range(a.shape[0]):
        torch.mul(b[:, k : k + 1], a[k : k + 1, :], out=product)  # exact: 24 bits times 24 bits fit in the 53 bits of fp64
        before.copy_(acc)
        torch.add(product, before, out=total)  # rounded to the nearest fp64
        torch.sub(total, product, out=part)
        torch.sub(total, part, out=lost)
        torch.sub(product, lost, out=lost)
        before.sub_(part)
        lost.add_(before)  # exactly what the rounding of ``total`` lost (the error term of the two-sum)
        inexact = lost != 0
        # Round to odd: go back to the fp64 value nearer to zero, then set the last bit.
        # The rounding to fp32 that follows then gives the one-rounding result of the fused operation.
        away = ((lost_bits ^ bits) < 0) & inexact
        bits -= away.to(torch.int64)
        bits |= inexact.to(torch.int64)
        acc.copy_(total)
    return acc


def lora_update(
    a: torch.Tensor, b: torch.Tensor, scale: float, kernel: str = FIXED_ORDER, *, device: str = "cpu", threads: int | None = None, chunk: int | None = None
) -> torch.Tensor:
    """``scale * (b @ a)`` in fp32, on the CPU, for LoRA matrices ``a`` (rank x columns) and ``b`` (rows x rank).

    ``device`` is where the fixed-order kernel computes: ``cpu`` or ``cuda:N``. ``threads`` and ``chunk``
    (elements of the result that one piece of work holds) change the time, never the bits.
    A kernel name that this code does not know stops the work.
    """
    if kernel not in KERNELS:
        raise VerifyError(f"unknown kernel {kernel}; this code knows {', '.join(KERNELS)}")
    if a.dim() != 2 or b.dim() != 2 or b.shape[1] != a.shape[0]:
        raise VerifyError("the LoRA matrices do not fit together")
    a, b = a.float(), b.float()
    if kernel == TORCH_MATMUL:
        return scale * (b @ a)
    a, b = a.to(device=device, dtype=torch.float64), b.to(device=device, dtype=torch.float64)
    factor = torch.tensor(scale, dtype=torch.float32, device=device)  # the scale is rounded to fp32, as torch does it
    on_cpu = a.device.type == "cpu"
    rows = max(1, (chunk or (_CPU_CHUNK if on_cpu else _GPU_CHUNK)) // a.shape[1])
    out = torch.empty(b.shape[0], a.shape[1], dtype=torch.float32)

    def work(start: int) -> None:
        out[start : start + rows] = (_fused_rows(a, b[start : start + rows]) * factor).cpu()

    starts = range(0, b.shape[0], rows)
    workers = min(len(starts), (threads or os.cpu_count() or 1) if on_cpu else 1)
    if workers <= 1:
        for start in starts:
            work(start)
    else:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(work, starts))
    return out


def check_arithmetic(device: str = "cpu", threads: int | None = None) -> None:
    """Known answers that a machine gives only with IEEE 754 arithmetic. A wrong answer stops the work.

    They fail when small numbers are flushed to zero, when the rounding is not to the nearest even,
    and when an fp64 sum is rounded two times.
    """
    if sys.byteorder != "little":
        raise VerifyError("this code writes the bytes of a little-endian machine")
    one, x, y = 1 + 2.0**-23, 2.0**-12 * (1 + 2.0**-23), 2.0**-12 * (1 - 2.0**-23)
    for chunk in (1, 4):  # one row for each thread, and all rows in one piece
        # one + x*y is a little below the middle of two fp32 numbers; two roundings give the wrong one
        tie = lora_update(torch.tensor([[1.0, 1.0], [y, -y]]), torch.tensor([[one, x], [-one, x]]), 1.0, device=device, threads=threads, chunk=chunk)
        small = lora_update(torch.tensor([[2.0**-40, 2.0**10]]), torch.tensor([[2.0**-100], [2.0**-140]]), 1.0, device=device, threads=threads, chunk=chunk)
        if tie.tolist() != [[one, one], [-one, -one]] or small.tolist() != [[2.0**-140, 2.0**-90], [0.0, 2.0**-130]]:
            raise VerifyError(f"the arithmetic of {device} does not give the known answers of the fixed-order kernel")
    tiny = torch.full((1 << 17,), 2.0**-140)  # large, so that the threads of torch do this add
    halves = torch.tensor([1 + 2.0**-8, 1 + 3 * 2.0**-8, 1 + 2.0**-8 + 2.0**-23]).to(torch.bfloat16).float()
    if not bool((tiny + tiny / 2 == 1.5 * 2.0**-140).all()) or halves.tolist() != [1.0, 1 + 2.0**-6, 1 + 2.0**-7]:
        raise VerifyError("the fp32 add or the cast to bf16 on this machine does not give the known answers")


def mark_pair(shape: str, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Made-up LoRA matrices ``(a, b)`` in one of the real shapes, from whole-number arithmetic: the same on every machine."""

    def grid(rows: int, columns: int, salt: int) -> torch.Tensor:
        values = (torch.arange(rows * columns, dtype=torch.int64) * 2654435761 + salt * 40503) % 65521
        return ((values - 32760).to(torch.float32) / 32768.0).reshape(rows, columns)

    rows, columns = SHAPES[shape]
    return grid(rank, columns, rank), grid(rows, rank, rank + 1) / 1024.0


def mark(update: torch.Tensor) -> str:
    """The first 16 characters of the sha256 of the bytes of an fp32 tensor."""
    return hashlib.sha256(_raw(update)).hexdigest()[:16]


# ---- the LoRA files ----


@dataclass(frozen=True)
class Lora:
    """One LoRA as the fold uses it: the scale and, for each weight of the head, the matrices A and B."""

    scale: float
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]


def read_pairs(path: Path, rank: int) -> Lora:
    """Read one LoRA directory with the checks of ``gpu/folder.read_lora``. The product is not computed here."""
    config = json.loads((Path(path) / "adapter_config.json").read_text(encoding="utf-8"))
    tensors = load_file(str(Path(path) / "adapter_model.safetensors"))
    if config.get("r") != rank:
        raise VerifyError("the LoRA rank differs from the rank recorded in the head")
    pairs = {}
    for key in sorted(tensors):
        if not key.startswith(_PREFIX) or not key.endswith((_A, _B)):
            raise VerifyError(f"unexpected LoRA tensor {key}")
        if key.endswith(_A):
            a, b = tensors[key], tensors.get(key[: -len(_A)] + _B)
            if b is None or a.dim() != 2 or b.dim() != 2 or a.shape[0] != rank or b.shape[1] != rank:
                raise VerifyError(f"the LoRA pair of {key} is missing or has a wrong shape")
            pairs[key[len(_PREFIX) : -len(_A)] + ".weight"] = (a, b)
    if 2 * len(pairs) != len(tensors):
        raise VerifyError("a LoRA B matrix has no A matrix")
    return Lora(config["lora_alpha"] / rank, pairs)


# ---- safetensors files, one tensor at a time ----


@dataclass(frozen=True)
class Entry:
    """Where one tensor is in a safetensors file."""

    path: Path
    dtype: str
    shape: tuple[int, ...]
    start: int
    nbytes: int


def read_index(path: Path) -> dict[str, Entry]:
    """The tensors of one safetensors file, from its header. No tensor is read."""
    with open(path, "rb") as file:
        (length,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(length))
    header.pop("__metadata__", None)
    index = {}
    for name, item in header.items():
        start, end = item["data_offsets"]
        index[name] = Entry(Path(path), item["dtype"], tuple(item["shape"]), 8 + length + start, end - start)
    return index


def _blocks(entry: Entry, block: int) -> Iterator[torch.Tensor]:
    """The values of one floating tensor, as flat tensors of at most ``block`` bytes."""
    dtype, size = _FLOATS[entry.dtype]
    with open(entry.path, "rb") as file:
        file.seek(entry.start)
        left = entry.nbytes
        while left:
            raw = bytearray(file.read(min(left, block - block % size)))
            if not raw:
                raise VerifyError(f"{entry.path.name} is shorter than its header says")
            left -= len(raw)
            yield torch.frombuffer(raw, dtype=_BITS[size]).view(dtype)


def _raw(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(_BITS[tensor.element_size()]).numpy().tobytes()


def header_bytes(entries: Sequence[tuple[str, str, tuple[int, ...], int]], metadata: dict[str, str] | None = _META) -> bytes:
    """The first bytes of a safetensors file that holds ``(name, dtype, shape, bytes)`` tensors of one dtype."""
    if len({dtype for _, dtype, _, _ in entries}) > 1:
        raise VerifyError("this writer puts tensors of one dtype in one file")
    table: dict[str, object] = {} if metadata is None else {"__metadata__": metadata}
    at = 0
    for name, dtype, shape, nbytes in sorted(entries):
        table[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [at, at + nbytes]}
        at += nbytes
    text = json.dumps(table, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text


class _Out:
    """One output file: the sha256 of all bytes, and the file itself when ``directory`` is given."""

    def __init__(self, directory: Path | None, name: str):
        self.name, self.sha = name, hashlib.sha256()
        self.file: BinaryIO | None = open(directory / name, "wb") if directory else None

    def write(self, data: bytes) -> None:
        self.sha.update(data)
        if self.file:
            self.file.write(data)

    def close(self) -> str:
        if self.file:
            self.file.close()
        return self.sha.hexdigest()


def _parts(index: dict[str, Entry], part_bytes: int) -> list[list[str]]:
    """The tensor names of each master part: name order, at most ``part_bytes`` of fp32 in one part."""
    parts: list[list[str]] = [[]]
    size = 0
    for name in sorted(index):
        nbytes = 4 * (index[name].nbytes // _FLOATS[index[name].dtype][1])
        if nbytes > part_bytes:
            raise VerifyError(f"{name} alone is larger than one upload")
        if parts[-1] and size + nbytes > part_bytes:
            parts.append([])
            size = 0
        parts[-1].append(name)
        size += nbytes
    return parts


def _check_floats(index: dict[str, Entry], where: str) -> None:
    for name, entry in index.items():
        if entry.dtype not in _FLOATS:
            raise VerifyError(f"{where}: {name} is {entry.dtype}; this tool handles the floating dtypes {', '.join(_FLOATS)}")


@dataclass(frozen=True)
class Folded:
    """The file hashes of one replayed fold. The two digests are the ones in a head record."""

    base_files: dict[str, str]
    master_files: dict[str, str]

    @property
    def base_hash(self) -> str:
        return manifest_digest(self.base_files)

    @property
    def master_digest(self) -> str:
        return manifest_digest(self.master_files)


def _new_dir(path: Path | None) -> Path | None:
    if path is not None:
        Path(path).mkdir(parents=True)
    return None if path is None else Path(path)


def upcast_master(base_dir: Path, out_master: Path | None, part_bytes: int = PART_BYTES) -> dict[str, str]:
    """The first fp32 master of a base (genesis): ``{file name: sha256}``. With ``out_master=None`` nothing is written."""
    out_master, files = _new_dir(out_master), {}
    for shard in sorted(Path(base_dir).glob("*.safetensors")):
        index = read_index(shard)
        _check_floats(index, shard.name)
        for number, names in enumerate(_parts(index, part_bytes)):
            out = _Out(out_master, f"{shard.stem}.part{number:02d}.safetensors")
            out.write(header_bytes([(name, "F32", index[name].shape, 4 * (index[name].nbytes // _FLOATS[index[name].dtype][1])) for name in names]))
            for name in names:
                for block in _blocks(index[name], _BLOCK):
                    out.write(_raw(block.float()))
            files[out.name] = out.close()
    return files


def replay_fold(
    master_dir: Path,
    base_dir: Path,
    loras: Sequence[Lora],
    out_master: Path | None,
    out_base: Path | None,
    *,
    kernel: str = FIXED_ORDER,
    part_bytes: int = PART_BYTES,
    device: str = "cpu",
    threads: int | None = None,
) -> Folded:
    """Fold ``loras``, in the given order, into the fp32 master. Returns the file hashes of the new master and base.

    ``base_dir`` gives the weight file names, the dtype of each tensor and the files that a fold does not change.
    An output directory may be ``None``: then that output is hashed and not written.
    Every check of ``gpu/folder.fold_files`` is here too, and each one fails closed.
    """
    if kernel not in KERNELS:
        raise VerifyError(f"unknown kernel {kernel}; this code knows {', '.join(KERNELS)}")
    if kernel == FIXED_ORDER:
        check_arithmetic(device, threads)
    master_dir, base_dir = Path(master_dir), Path(base_dir)
    out_master, out_base = _new_dir(out_master), _new_dir(out_base)
    waiting = {name for lora in loras for name in lora.pairs}
    unread = {file.name for file in master_dir.iterdir()}
    base_files: dict[str, str] = {}
    master_files: dict[str, str] = {}
    for shard in sorted(base_dir.glob("*.safetensors")):
        base, master = read_index(shard), {}
        for part in sorted(master_dir.glob(f"{shard.stem}.part*.safetensors")):
            master.update(read_index(part))
            unread.discard(part.name)
        if sorted(master) != sorted(base):
            raise VerifyError(f"{shard.name}: the fp32 master and the base hold different tensors")
        _check_floats(base, shard.name)
        for name, entry in master.items():
            if entry.dtype != "F32":
                raise VerifyError(f"{name} is {entry.dtype} in the master; a master is fp32 and is never upcast here")
            if entry.shape != base[name].shape:
                raise VerifyError(f"{name} has one shape in the master and another in the base")
        base_out = _Out(out_base, shard.name)
        base_out.write(header_bytes([(name, entry.dtype, entry.shape, entry.nbytes) for name, entry in base.items()]))
        for number, names in enumerate(_parts(master, part_bytes)):
            master_out = _Out(out_master, f"{shard.stem}.part{number:02d}.safetensors")
            master_out.write(header_bytes([(name, "F32", master[name].shape, master[name].nbytes) for name in names]))
            for name in names:
                entry, dtype = master[name], _FLOATS[base[name].dtype][0]
                updates = [lora for lora in loras if name in lora.pairs]
                for weight in _blocks(entry, entry.nbytes if updates else _BLOCK):
                    for lora in updates:
                        a, b = lora.pairs[name]
                        if (b.shape[0], a.shape[1]) != entry.shape:
                            raise VerifyError(f"the LoRA update of {name} has a wrong shape")
                        weight = weight + lora_update(a, b, lora.scale, kernel, device=device, threads=threads).reshape(-1)
                    if not bool(torch.isfinite(weight).all()):
                        raise VerifyError(f"{name} is not finite after the fold")
                    master_out.write(_raw(weight))
                    base_out.write(_raw(weight.to(dtype)))
                waiting.discard(name)
            master_files[master_out.name] = master_out.close()
        base_files[base_out.name] = base_out.close()
    if unread or not base_files:
        raise VerifyError("the fp32 master does not hold exactly the weight files of the base")
    if waiting:
        raise VerifyError(f"{len(waiting)} LoRA tensors match no weight of the head, for example {sorted(waiting)[0]}")
    for file in sorted(base_dir.iterdir()):  # config and tokenizer files do not change
        if file.is_file() and file.suffix != ".safetensors":
            base_files[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
            if out_base:
                shutil.copy2(file, out_base / file.name)
    return Folded(base_files, master_files)


# ---- the command ----


def _compare(label: str, files: dict[str, str], record: dict[str, str] | None) -> bool:
    """Print one line for each file. With a record: EQUAL or DIFFERENT for each file."""
    equal = record is None or sorted(files) == sorted(record)
    for name in sorted(files):
        same = "" if record is None else ("  EQUAL" if record.get(name) == files[name] else "  DIFFERENT")
        equal = equal and "DIFFERENT" not in same
        print(f"{label} {files[name]}  {name}{same}")
    print(f"{label} digest {manifest_digest(files)}" + ("" if record is None else f"  record {manifest_digest(record)}  {'EQUAL' if equal else 'DIFFERENT'}"))
    return equal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.fold_kernel", description="Replay one fold of the head and compare its file hashes with a head record.")
    commands = parser.add_subparsers(dest="command", required=True)
    up = commands.add_parser("upcast", help="build the first fp32 master from the genesis base")
    up.add_argument("base_dir", type=Path)
    fold = commands.add_parser("fold", help="fold LoRAs into a master")
    fold.add_argument("master_dir", type=Path)
    fold.add_argument("base_dir", type=Path)
    fold.add_argument("--lora", action="append", required=True, metavar="DIR:RANK", help="one LoRA directory and its rank in the head record; give it one time for each LoRA, in the order of the head")
    fold.add_argument("--kernel", choices=KERNELS, default=FIXED_ORDER, help="the kernel that the step names in the head chain")
    marks = commands.add_parser("marks", help="run fixed-order-v1 on made-up matrices of the real shapes and compare with the known marks")
    for command in (fold, marks):
        command.add_argument("--device", default="cpu", help="where fixed-order-v1 computes: cpu or cuda:N")
        command.add_argument("--threads", type=int, default=None, help="CPU threads; they change the time of fixed-order-v1, never its bits")
    fold.add_argument("--out-base", type=Path, default=None, help="write the new base here; without it the base is only hashed")
    for command in (up, fold):
        command.add_argument("--out-master", type=Path, default=None, help="write the fp32 master here; without it the master is only hashed")
        command.add_argument("--part-bytes", type=int, default=PART_BYTES)
        command.add_argument("--record", type=Path, default=None, help="a head record (public/heads/<version_id>.json) to compare the file hashes with")
    args = parser.parse_args(argv)
    started = time.monotonic()
    if args.command == "marks":
        check_arithmetic(args.device, args.threads)
        equal = True
        for (shape, rank, scale), want in MARKS.items():
            a, b = mark_pair(shape, rank)
            at = time.monotonic()
            got = mark(lora_update(a, b, scale, device=args.device, threads=args.threads))
            equal = equal and got == want
            print(f"{FIXED_ORDER} {args.device} {shape} rank {rank} scale {scale:.6g}: {got}  {'EQUAL' if got == want else 'DIFFERENT'}  {time.monotonic() - at:.2f} s")
        print(f"{time.monotonic() - started:.0f} s")
        return 0 if equal else 1
    record = json.loads(args.record.read_text(encoding="utf-8")) if args.record else {}
    if args.command == "upcast":
        result = Folded({}, upcast_master(args.base_dir, args.out_master, args.part_bytes))
        equal = True
    else:
        if args.threads and args.kernel == TORCH_MATMUL:
            torch.set_num_threads(args.threads)
        loras = [read_pairs(Path(item.rsplit(":", 1)[0]), int(item.rsplit(":", 1)[1])) for item in args.lora]
        result = replay_fold(args.master_dir, args.base_dir, loras, args.out_master, args.out_base, kernel=args.kernel, part_bytes=args.part_bytes, device=args.device, threads=args.threads)
        equal = _compare("base", result.base_files, record.get("base_files"))
    equal = _compare("master", result.master_files, record.get("master_files")) and equal
    print(f"{time.monotonic() - started:.0f} s")
    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main())
