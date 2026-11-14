"""The replay of a fold: the bytes of the GPU box's fold, and a product that no library decides."""

from __future__ import annotations

import json
import math
import struct
import threading
import time
from fractions import Fraction
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import load_file, save_file  # noqa: E402 - after the skip guard

from sn120_v2.canonical import manifest_digest  # noqa: E402
from sn120_v2.errors import VerifyError  # noqa: E402
from sn120_v2 import fold_kernel  # noqa: E402
from sn120_v2.fold_kernel import FIXED_ORDER, MARKS, TORCH_MATMUL, Folded, check_arithmetic, header_bytes, lora_update, main, mark, mark_pair, read_index, read_pairs, replay_fold, upcast_master  # noqa: E402
from sn120_v2.gpu.folder import fold_files, read_lora, upcast  # noqa: E402
from sn120_v2.gpu.resident import hash_tree  # noqa: E402

PART = 1500  # bytes of fp32 in one master part: small, so each weight file of the tiny head has some parts
LAYER = "model.language_model.layers.{}.mlp.{}.weight"


def f32(value: float) -> float:
    """The fp32 number nearest to ``value``, as a Python float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def nearest_f32(exact: Fraction) -> float:
    """``exact`` rounded one time to fp32, ties to the even last bit. Slow and plain: this is the reference."""
    near = torch.tensor(float(exact), dtype=torch.float32)
    best = None
    for other in (float("-inf"), float("inf")):
        for candidate in (near, torch.nextafter(near, torch.tensor(other, dtype=torch.float32))):
            even = candidate.view(torch.int32).item() % 2 == 0
            rank = (abs(Fraction(candidate.item()) - exact), not even)
            if best is None or rank < best[0]:
                best = (rank, candidate.item())
    return best[1]


def negative(value: float) -> bool:
    return math.copysign(1.0, value) < 0


def reference(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    """fixed-order-v1 one element at a time, with exact fractions: the definition, not the fast code."""
    out = torch.empty(b.shape[0], a.shape[1], dtype=torch.float32)
    for i in range(b.shape[0]):
        for j in range(a.shape[1]):
            acc = 0.0
            for k in range(a.shape[0]):
                left, right = b[i, k].item(), a[k, j].item()
                exact = Fraction(left) * Fraction(right) + Fraction(acc)  # one fused multiply-add
                # IEEE 754: an exact sum of zero is +0, but (-0) + (-0) is -0
                acc = nearest_f32(exact) if exact else (-0.0 if negative(left) != negative(right) and negative(acc) else 0.0)
            exact = Fraction(f32(scale)) * Fraction(acc)
            out[i, j] = nearest_f32(exact) if exact else (-0.0 if negative(scale) != negative(acc) else 0.0)
    return out


def same_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.shape == right.shape and bool((left.view(torch.int32) == right.view(torch.int32)).all())


def wide(shape: tuple[int, int], seed: int) -> torch.Tensor:
    """fp32 values over many powers of two, so that sums lose low bits and cancel."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator) * torch.pow(2.0, torch.randint(-30, 6, shape, generator=generator).float())


@pytest.fixture
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


# ---- the product ----


def test_fixed_order_is_one_fused_multiply_add_for_each_rank_in_order_and_the_scale_after_the_sum():
    a, b = wide((5, 9), 1), wide((7, 5), 2)
    for scale in (2.0, 37.7 / 16):
        assert same_bits(lora_update(a, b, scale), reference(a, b, scale))
    turned = lora_update(a.flip(0), b.flip(1), 2.0)  # the same products, summed in the other order
    assert torch.allclose(turned, lora_update(a, b, 2.0), rtol=1e-4) and not same_bits(turned, lora_update(a, b, 2.0))


def test_fixed_order_is_exact_where_a_plain_fp64_sum_rounds_two_times():
    # The exact sum is 1 + 2^-23 + 2^-24 - 2^-70: a little below the middle of two fp32 numbers.
    # One rounding gives 1 + 2^-23. An fp64 sum first rounds to the middle, and then to 1 + 2^-22.
    one, x, y = 1 + 2.0**-23, 2.0**-12 * (1 + 2.0**-23), 2.0**-12 * (1 - 2.0**-23)
    a = torch.tensor([[1.0, 1.0], [y, -y]], dtype=torch.float32)
    b = torch.tensor([[one, x], [-one, x]], dtype=torch.float32)
    got = lora_update(a, b, 1.0)
    assert same_bits(got, reference(a, b, 1.0))
    assert got.tolist() == [[one, one], [-one, -one]]
    plain = (b.double()[:, :1] * a.double()[:1, :] + b.double()[:, 1:] * a.double()[1:, :]).float()
    assert plain[0, 0].item() == 1 + 2.0**-22  # the fault that rounding to odd prevents
    check_arithmetic()  # the same cases, as the check that every replay runs first


def test_fixed_order_keeps_very_small_numbers_and_the_sign_of_a_sum():
    tiny = lora_update(torch.tensor([[2.0**-40, 2.0**10]]), torch.tensor([[2.0**-100], [2.0**-140]]), 1.0)
    assert tiny.tolist() == [[2.0**-140, 2.0**-90], [0.0, 2.0**-130]]  # no flush to zero, and one rounding below the smallest number
    a, b = torch.tensor([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]), torch.tensor([[3.0, -3.0], [-0.0, 0.0], [-2.0, 0.0]])
    got = lora_update(a, b, 2.0)
    assert same_bits(got, reference(a, b, 2.0))
    assert same_bits(got, torch.tensor([[0.0, -12.0, 0.0], [0.0, 0.0, 0.0], [-4.0, 4.0, 0.0]]))  # a sum that cancels is +0


def near_the_middle(rows: int, columns: int, seed: int, low: int, high: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank 3. After rank 0 each sum is an fp32 number ``c``. Rank 1 adds a little less than one half of the
    last bit of ``c``: the right result is ``c``, and an fp64 sum that is then rounded to fp32 gives the next number.
    Rank 2 adds a product of any size. The powers of two go from ``low`` to ``high``."""
    generator = torch.Generator().manual_seed(seed)
    power = lambda count: torch.randint(low, high, (count,), generator=generator).double()  # noqa: E731
    sign = lambda count: torch.randint(0, 2, (count,), generator=generator).double() * 2 - 1  # noqa: E731
    row_power, column_power = power(rows), power(columns)
    odd = 1 + (2 * torch.randint(0, 2**22, (columns,), generator=generator).double() + 1) * 2.0**-23
    b = torch.stack([sign(rows) * 2.0**row_power, 2.0 ** (row_power - 12) * (1 + 2.0**-23), sign(rows) * 2.0 ** power(rows) * 1.2345678], dim=1)
    a = torch.stack([sign(columns) * odd * 2.0**column_power, sign(columns) * 2.0 ** (column_power - 12) * (1 - 2.0**-23), 2.0 ** power(columns) * 0.87654321])
    return a.float(), b.float()


def plain(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The fault that the kernel must not have: each step as an fp64 sum that is then rounded to fp32."""
    acc = torch.zeros(b.shape[0], a.shape[1])
    for k in range(a.shape[0]):
        acc = (b.double()[:, k : k + 1] * a.double()[k : k + 1, :] + acc.double()).float()
    return acc


@pytest.mark.parametrize(("low", "high"), [(-20, 20), (-90, -55)])
def test_the_quick_way_and_the_careful_way_give_the_bits_of_the_definition(low, high, monkeypatch):
    # (-20, 20): sums in the middle of two fp32 numbers. (-90, -55): the same below 2**-126, where fp32 keeps fewer bits.
    careful = []
    one_rounding = fold_kernel._one_rounding
    monkeypatch.setattr(fold_kernel, "_one_rounding", lambda product, before: careful.append(product.numel()) or one_rounding(product, before))
    a, b = near_the_middle(12, 14, 7, low, high)
    got = lora_update(a, b, 1.0)
    assert 0 < sum(careful) < 3 * got.numel()  # some elements went the careful way, and not all
    assert same_bits(got, reference(a, b, 1.0))
    assert not same_bits(got, plain(a, b))  # this data finds the fault of two roundings
    a, b = near_the_middle(150, 130, 8, low, high)
    every = fold_kernel._fused_rows(a.double(), b.double(), every=True)
    assert same_bits(lora_update(a, b, 1.0), every) and not same_bits(every, plain(a, b))
    for seed in (9, 10):  # and data with no plan
        a, b = wide((5, 300), seed) * 2.0**low, wide((200, 5), seed + 50) * 2.0**low
        assert same_bits(lora_update(a, b, 1.0), fold_kernel._fused_rows(a.double(), b.double(), every=True))


def test_fixed_order_at_the_ends_of_fp32():
    low, one, edge = 1 - 2.0**-23, 1 + 2.0**-23, 2.0**-126 - 2.0**-149
    # the largest number below 2**-126, plus a little less than one half of its last bit: it stays
    a = torch.tensor([[2.0**-63 * low, -(2.0**-63) * low], [2.0**-75 * low, -(2.0**-75) * low]])
    b = torch.tensor([[2.0**-63, 2.0**-75 * one], [-(2.0**-63), -(2.0**-75) * one]])
    got = lora_update(a, b, 1.0)
    assert got.tolist() == [[edge, -edge], [-edge, edge]] and same_bits(got, reference(a, b, 1.0))
    assert plain(a, b)[0, 0].item() == 2.0**-126  # two roundings go up to the next number
    # a sum too small for fp32 is zero with the sign of the sum, and the next rank starts from it
    a, b = torch.tensor([[2.0**-80, -(2.0**-80)], [3.0, 5.0]]), torch.tensor([[2.0**-80, 2.0**-140], [2.0**-80, -0.0], [2.0**-80, 0.0]])
    got = lora_update(a, b, 1.0)
    assert same_bits(got, torch.tensor([[3 * 2.0**-140, 5 * 2.0**-140], [0.0, -0.0], [0.0, 0.0]])) and same_bits(got, reference(a, b, 1.0))
    # more than the largest fp32 number: not finite, so the fold stops
    huge = lora_update(torch.tensor([[2.0**100, 1.0]]), torch.tensor([[2.0**100], [2.0**127]]), 2.0)
    assert huge.tolist() == [[float("inf"), 2.0**101], [float("inf"), float("inf")]]


def test_the_check_of_the_arithmetic_stops_a_machine_that_flushes_small_numbers():
    check_arithmetic(threads=2)
    assert torch.set_flush_denormal(True)
    try:
        with pytest.raises(VerifyError, match="known answers"):
            check_arithmetic()
    finally:
        torch.set_flush_denormal(False)
    check_arithmetic()


def test_whole_tensor_marks_fail_on_any_change_of_bits():
    for shape, rank, scale in (("kv", 64, 2.0), ("kv", 16, 37.7 / 16), ("o", 16, 2.0)):
        a, b = mark_pair(shape, rank)
        update = lora_update(a, b, scale)
        assert mark(update) == MARKS[shape, rank, scale]
        update.view(torch.int32)[-1, -1] ^= 1  # one unit in the last place of one element
        assert mark(update) != MARKS[shape, rank, scale]
    assert {shape for shape, _, _ in MARKS} == {"gate_up", "down", "q", "kv", "o"} and {rank for _, rank, _ in MARKS} == {16, 64}


def test_threads_and_chunks_change_no_bit_of_fixed_order():
    a, b = wide((4, 33), 3), wide((21, 4), 4)
    first = lora_update(a, b, 2.0, threads=1)
    before = torch.get_num_threads()
    assert same_bits(first, lora_update(a, b, 2.0, threads=3, chunk=40))
    assert same_bits(first, lora_update(a, b, 2.0, threads=2, chunk=1))
    assert torch.get_num_threads() == before  # the setting of the caller comes back


def test_an_unknown_kernel_and_matrices_that_do_not_fit_are_refused(tmp_path):
    with pytest.raises(VerifyError, match="unknown kernel"):
        lora_update(torch.ones(2, 3), torch.ones(4, 2), 1.0, "fixed-order-v2")
    with pytest.raises(VerifyError, match="do not fit"):
        lora_update(torch.ones(2, 3), torch.ones(4, 5), 1.0)
    with pytest.raises(VerifyError, match="unknown kernel"):
        replay_fold(tmp_path, tmp_path, [], None, None, kernel="fast")


# ---- the files ----


def write_base(path: Path) -> Path:
    """A tiny bf16 base: two weight files, a config file, tensor names of the launch model."""
    path.mkdir()
    generator = torch.Generator().manual_seed(5)
    make = lambda *shape: torch.randn(shape, generator=generator).to(torch.bfloat16)  # noqa: E731
    first = {LAYER.format(0, "down_proj"): make(8, 24), LAYER.format(0, "up_proj"): make(24, 8), "model.language_model.embed_tokens.weight": make(40, 8)}
    second = {LAYER.format(1, "down_proj"): make(8, 24), LAYER.format(1, "up_proj"): make(24, 8), "model.language_model.norm.weight": make(8), "lm_head.bias": make(3)}
    save_file(first, str(path / "model-00001-of-00002.safetensors"))  # the genesis files have no metadata
    save_file(second, str(path / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    (path / "config.json").write_text('{"model_type": "tiny"}')
    return path


def write_lora(path: Path, rank: int, alpha: float, layers: tuple[int, ...], seed: int) -> Path:
    path.mkdir()
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in layers:
        for module, rows, columns in (("down_proj", 8, 24), ("up_proj", 24, 8)):
            key = f"base_model.model.model.language_model.layers.{layer}.mlp.{module}"
            tensors[f"{key}.lora_A.weight"] = torch.randn(rank, columns, generator=generator) * 0.05
            tensors[f"{key}.lora_B.weight"] = torch.randn(rows, rank, generator=generator) * 0.05
    save_file(tensors, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": alpha}))
    return path


@pytest.fixture
def head(tmp_path) -> tuple[Path, Path, list[tuple[Path, int]]]:
    """A base, its fp32 master as the GPU box writes it, and two LoRAs of different rank."""
    base = write_base(tmp_path / "base")
    upcast(base, tmp_path / "master", part_bytes=PART)
    loras = [(write_lora(tmp_path / "lora-a", 4, 37.7, (0, 1), 6), 4), (write_lora(tmp_path / "lora-b", 2, 4.0, (1,), 7), 2)]
    return base, tmp_path / "master", loras


def test_the_header_is_the_one_that_safetensors_writes(tmp_path):
    tensors = {"b.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3), "a.weight": torch.ones(5), "c": torch.zeros(1, 1, 2)}
    for metadata in ({"format": "pt"}, None):
        save_file(tensors, str(tmp_path / "file.safetensors"), metadata=metadata)
        entries = [(name, "F32", tuple(tensor.shape), 4 * tensor.numel()) for name, tensor in tensors.items()]
        ours = header_bytes(entries, metadata) + b"".join(tensors[name].numpy().tobytes() for name in sorted(tensors))
        assert ours == (tmp_path / "file.safetensors").read_bytes()
    with pytest.raises(VerifyError, match="one dtype"):
        header_bytes([("a", "F32", (1,), 4), ("b", "BF16", (1,), 2)])


def test_the_first_master_has_the_bytes_of_the_gpu_box_upcast(head, tmp_path):
    base, master, _ = head
    files = upcast_master(base, tmp_path / "ours", part_bytes=PART)
    assert files == hash_tree(master) and len(files) > 2
    assert hash_tree(tmp_path / "ours") == files  # what was written is what was hashed
    assert upcast_master(base, None, part_bytes=PART) == files


def test_torch_matmul_replays_the_fold_of_the_gpu_box_byte_for_byte(head, tmp_path, one_thread):
    base, master, loras = head
    fold_files(master, base, [read_lora(path, rank) for path, rank in loras], tmp_path / "box-master", tmp_path / "box-base", part_bytes=PART)
    pairs = [read_pairs(path, rank) for path, rank in loras]
    ours = replay_fold(master, base, pairs, tmp_path / "our-master", tmp_path / "our-base", kernel=TORCH_MATMUL, part_bytes=PART)
    assert ours.master_files == hash_tree(tmp_path / "box-master") and ours.base_files == hash_tree(tmp_path / "box-base")
    assert ours.master_digest == manifest_digest(ours.master_files) and ours.base_hash == manifest_digest(ours.base_files)
    assert hash_tree(tmp_path / "our-master") == ours.master_files and hash_tree(tmp_path / "our-base") == ours.base_files
    assert replay_fold(master, base, pairs, None, None, kernel=TORCH_MATMUL, part_bytes=PART) == ours  # hash only: the same answer
    changed = LAYER.format(1, "down_proj")
    before, after = load_file(str(base / "model-00002-of-00002.safetensors")), load_file(str(tmp_path / "our-base" / "model-00002-of-00002.safetensors"))
    assert not torch.equal(before[changed], after[changed]) and torch.equal(before["lm_head.bias"], after["lm_head.bias"])


def test_fixed_order_folds_the_exact_products_in_the_order_of_the_head(head, tmp_path):
    base, master, loras = head
    pairs = [read_pairs(path, rank) for path, rank in loras]
    ours = replay_fold(master, base, pairs, tmp_path / "new-master", tmp_path / "new-base", part_bytes=PART, threads=2)
    name, shard = LAYER.format(1, "up_proj"), "model-00002-of-00002"
    old = {key: value for part in sorted(master.glob(f"{shard}.part*")) for key, value in load_file(str(part)).items()}
    want = old[name]
    for lora in pairs:  # the order of the head: an fp32 add for each LoRA
        want = want + reference(*lora.pairs[name], lora.scale)
    new = {key: value for part in sorted((tmp_path / "new-master").glob(f"{shard}.part*")) for key, value in load_file(str(part)).items()}
    assert same_bits(new[name], want)
    folded = load_file(str(tmp_path / "new-base" / f"{shard}.safetensors"))
    assert folded[name].dtype == torch.bfloat16 and torch.equal(folded[name], want.to(torch.bfloat16))
    assert same_bits(new["model.language_model.norm.weight"], old["model.language_model.norm.weight"])
    again = replay_fold(master, base, pairs, None, None, part_bytes=PART, threads=1)
    assert again == ours
    assert replay_fold(master, base, pairs[::-1], None, None, part_bytes=PART) != ours  # the order is part of the result


def test_more_devices_share_one_fold_and_change_no_byte(head, tmp_path):
    base, master, loras = head
    pairs = [read_pairs(path, rank) for path, rank in loras]
    one = replay_fold(master, base, pairs, None, None, part_bytes=PART)
    three = replay_fold(master, base, pairs, tmp_path / "new-master", tmp_path / "new-base", part_bytes=PART, device=["cpu", "cpu", "cpu"])
    assert three == one and hash_tree(tmp_path / "new-master") == one.master_files and hash_tree(tmp_path / "new-base") == one.base_files
    assert replay_fold(master, base, pairs, None, None, part_bytes=PART, device="cpu, cpu") == one
    with pytest.raises(VerifyError, match="no device"):
        replay_fold(master, base, pairs, None, None, part_bytes=PART, device="cpu,")
    wrong = read_pairs(loras[1][0], 2)
    name = sorted(wrong.pairs)[-1]
    wrong.pairs[name] = (wrong.pairs[name][0][:, :5], wrong.pairs[name][1])
    with pytest.raises(VerifyError, match="wrong shape"):
        replay_fold(master, base, [wrong], None, None, part_bytes=PART, device=["cpu", "cpu"])


def test_the_results_of_more_devices_come_in_order_and_a_device_has_one_item_at_a_time():
    busy, lock = set(), threading.Lock()

    def work(item: int, device: str) -> int:
        with lock:
            assert device not in busy
            busy.add(device)
        time.sleep(0.002 * (5 - item % 5))  # a later item is often ready first
        with lock:
            busy.discard(device)
        return item * item

    assert list(fold_kernel._in_order(work, range(40), ["x", "y", "z"])) == [item * item for item in range(40)]
    assert list(fold_kernel._in_order(work, range(5), ["x"])) == [0, 1, 4, 9, 16]
    started = []

    def fails(item: int, device: str) -> int:
        started.append(item)
        if item == 3:
            raise VerifyError("item three")
        return item

    with pytest.raises(VerifyError, match="item three"):
        list(fold_kernel._in_order(fails, range(40), ["x", "y"]))
    assert len(started) < 40  # after an error no more work starts


def test_small_blocks_and_the_reading_thread_change_no_byte(head, tmp_path, monkeypatch):
    base, master, loras = head
    pairs = [read_pairs(path, rank) for path, rank in loras]
    whole = replay_fold(master, base, pairs, None, None, part_bytes=PART)
    monkeypatch.setattr(fold_kernel, "_BLOCK", 32)  # the largest tensor is 1280 bytes of fp32: 40 blocks in two buffers
    assert upcast_master(base, None, part_bytes=PART) == hash_tree(master)
    small = replay_fold(master, base, pairs, tmp_path / "new-master", tmp_path / "new-base", part_bytes=PART)
    assert small == whole and hash_tree(tmp_path / "new-master") == whole.master_files and hash_tree(tmp_path / "new-base") == whole.base_files


def test_the_reading_thread_is_one_item_ahead_and_passes_an_error():
    taken = []

    def items():
        for item in range(6):
            taken.append(item)
            yield item
        raise VerifyError("the end of the items")

    seen = []
    with pytest.raises(VerifyError, match="the end of the items"):
        for item in fold_kernel._ahead(items()):
            time.sleep(0.01)
            assert len(taken) <= item + 2
            seen.append(item)
    assert seen == list(range(6))
    assert list(fold_kernel._ahead(iter([]))) == []


def test_the_command_compares_with_a_head_record(head, tmp_path, capsys):
    base, master, loras = head
    ours = replay_fold(master, base, [read_pairs(path, rank) for path, rank in loras], None, None, part_bytes=PART)
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"base_files": ours.base_files, "master_files": ours.master_files}))
    args = ["fold", str(master), str(base), *[item for path, rank in loras for item in ("--lora", f"{path}:{rank}")], "--part-bytes", str(PART), "--record", str(record)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert f"base digest {ours.base_hash}" in out and f"master digest {ours.master_digest}" in out and "DIFFERENT" not in out
    assert main([*args, "--kernel", FIXED_ORDER, "--threads", "2"]) == 0 and "DIFFERENT" not in capsys.readouterr().out
    other = dict(ours.master_files)
    other[sorted(other)[-1]] = "0" * 64
    record.write_text(json.dumps({"base_files": ours.base_files, "master_files": other}))
    assert main(args) == 1 and capsys.readouterr().out.count("DIFFERENT") == 2  # the file and the digest
    assert main(["upcast", str(base), "--part-bytes", str(PART)]) == 0
    assert f"master digest {manifest_digest(hash_tree(master))}" in capsys.readouterr().out


def test_a_replay_fails_closed(head, tmp_path):
    base, master, loras = head
    pairs = [read_pairs(path, rank) for path, rank in loras]

    def replay(loras=pairs, master_dir=master, base_dir=base) -> Folded:
        return replay_fold(master_dir, base_dir, loras, None, None, part_bytes=PART)

    with pytest.raises(VerifyError, match="rank differs"):
        read_pairs(loras[0][0], 8)
    stray = write_lora(tmp_path / "stray", 2, 4.0, (9,), 8)
    with pytest.raises(VerifyError, match="match no weight"):
        replay([read_pairs(stray, 2)])
    wrong = read_pairs(loras[1][0], 2)
    name = next(iter(wrong.pairs))
    wrong.pairs[name] = (wrong.pairs[name][0][:, :5], wrong.pairs[name][1])
    with pytest.raises(VerifyError, match="wrong shape"):
        replay([wrong])
    huge = read_pairs(loras[1][0], 2)
    huge.pairs[name] = (huge.pairs[name][0] * 3e38, huge.pairs[name][1] * 3e38)
    with pytest.raises(VerifyError, match="not finite"):
        replay([huge])
    short = tmp_path / "short"
    short.mkdir()
    for file in sorted(master.iterdir())[1:]:
        (short / file.name).write_bytes(file.read_bytes())
    with pytest.raises(VerifyError, match="different tensors"):
        replay(master_dir=short)
    (short / "other.part00.safetensors").write_bytes(sorted(master.iterdir())[0].read_bytes())
    with pytest.raises(VerifyError):
        replay(master_dir=short)
    half = tmp_path / "half"
    half.mkdir()
    for file in sorted(base.glob("*.safetensors")):
        (half / f"{file.stem}.part00.safetensors").write_bytes(file.read_bytes())
    with pytest.raises(VerifyError, match="never upcast"):
        replay(master_dir=half)  # bf16 files are no master
    with pytest.raises(VerifyError, match="larger than one upload"):
        replay_fold(master, base, pairs, None, None, part_bytes=100)
    assert {entry.dtype for entry in read_index(sorted(master.iterdir())[0]).values()} == {"F32"}
