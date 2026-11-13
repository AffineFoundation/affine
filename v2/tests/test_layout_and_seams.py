from __future__ import annotations

import pytest

from sn120_v2 import layout, seams
from sn120_v2.errors import ContractError, V2Error
from sn120_v2 import errors


def test_miner_object_keeps_the_existing_private_key():
    assert layout.miner_object("5Hotkey") == "private/ff/v1/miners/5Hotkey/batch.json"


def test_every_box_upload_lands_in_the_pending_zone():
    keys = [
        layout.pending_result("r000001", "r000001-l0"),
        layout.pending_lora_prefix("r000001", "r000001-l0", 3),
        layout.pending_fold_prefix("abc"),
    ]
    assert all(key.startswith("pending/") for key in keys)
    assert layout.pending_lora_prefix("r000001", "r000001-l0", 3).endswith("/loras/003")
    assert layout.pending_result("r000001", "r000001-l0").startswith(layout.pending_round_prefix("r000001") + "/")


def test_frozen_batches_and_tickets_are_private():
    assert layout.frozen_batch("r000001", "5Hotkey").startswith("private/")
    assert layout.ticket("r000001", "r000001-l0").startswith("private/")


def test_the_fp32_master_never_shares_a_prefix_with_the_base():
    base = layout.public_base_prefix("a" * 64)
    master = layout.public_master_prefix("a" * 64)
    assert base != master
    assert not master.startswith(base) and not base.startswith(master)


def test_public_artifacts_use_content_addressed_keys():
    assert layout.public_lora_prefix("f" * 64) == "public/loras/" + "f" * 64
    assert layout.head_manifest("e" * 64) == "public/heads/" + "e" * 64 + ".json"
    assert layout.winner_batch("r000001") == "public/rounds/r000001/batch.json"
    assert layout.CURRENT == "public/current.json"


@pytest.mark.parametrize("bad", ["", ".", "..", "../x", "a/b", "a b", ".hidden", "x\ny"])
def test_a_key_component_cannot_escape_its_zone(bad):
    with pytest.raises(ContractError):
        layout.frozen_batch("r000001", bad)
    with pytest.raises(ContractError):
        layout.pending_result(bad, "lease")


PROTOCOLS = (
    "ObjectStore",
    "Presigner",
    "Admission",
    "TaskSampler",
    "Trainer",
    "Evaluator",
    "Folder",
    "RoundRunner",
    "Publisher",
    "ChainReader",
    "WeightSetter",
)


def test_every_seam_is_a_runtime_checkable_protocol():
    for name in PROTOCOLS:
        protocol = getattr(seams, name)
        assert getattr(protocol, "_is_protocol", False), name
        assert not isinstance(object(), protocol), name


def test_a_class_with_the_right_methods_satisfies_a_seam():
    class OneBox:
        def run(self, ticket):
            raise NotImplementedError

    class Writer:
        def set_weights(self, weights):
            raise NotImplementedError

    assert isinstance(OneBox(), seams.RoundRunner)
    assert isinstance(Writer(), seams.WeightSetter)
    assert not isinstance(OneBox(), seams.WeightSetter)


def test_every_error_type_shares_one_base():
    names = [
        "ConfigError",
        "InfraError",
        "BatchRejected",
        "VerifyError",
        "HeadMismatchError",
        "LeaseError",
        "ContractError",
        "FencedError",
        "JournalCorrupt",
    ]
    for name in names:
        assert issubclass(getattr(errors, name), V2Error), name
