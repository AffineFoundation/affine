"""The live prod-shaped off-mode fixture used by test_pool_rotation.py."""
from sn120_v2.params import Params


def fixture() -> Params:
    """Exact live production params supplied for the 2026-11-19 merge.

    Serving values ``kv_cache_dtype=fp8`` and ``max_num_seqs=512`` remain
    explicit. Rotation fields are absent and therefore load at their defaults,
    preserving the supplied params_hash. This is public configuration only;
    no host, credential, or live service is read by this fixture.
    """
    from pathlib import Path
    return Params.from_json((Path(__file__).with_suffix(".json")).read_bytes())
