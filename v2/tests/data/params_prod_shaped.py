"""The prod-shaped off-mode fixture used by test_pool_rotation.py."""
from sn120_v2.params import Params


def fixture() -> Params:
    """Current defaults plus PR 274's live worker-image/code pin shape.

    PR 274 changes the pool/grader, not a Params field. An empty env_pin keeps
    that choice out of this test. The all-zero image digest is a placeholder,
    not a credential or a digest claimed to be in production.
    """
    return Params.from_dict({
        "env_pin": "",
        "recipe": {"image_digests": ["sha256:0000000000000000000000000000000000000000000000000000000000000000"]},
    })
