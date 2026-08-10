"""Trajectory-chunk corpus helpers (schema_version=2).

Production D stores conversations once per trajectory and a Parquet turn
index for stratified sampling. Eval materializes prefixes only for the
drawn slice. See affine/scripts/corpus_push.py and evalsrv/corpus.py.
"""

from .materialize import materialize_turn, stratum_key
from .pack import PackResult, pack_turns_to_chunks

__all__ = [
    "PackResult",
    "materialize_turn",
    "pack_turns_to_chunks",
    "stratum_key",
]
