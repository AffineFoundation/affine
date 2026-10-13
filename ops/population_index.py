"""Prospective host-only linear eligible-pair join. No cached admissions or policy change."""
import hashlib
import inspect
from pathlib import Path

ORIGINAL_FILE_SHA256 = 'e66cf45dacdb6b86afa04397c113c2a743eab62d8a432b4b911fa23c93a9a305'
ORIGINAL_FUNCTION_SHA256 = 'ee4fd95c3e155fd5db3a2f158e5b6c9696ac4e1dd31f58f6a950f42cd08d4317'
FIELDS = ('miner', 'commitment_sha256', 'batch_sha256', 'proof_sha256')


def _index(rows):
    result = {}
    for row in rows:
        result.setdefault(tuple(row[k] for k in FIELDS), []).append(row)
    return result


def _matches(index, rows, pair):
    # Only ordinary JSON strings use hashing. Other malformed/non-string values
    # retain the original equality/error path, including unhashable values.
    if all(type(pair[k]) is str for k in FIELDS):
        return index.get(tuple(pair[k] for k in FIELDS), [])
    return [r for r in rows if all(r[k] == v for k, v in pair.items())]


def bind(original):
    """Bind only exact admitted function/file bytes; invocation-local index only."""
    path = Path(inspect.getsourcefile(original))
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_FILE_SHA256:
        raise ValueError('original population module pin changed')
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_FUNCTION_SHA256:
        raise ValueError('original population function pin changed')
    old = "  ids=[]\n  for pair in eligible_pairs:\n   matching=[r for r in rows if all(r[k]==v for k,v in pair.items())]"
    new = "  ids=[]\n  eligible_index=_host_eligible_index(rows)\n  for pair in eligible_pairs:\n   matching=_host_eligible_matches(eligible_index,rows,pair)"
    if source.count(old) != 1:
        raise ValueError('original exact eligible join changed')
    namespace = dict(original.__globals__, _host_eligible_index=_index,
                     _host_eligible_matches=_matches)
    exec(compile(source.replace(old, new), __file__ + ':pinned-register-population', 'exec'), namespace)
    return namespace[original.__name__]
