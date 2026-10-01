"""Versioned selection of signed per-index harnesses, independent of environment.

Callers authenticate the containing manifest and validate exact mining index
coverage. Heldout evaluation supplies its own separately approved harness.
"""
import copy
from .harness import normalize

VERSION='indexed-harness-v1'


def _indices(values):
    if (not isinstance(values,list) or not 1<=len(values)<=10000
        or any(type(i)is not int or i<0 for i in values)
        or len(set(values))!=len(values)):
        raise ValueError('exact bounded mining index population')
    return values


def validate(config,indices):
    """Return a normalized copy; wrappers may not silently fall back."""
    if not isinstance(config,dict):raise ValueError('harness configuration required')
    if config.get('version')!=VERSION:
        if 'by_index' in config:raise ValueError('per-index choices need versioned wrapper')
        return normalize(copy.deepcopy(config))
    indices=_indices(indices)
    if set(config)!={'version','by_index'} or not isinstance(config['by_index'],dict):
        raise ValueError('exact indexed harness fields')
    rows=config['by_index']
    if set(rows)!={str(i)for i in indices}:
        raise ValueError('exact signed mining index coverage')
    normalized={}
    for key,value in rows.items():
        if not isinstance(value,dict) or value.get('version')==VERSION or 'by_index'in value:
            raise ValueError('nested indexed harness unsupported')
        normalized[key]=normalize(copy.deepcopy(value))
    return dict(version=VERSION,by_index=normalized)


def resolve(config,index,indices):
    """Choose only an explicitly authorized mining index, even for legacy maps."""
    indices=_indices(indices)
    if type(index)is not int or index not in indices:
        raise ValueError('sample index not authorized for mining')
    normalized=validate(config,indices)
    if normalized.get('version')==VERSION:return normalized['by_index'][str(index)]
    return normalized
