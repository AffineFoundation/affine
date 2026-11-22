"""Canonical JSON, SHA-256, and the ``Record`` base for every shared type.

Every hash in v2 goes through this module, so two processes that hold the
same data always compute the same id.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import types
import typing
from pathlib import Path
from typing import Any, Mapping, Self

from .errors import ContractError

_HEX_DIGITS = frozenset("0123456789abcdef")
_CANONICAL = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _check_plain(value: Any, where: str) -> None:
    """Allow only values that have exactly one JSON form."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{where}: NaN and Infinity have no JSON form")
        return
    if isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            _check_plain(item, f"{where}[{position}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{where}: object key {key!r} is not a string")
            _check_plain(item, f"{where}.{key}")
        return
    raise ContractError(f"{where}: {type(value).__name__} has no JSON form")


def canonical_json(value: Any) -> bytes:
    """The one byte form used for hashing: sorted keys, no spaces, UTF-8."""
    _check_plain(value, "value")
    return _CANONICAL.encode(value).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    """SHA-256 of the canonical JSON of ``value``."""
    return sha256_hex(canonical_json(value))


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and _HEX_DIGITS.issuperset(value)


def manifest_digest(files: Mapping[str, str]) -> str:
    """Identity of a set of files: SHA-256 of the canonical ``{file name: sha256}`` object.

    The published head id uses this same rule, so the id of a file set does
    not change between v1 and v2. File names must be ASCII: then every JSON
    writer produces the same bytes for the object.
    """
    if not files:
        raise ContractError("a manifest needs at least one file")
    for file_name, file_hash in files.items():
        if not isinstance(file_name, str) or not file_name or not file_name.isascii():
            raise ContractError(f"manifest file name {file_name!r} must be non-empty ASCII")
        if not is_sha256(file_hash):
            raise ContractError(f"manifest entry {file_name!r} needs a lowercase 64-hex sha256")
    return digest(dict(files))


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes, read as a stream."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@functools.lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def to_jsonable(obj: Any, tp: Any = Any) -> Any:
    """Turn records, tuples and mappings into plain JSON values.

    ``tp`` is the declared type. A whole number in a float field is written
    as a float, so a record's digest is the same before and after a JSON
    round trip.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        hints = _hints(type(obj))
        return {f.name: to_jsonable(getattr(obj, f.name), hints[f.name]) for f in dataclasses.fields(obj)}
    if tp is float and type(obj) is int:
        return float(obj)
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in args if a is not type(None)]
        return to_jsonable(obj, inner[0] if len(inner) == 1 else Any)
    if isinstance(obj, (list, tuple)):
        if origin is tuple and args and args[-1] is not Ellipsis and len(args) == len(obj):
            return [to_jsonable(item, item_type) for item, item_type in zip(obj, args)]
        item_type = args[0] if origin is tuple and args else Any
        return [to_jsonable(item, item_type) for item in obj]
    if isinstance(obj, Mapping):
        item_type = args[1] if origin is dict and len(args) == 2 else Any
        return {key: to_jsonable(item, item_type) for key, item in obj.items()}
    return obj


def from_jsonable(tp: Any, data: Any, where: str = "value") -> Any:
    """Strictly rebuild a value of type ``tp`` from plain JSON. Unknown or mistyped fields raise."""
    if tp is Any:
        return data
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if data is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        if len(inner) != 1:
            raise ContractError(f"{where}: unsupported union {tp}")
        return from_jsonable(inner[0], data, where)
    if origin is tuple:
        if not isinstance(data, (list, tuple)):
            raise ContractError(f"{where}: expected a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_jsonable(args[0], v, f"{where}[{i}]") for i, v in enumerate(data))
        if len(args) != len(data):
            raise ContractError(f"{where}: expected {len(args)} items, got {len(data)}")
        return tuple(from_jsonable(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, data)))
    if origin is dict or tp is dict:
        if not isinstance(data, Mapping):
            raise ContractError(f"{where}: expected an object")
        item_type = args[1] if len(args) == 2 else Any
        return {key: from_jsonable(item_type, item, f"{where}.{key}") for key, item in data.items()}
    if dataclasses.is_dataclass(tp):
        if not isinstance(data, Mapping):
            raise ContractError(f"{where}: expected an object for {tp.__name__}")
        hints = _hints(tp)
        unknown = sorted(set(data) - set(hints))
        if unknown:
            raise ContractError(f"{where}: unknown field(s) {unknown} for {tp.__name__}")
        built = {key: from_jsonable(hints[key], item, f"{where}.{key}") for key, item in data.items()}
        try:
            return tp(**built)
        except TypeError as exc:
            raise ContractError(f"{where}: {exc}") from exc
    if tp is bool:
        if type(data) is not bool:
            raise ContractError(f"{where}: expected true or false")
        return data
    if tp is int:
        if type(data) is not int:
            raise ContractError(f"{where}: expected an integer")
        return data
    if tp is float:
        if type(data) not in (int, float) or not math.isfinite(data):
            raise ContractError(f"{where}: expected a finite number")
        return float(data)
    if tp is str:
        if not isinstance(data, str):
            raise ContractError(f"{where}: expected a string")
        return data
    raise ContractError(f"{where}: unsupported type {tp}")


class Record:
    """Base for shared dataclasses: strict JSON round trip and a stable digest."""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return from_jsonable(cls, data, cls.__name__)

    def to_json(self) -> bytes:
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, raw: bytes | str) -> Self:
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ContractError(f"{cls.__name__}: not JSON: {exc}") from exc
        return cls.from_dict(data)

    def digest(self) -> str:
        """SHA-256 of this record's canonical JSON. This is the seal of a ticket or a result."""
        return sha256_hex(self.to_json())
