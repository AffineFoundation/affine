"""Expected fp32 master digest lookup (BRIEF §1.6).

Order: chain override → public manifest → index → TRAINING-RESULT.
On failure return None (never an empty string) and optionally alert.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


AlertFn = Callable[[str, Mapping[str, Any]], None]


def _hex64(value: Any) -> Optional[str]:
    if not isinstance(value, str) or len(value) != 64:
        return None
    if any(c not in '0123456789abcdef' for c in value):
        return None
    return value


def lookup_expected_master(
    head: str,
    *,
    chain_override: Optional[Mapping[str, str]] = None,
    public_manifest: Optional[Mapping[str, Any]] = None,
    index: Optional[Mapping[str, str]] = None,
    training_result: Optional[Mapping[str, Any]] = None,
    alert: Optional[AlertFn] = None,
) -> Optional[str]:
    """Resolve M(H). Returns lowercase hex64 or None (fail closed / pause dequeue)."""
    head = _hex64(head)
    if head is None:
        if alert:
            alert('expected-master-bad-head', {})
        return None

    if chain_override:
        dig = _hex64(chain_override.get(head) or chain_override.get('digest'))
        if dig:
            return dig

    if public_manifest:
        # shapes: {head: digest} or {digest: ...} or {master_digest: ...}
        dig = _hex64(
            public_manifest.get(head)
            or public_manifest.get('master_digest')
            or public_manifest.get('digest')
        )
        if dig:
            return dig
        files = public_manifest.get('files')
        if isinstance(files, Mapping):
            dig = _hex64(files.get('meta.json') or files.get('digest'))
            if dig:
                return dig

    if index:
        dig = _hex64(index.get(head))
        if dig:
            return dig

    if training_result:
        dig = _hex64(
            training_result.get('master_digest')
            or training_result.get('input_master_digest')
            or (training_result.get('fp32_master') or {}).get('digest')
        )
        if dig:
            return dig

    if alert:
        alert('expected-master-unpublished', {'head': head[:12]})
    return None


def load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def lookup_from_paths(
    head: str,
    *,
    chain_override_path: Optional[Path] = None,
    public_manifest_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
    training_result_path: Optional[Path] = None,
    alert: Optional[AlertFn] = None,
) -> Optional[str]:
    return lookup_expected_master(
        head,
        chain_override=load_json(chain_override_path) if chain_override_path else None,
        public_manifest=load_json(public_manifest_path) if public_manifest_path else None,
        index=load_json(index_path) if index_path else None,
        training_result=load_json(training_result_path) if training_result_path else None,
        alert=alert,
    )
