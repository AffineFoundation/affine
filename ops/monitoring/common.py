"""Read-only helpers for Affine board monitoring."""
from __future__ import annotations
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    return json.loads((ROOT / path).read_text())

def get_json(url, timeout=8, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def pick(obj, keys):
    return {k: obj.get(k) for k in keys.split()}

def age(value):
    if value is None:
        return None
    try:
        stamp = float(value) if isinstance(value, (int, float)) else datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        return max(0, time.time() - stamp)
    except (ValueError, TypeError):
        return None

def duration(seconds):
    if seconds is None: return 'Unknown'
    if seconds < 60: return f'{seconds:.0f}s'
    if seconds < 3600: return f'{seconds / 60:.1f}m'
    return f'{seconds / 3600:.1f}h'

def metric(label, value, detail='', tone=''):
    return dict(label=label, value=value, detail=detail, tone=tone)

def table(title, columns, rows):
    return dict(title=title, columns=columns, rows=rows)

def panel(title, subtitle, *, status='ok', metrics=None, sections=None, bars=None, notes=None, sources=None, charts=None):
    return dict(title=title, subtitle=subtitle, status=status, metrics=metrics or [], sections=sections or [], bars=bars or [], notes=notes or [], sources=sources or [], charts=charts or [], collected_at=now_iso())
