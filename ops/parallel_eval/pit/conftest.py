"""Pytest hermetic path guard for PIT tests.

* Seeds PE_STATE / PIT_STATE / gate roots into ``tmp_path`` when unset so a
  missing env cannot fall through to live host state.
* Fails any test that writes under ``/home/const`` or outside ``tmp_path`` /
  the process tempfile root.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


@pytest.fixture(autouse=True)
def _pit_hermetic_paths(tmp_path, monkeypatch):
    pe = tmp_path / 'pe'
    life = tmp_path / 'life'
    gate = tmp_path / 'gate'
    pe.mkdir()
    (pe / 'pit').mkdir()
    (life / 'epochs').mkdir(parents=True)
    (gate / 'epochs').mkdir(parents=True)
    (gate / 'heldout').mkdir(parents=True)
    k1 = tmp_path / 'K1-TURNS.jsonl'
    k1.write_text('')
    fp32 = tmp_path / 'fp32'
    fp32.mkdir()
    (tmp_path / 'pe-life' / 'pending').mkdir(parents=True)

    defaults = {
        'PE_STATE': str(pe),
        'PIT_STATE': str(pe / 'pit'),
        'PE_LIFE': str(life),
        'LIFE_ROOT': str(life),
        'PE_LIFE_STATE': str(tmp_path / 'pe-life'),
        'PE_GATE_EP': str(gate / 'epochs'),
        'HG_GATE_STATE': str(gate),
        'PE_HELDOUT': str(gate / 'heldout'),
        'K1_TURNS': str(k1),
        'PIT_FP32_MASTER_INDEX': str(fp32),
    }
    for key, value in defaults.items():
        if not os.environ.get(key):
            monkeypatch.setenv(key, value)

    allowed_roots = (
        tmp_path.resolve(),
        Path(tempfile.gettempdir()).resolve(),
    )
    forbidden = Path('/home/const')

    real_write_text = Path.write_text
    real_write_bytes = Path.write_bytes
    real_replace = os.replace
    real_open = open

    def _check_write(path_like) -> None:
        path = Path(path_like)
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if _is_under(resolved, forbidden) or str(resolved).startswith(str(forbidden)):
            raise AssertionError(
                'hermetic guard: refused write under host path %s' % resolved
            )
        if any(_is_under(resolved, root) for root in allowed_roots):
            return
        raise AssertionError(
            'hermetic guard: path %s is outside tmp_path/tempfile' % resolved
        )

    def write_text(self, data, *args, **kwargs):
        _check_write(self)
        return real_write_text(self, data, *args, **kwargs)

    def write_bytes(self, data, *args, **kwargs):
        _check_write(self)
        return real_write_bytes(self, data, *args, **kwargs)

    def replace(src, dst, *args, **kwargs):
        _check_write(dst)
        return real_replace(src, dst, *args, **kwargs)

    def guarded_open(file, mode='r', *args, **kwargs):
        mode_s = str(mode)
        if any(c in mode_s for c in 'wxa'):
            _check_write(file)
        elif '+' in mode_s and 'r' in mode_s:
            # read+write — still a write surface
            _check_write(file)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', write_text)
    monkeypatch.setattr(Path, 'write_bytes', write_bytes)
    monkeypatch.setattr(os, 'replace', replace)
    monkeypatch.setattr('builtins.open', guarded_open)
    yield
