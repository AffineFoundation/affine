#!/usr/bin/env python3
"""Alias module: host_wiring also looks for ``pe_upload_verify``."""
from __future__ import annotations

try:
    from .pe_verify import VerifyError, verify_green_uploads, verify_uploads
except ImportError:  # deployed as flat pe_*.py on the host
    from pe_verify import VerifyError, verify_green_uploads, verify_uploads  # type: ignore

__all__ = ['VerifyError', 'verify_uploads', 'verify_green_uploads']
