"""Process resource measurements for train logs, never hashed artifacts."""
from __future__ import annotations

import resource


def resource_metrics() -> dict[str, int]:
    import torch
    cuda = torch.cuda.is_available()
    return {
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()) if cuda else 0,
        "cuda_max_memory_reserved": int(torch.cuda.max_memory_reserved()) if cuda else 0,
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
