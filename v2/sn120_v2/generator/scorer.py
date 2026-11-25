"""Generator scorer entry point with the installed-king guard.

The pod implementation lives in ``runner`` (scorer PR #298). Keeping this small
entry point with dispatch means an incomplete pod source cannot silently bypass
the exact job ``head_version`` check before inference begins.
"""
from __future__ import annotations

import sys

from .ensure_head import require_installed


def main(argv: list[str] | None = None) -> int:
    from .runner import ScoringJob, main as run_scorer

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "run":
        if len(arguments) < 2:
            raise SystemExit("Generator scorer run needs a private job JSON")
        job = ScoringJob.load(arguments[1])
        require_installed(job.head_version)
    return run_scorer(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
