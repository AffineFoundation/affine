"""The deterministic-check version, in a module with no dependencies.

It is part of ``guard_version``, so the judge needs it; keeping it here lets the
judge (and ``judge-requests`` on a host without the math grader) import it
without pulling in ``checks.py`` and its math-verify dependency.
"""

CHECK_VERSION = "guard-checks-v1"  # bump with any change to a check's meaning; it is part of guard_version
