"""Affine SN120 v2: batch -> LoRA tournament.

Miners submit text batches. The validator trains one bounded LoRA per batch,
scores every LoRA and a null copy of the head M on the same public tasks and
seeds, and applies at most one winner per round, only if it beats M by more
than one standard error (z > 1).

Import from the submodules directly. This file stays empty of re-exports so
parallel workstreams never conflict on it.
"""

__version__ = "0.1.0"
