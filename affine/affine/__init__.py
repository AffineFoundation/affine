"""Affine — Bittensor SN120 king-of-the-hill validator.

Miners submit HF checkpoints via on-chain commit-reveal; the validator
downloads the pinned revision, duels it against the reigning king under the
frozen S* scoring rule (thought-injection vs a frontier teacher), crowns
winners, runs advisory tau2 benchmarks, and publishes state to Hippius.
"""

__version__ = "0.1.0"
