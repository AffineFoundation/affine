"""Affine eval server: runs on the GPU pod.

Serves the frozen teacher + reigning king warm via vLLM, loads challengers
per duel, produces S* verdicts, and runs advisory tau2 benchmarks. Driven
over HTTP by the root validator (see affine/eval_client.py).
"""
