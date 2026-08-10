"""Continuous SWE-rebench data generation for the Affine turn corpus.

Runs on a dedicated CPU-only machine (docker required): rolls mini-swe-agent
over SWE-rebench instances via an OpenRouter-served model, keeps only
trajectories the swebench harness verifies as resolved, slices them into
duel-compatible per-turn records, and uploads sha-pinned shards to a HF
dataset repo. Never touches the pinned production corpus in affine.toml.
"""
