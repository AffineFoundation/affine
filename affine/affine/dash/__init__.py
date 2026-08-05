"""Read-only hot-path dashboard API (affine-dash).

Serves live snapshot / history / duel projections from local validator state.
Hippius remains the cold public archive for miners; this process never writes
chain or eval state.
"""
