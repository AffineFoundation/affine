"""Prospective in-process state channel for operator-trusted native toolsets.

Install before setup_task: native toolsets may seed verifier state during setup.
This helper is not yet dispatched by the signed production adapter. It never
loads submitted code and does not replace original native DB/grader semantics.
"""
import asyncio
import hashlib


def bind_state_channel(toolset, trace):
    if not isinstance(trace.state, toolset._state_cls):
        raise ValueError('native toolset/trace state class mismatch')
    lock = asyncio.Lock()
    audit = []

    async def pull():
        # Toolset._with_state installs the returned object in its native
        # context variable. All calls are serialized by the ordinary harness.
        return trace.state.model_copy(deep=True)

    async def push(before):
        async with lock:
            current = toolset.state
            after = toolset._state_adapter.dump_json(current)
            if before != after:
                trace.state = toolset._state_adapter.validate_json(after)
                audit.append(dict(before_sha256=hashlib.sha256(before).hexdigest(),
                                  after_sha256=hashlib.sha256(after).hexdigest()))

    toolset._pull_state = pull
    toolset._push_state = push
    return audit
