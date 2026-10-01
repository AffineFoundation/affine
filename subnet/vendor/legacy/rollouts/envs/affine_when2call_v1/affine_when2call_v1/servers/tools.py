"""Simulated tool server for affine-when2call-v1.

The task's tool schemas come from the dataset row (xlam / BFCL function
format: name, description, parameters.properties, required). Each one is
registered as an MCP tool whose signature is built from the schema, so the
model sees exactly the schema When2Call served. Calling a tool records
nothing real: it answers with a simulated result. The grade reads the
model's tool call from the trace (name + arguments against the When2Call
label), not the tool's return.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import verifiers.v1 as vf

SIMULATED = "Tool call received (simulated environment: no live data). Finish your reply to the user."

_TYPE_MAP = (
    ("bool", bool), ("int", int), ("float", float), ("number", float),
    ("list", list), ("array", list), ("set", list), ("tuple", list),
    ("dict", dict), ("object", dict),
)


def py_type(type_str: str) -> type:
    low = (type_str or "").strip().lower()
    for key, typ in _TYPE_MAP:
        if low.startswith(key):
            return typ
    return str


def parse_tool(spec: str | dict) -> dict | None:
    """One dataset tool (JSON string or dict) -> {name, description, params:
    [(name, type, description, required)]}; None if unusable."""
    try:
        d = json.loads(spec) if isinstance(spec, str) else dict(spec)
    except (TypeError, ValueError):
        return None
    name = str(d.get("name") or "").strip()
    if not name or not name.replace("_", "a").replace(".", "a").isalnum():
        return None
    params = d.get("parameters") or {}
    props = params.get("properties") if isinstance(params, dict) else None
    props = props if isinstance(props, dict) else {}
    required = set(d.get("required") or params.get("required") or [])
    out = []
    for pname, pspec in props.items():
        pspec = pspec if isinstance(pspec, dict) else {}
        ptype = str(pspec.get("type") or "str")
        is_req = pname in required and "optional" not in ptype.lower()
        out.append((str(pname), ptype, str(pspec.get("description") or ""), is_req))
    return {"name": name, "description": str(d.get("description") or ""), "params": out}


def make_tool_fn(tool: dict, record):
    """A plain function whose signature mirrors the schema; FastMCP derives
    the advertised JSON schema from it. `record(name, arguments)` is called
    on every invocation."""
    name = tool["name"]
    params = []
    annotations: dict[str, Any] = {}
    for pname, ptype, _desc, is_req in tool["params"]:
        typ = py_type(ptype)
        if is_req:
            params.append(inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY, annotation=typ))
            annotations[pname] = typ
        else:
            params.append(inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY,
                                            annotation=typ | None, default=None))
            annotations[pname] = typ | None
    annotations["return"] = str

    def impl(**kwargs) -> str:
        record(name, {k: v for k, v in kwargs.items() if v is not None})
        return SIMULATED

    impl.__name__ = name
    impl.__qualname__ = name
    impl.__doc__ = tool["description"] or f"Call {name}."
    impl.__signature__ = inspect.Signature(params, return_annotation=str)  # type: ignore[attr-defined]
    impl.__annotations__ = annotations
    return impl


class When2CallToolsetConfig(vf.ToolsetConfig):
    tools: list[str] = []
    """The row's tool schemas, JSON strings in the dataset's own format."""


class When2CallToolset(vf.Toolset[When2CallToolsetConfig]):
    # None = bare tool names, exactly as When2Call presents them.
    TOOL_PREFIX = None

    def register(self, mcp) -> None:
        self.calls: list[dict] = []
        seen: set[str] = set()
        for spec in self.config.tools:
            tool = parse_tool(spec)
            if tool is None or tool["name"] in seen:
                continue
            seen.add(tool["name"])
            fn = make_tool_fn(tool, lambda n, a: self.calls.append({"name": n, "arguments": a}))
            mcp.add_tool(fn, name=tool["name"], description=tool["description"] or None)


if __name__ == "__main__":
    When2CallToolset.run()
