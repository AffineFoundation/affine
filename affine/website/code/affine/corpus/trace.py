"""Reading a rollout trace: the message graph and what the model saw.

verifiers stores a rollout as a message *graph* (`parent` per node); the
interception proxy adds one node per message of each request, so the path
from the root to a sampled assistant node is, by construction, exactly the
prompt the model was sent plus its reply. That is the harness-independent
reading of "what the model saw": echo re-serialization (pi), dropped turns
(mini-swe drops a FormatError reply from history), compaction,
summarization and subagent branches all come out right with no pattern
for any of them — whatever a harness did to history between calls is in
the path, and nothing else is.

Lives in the affine member (not rollouts) because the fold derives D from
published traces on the validator box; rollouts re-exports these names.
"""

from __future__ import annotations


class ToolParityError(ValueError):
    """The baked plain-text prefix does not render to the bytes the model
    saw with native tools — the trajectory is not admissible."""


class TraceShapeError(ValueError):
    """The message graph has no single root, or a node points at a parent
    that does not exist — the trace cannot be walked."""


def message_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return content or ""


def node_message(nd: dict) -> dict | None:
    m = nd["message"]
    role = m.get("role")
    if role not in ("system", "user", "assistant", "tool"):
        return None
    mm = {"role": role, "content": message_text(m.get("content"))}
    if role == "assistant" and m.get("tool_calls"):
        mm["tool_calls"] = m["tool_calls"]
    if role == "tool":
        for k in ("tool_call_id", "name"):
            if m.get(k):
                mm[k] = m[k]
    return mm


def sampled_paths(trace: dict) -> list[list[dict]]:
    """One conversation per model reply: the root-to-node path of every
    sampled assistant node, in node order. Non-sampled assistant nodes on a
    path (a harness re-stating or rewriting an earlier reply) stay as
    prefix history; only sampled nodes are replies to score.

    Traces without `parent` (mini_swe adapter, older dumps) are linear by
    construction: the path to node i is nodes[:i+1]."""
    nodes = trace["nodes"]
    if not nodes:
        return []
    linear = "parent" not in nodes[-1]
    msgs = [node_message(nd) for nd in nodes]
    if not linear:
        for i, nd in enumerate(nodes):
            parent = nd.get("parent")
            if parent is not None and not 0 <= parent < i:
                raise TraceShapeError(f"node {i} has parent {parent}")
        if sum(nd.get("parent") is None for nd in nodes) != 1:
            raise TraceShapeError("graph does not have exactly one root")
    paths: list[list[dict]] = []
    for i, nd in enumerate(nodes):
        if msgs[i] is None or msgs[i]["role"] != "assistant":
            continue
        if not nd.get("sampled"):
            continue
        if linear:
            chain = range(i + 1)
        else:
            chain = []
            j: int | None = i
            while j is not None:
                chain.append(j)
                j = nodes[j].get("parent")
            chain.reverse()
        paths.append([msgs[j] for j in chain if msgs[j] is not None])
    return paths


def has_tool_traffic(trace: dict, msgs: list[dict]) -> bool:
    return bool(trace.get("tools")) or any(
        m["role"] == "tool" or m.get("tool_calls") for m in msgs)


def trace_conversations(trace: dict, baker=None) -> list[list[dict]]:
    """Plain `[{role, content}]` conversations for the slicer, one per
    sampled reply (see sampled_paths); each ends with the reply.

    Traces without tools reduce to system/user/assistant content. A
    tool-using trace needs `baker` (affine.toolbake.ToolBaker): its schemas,
    tool calls and tool results are folded into content the teacher's
    template renders byte-identically to the native tools form;
    ToolParityError when it does not, and when no baker is available — such
    a trace can never be stored plain."""
    tools = trace.get("tools") or []
    out: list[list[dict]] = []
    for msgs in sampled_paths(trace):
        if not has_tool_traffic(trace, msgs):
            out.append([{"role": m["role"], "content": m["content"]}
                        for m in msgs if m["role"] != "tool"])
            continue
        if baker is None:
            raise ToolParityError("tool-using trace but no ToolBaker configured")
        # A conversation the teacher's template refuses to render at all
        # (e.g. Codex CLI's two leading system messages, 2026-09-07 probe:
        # "System message must be at the beginning") is not admissible
        # either — surface it as a parity failure so the fold / pod yield
        # path drops the trajectory instead of dying on a jinja exception.
        try:
            baked = baker.bake(msgs, tools)
            parity = baker.parity_ok(msgs, tools, baked)
        except ToolParityError:
            raise
        except Exception as e:  # jinja TemplateError, tokenizer ValueError
            raise ToolParityError(
                f"template cannot render conversation: {type(e).__name__}: {e}"
            ) from e
        if not parity:
            raise ToolParityError("baked prefix != template(messages, tools)")
        out.append(baked)
    return out


TURN_CAP_STOP = "max_turns"
TURN_CAP_ARTIFACT = "rollout stopped: max_turns"


def is_turn_cap_artifact(err: dict, trace: dict) -> bool:
    """The error is only the turn cap surfacing through a harness.

    When interception refuses the call past `max_turns`, ACP agents (Claude
    Code and friends) raise the refusal as their own runtime error, so the
    trace carries `stop_condition == max_turns` AND a HarnessError whose
    message quotes "rollout stopped: max_turns". Nothing failed: the model
    used its whole turn budget. Live 2026-09-10: 15-20 % of the teacher's
    Claude Code rollouts and 24/24 of the king's looked like this and were
    dropped as errored — the king looping to the cap is exactly the
    failure the king seat exists to capture."""
    if trace.get("stop_condition") != TURN_CAP_STOP:
        return False
    return TURN_CAP_ARTIFACT in str(err.get("message") or "")


def real_errors(trace: dict) -> list[dict]:
    """`trace["errors"]` without turn-cap artifacts."""
    return [e for e in (trace.get("errors") or [])
            if not is_turn_cap_artifact(e, trace)]


def trace_error_type(trace: dict) -> str | None:
    errors = real_errors(trace)
    if not errors:
        return None
    return errors[0].get("type") or errors[0].get("error") or "unknown"
