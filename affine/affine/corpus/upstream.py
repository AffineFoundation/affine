"""Detect a successful fetch of upstream code inside a rollout.

The sandboxes that build SWE-style rollouts can reach the public internet.
The teacher uses that path to read the task's own fix: `curl` of a GitHub
commit or pull-request patch, `git clone` of the upstream repo, or
`pip download` of a newer release. A later turn then sees that text in its
prefix, so the teacher reference is "apply what upstream wrote".

This detector is the one used for the 2026-09-24 sample. It is an upper
bound. A curl of an issue page counts. A command that copies the patch into
the work tree is a subset of these hits, not a separate gate.

A hit needs both parts. The assistant command must match, and the tool
response after it must be non-empty and not a short error (a 404 JSON body
is not a leak). The fold drops a turn when the prefix already contains such
a pair. The turn that first runs the fetch is kept: its prefix does not
hold the answer yet.
"""

from __future__ import annotations

import json
import re

# Hosts the teacher uses to read the official fix or a released package.
# `*.github.com` covers api.github.com and codeload.github.com.
_FETCH_HOST = re.compile(
    r"https?://(?:[^/\s\"']+\.)?"
    r"(?:github\.com|githubusercontent\.com|pypi\.org|pythonhosted\.org|pypi\.python\.org)\b",
    re.IGNORECASE,
)
_CURL = re.compile(r"(?:^|[^\w./-])(?:curl|wget)\b", re.IGNORECASE)
_GIT_CLONE = re.compile(r"\bgit\s+clone\b", re.IGNORECASE)
_GIT_FETCH = re.compile(r"\bgit\s+fetch\b", re.IGNORECASE)
_URL = re.compile(r"https?://|git@", re.IGNORECASE)
_UPSTREAM_WORD = re.compile(r"\bupstream\b", re.IGNORECASE)
_PIP_DOWNLOAD = re.compile(r"\bpip3?\s+download\b", re.IGNORECASE)
_API_OBJECT = re.compile(
    r"api\.github\.com/repos/[^/\s\"']+/[^/\s\"']+/(?:pulls|commits)\b",
    re.IGNORECASE,
)

_FENCE = re.compile(
    r"```(?:bash|sh|shell)[^\n]*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_PARAM_COMMAND = re.compile(
    r"<parameter=command>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
_JSON_COMMAND = re.compile(
    r"""["']command["']\s*:\s*["']((?:\\.|[^"'\\])*)["']""",
)
_TOOL_RESPONSE = re.compile(
    r"<tool_response>\s*(.*?)\s*</tool_response>",
    re.DOTALL,
)

# Short bodies that match these are failed fetches. A long patch or issue
# page can mention the same words and still counts.
_SHORT_ERROR = re.compile(
    r"(curl:\s*\(\d+\)"
    r"|Could not resolve host"
    r"|Failed to connect"
    r"|Connection refused"
    r"|fatal:\s"
    r"|\"message\":\s*\"Not Found\""
    r"|\"status\":\s*\"?404\"?"
    r"|\"message\":\s*\"Moved Permanently\""
    r"|HTTP Error 40[34]"
    r"|ERROR:\s+Could not find"
    r"|No matching distribution)",
    re.IGNORECASE,
)
_SHORT_ERROR_MAX = 600


def is_upstream_fetch(command: str) -> bool:
    """True when a shell command asks the network for upstream code."""
    if not command or not command.strip():
        return False
    if _CURL.search(command) and (_FETCH_HOST.search(command) or _API_OBJECT.search(command)):
        return True
    if _API_OBJECT.search(command) and not _CURL.search(command):
        # A client other than curl (python, gh) hitting the pulls/commits API.
        return True
    if _PIP_DOWNLOAD.search(command):
        return True
    if _GIT_CLONE.search(command) and _URL.search(command):
        return True
    if _GIT_FETCH.search(command) and (_URL.search(command) or _UPSTREAM_WORD.search(command)):
        return True
    return False


def response_ok(text: str) -> bool:
    """The tool came back with something the model can read.

    An empty body, or a short GitHub/curl error, is not a leak. The patch
    text, a clone's log, or a pull-request JSON body is.
    """
    body = (text or "").strip()
    if len(body) < 8:
        return False
    if len(body) < _SHORT_ERROR_MAX and _SHORT_ERROR.search(body):
        return False
    return True


def _json_unescape(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw


def _struct_command(call: dict) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    if not isinstance(fn, dict):
        return ""
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return args
    if isinstance(args, dict):
        for key in ("command", "cmd", "script"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def bash_commands(text: str, tool_calls: list | None = None) -> list[str]:
    """Shell command strings in one assistant message, in order.

    Covers a closed ```bash fence, a Qwen `<tool_call>` command parameter,
    and a structured `tool_calls` entry on a raw trace node.
    """
    commands: list[str] = []
    commands.extend(_FENCE.findall(text or ""))
    for body in _TOOL_CALL.findall(text or ""):
        params = _PARAM_COMMAND.findall(body)
        if params:
            commands.extend(params)
            continue
        match = _JSON_COMMAND.search(body)
        if match:
            commands.append(_json_unescape(match.group(1)))
    for call in tool_calls or []:
        if isinstance(call, dict):
            command = _struct_command(call)
            if command:
                commands.append(command)
    return commands


def _as_message(node: dict) -> dict:
    message = node.get("message") if isinstance(node.get("message"), dict) else node
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text")
    return {
        "role": message.get("role"),
        "content": content or "",
        "tool_calls": message.get("tool_calls") or node.get("tool_calls"),
    }


def _tagged_responses(messages: list[dict]) -> list[str]:
    found: list[str] = []
    for message in messages:
        found.extend(_TOOL_RESPONSE.findall(message.get("content") or ""))
    return found


def _span_has_fetch(commands: list[str], responses: list[str], tagged: list[str]) -> bool:
    if not any(is_upstream_fetch(command) for command in commands):
        return False
    if tagged:
        # One tool result per command. Zip stops at the shorter list so a
        # later user prompt in the same span is not treated as the result.
        for command, response in zip(commands, tagged):
            if is_upstream_fetch(command) and response_ok(response):
                return True
        return False
    blob = "\n".join(responses).strip()
    return bool(blob) and response_ok(blob)


def prefix_has_upstream_fetch(messages: list[dict]) -> bool:
    """True when some earlier turn already fetched upstream content."""
    return _conversation_has_fetch([_as_message(m) for m in messages])


def rollout_has_upstream_fetch(nodes: list[dict]) -> bool:
    """True when any node in the rollout fetched upstream content.

    Node order is the trace's commit order. A forked graph can mis-pair a
    command with a sibling branch's result; the fold uses the baked prefix,
    which follows one path, and that check is the one that admits turns.
    """
    return _conversation_has_fetch([_as_message(n) for n in nodes])


def _conversation_has_fetch(messages: list[dict]) -> bool:
    index = 0
    count = len(messages)
    while index < count:
        message = messages[index]
        if message.get("role") != "assistant":
            index += 1
            continue
        nxt = index + 1
        span: list[dict] = []
        while nxt < count and messages[nxt].get("role") != "assistant":
            span.append(messages[nxt])
            nxt += 1
        commands = bash_commands(message.get("content") or "", message.get("tool_calls"))
        tagged = _tagged_responses(span)
        plain = [m.get("content") or "" for m in span]
        if _span_has_fetch(commands, plain, tagged):
            return True
        index = nxt
    return False
