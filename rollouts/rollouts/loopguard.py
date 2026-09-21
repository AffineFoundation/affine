"""Loop guard: end a rollout early once the agent repeats itself.

Why: the reign-11 king fails by looping — in its failed agent rollouts
49-100 % contain a loop, and once it loops it usually runs to the 80-turn
cap (80 model calls, 4-6 M prompt tokens, ~3,000 s). The fold keeps only the
loop's FIRST turn (`king_loop_onset`) and drops the repeats as references
leaked into the prefix, so everything after the onset is pure wall time.
Container utilization in agentic batches was 0.46 and 40 % of batches ended
at the 3,600 s wall (king-datagen throughput report, 2026-09-11).

Rule (the `label_loops.py` definition, exact form): the last N assistant
actions are identical after whitespace normalization AND the observations
between them are identical on their first 200 normalized characters. Short
cycles (period 2-3, e.g. re-reading two wiki sections in turn) count too
once they have run >= 3 full periods and >= N turns (`is_loop`). The
onset and its repeats stay in the trace (the fold needs them); the rollout
ends with `stop_condition = "loop_guard"`, which `affine.corpus.view.
rollout_outcome` classifies as a FAILED rollout (not errored).

Two agent loops exist on a datagen pod, so there are two hooks over one
detector:

* verifiers harnesses (bash / pi / claude_code / kimi / hermes / terminus /
  null / mini_swe_textbased): a `@stop` method installed on the `Task`
  base class. verifiers runs every Task `@stop` before each model call
  (`RolloutSession.refused`) and, when one fires, refuses the call and
  stamps the method's name as the stop condition — the same mechanism as
  `max_turns`. The check sees committed turns only, so it needs N actions
  and the N-1 observations between them. Installed by
  `loopguard_site/sitecustomize.py`, which the runner puts first on the
  eval subprocess's PYTHONPATH; a process without
  ROLLOUTS_LOOP_GUARD_REPEATS in its env is untouched.
* mini-swe-agent batch runner (`mini-extra swebench`, the swerebench_main
  source): `run_mini_swe_guarded` patches the batch's agent class to end
  the agent with exit status `loop_guard` once the N-th identical
  (action, observation) pair lands. The trajectory adapter maps the exit
  status to `stop_condition`.

Both read the threshold from ROLLOUTS_LOOP_GUARD_REPEATS (0 / unset = off).
Per policy: `loop_guard_repeats` in policies.toml; default 6 for `king_*`
policies, off for the teacher (its loop rate is 8-13 % and its loops are
one turn long).
"""

from __future__ import annotations

import json
import os
import re
import sys

ENV_REPEATS = "ROLLOUTS_LOOP_GUARD_REPEATS"
STOP_CONDITION = "loop_guard"
DEFAULT_KING_REPEATS = 6
# Longest action cycle the guard recognizes (1 = the plain "same action N
# times" rule; 2 and 3 = A B A B ... / A B C A B C ... with the same
# observations). Cycles need >= 3 full periods and >= `repeats` turns.
MAX_PERIOD = 3
OBS_HEAD = 200
SITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loopguard_site")

_WS_RE = re.compile(r"\s+")
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)\n```", re.DOTALL)


def repeats_from_env(env: dict | None = None) -> int:
    raw = (env if env is not None else os.environ).get(ENV_REPEATS, "")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def norm_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def action_key(content: str | None, tool_calls: list | None = None) -> str:
    """The reply's action, whitespace-normalized: its tool calls (name +
    canonical JSON arguments) when it has any, else the body of its last
    fenced block (mini-swe's ```bash action), else the whole visible reply
    (Terminus JSON batches, boxed answers, prose). Empty = no action."""
    if tool_calls:
        parts = []
        for call in tool_calls:
            fn = call.get("function") if isinstance(call, dict) else None
            if fn is None:
                fn = call if isinstance(call, dict) else {
                    "name": getattr(call, "name", ""),
                    "arguments": getattr(call, "arguments", ""),
                }
            parts.append(f"{fn.get('name')}({_canon_args(fn.get('arguments'))})")
        return norm_ws("\n".join(parts))
    text = content or ""
    fences = _FENCE_RE.findall(text)
    if fences:
        return norm_ws(fences[-1])
    return norm_ws(text)


def _canon_args(args) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return args.strip()
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(args)


def obs_head(text: str) -> str:
    return norm_ws(text)[:OBS_HEAD]


def is_loop(pairs: list[tuple[str, str | None]], repeats: int,
            max_period: int = MAX_PERIOD) -> bool:
    """`pairs` = (action, observation head) per completed turn, oldest first;
    the newest observation may be None (not seen yet).

    Period 1 (the spec): the last `repeats` actions are one non-empty
    action and every known observation between them is the same. Periods
    2..`max_period` catch the short cycles the king also runs to the cap
    (A B A B ...: re-reading two wiki sections in turn, seen live
    2026-09-11): the last `max(repeats, 3 * period)` turns repeat with that
    period in both action and known observation, i.e. at least three full
    cycles and never fewer turns than the period-1 rule. Longer cycles fall
    to `_revisits`: `repeats` consecutive turns that each exactly repeat an
    earlier (action, observation) pair."""
    if repeats <= 0 or len(pairs) < repeats:
        return False
    for period in range(1, max_period + 1):
        window = max(repeats, 3 * period)
        if len(pairs) < window:
            continue
        tail = pairs[-window:]
        acts = [a for a, _ in tail]
        if any(not a for a in acts):
            continue
        if any(acts[i] != acts[i - period] for i in range(period, window)):
            continue
        known = [o for _, o in tail[:-1]]
        if len(known) <= period or any(o is None for o in known):
            continue
        if any(known[i] != known[i - period] for i in range(period, len(known))):
            continue
        return True
    return _revisits(pairs, repeats)


def _revisits(pairs: list[tuple[str, str | None]], repeats: int) -> bool:
    """The fold's `in_loop` shape for cycles of any length (the king
    re-reading every section of a page, over and over): each of the last
    `repeats` turns with a known observation exactly repeats an earlier
    (action, observation) pair of the rollout, and the newest action — whose
    observation may not be in yet — repeats an earlier action too."""
    if pairs[-1][1] is None:
        known, newest = pairs[:-1], pairs[-1][0]
        if not newest or all(a != newest for a, _ in known):
            return False
    else:
        known = pairs
    if len(known) <= repeats:
        return False
    seen: set[tuple[str, str]] = set()
    first_checked = len(known) - repeats
    for idx, (action, obs) in enumerate(known):
        if idx >= first_checked:
            if not action or obs is None or (action, obs) not in seen:
                return False
        if action and obs is not None:
            seen.add((action, obs))
    return True


# -- verifiers hook -----------------------------------------------------------

def _trace_pairs(trace) -> list[tuple[str, str | None]]:
    """(action, observation head) for each sampled assistant node on the
    trace's current path (root -> newest node). The observation of a reply
    is the text of the user/tool messages committed after it and before
    the next sampled reply; the newest reply has none yet."""
    nodes = trace.nodes
    if not nodes:
        return []
    path: list[int] = []
    nid: int | None = len(nodes) - 1
    while nid is not None:
        path.append(nid)
        nid = nodes[nid].parent
    path.reverse()
    pairs: list[tuple[str, str | None]] = []
    obs_parts: list[str] = []
    for i in path:
        node = nodes[i]
        msg = node.message
        role = getattr(msg, "role", "")
        if node.sampled and role == "assistant":
            if pairs:
                a, _ = pairs[-1]
                pairs[-1] = (a, obs_head("\n".join(obs_parts)))
            obs_parts = []
            calls = [{"name": c.name, "arguments": c.arguments}
                     for c in (msg.tool_calls or [])]
            pairs.append((action_key(msg.content, calls), None))
        elif role in ("user", "tool") and pairs:
            obs_parts.append(_content_text(msg.content))
    if pairs and obs_parts:
        a, _ = pairs[-1]
        pairs[-1] = (a, obs_head("\n".join(obs_parts)))
    return pairs


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for part in content or []:
        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")
        if text:
            out.append(text)
    return "\n".join(out)


def install_verifiers_stop(repeats: int | None = None) -> bool:
    """Put a `@stop` named `loop_guard` on verifiers' Task base class so
    every taskset inherits it. Idempotent; False when verifiers is not
    importable in this interpreter (a foreign uv script env) or the guard
    is off."""
    n = repeats if repeats is not None else repeats_from_env()
    if n <= 0:
        return False
    try:
        from verifiers.v1.task import Task
        from verifiers.v1.utils.decorators import stop
    except Exception:
        return False
    if getattr(Task, STOP_CONDITION, None) is not None:
        return True

    async def loop_guard(self, trace) -> bool:
        pairs = _trace_pairs(trace)
        if not is_loop(pairs, n):
            return False
        print(f"[loop_guard] rollout {trace.id}: action repeated {n}x with the "
              f"same observation after {len(pairs)} turns; stopping",
              file=sys.stderr, flush=True)
        return True

    Task.loop_guard = stop(loop_guard)
    return True


# -- mini-swe-agent batch hook -----------------------------------------------

def run_mini_swe_guarded(argv: list[str], repeats: int | None = None) -> None:
    """`mini-extra swebench` with the loop guard on its agent class.

    The batch runner constructs its own `ProgressTrackingAgent` (it does not
    honour `agent_class`), so its `execute_actions` is wrapped: after the
    observations of an action land, the N-th identical (action, observation
    head) pair appends an `exit` message with `exit_status = loop_guard`,
    which ends `DefaultAgent.run` the way LimitsExceeded does."""
    from minisweagent.run.benchmarks import swebench as sb

    n = repeats if repeats is not None else repeats_from_env()
    if n > 0:
        agent_cls = sb.ProgressTrackingAgent
        original = agent_cls.execute_actions

        def execute_actions(self, message: dict) -> list[dict]:
            observations = original(self, message)
            pairs = getattr(self, "_loop_guard_pairs", None)
            if pairs is None:
                pairs = self._loop_guard_pairs = []
            action = "\n".join(
                str(a.get("command", a)) for a in
                message.get("extra", {}).get("actions", []))
            pairs.append((norm_ws(action),
                          obs_head("\n".join(str(o.get("content", ""))
                                             for o in observations))))
            if is_loop(pairs, n):
                self.add_messages({
                    "role": "exit", "content": STOP_CONDITION,
                    "extra": {"exit_status": STOP_CONDITION, "submission": ""},
                })
            return observations

        agent_cls.execute_actions = execute_actions
    sys.argv = [sys.argv[0], *argv]
    sb.app()


if __name__ == "__main__":
    run_mini_swe_guarded(sys.argv[1:])
