# /// script
# requires-python = ">=3.10"
# dependencies = ["mini-swe-agent=={version}", "litellm[proxy]"]
# ///
"""Resume a mini-swe-agent *textbased* rollout from a stored prefix.

Runs inside the task container (same place the `mini_swe_textbased` harness
program runs). Two phases:

1. Replay: every assistant message of the prefix carries exactly one
   ```mswea_bash_command block that the original run executed (mini-swe drops
   format-error replies from history, so they are not in the prefix). Run
   them in order with mini-swe's own LocalEnvironment so the filesystem ends
   up where the king left it. Each replayed output is compared with the
   recorded observation (a fidelity signal, not a gate).
2. Continue: seed mini-swe's DefaultAgent with the prefix and step it until
   it exits (Submitted / LimitsExceeded / RepeatedFormatError), with the
   model routed through the interception endpoint — the trace records only
   the continuation.

argv: --state-file --base-url --api-key --model --report --temperature
      --replay-timeout
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError, InterruptAgentFlow
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

ACTION_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
RETURNCODE_RE = re.compile(r"<returncode>(-?\d+)</returncode>")
OUTPUT_RE = re.compile(r"<output>\n?(.*?)</output>", re.DOTALL)
SUBMIT_MARK = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-file", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--replay-timeout", type=int, default=900)
    ap.add_argument("--command-timeout", type=int, default=86400)
    return ap.parse_args()


def replay_actions(messages: list[dict]) -> list[tuple[int, str, dict | None]]:
    """(message index, command, recorded observation) for every executed
    prefix action. The observation is the next user message when it looks
    like a mini-swe observation."""
    out = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        found = ACTION_RE.findall(m.get("content") or "")
        if len(found) != 1:
            continue
        obs = None
        if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
            txt = messages[i + 1].get("content") or ""
            rc = RETURNCODE_RE.search(txt)
            if rc:
                body = OUTPUT_RE.search(txt)
                obs = {"returncode": int(rc.group(1)),
                       "output": body.group(1) if body else None}
        out.append((i, found[0].strip(), obs))
    return out


def main() -> None:
    args = parse_args()
    state = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    messages = state["messages"]
    cfg = yaml.safe_load(get_config_path("mini_textbased").read_text())
    env_cfg = dict(cfg.get("environment") or {})
    env_cfg.pop("environment_class", None)
    env_cfg["timeout"] = args.command_timeout
    env = LocalEnvironment(**env_cfg)

    report: dict = {"replay": [], "replay_n": 0, "replay_match": 0,
                    "replay_rc_match": 0, "replay_seconds": 0.0}
    t0 = time.time()
    for msg_idx, command, obs in replay_actions(messages):
        if command.strip().splitlines() and \
                command.strip().splitlines()[0].strip().startswith("echo " + SUBMIT_MARK):
            # A replayed submit would end the run; the king never got here
            # (the state is inside a failed rollout), so it cannot occur.
            continue
        try:
            got = env.execute({"command": command}, timeout=args.replay_timeout)
        except InterruptAgentFlow:
            got = {"output": SUBMIT_MARK, "returncode": 0}
        entry = {"msg_idx": msg_idx, "rc": got.get("returncode"),
                 "rc_match": None, "output_match": None}
        if obs is not None:
            entry["rc_match"] = got.get("returncode") == obs["returncode"]
            if obs["output"] is not None:
                entry["output_match"] = (got.get("output") or "").strip() == \
                    (obs["output"] or "").strip()
        report["replay"].append(entry)
        report["replay_n"] += 1
        report["replay_match"] += 1 if entry["output_match"] else 0
        report["replay_rc_match"] += 1 if entry["rc_match"] else 0
    report["replay_seconds"] = round(time.time() - t0, 1)
    Path(args.report).write_text(json.dumps(report))

    model_cfg = dict(cfg.get("model") or {})
    model_cfg.pop("model_class", None)
    kwargs = dict(model_cfg.pop("model_kwargs", {}) or {})
    kwargs.update({
        "custom_llm_provider": "openai",
        "api_base": args.base_url,
        "api_key": args.api_key,
        "temperature": args.temperature,
    })
    model = LitellmTextbasedModel(model_name=args.model, model_kwargs=kwargs,
                                  cost_tracking="ignore_errors", **model_cfg)
    agent_cfg = dict(cfg.get("agent") or {})
    agent_cfg = {k: v for k, v in agent_cfg.items()
                 if k in ("system_template", "instance_template",
                          "max_consecutive_format_errors")}
    agent = DefaultAgent(model, env, step_limit=0, cost_limit=0, **agent_cfg)

    # DefaultAgent.run() would rebuild the first two messages; seed the
    # conversation instead and drive step() exactly as run() does.
    agent.messages = [{"role": m["role"], "content": m.get("content") or ""}
                      for m in messages]
    first_reply = None
    exit_status = ""
    while True:
        try:
            msg = agent.query()
            if first_reply is None:
                first_reply = msg.get("content")
            agent.execute_actions(msg)
            agent.n_consecutive_format_errors = 0
        except FormatError as e:
            if first_reply is None:
                # The reply itself is never added to history; it rides in
                # the format-error message's extra.
                first_reply = ((e.messages[0].get("extra") or {}).get("model_response")
                               if e.messages else "")
            agent.n_consecutive_format_errors += 1
            if 0 < agent.config.max_consecutive_format_errors <= agent.n_consecutive_format_errors:
                agent.add_messages(*e.messages, {
                    "role": "exit", "content": "RepeatedFormatError",
                    "extra": {"exit_status": "RepeatedFormatError", "submission": ""}})
            else:
                agent.add_messages(*e.messages)
        except InterruptAgentFlow as e:
            agent.add_messages(*e.messages)
        except Exception as e:  # noqa: BLE001 - surfaced through the report + exit code
            report["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
            report["first_reply"] = first_reply
            report["exit_status"] = "error"
            Path(args.report).write_text(json.dumps(report))
            raise
        if agent.messages[-1].get("role") == "exit":
            exit_status = agent.messages[-1].get("extra", {}).get("exit_status", "")
            break
    report["exit_status"] = exit_status
    report["first_reply"] = first_reply
    report["n_calls"] = agent.n_calls
    Path(args.report).write_text(json.dumps(report))
    if exit_status not in ("Submitted", "LimitsExceeded", "RepeatedFormatError"):
        sys.exit(1)


if __name__ == "__main__":
    main()
