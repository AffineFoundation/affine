"""The task pool and the two prompt renderings a miner needs.

pool()                   the 7,496 public math tasks (data/original-math7496.tasks.json, sha256 checked) as a list of
                         {'prompt', 'answer'}; the list index is the task id the gate uses (gate_core.draw_sample over range(7496)).
batch_prompt(problem)    the prompt string for a batch file: MANDATE + "\\n\\n" + problem. The trainer renders it as ONE user message
                         (ff_format.render: apply_chat_template([user], add_generation_prompt=True), no system message). Every GREEN
                         file so far (e354/e356/e358/e360/e368) used exactly this string.
batch_ids(tok, prompt)   the token ids the trainer conditions on for that prompt (sample training rollouts from THESE ids).
gate_ids(tok, problem)   the token ids the gate probe samples from: [system: MANDATE, user: problem], enable_thinking=True.
"""
import hashlib, json, sys
from pathlib import Path
KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import kitpath  # noqa: E402,F401  (puts the verbatim validator modules on sys.path)
import ff_format  # noqa: E402
MANDATE = ("Solve the math problem. Reason step by step in plain text, then end your response with the final answer in `\\boxed{}`. "
           "Emit exactly one `\\boxed{}` block and nothing after it.")


def pool():
    raw = Path(kitpath.POOL_FILE).read_bytes()
    if hashlib.sha256(raw).hexdigest() != kitpath.POOL_SHA256:
        raise SystemExit('task pool sha256 mismatch')
    return [dict(prompt=t['data']['prompt'], answer=t['data']['answer']) for t in json.loads(raw)]


def batch_prompt(problem):
    return MANDATE + '\n\n' + problem


def batch_ids(tok, prompt):
    return list(ff_format.render(tok, prompt))


def gate_ids(tok, problem):
    ids = tok.apply_chat_template([dict(role='system', content=MANDATE), dict(role='user', content=problem)], tokenize=True,
                                  return_dict=False, add_generation_prompt=True, enable_thinking=True)
    return list(ids['input_ids'] if isinstance(ids, dict) else ids)
