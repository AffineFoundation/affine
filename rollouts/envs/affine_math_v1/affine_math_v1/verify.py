# /// script
# dependencies = ["math-verify"]
# ///
"""Grade one \\boxed{} answer inside the rollout's runtime (`uv run`).

argv[1] = gold answer (already the bare content of the reference \\boxed{}),
argv[2] = the model's full reply. Prints 1.0 when the LAST complete
\\boxed{...} in the reply is equivalent to the gold under math-verify, else
0.0. The last-boxed rule and brace balancing mirror affine/dialects.py
`boxed` (the duel scores the same span), kept inline because this script
runs with math-verify as its only dependency.
"""

import sys

from math_verify import parse, verify

OPEN = "\\boxed{"


def last_boxed(text: str) -> str | None:
    found = None
    start = text.find(OPEN)
    while start != -1:
        depth = 0
        for i in range(start + len(OPEN) - 1, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    found = text[start:i + 1]
                    break
        start = text.find(OPEN, start + len(OPEN))
    return found


gold, reply = sys.argv[1], sys.argv[2]
pred = last_boxed(reply)
if pred is None:
    print(0.0)
    sys.exit(0)
try:
    score = 1.0 if verify(parse(OPEN + gold + "}"), parse(pred)) else 0.0
except Exception:  # noqa: BLE001 - malformed LaTeX can fail anywhere in math-verify
    score = 0.0
print(score)
