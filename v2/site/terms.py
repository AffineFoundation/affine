"""Public wording for the v2 status site: Jacob's approved terms.

``submission``  one LoRA a miner uploads (never call one submission a batch)
``round``        one evaluation cycle (r000011); never ``epoch`` for this
``batch``        the submissions seated in a round (up to six now)
``king``         the round's winner
``tasks``        the eval problems (2000 per submission)
``rollouts``     each attempt a submission makes at a task
``looks``        stages rollouts are scored in, with an early-stop check

``rewrite_served`` adjusts text the site copies to the public root (llms.txt).
Technical identifiers stay: ``batch.json``, ``batch_sha256``, ``affine-batch-v*``,
``batch_cap``, field names, and code paths.

``_SEATS`` is not a term: it states a rule. A round limits seats by hotkey only,
and the coldkey is not used. The replacement is, byte for byte, the two lines
that the publisher prints from the commit that states the rule (``0b800898``).
So the rule matches nothing once the box runs that commit, and it can go then.
"""

from __future__ import annotations

import re

# Longer phrases first so shorter ones do not fire inside them.
_EVALUATED = (
    (re.compile(r"until the entry is evaluated", re.I), "until the entry has had its one turn"),
    (re.compile(r"until its queue entry is evaluated", re.I), "until its queue entry has had its one turn"),
    (re.compile(r"whose entry was evaluated", re.I), "that already had their one turn"),
    (re.compile(r"whose queue entry is not evaluated yet", re.I), "whose queue entry has not had its one turn yet"),
    (re.compile(r"entry was evaluated", re.I), "entry already had its one turn"),
    (re.compile(r"is not evaluated yet", re.I), "has not had their one turn yet"),
    (re.compile(r"not evaluated yet", re.I), "has not had their one turn yet"),
    (re.compile(r"already evaluated", re.I), "already had their one turn"),
    (re.compile(r"\bevaluated\b", re.I), "had their one turn"),
)

# User-facing "batch" meaning one miner upload → submission. Leave wire names alone.
_SUBMISSION = (
    (re.compile(r"\badmitted batch\b"), "admitted submission"),
    (re.compile(r"\beach admitted batch\b"), "each admitted submission"),
    (re.compile(r"\bYour batch\b"), "Your submission"),
    (re.compile(r"\byour batch\b"), "your submission"),
    (re.compile(r"\ba batch made\b"), "a submission made"),
    (re.compile(r"\bthe batch is not trained\b"), "the submission is not trained"),
    (re.compile(r"\bof the batch\b"), "of the submission"),
    (re.compile(r"\bA failure of the box in a training \(not of the batch\)"),
     "A failure of the box in a training (not of the submission)"),
    (re.compile(r"\bat most 6 batches\b"), "at most 6 submissions"),
    (re.compile(r"\bneeds at least 1\b"), "needs at least 1 submission"),
    (re.compile(r"\bBatches train\b"), "Submissions train"),
    (re.compile(r"\bEach batch trains\b"), "Each submission trains"),
    (re.compile(r"\beach other batch\b"), "each other submission"),
    (re.compile(r"\bA batch that started\b"), "A submission that started"),
    (re.compile(r"\bthe batch is `not_trained`\b"), "the submission is `not_trained`"),
    (re.compile(r"\bno batch had\b"), "no submission had"),
    (re.compile(r"\ba batch that does\b"), "a submission that does"),
    (re.compile(r"\byour batch had\b"), "your submission had"),
    (re.compile(r"\bone other batch\b"), "one other submission"),
    (re.compile(r"\bbefore your batch was trained\b"), "before your submission was trained"),
    (re.compile(r"\bOnly a winner has a public batch\b"), "Only a winner has a public submission"),
    (re.compile(r"\bthe batch sha256\b"), "the submission sha256"),
    (re.compile(r"\bTrain and score your batch\b"), "Train and score your submission"),
    (re.compile(r"\bThe round has one batch\b"), "The round has one submission"),
    (re.compile(r"\bnumber of batches\b"), "number of submissions"),
    (re.compile(r"\bfor all batches\b"), "for all submissions"),
    (re.compile(r"\bwith 8 batches or\b"), "with 8 submissions or"),
    (re.compile(r"\b8 batches at a time\b"), "8 submissions at a time"),
    (re.compile(r"\beach batch trains on one GPU\b"), "each submission trains on one GPU"),
    (re.compile(r"\bfloor\(8 / batches\)\b"), "floor(8 / submissions)"),
    (re.compile(r"--max-loras '<batches of the round>'"), "--max-loras '<submissions of the round>'"),
)


_SEATS = (
    (
        re.compile(
            r"^- One entry for each miner in a round\. A second seat of the same hotkey or coldkey waits for a later round\.$",
            re.M,
        ),
        "- One seat for a hotkey in a round. A second seat of the same hotkey waits for a later round. The coldkey is not\n"
        "  used: one coldkey can have several seats, each with its own hotkey, in the same round.",
    ),
)


def rewrite_served(text: str | bytes) -> bytes:
    """Apply approved public wording to text the site serves (llms.txt)."""
    raw = text.decode() if isinstance(text, (bytes, bytearray)) else text
    for pattern, repl in _EVALUATED:
        raw = pattern.sub(repl, raw)
    for pattern, repl in _SUBMISSION:
        raw = pattern.sub(repl, raw)
    for pattern, repl in _SEATS:
        raw = pattern.sub(repl, raw)
    return raw.encode()
