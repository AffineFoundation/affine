"""Prospective CPU-only K2/L2 identity and pairing controls. No live call sites.

Inputs are normalized metadata from an authenticated adapter, not a replacement
for manifest authentication, prescribed draws, native grading or inference checks.
All classifications remain claims until audited. No persistent receipt is written.
"""
from dataclasses import dataclass
from hashlib import sha256
import json


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _trace_json(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("trace object keys must be strings")
        for child in value.values():
            _trace_json(child)
    elif isinstance(value, list):
        for child in value:
            _trace_json(child)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("canonical JSON trace required")


def _text(value):
    if not isinstance(value, str) or not value:
        raise ValueError("nonempty identity required")
    return value


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("canonical SHA256 required")
    return value


@dataclass(frozen=True)
class ApprovedTask:
    epoch: str
    checkpoint: str
    taskset_sha256: str
    env_id: str
    index: int
    task_sha256: str
    harness_sha256: str
    sampling_context_sha256: str
    approved_attempts: tuple

    def __post_init__(self):
        _text(self.epoch)
        _text(self.checkpoint)
        _text(self.env_id)
        for value in (self.taskset_sha256, self.task_sha256,
                      self.harness_sha256, self.sampling_context_sha256):
            _sha(value)
        if type(self.index) is not int or self.index < 0:
            raise ValueError("task index required")
        if (type(self.approved_attempts) is not tuple or not self.approved_attempts or
                any(type(a) is not int or a < 0 for a in self.approved_attempts) or
                len(set(self.approved_attempts)) != len(self.approved_attempts)):
            raise ValueError("distinct approved attempt identifiers required")

    def task_binding(self):
        return dict(checkpoint=self.checkpoint, taskset_sha256=self.taskset_sha256,
                    env_id=self.env_id, index=self.index, task_sha256=self.task_sha256)

    def slot_id(self, miner):
        return digest(dict(kind="task-slot-v1", epoch=self.epoch,
                           miner=_text(miner), task=self.task_binding()))


def identities(task, rollout):
    """Recompute identities. Metadata wrappers never make a new trajectory.

    The canonical trace schema here deliberately requires complete per-turn
    prompt/output token IDs and actions/observations. Production needs an adapter
    for its actual harness fields; absent actions are explicitly empty lists.
    """
    if not isinstance(rollout, dict):
        raise ValueError("rollout mapping required")
    if rollout.get("epoch") != task.epoch:
        raise ValueError("rollout epoch binding")
    for key, expected in task.task_binding().items():
        if rollout.get(key) != expected:
            raise ValueError("rollout task/checkpoint binding")
    if (rollout.get("harness_sha256") != task.harness_sha256 or
            rollout.get("sampling_context_sha256") != task.sampling_context_sha256):
        raise ValueError("approved execution binding")
    attempt = rollout.get("attempt")
    if type(attempt) is not int or attempt not in task.approved_attempts:
        raise ValueError("unapproved prescribed attempt")
    turns = rollout.get("turns")
    if not isinstance(turns, list) or not 1 <= len(turns) <= 32:
        raise ValueError("complete turn trace required")
    trace = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("turn mapping required")
        for key in ("prompt", "output"):
            tokens = turn.get(key)
            if (not isinstance(tokens, list) or not 1 <= len(tokens) <= (8192 if key == "prompt" else 2048) or
                    any(type(t) is not int or not 0 <= t < 200000 for t in tokens)):
                raise ValueError("canonical token trace required")
        if len(turn["prompt"]) + len(turn["output"]) > 8192:
            raise ValueError("turn context budget")
        for key in ("actions", "observations"):
            if not isinstance(turn.get(key), list):
                raise ValueError("explicit action/observation trace required")
        _trace_json([turn["actions"], turn["observations"]])
        trace.append({key: turn[key] for key in ("prompt", "output", "actions", "observations")})
    if len(json.dumps(trace, sort_keys=True, allow_nan=False).encode()) > 1048576:
        raise ValueError("canonical trace byte budget")
    execution = digest(dict(kind="execution-v1", task=task.task_binding(),
                            harness_sha256=task.harness_sha256,
                            sampling_context_sha256=task.sampling_context_sha256,
                            attempt=attempt))
    content = digest(dict(kind="trajectory-content-v1", task=task.task_binding(),
                          harness_sha256=task.harness_sha256, turns=trace))
    return execution, content


def select_pairs(task, miner, rollouts, *, quota=1):
    """Return nonoverlapping deterministic pairs or refuse incomplete quota.

    Repeated attempts with inconsistent content/labels are refused. Repeated
    contents across attempts are duplicates, not evidence of cheating. Labels
    cannot create a second identity. Sorting before dedupe makes upload order
    irrelevant. Caller must supply the whole cumulative task-slot revision.
    Default quota remains one; no production contract imports this helper.
    """
    if type(quota) is not int or quota not in (1, 2):
        raise ValueError("prospective quota must be one or two")
    if not isinstance(rollouts, list) or not rollouts or len(rollouts) > len(task.approved_attempts) * 4:
        raise ValueError("bounded cumulative revision required")
    attempts = {}
    contents = {}
    for rollout in rollouts:
        execution, content = identities(task, rollout)
        label = rollout.get("classification")
        if label not in ("positive", "negative"):
            raise ValueError("explicit claimed classification required")
        old = attempts.get(execution)
        if old is not None and old != (content, label):
            raise ValueError("conflicting prescribed attempt")
        attempts[execution] = (content, label)
        if content in contents and contents[content] != label:
            raise ValueError("conflicting claimed content classification")
        contents[content] = label
    unique = {}
    for execution, (content, label) in sorted(attempts.items()):
        unique.setdefault(content, dict(execution_id=execution, content_id=content, classification=label))
    positives = sorted((r for r in unique.values() if r["classification"] == "positive"), key=lambda r: r["content_id"])
    negatives = sorted((r for r in unique.values() if r["classification"] == "negative"), key=lambda r: r["content_id"])
    if len(positives) < quota or len(negatives) < quota:
        raise ValueError("distinct success/failure quota not met")
    pairs = [dict(positive=p, negative=n) for p, n in zip(positives[:quota], negatives[:quota])]
    chosen = [r["content_id"] for pair in pairs for r in (pair["positive"], pair["negative"])]
    if len(set(chosen)) != 2 * quota:
        raise ValueError("pair member reuse")
    revision = dict(kind="selected-task-revision-v1", slot_id=task.slot_id(miner),
                    quota=quota, pairs=pairs, contribution_units=1,
                    pair_weight_within_task=1 / quota)
    return dict(revision, revision_id=digest(revision),
                duplicate_content_count=len(attempts) - len(unique))
