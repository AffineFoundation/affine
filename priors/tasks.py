"""Active-set task loading. Task ids are env-prefixed: math:<index into original-math7496>, dapo:<idx in the pinned DAPO pool>.
Both envs use the affine_math system prompt and a single user turn with the bare problem (exactly how the live affine_math
MathTask builds its messages: [system=data.system_prompt, user=data.prompt])."""
import hashlib, json
MATH_SYSTEM = ("Solve the math problem. Reason step by step in plain text, then end your response with the final answer in "
               "`\\boxed{}`. Emit exactly one `\\boxed{}` block and nothing after it.")
ENV_PREFIX = {"affine_math": "math", "dapo": "dapo"}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load_affine_math(path):
    rows = json.load(open(path)); out = []
    for i, t in enumerate(rows):
        d = t["data"]
        assert d["system_prompt"] == MATH_SYSTEM and d["prompt"] == d["problem"]
        out.append(dict(task_id="math:%d" % i, env="affine_math", system=d["system_prompt"], prompt=d["prompt"], answer=str(d["answer"])))
    return out


def load_dapo(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        out.append(dict(task_id=r["id"], env="dapo", system=MATH_SYSTEM, prompt=r["problem"], answer=str(r["answer"])))
    return out


LOADERS = {"affine_math": load_affine_math, "dapo": load_dapo}


def load_active(envs: dict):
    """envs = {env: {"path": ..., "sha256": optional pin}} -> list of task dicts (stable order)."""
    out = []
    for env, spec in envs.items():
        if spec.get("sha256") and file_sha256(spec["path"]) != spec["sha256"]:
            raise RuntimeError("%s pool sha256 mismatch" % env)
        out += LOADERS[env](spec["path"])
    return out
