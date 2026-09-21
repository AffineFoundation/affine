"""Synthetic test of the wvk-19 validator path: `_confirm_crown` with a fake
eval client returning a canned pod confirmation stamp, both outcomes.
No pod, no chain, no state file writes."""
import asyncio
import sys
import types

sys.path.insert(0, "/home/const/subnet120/affine")
from affine import validator as V  # noqa: E402
from affine.state import QueueEntry  # noqa: E402
from evalsrv.dueling import confirmation_stamp  # noqa: E402
from affine.score import DuelResult  # noqa: E402


def pod_verdict(slice_margin: float, se: float = 0.00075, n: int = 1290):
    """What the pod would return for a per_duel confirmation request."""
    res = DuelResult(challenger="c", king="k", margin=slice_margin, se=se,
                     z=slice_margin / se, k_sigma=2.0, challenger_wins=False,
                     n_paired_turns=n, min_margin=0.002)
    confirm = {"challenge_id": "chal-test", "slice_index": 1, "rule": "per_duel",
               "k_sigma": 2.0, "min_margin": 0.002,
               "base": {"n": 1290, "margin": 0.0030, "se": 0.00075}}
    stamp = confirmation_stamp(confirm, {"index": 1, "seed": 424242, "n": 1300, "digest": "d"}, res)
    return {"confirmation": stamp, "job_id": "duel-synthetic", "challenger_wins": False,
            "rejection_reason": None}


class FakeEval:
    def __init__(self, v):
        self.v = v
        self.calls = []

    async def run_duel(self, **kw):
        self.calls.append(kw)
        return self.v


class FakeState:
    current_eval = None
    weight_fingerprints = {}

    def set_phase(self, *a, **k): pass


class FakeDash:
    def flush(self, *a, **k): pass


class FakeWatchdog:
    def beat(self): pass


async def run(slice_margin):
    v = V.Validator.__new__(V.Validator)
    v.cfg = types.SimpleNamespace(duel=types.SimpleNamespace(k_sigma=2.0, confirmation_required=True))
    v.eval_client = FakeEval(pod_verdict(slice_margin))
    v.state = FakeState()
    v.dashboard = FakeDash()
    v.watchdog = FakeWatchdog()
    published = []

    async def _pub(entry, verdict):
        published.append((entry.challenge_id, verdict.get("confirmation_of")))
    v._publish_eval_artifact = _pub
    entry = QueueEntry(challenge_id="chal-test", hotkey="5Fake", repo="r2://b/p/", revision="r" * 64,
                       block=1, queued_at="")
    king = types.SimpleNamespace(repo="r2://k/", revision="k" * 64)
    first = {"margin": 0.0030, "se": 0.00075, "z": 4.0, "n_paired_turns": 1290, "near_miss": {"slices": [{}]}}
    info = types.SimpleNamespace(total_safetensors_bytes=1)
    conf = await v._confirm_crown(entry, king, first, "0xhash", {"min_margin_effective": 0.002}, info)
    kw = v.eval_client.calls[0]
    assert kw["confirm"]["rule"] == "per_duel" and kw["confirm"]["slice_index"] == 1
    assert kw["confirm"]["base"]["margin"] == 0.0030 and kw["confirm"]["min_margin"] == 0.002
    assert published == [("chal-test-confirm", "chal-test")]
    return conf


for m in (0.0025, -0.0003, 0.0005):
    c = asyncio.run(run(m))
    print(f"slice2 margin {m:+.4f}: seed={c['seed']} n={c['n']} z={c['z']:.2f} pooled={c['pooled_margin']:.5f} "
          f"pooled_se={c['pooled_se']:.5f} pooled_z={c['pooled_z']:.2f} bar={c['bar']:.5f} passed={c['passed']}")
    # what _process_challenge does with it
    verdict = {"challenger_wins": True}
    verdict["confirmation"] = c
    if not c["passed"]:
        verdict["challenger_wins"] = False
        verdict["rejection_reason"] = "confirmation_failed"
    print("   -> challenger_wins", verdict["challenger_wins"], "rejection_reason", verdict.get("rejection_reason"))
print("OK")
