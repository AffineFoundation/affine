"""State.load must not requeue an in-flight entry that already has a verdict.

Today's case (2026-09-21 17:51): `chal-00637`'s verdict row was appended at
17:50:58, the validator was restarted at 17:51:37 before the periodic flush
had cleared `in_flight`, and `State.load` put chal-00637 back at the queue
head — a second duel for a judged submission. Same pitfall as chal-00308
(2026-08-07) and chal-00366 (wvk-13 flip).
"""
import json
import tempfile
import unittest
from pathlib import Path

from affine.state import State


def _entry(cid: str) -> dict:
    return {"challenge_id": cid, "hotkey": "5X", "repo": "r2://b/p/", "revision": "d" * 64,
            "block": 1, "queued_at": "2026-09-21T17:00:00+00:00", "retry_count": 0,
            "infra_retry_count": 0, "deferred_after": 0}


class TestInFlightRecovery(unittest.TestCase):
    def _state(self, in_flight: dict | None, history_rows: list[dict]) -> State:
        d = Path(tempfile.mkdtemp())
        (d / "state.json").write_text(json.dumps({
            "king": None, "queue": [_entry("chal-00638")], "in_flight": in_flight,
            "seen_hotkeys": [], "completed_revisions": [],
            "stats": {"queued": 1, "accepted": 0, "rejected": 0, "failed": 0}, "id_counter": 638}))
        with open(d / "history.jsonl", "w") as f:
            for r in history_rows:
                f.write(json.dumps(r) + "\n")
        st = State(d)
        st.load()
        return st

    def test_judged_in_flight_is_dropped(self):
        st = self._state(_entry("chal-00637"), [
            {"event": "verdict", "at": "2026-09-21T17:50:58+00:00",
             "challenge_id": "chal-00637", "accepted": False}])
        self.assertEqual([e.challenge_id for e in st.queue], ["chal-00638"])
        self.assertIsNone(st.in_flight)

    def test_unjudged_in_flight_is_requeued_at_front(self):
        st = self._state(_entry("chal-00637"), [
            {"event": "verdict", "at": "2026-09-21T16:00:00+00:00",
             "challenge_id": "chal-00636", "accepted": False}])
        self.assertEqual([e.challenge_id for e in st.queue], ["chal-00637", "chal-00638"])
        self.assertIsNone(st.in_flight)

    def test_clear_in_flight_flushes_immediately(self):
        st = self._state(None, [])
        entry = st.pop_next()  # sets in_flight and flushes
        self.assertEqual(json.loads((st.dir / "state.json").read_text())["in_flight"]["challenge_id"],
                         "chal-00638")
        st.record_verdict(entry, {"challenger_wins": False})
        on_disk = json.loads((st.dir / "state.json").read_text())
        self.assertIsNone(on_disk["in_flight"])


if __name__ == "__main__":
    unittest.main()
