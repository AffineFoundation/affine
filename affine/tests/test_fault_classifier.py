"""Infra faults must never spend the miner's retry budget.

Fixtures are the stored validator log lines that motivated the rule:
  chal-00568, 2026-09-17 10:17:26 — "requeued ... (retry 1, counted=True) due to
    eval server error: Server error '502 Bad Gateway' for url
    'http://127.0.0.1:9100/v1/completions'"
  chal-00520, 2026-09-15 — the swarm answered 503 the same way (log rotated;
    reconstructed text).
Both came through the pod's SSE error event with no Fault code, were typed
TransientEvalError, and were counted. Miner-attributable outcomes (unservable
verdicts, probe/format rejections) are verdicts, not exceptions — they never
reach this path.
"""
import unittest

from affine.eval_client import (Fault, InfraFaultError, TransientEvalError,
                                classify_server_error, is_infra_message)

CASE_00568 = ("Server error '502 Bad Gateway' for url "
              "'http://127.0.0.1:9100/v1/completions'")
CASE_00520 = ("Server error '503 Service Unavailable' for url "
              "'http://127.0.0.1:9100/v1/completions'")


class TestFaultClassifier(unittest.TestCase):
    def test_stored_cases_are_infra(self):
        for text in (CASE_00568, CASE_00520):
            exc = classify_server_error(text, None)
            self.assertIsInstance(exc, InfraFaultError, text)
            self.assertEqual(exc.code, Fault.UPSTREAM)
            self.assertTrue(is_infra_message(text))

    def test_explicit_codes_win(self):
        exc = classify_server_error("teacher not servable", Fault.TEACHER)
        self.assertIsInstance(exc, InfraFaultError)
        self.assertEqual(exc.code, Fault.TEACHER)

    def test_transport_and_stream_loss_are_infra(self):
        for text in (
            "challenger: ConnectError:All connection attempts failed",
            "duel stream broke: peer closed connection without sending complete message body",
            "duel job duel-abc vanished (404); evalsrv likely restarted mid-duel",
            "RemoteProtocolError('Server disconnected without sending a response.')",
            "Client error '500 Internal Server Error' for url 'http://localhost:8002/v1/completions'",
        ):
            self.assertTrue(is_infra_message(text), text)
            self.assertIsInstance(classify_server_error(text, None), InfraFaultError, text)

    def test_unknown_text_stays_generic(self):
        exc = classify_server_error("slice digest mismatch: expected abc got def", None)
        self.assertIsInstance(exc, TransientEvalError)
        self.assertNotIsInstance(exc, InfraFaultError)
        self.assertFalse(is_infra_message("protocol:pass_rate=0.70<0.9"))
        self.assertFalse(is_infra_message("unservable:challenger failed to load in vLLM: "
                                          "FileNotFoundError: weight files referenced in index"))


if __name__ == "__main__":
    unittest.main()
