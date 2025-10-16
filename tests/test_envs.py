import numpy as np

from affine.envs import mult8, tictactoe


def test_mult8_deterministic_prompt() -> None:
    env = mult8.Mult8Env()
    obs1, info1 = env.reset(options={"challenge_id": "abcd"})
    obs2, info2 = env.reset(options={"challenge_id": "abcd"})
    assert obs1 == obs2
    assert info1["challenge_id"] == info2["challenge_id"]


def test_mult8_verify() -> None:
    env = mult8.Mult8Env()
    obs, info = env.reset(options={"challenge_id": "1234"})
    expected = info["expected"]
    assert env.verify(str(expected), info).ok
    assert not env.verify("42", info).ok


def test_mult8_verify_challenge_reconstruction() -> None:
    env = mult8.Mult8Env()
    _obs, info = env.reset(options={"challenge_id": "abcd"})
    expected = info["expected"]
    stripped = {k: v for k, v in info.items() if k != "expected"}
    verifier = mult8.Mult8Env()
    assert verifier.verify(str(expected), stripped).ok


def test_mult8_expected_mismatch_detected() -> None:
    env = mult8.Mult8Env()
    _obs, info = env.reset(options={"challenge_id": "abcd"})
    expected = info["expected"]
    tampered = dict(info)
    tampered["expected"] = expected + 1
    verdict = env.verify(str(expected), tampered)
    assert not verdict.ok
    assert verdict.reason == "expected-mismatch"


def test_tictactoe_illegal_move_detected() -> None:
    env = tictactoe.TicTacToeEnv()
    _obs, info = env.reset(options={"challenge_id": "aa"})
    verdict = env.verify([0, 0], info)
    assert not verdict.ok


def test_tictactoe_opening_deterministic() -> None:
    env1 = tictactoe.TicTacToeEnv()
    obs1, info1 = env1.reset(options={"challenge_id": "1"})
    env2 = tictactoe.TicTacToeEnv()
    obs2, info2 = env2.reset(options={"challenge_id": "1"})
    assert np.array_equal(obs1, obs2)
    assert info1["opening_moves"] == info2["opening_moves"]


def test_tictactoe_verify_matches_playthrough() -> None:
    env = tictactoe.TicTacToeEnv()
    observation, info = env.reset(options={"challenge_id": "2"})
    moves = []
    terminated = False
    truncated = False
    for _ in range(5):
        empty_slots = [idx for idx, value in enumerate(observation.ravel()) if value == 0]
        move = empty_slots[0]
        observation, reward, terminated, truncated, info = env.step(move)
        moves.append(move)
        if terminated or truncated:
            break
    assert terminated
    verifier = tictactoe.TicTacToeEnv()
    verdict = verifier.verify(moves, info)
    assert verdict.reason == info["reason"]
