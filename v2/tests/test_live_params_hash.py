"""The public contract is live; only an unavailable fetch skips this check."""

import pytest

from sn120_v2 import check_live_params_hash
from sn120_v2.params import Params


@pytest.fixture(scope="module")
def live_document():
    try:
        return check_live_params_hash.fetch_mining_json()
    except check_live_params_hash.ParamsFetchError as exc:
        pytest.skip(f"live mining.json fetch unavailable: {exc}")


def test_live_params_hash(live_document):
    recomputed = Params.from_public_dict(live_document["params"]).params_hash
    assert recomputed == live_document["params_hash"]


def test_cli_reports_a_successful_match(monkeypatch, capsys):
    document = {"params": Params().to_dict(), "params_hash": Params().params_hash}
    monkeypatch.setattr(check_live_params_hash, "fetch_mining_json", lambda: document)
    assert check_live_params_hash.main() == 0
    assert "params_hash matches" in capsys.readouterr().out


def test_cli_fails_on_a_hash_mismatch(monkeypatch, capsys):
    document = {"params": Params().to_dict(), "params_hash": "0" * 64}
    monkeypatch.setattr(check_live_params_hash, "fetch_mining_json", lambda: document)
    assert check_live_params_hash.main() == 1
    assert "params_hash mismatch" in capsys.readouterr().err


def test_cli_distinguishes_a_fetch_failure(monkeypatch, capsys):
    def fail():
        raise check_live_params_hash.ParamsFetchError("test fetch failure")

    monkeypatch.setattr(check_live_params_hash, "fetch_mining_json", fail)
    assert check_live_params_hash.main() == 2
    assert "test fetch failure" in capsys.readouterr().err
