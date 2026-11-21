"""Large-set, incremental Generator link issuance with a concurrent fake store."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest
from nacl.signing import SigningKey

from sn120_v2.generator.config import Config
from sn120_v2.generator.control import issue_links
from sn120_v2.generator.epoch import epoch_deadline, epoch_start
from sn120_v2.store import LocalStore

EPOCH = "20261120T0000Z"
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def address(key):
    raw = bytes([42]) + bytes(key.verify_key)
    number = int.from_bytes(raw + hashlib.blake2b(b"SS58PRE" + raw, digest_size=64).digest()[:2], "big")
    text = ""
    while number:
        number, digit = divmod(number, 58)
        text = B58[digit] + text
    return text


def hotkeys(count):
    return [address(SigningKey(hashlib.sha256(f"issuance-{number}".encode()).digest())) for number in range(count)]


class Store(LocalStore):
    def __init__(self, root):
        super().__init__(root)
        self.puts = []
        self.active = 0
        self.max_active = 0
        self.gate = None
        self.index_at_completion = None
        self.site = None
        self.lock = threading.Lock()

    def presign_get(self, key, ttl_s):
        return "https://store.invalid/" + key

    def presign_put(self, key, ttl_s):
        return "https://store.invalid/" + key, {}

    def get(self, key):
        try:
            return super().get(key)
        except KeyError:
            raise KeyError(key) from None

    def put(self, key, raw):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if key == "public/generator/links.json":
                assert self.site is not None
                index = json.loads(raw)
                for hotkey in index["hotkeys"]:
                    assert (self.site / "generator" / "links" / f"{hotkey}.json").is_file()
                    assert super().get(f"public/generator/links/{hotkey}.json")
            if self.gate is not None and key != "public/generator/links.json":
                self.gate.wait(5)
            super().put(key, raw)
            with self.lock:
                self.puts.append(key)
                if key == "public/generator/links.json":
                    self.index_at_completion = len(self.puts)
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def setup(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    store = Store(tmp_path / "store")
    store.site = site
    config = Config(site, tmp_path / "work", tmp_path / "pool", first_epoch=EPOCH)
    current = {"keys": []}
    def registrations(client, clock):
        return {index + 1: hotkey for index, hotkey in enumerate(current["keys"])}, set(current["keys"])
    monkeypatch.setattr("sn120_v2.generator.control.read_registrations", registrations)
    return config, store, current


def test_1000_keys_concurrent_index_last_then_incremental_prefetch(setup):
    config, store, current = setup
    start = int(epoch_start(EPOCH).timestamp()) + 60
    current["keys"] = hotkeys(1000)
    store.gate = threading.Event()
    result = {}

    def issue():
        result.update(issue_links(config, now=start, client=None, store=store))

    # The index assertion below covers every local file and remote grant.
    thread = threading.Thread(target=issue)
    thread.start()
    time.sleep(0.15)
    store.gate.set()
    thread.join(120)
    assert not thread.is_alive()
    assert result["count"] == 1000
    assert store.max_active > 1
    assert store.puts[-1] == "public/generator/links.json"
    assert store.index_at_completion == 1001
    assert len(store.puts) == 1001
    state = json.loads((config.work_dir / "links-state.json").read_text())
    assert len(state["issued"]) == 1000

    store.puts.clear()
    same = issue_links(config, now=start + 601, client=None, store=store)
    assert same["status"] == "unchanged"
    assert store.puts == ["public/generator/links.json"]

    current["keys"] = current["keys"] + hotkeys(1001)[-1:]
    store.puts.clear()
    added = issue_links(config, now=start + 1202, client=None, store=store)
    assert added["count"] == 1
    assert store.puts == [f"public/generator/links/{current['keys'][-1]}.json", "public/generator/links.json"]
    assert len(json.loads(store.get("public/generator/links.json"))["hotkeys"]) == 1001

    store.puts.clear()
    prefetch = issue_links(config, now=int(epoch_deadline(EPOCH).timestamp()) - 300, client=None, store=store)
    assert prefetch["count"] == 1001
    assert len(store.puts) == 1002 and store.puts[-1] == "public/generator/links.json"
    for hotkey in current["keys"]:
        assert "next" in json.loads((config.site_dir / "generator" / "links" / f"{hotkey}.json").read_text())

    # No deletion: a deregistered grant remains until its deadline and in the index.
    removed = current["keys"].pop()
    store.puts.clear()
    issue_links(config, now=int(epoch_deadline(EPOCH).timestamp()) - 240, client=None, store=store)
    assert removed in json.loads(store.get("public/generator/links.json"))["hotkeys"]
    assert (config.site_dir / "generator" / "links" / f"{removed}.json").is_file()
    assert store.get(f"public/generator/links/{removed}.json")
