"""The wording that the site gives to the served llms.txt (``v2/site/terms.py``): the limit of a round by hotkey."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

from sn120_v2 import sim
from sn120_v2.publisher import render

TERMS = Path(__file__).parents[1] / "site" / "terms.py"
terms = SimpleNamespace(**runpy.run_path(str(TERMS), run_name="terms"))  # a file of v2/site, not a module of the package

OLD = "- One entry for each miner in a round. A second seat of the same hotkey or coldkey waits for a later round."
NEW = (
    "- One seat for a hotkey in a round. A second seat of the same hotkey waits for a later round. The coldkey is not\n"
    "  used: one coldkey can have several seats, each with its own hotkey, in the same round."
)


def served(text: str) -> str:
    return terms.rewrite_served(text).decode()


def test_the_served_llms_txt_states_the_limit_of_a_round_by_hotkey_and_no_limit_by_coldkey(tmp_path):
    world = sim.SimWorld(tmp_path / "world", sim.sim_params())
    world.register(11, 0.4)
    world.round()
    llms = render(json.loads(json.dumps(world.publisher.snapshots[-1])), "https://site.example")[1].decode()
    assert (OLD in llms) != (NEW in llms), "the publisher states the rule one time, in the old words or in the new ones"
    text = served(llms)
    assert [line for line in text.splitlines() if "coldkey" in line.lower()] == NEW.splitlines()
    assert text == served(llms.replace(OLD, NEW)), "the same file whether the publisher or the site states the rule"


def test_the_rule_changes_its_one_line_only_and_matches_nothing_in_the_new_lines():
    before = f"- Order: registration block, then UID.\n{OLD}\n- `no_decision`: our infrastructure failed.\n"
    after = served(before)
    assert after == before.replace(OLD, NEW) and after.count("\n") == before.count("\n") + 1
    assert served(after) == after and not any(pattern.search(after) for pattern, _ in terms._SEATS)
    for other in (OLD + " More.", "  " + OLD, OLD.replace("or coldkey", "or of the same owner coldkey"), OLD[2:]):
        assert served(other + "\n") == other + "\n", "only the whole line of the publisher is replaced"
