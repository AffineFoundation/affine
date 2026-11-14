"""llms.txt against the params: no value of the text may stay behind when a parameter changes.

Three checks on the rendered text, and one on a published pair:

* each parameter alone: a changed value changes the text;
* every parameter at once: a number of the text that did not move must stand in ``FIXED``, in a phrase that is
  built from the code constant that gives the number;
* the text is the same from a snapshot and from the mining.json that was read back;
* ``publisher.check_pair`` and ``python -m sn120_v2.publisher check`` on the two files as a reader gets them.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import json
import re
import shlex
from pathlib import Path

import pytest

import sn120_v2
from sn120_v2 import admission, irt, judging, layout, public, publisher, queue, sampling, sim, worker
from sn120_v2.errors import ConfigError
from sn120_v2.gpu.evaluator import ServeSettings, server_argv
from sn120_v2.journal import JOURNAL_FILE
from sn120_v2.params import Params, Recipe

PACKAGE = Path(sn120_v2.__file__).parent

SITE, LINKS = "https://example.org", "https://example.org/links"
NOW = 1_800_000_000
ID = re.compile(r"\b[0-9a-f]{64}\b")
TIME = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC")
NUMBER = re.compile(r"(?<![A-Za-z_\d])\d")  # a digit that starts a number: not the 2 of v2, not the 256 of sha256
SECTION = re.compile(r"\b[Ss]ections? (\d+)\b")
HEADING = re.compile(r"^## (\d+)\. ", re.MULTILINE)

# A parameter that the sentences of llms.txt do not state: only the table of section 11 has its value.
ONLY_IN_THE_TABLE = {
    "calibration_tol": "a rule for a round on more than one box; a round runs on one box",
    "apply_max_attempts": "how often the validator tries to publish a winner before the round is abandoned",
    "weights_rate_limit_blocks": "a limit of the weights writer",
    "budget_usd_per_hour": "the operator's budget",
}
# A recipe field that only the recipe table of section 2 states (with its note), and no sentence.
ONLY_IN_THE_RECIPE_TABLE = {"method", "dropout", "kl", "image_digests"}


def made_world(tmp_path: Path) -> sim.SimWorld:
    return sim.SimWorld(tmp_path / "world", sim.sim_params())


def text_of(world: sim.SimWorld, params: Params) -> str:
    """llms.txt of the world's state under these params: what the box publishes after a change of the params."""
    snap = public.snapshot(world.controller.state, params, NOW)
    return publisher.render(snap, SITE, LINKS)[1].decode()


def launch_params() -> Params:
    """The defaults of the code, with the one worker image that the sim knows."""
    return Params(recipe=Recipe(image_digests=(sim.SIM_IMAGE,)))


def moved(value):
    """Another value of the same kind. It rises with the value, so every order rule of ``Params`` still holds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if isinstance(value, int):
        return 2 * value + 3
    return value / 2 + 0.013 if 0 <= value <= 1 else value * 1.5 + 0.25


def all_moved(params: Params) -> Params:
    """Every number of the params and of the recipe at another value, the bounds of a miner too."""
    def shift(record, **special):
        names = [f.name for f in dataclasses.fields(record) if f.name not in ("recipe", "miner_bounds", *special)]
        return dataclasses.replace(record, **{name: moved(getattr(record, name)) for name in names}, **special)

    bounds = tuple(shift(bound) for bound in params.recipe.miner_bounds)
    recipe = shift(params.recipe, miner_bounds=bounds, global_norm_cap=1.75)
    return shift(params, recipe=recipe, looks=(0.2, 0.6, 1.0), irt_middle_band=(0.3, 0.7))


def sentences(text: str) -> str:
    """The text without its tables and without ids: what a param must move to count as stated in a sentence."""
    lines = [line for line in text.split(f"{publisher.PARAMS_HEADING}\n")[0].splitlines() if not line.startswith("|")]
    return ID.sub("<id>", " ".join(" ".join(lines).split()))


OTHER = {
    "looks": (0.2, 0.6, 1.0), "irt_middle_band": (0.3, 0.7), "global_norm_cap": 1.75, "target_modules": ("q_proj",),
    "image_digests": (sim.SIM_IMAGE, "other-image"), "dtype": "float32", "env_pin": "pin-1", "method": "other-method", "code_digest": "c" * 64,
}  # fmt: skip


def one_changed(record, name: str):
    """The record with one field at another value that the rules of the record allow."""
    value = getattr(record, name)
    if name in OTHER:
        tries = [OTHER[name]]
    elif isinstance(value, int):
        tries = [value + 1, value - 1]
    else:
        tries = [moved(value), value * 0.875]
    for other in tries:
        try:
            return other, dataclasses.replace(record, **{name: other})
        except ConfigError:
            continue
    raise AssertionError(f"{name}: no other value is valid")


def test_each_param_alone_moves_the_text(tmp_path):
    """A changed parameter with an unchanged text fails here. So does a new parameter that the text does not state."""
    world, base = made_world(tmp_path), launch_params()
    before = text_of(world, base)
    for record, only_table in ((base, ONLY_IN_THE_TABLE), (base.recipe, ONLY_IN_THE_RECIPE_TABLE)):
        for field in dataclasses.fields(record):
            name = field.name
            if name in ("recipe", "miner_bounds", "env"):  # the recipe and its bounds: field by field below; env: one value exists
                continue
            value, changed = one_changed(record, name)
            params = changed if record is base else dataclasses.replace(base, recipe=changed)
            after = text_of(world, params)
            assert f"| `{name}` | {publisher.show(value)} |" in after, f"{name}: the table does not state the new value"
            assert f"| `{name}` | {publisher.show(getattr(record, name))} |" not in after, f"{name}: the table still states the old value"
            in_sentences = sentences(after) != sentences(before)
            assert in_sentences != (name in only_table), f"{name}: stated in a sentence = {in_sentences}; correct the list of this test"
    for position, bound in enumerate(base.recipe.miner_bounds):  # each end of each bound of a miner
        for end, factor in (("low", 0.5), ("high", 2)) if not bound.choices else ():
            bounds = list(base.recipe.miner_bounds)
            bounds[position] = dataclasses.replace(bound, **{end: type(getattr(bound, end))(getattr(bound, end) * factor)})
            after = text_of(world, dataclasses.replace(base, recipe=dataclasses.replace(base.recipe, miner_bounds=tuple(bounds))))
            row = next(line for line in after.splitlines() if line.startswith(f"| `train.{bound.key}` |"))
            assert publisher.bound_text(bounds[position]) in row and publisher.bound_text(bound) not in row
    assert set(ONLY_IN_THE_TABLE) < {f.name for f in dataclasses.fields(Params)} and ONLY_IN_THE_RECIPE_TABLE < {f.name for f in dataclasses.fields(Recipe)}


def fixed_phrases(params: Params, text: str) -> list[str]:
    """Each phrase of the text with a number that no parameter gives. The digits of these phrases may stay."""
    upload = publisher.contract(params, SITE, LINKS)["upload"]
    serve = ServeSettings()
    return [
        # Built from the constant of the code that gives the number.
        irt.MODEL,
        irt.DRAW_RULE,
        f"at most once each {admission.FAULT_HOLD_S // 60} minutes",
        f"after {admission.FAULT_TRIES} failed checks in a row",
        f"After {admission.FAULT_TRIES} counted checks in a row",
        f"Seconds per turn = {queue.SECONDS_PER_DAY} /",
        f"charged in the last {queue.SECONDS_PER_DAY // 3600} hours",
        f"A new file is written every {publisher.show(upload['reissue_every_seconds'] / 60)} minutes",
        f"Each link works for {publisher.show(upload['link_ttl_seconds'] / 3600)} hours",
        f"a text of more than {public.TEXT_LIMIT} characters",
        f"the first {sampling.SEED_BITS} bits of sha256",
        publisher.OPTIMIZER,  # typed in the publisher; a test below compares it with the training job
        publisher.GRADER_VERSION,  # the same, with the grader
        f"at most {publisher.BOX_TRIES} tries in all, inside the training time",
        f"later than {worker.AT_THE_LEASE_S} s after the start of the lease",
        f"The chunk runs again: at most {publisher.BOX_TRIES} runs in all",
        f"not usable: at most {serve.retries + 1} tries in all",
        f"graded again: at most {serve.grade_tries} grades in all",
        f"--max-num-seqs {serve.clients_per_server}",
        f"--limit-mm-per-prompt {shlex.quote(server_argv('', 0, params)[-1])}",
        f"higher by more than {publisher.show(judging.TIE_TOLERANCE)}",
        f"| `looks` | {publisher.show(params.looks)} |",  # the last look is always 1: a rule of Params
        # The state of this world, not a parameter.
        "Seats that wait: 0.",
        # A fact of the chain, of a key format or of HTTP. No constant of this code gives it.
        "The chain shows it after about 3 minutes",
        "the 32-byte Ed25519 seed of your hotkey as 64 hex characters",
        "HTTP 404 on your link file",
        "The site can answer HTTP 403 to a read", "HTTP 403 does not say that a file is missing", "a missing file gives HTTP 404",
        "UTF-8 JSON",
        # A formula, a count that is part of a rule, or a step of a list: the digit is a word of the rule.
        "(1) pin head M", "(2) copy the next valid files", "(3) train one LoRA", "(4) score every LoRA", "(5) apply at most one winner",
        "(6) publish", "(7) start again",
        "0 = the groups of your file",
        "is clipped to 1 +- this value",
        "KL term; 0 = off",
        "`group_size`: 0 means one group per prompt",
        "A value G above 0 cuts",
        "||(alpha / rank) * B @ A||_F^2)",
        '"reward": 1.0',
        '"reward": 0.0',
        "Each prompt has at least 1 rollout",
        "counts 1 token more there",
        "its official grade is 0.",
        "(0 = no such rule)",
        "its advantage is not 0 and it has at least 1 output token",
        "counts its tokens + 1 (the end-of-turn token)",
        "A rollout with advantage 0 counts 0.",
        "sqrt(n / (n - 1) * sum((w_t * (d_t - delta))^2))",
        "on that UID: 1 / (the number of such kings)",
        "from -1/sqrt(inputs) to +1/sqrt(inputs)",
        "clip(ratio, 1 - `ppo_clip`, 1 + `ppo_clip`)",
        "With A below 0 the token loss is at most",
        "also a row with advantage 0.",
        "1.0 for a pass and 0.0 for every other result",
        "j counts from 0.",
        "(i counts from 0)",
        "an HTTP status of 500 or above",
        "an HTTP status below 500",
        "When se is 0, z is 0.",
        "spent(t_(k-1))",
        "ln(1 + (e - 1) * t), and spent = 0 before the first look",
        "the normal quantile of 1 - a_k",
        "or when its se is 0.",
    ]  # fmt: skip


def test_a_number_that_no_param_moves_is_named_with_its_source(tmp_path):
    """Render with every parameter at another value. A number that stays must be in a phrase of ``fixed_phrases``.

    So a number that was typed into a sentence fails: either a parameter should give it (then the text diverges
    when the parameter changes), or a constant of the code gives it and the phrase is built from that constant.
    """
    world, base = made_world(tmp_path), launch_params()
    texts = [text_of(world, params) for params in (base, all_moved(base))]
    flat = [TIME.sub("<time>", ID.sub("<id>", " ".join(one.split()))) for one in texts]
    text = flat[0]
    words = [[(found.group(), found.start()) for found in re.finditer(r"\S+", one)] for one in flat]
    same = difflib.SequenceMatcher(None, [word for word, _ in words[0]], [word for word, _ in words[1]], autojunk=False)
    stayed = [words[0][position] for tag, i1, i2, _, _ in same.get_opcodes() if tag == "equal" for position in range(i1, i2)]
    digits = [start + found.start() for word, start in stayed for found in NUMBER.finditer(word)]
    assert len(digits) > 20, "the two texts must share their fixed numbers"

    phrases = fixed_phrases(base, text)
    missing = [phrase for phrase in phrases if phrase not in text]
    assert not missing, f"fixed_phrases names a phrase that the text does not hold: {missing}"
    spans = [found.span() for phrase in phrases for found in re.finditer(re.escape(phrase), text)]
    spans += [found.span() for pattern in (SECTION, re.compile(r"## \d+\. ")) for found in pattern.finditer(text)]
    typed = [f"...{text[max(0, at - 70) : at]}>>{text[at : at + 30]}..." for at in digits if not any(low <= at < high for low, high in spans)]
    assert not typed, "a number in llms.txt that no parameter and no named constant gives:\n" + "\n".join(typed)
    unused = [phrase for phrase in phrases if not any(low <= at < high for found in re.finditer(re.escape(phrase), text) for low, high in [found.span()] for at in digits)]
    assert not unused, f"fixed_phrases names a phrase with no fixed number: {unused}"

    first, second = (publisher.stated_params(one) for one in texts)
    assert first.keys() == second.keys() == {f.name for f in dataclasses.fields(Params)} - {"recipe"}
    assert {name for name in first if first[name] == second[name]} == {"env", "env_pin", "dtype"}, "all_moved must move every number of the params"
    headings = HEADING.findall(texts[0])
    assert headings == [str(number) for number in range(1, len(headings) + 1)] and set(SECTION.findall(text)) <= set(headings)


def assigned(path: Path, name: str) -> object:
    """The value that a module assigns to ``name`` at its top level, read from the source: the module is not imported."""
    (value,) = [
        ast.literal_eval(node.value)
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    return value


def test_the_typed_constants_of_the_text_are_those_of_the_code():
    """The publisher cannot import the training job (torch) or the grader (sympy). It types two of their texts."""
    assert publisher.OPTIMIZER == assigned(PACKAGE / "gpu" / "train_job.py", "OPTIMIZER")
    assert publisher.GRADER_VERSION == assigned(PACKAGE / "envs" / "math_grader.py", "GRADER_VERSION")
    # The text states the tries of the round runner and the settings of the servers as the code has them by default.
    # The stack of the box must not set others.
    stack = (PACKAGE / "assembly.py").read_text()
    assert "LocalRunner(trainer, evaluator, lambda: int(time.time()), box.image_digest)" in stack
    assert "VllmEvaluator(store, box, pool, GradePool())" in stack


def played(tmp_path: Path) -> sim.SimWorld:
    world = made_world(tmp_path)
    world.register(11, 0.4)  # wins
    world.register(12, None)  # no file: the seat waits
    world.register(13, 0.0)
    world.round()
    world.register(14, 0.3)
    return world


def test_llms_txt_is_the_same_text_from_the_snapshot_and_from_the_mining_json_that_was_read_back(tmp_path):
    """mining.json has sorted keys. The check of a published pair renders the text again from that file."""
    world = played(tmp_path)
    phases = set()
    for snap in world.publisher.snapshots:
        mining, llms = publisher.render(snap, SITE, LINKS)
        assert publisher.llms_text(json.loads(mining)) == llms.decode()
        phases.add(snap["phase"])
    assert phases >= {"IDLE", "RUNNING"} and snap["kings"] and snap["queue"]["next"]
    text = llms.decode()
    assert "| position | uid | hotkey | registration_block | file_valid | front | demoted | held |" in text  # the order of public.snapshot
    assert "| uid | hotkey | round_id | head_version_id | applied_at | z |" in text
    assert tuple(snap["queue"]["next"][0]) == publisher.QUEUE_COLUMNS and tuple(snap["kings"][0]) == publisher.KING_COLUMNS
    running = next(snap for snap in world.publisher.snapshots if snap["phase"] == "RUNNING")
    assert [key for key, value in running["round"].items() if not isinstance(value, (dict, list))] == list(publisher.ROUND_FIELDS[:12])
    assert publisher.in_order({"z": 1, "uid": 2, "a": 3}, publisher.KING_COLUMNS) == ["uid", "z", "a"]


def test_the_check_of_a_published_pair_fails_when_llms_txt_and_the_params_diverge(tmp_path, capsys):
    world = played(tmp_path)
    snap = world.publisher.snapshots[-1]
    mining, llms = publisher.render(snap, SITE, LINKS)
    assert publisher.check_pair(mining, llms) == [] and publisher.check_pair(mining, llms, world.params) == []

    # The params changed and llms.txt is the text of the old ones: each stated value that differs is named.
    newer = dataclasses.replace(world.params, n_tasks=world.params.n_tasks + 8, z_bar=1.5)
    new_mining, new_llms = publisher.render(public.snapshot(world.controller.state, newer, snap["generated_at"]), SITE, LINKS)
    problems = publisher.check_pair(new_mining, llms)
    assert f"`n_tasks`: llms.txt states {world.params.n_tasks}, mining.json has {newer.n_tasks}" in problems
    assert "`z_bar`: llms.txt states 1, mining.json has 1.5" in problems
    assert any("are not a pair" in problem for problem in problems) and any("does not name the `params_hash`" in problem for problem in problems)
    assert publisher.check_pair(new_mining, new_llms) == []

    # The state has other params than the published files: a set-params that was not published.
    assert publisher.check_pair(mining, llms, newer) == ["the published params are not the params of the state: n_tasks, z_bar differ"]

    # One number of the text was changed by hand, in the table and in a sentence.
    edited = llms.decode().replace(f"| `kings_window` | {world.params.kings_window} |", "| `kings_window` | 99 |")
    assert publisher.check_pair(mining, edited.encode(), whole_text=False) == [f"`kings_window`: llms.txt states 99, mining.json has {world.params.kings_window}"]
    edited = llms.decode().replace("answer may be in at most", "answer may be in at most 5 rollouts or")
    assert publisher.check_pair(mining, edited.encode(), whole_text=False) == [], "a sentence is covered only by the whole text"
    (problem,) = publisher.check_pair(mining, edited.encode())
    assert problem.startswith("llms.txt is not the text that this code renders from mining.json: 1 lines differ. The first, at line ") and "in at most 5 rollouts or" in problem

    # mining.json was changed after it was written, or it is not a mining.json at all.
    doc = json.loads(mining)
    doc["params"]["kings_window"] += 1
    problems = publisher.check_pair(json.dumps(doc).encode(), llms)
    assert "mining.json: the content does not have the `snapshot_digest` that the file names" in problems
    assert any(problem.startswith("mining.json: `params_hash` is not the id of its own params") for problem in problems)
    assert publisher.check_pair(b"<html>not found</html>", llms)[0].startswith("mining.json or llms.txt cannot be read: ")
    doc = json.loads(mining)
    del doc["head"]
    assert publisher.check_pair(json.dumps(doc).encode(), llms)[-1] == "this code cannot render the published mining.json: KeyError: 'head'"

    # The command, on files and on a state directory.
    files = [tmp_path / "mining.json", tmp_path / "llms.txt", tmp_path / "old-llms.txt"]
    for path, data in zip(files, (new_mining, new_llms, llms)):
        path.write_bytes(data)
    command = ["check", "--mining-json", str(files[0]), "--llms-txt"]
    assert publisher.main([*command, str(files[1])]) == 0
    assert capsys.readouterr().out.startswith(f"ok: {files[1]} is true to the params of {files[0]} (every line; params_hash {newer.params_hash}; ")
    assert publisher.main([*command, str(files[2]), "--tries", "2", "--wait-s", "0"]) == 1
    assert f"- `n_tasks`: llms.txt states {world.params.n_tasks}, mining.json has {newer.n_tasks}" in capsys.readouterr().err
    assert (world.root / JOURNAL_FILE).exists()
    assert publisher.main([*command, str(files[1]), "--state-dir", str(world.root)]) == 1, "the state has the params before the change"
    assert "the published params are not the params of the state: n_tasks, z_bar differ" in capsys.readouterr().err
    assert publisher.main([*command, str(files[1]), "--state-dir", str(tmp_path / "nothing")]) == 2 and "the check did not run" in capsys.readouterr().err
    assert publisher.main(["check", "--mining-json", str(tmp_path / "missing.json"), "--llms-txt", str(files[1])]) == 2
    with pytest.raises(SystemExit):
        publisher.main(["check"])
    assert layout.LLMS_TXT.endswith("/llms.txt") and layout.MINING_JSON.endswith("/mining.json")  # the two names under --url
