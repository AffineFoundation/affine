"""llms.txt names the local train-and-score tool, and the publisher does not import it.

The tool (``sn120_v2.replicate``) needs the GPU install, and the publisher runs with the standard library alone.
So the text of section 10 types the names of the tool. These tests fail when a name of the text is no longer a
name of the code: a command, an argument, the backend function, the document, an install extra, a module that
reads an address with no client name.
"""

from __future__ import annotations

import ast
import importlib
import re
import shlex
import tomllib
from pathlib import Path

import sn120_v2
from sn120_v2 import public, publisher, replicate, sim

PACKAGE = Path(sn120_v2.__file__).parent
V2 = PACKAGE.parent
SITE, LINKS = "https://files.example", "https://files.example/links"
PLACEHOLDERS = {"<N>": "1"}  # a value of the text that the parser of the tool must read as a number
COMMAND = re.compile(r"`(python [^`]+)`")


def llms(tmp_path: Path) -> str:
    """llms.txt of a new world, on one line."""
    params = sim.sim_params()
    snap = public.snapshot(sim.SimWorld(tmp_path / "world", params).controller.state, params, 1_800_000_000)
    return " ".join(publisher.render(snap, SITE, LINKS)[1].decode().split())


def commands(text: str) -> list[list[str]]:
    """Each command line that llms.txt shows, as its arguments."""
    return [[PLACEHOLDERS.get(word, word) for word in shlex.split(found)] for found in COMMAND.findall(text)]


def test_each_command_of_the_tool_in_llms_txt_is_a_command_of_the_tool(tmp_path):
    lines = [argv[3:] for argv in commands(llms(tmp_path)) if argv[:3] == ["python", "-m", publisher.REPLICATE_MODULE]]
    known = ("compare", "fetch", "run")
    assert all(argv[0] in known for argv in lines if len(argv) == 1), "a short name of a command (with no argument) is a command of the tool"
    parsed = [replicate.parser().parse_args(argv) for argv in lines if len(argv) > 1]  # argparse stops the test at a name it does not know
    assert tuple(sorted(args.command for args in parsed)) == known
    assert all(args.mining_json == f"{SITE}/mining.json" for args in parsed), "the commands read the public mining.json"
    run = next(args for args in parsed if args.command == "run")
    assert (run.gpus, run.batch, run.round, run.round_file, run.round_seed) == (1, "batch.json", None, None, None)


def test_the_tool_that_llms_txt_names_is_in_the_code(tmp_path):
    assert publisher.REPLICATE_MODULE == replicate.__name__
    assert callable(getattr(importlib.import_module(publisher.REPLICATE_MODULE), publisher.REPLICATE_BACKEND))
    assert (V2.parent / publisher.REPLICATE_DOC).is_file(), "llms.txt links to a document that is not in the code"
    text = llms(tmp_path)
    assert f"{publisher.REPO}/blob/v2/{publisher.REPLICATE_DOC}" in text
    assert "not published yet" not in text, "the GPU code of the box is published"
    assert "has not run on a real GPU yet" in text

    extras = tomllib.loads((V2 / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    installs = re.findall(r'pip install -e "(?:v2|\.)\[([a-z,]+)\]"', text)
    assert installs and all(extra in extras for line in installs for extra in line.split(","))

    # The miner tool takes the backend of the text, with the arguments of the text.
    reproduce = next(argv for argv in commands(text) if argv[1:3] == ["v2/miner/affine_miner.py", "reproduce"])
    assert reproduce[reproduce.index("--backend") + 1] == f"{publisher.REPLICATE_MODULE}:{publisher.REPLICATE_BACKEND}"
    source = (V2 / "miner" / "affine_miner.py").read_text()
    assert all(f'"{word}"' in source for word in reproduce if word.startswith("--"))


def reads_with_no_headers(path: Path) -> list[int]:
    """The lines of a module that open an address and give no headers: such a read has the default client name of urllib."""
    tree = ast.parse(path.read_text())
    requests = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    bare = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "urlopen":
            first = node.args[0] if node.args else None
            request = requests.get(first.id) if isinstance(first, ast.Name) else first
            named = isinstance(request, ast.Call) and getattr(request.func, "attr", "") == "Request" and any(key.arg == "headers" for key in request.keywords)
            if not named:
                bare.append(node.lineno)
    return bare


def test_each_command_that_llms_txt_names_sends_a_client_name(tmp_path):
    """The public site answers 403 to the default client name of urllib. llms.txt says that its commands send their own."""
    named = set()
    for argv in commands(llms(tmp_path)):
        named.add(V2.parent / argv[1] if argv[1] != "-m" else V2 / (argv[2].replace(".", "/") + ".py"))
    assert {PACKAGE / "replicate.py", PACKAGE / "links.py", V2 / "miner" / "affine_miner.py"} <= named
    assert all(path.is_file() for path in named), "llms.txt names a command that is not in the code"
    checked = named | {PACKAGE / "publisher.py"}  # the check of a published pair, `python -m sn120_v2.publisher check`
    assert {path.name: reads_with_no_headers(path) for path in checked} == {path.name: [] for path in checked}
    probe = tmp_path / "probe.py"
    probe.write_text("import urllib.request\nurllib.request.urlopen(address, timeout=30)\n")
    assert reads_with_no_headers(probe) == [2], "the check finds a read with no headers"
