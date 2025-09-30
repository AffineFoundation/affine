from __future__ import annotations

import json
from typing import Optional

import typer
from rich import print

from ..core.sandbox import Sandbox
from ..core.lifecycle import connect as connect_fn
from ..core.templates import Templates


app = typer.Typer(name="qs", help="Quixand CLI")
sandbox_app = typer.Typer(name="sandbox", help="Sandbox lifecycle and exec")
files_app = typer.Typer(name="files", help="Filesystem operations")
templates_app = typer.Typer(name="templates", help="Template management")

app.add_typer(sandbox_app, name="sandbox")
app.add_typer(files_app, name="files")
app.add_typer(templates_app, name="templates")


@sandbox_app.command("create")
def sandbox_create(template: Optional[str] = typer.Option(None), timeout: int = typer.Option(300), env: list[str] = typer.Option(None), metadata: Optional[str] = typer.Option(None)):
	md = json.loads(metadata) if metadata else None
	env_map = {}
	if env:
		for kv in env:
			if "=" in kv:
				k, v = kv.split("=", 1)
				env_map[k] = v
	sbx = Sandbox(template=template, timeout=timeout, metadata=md, env=env_map)
	print({"id": sbx.id})


@sandbox_app.command("connect")
def sandbox_connect(id: str):
	sbx = connect_fn(id)
	print({"id": sbx.id})


@sandbox_app.command("exec")
def sandbox_exec(id: str, cmd: list[str]):
	sbx = connect_fn(id)
	res = sbx.run(cmd)
	print({"exit_code": res.exit_code, "text": res.text})


@sandbox_app.command("ls")
def sandbox_ls():
	from ..config import Config
	import json
	state = {}
	try:
		state = json.loads(Config.state_file().read_text())
	except Exception:
		state = {}
	print(state)


@sandbox_app.command("refresh-timeout")
def sandbox_refresh_timeout(id: str, seconds: int):
	sbx = connect_fn(id)
	sbx.refresh_timeout(seconds)
	print({"ok": True, "timeout": seconds})


@sandbox_app.command("run-code")
def sandbox_run_code(id: str, code: str):
	sbx = connect_fn(id)
	ex = sbx.run_code(code)
	print({"ok": ex.ok, "text": ex.text})


@sandbox_app.command("kill")
def sandbox_kill(id: str):
	sbx = connect_fn(id)
	sbx.shutdown()
	print({"ok": True})


@files_app.command("put")
def files_put(id: str, local: str, remote: str):
	sbx = connect_fn(id)
	sbx.files.put(local, remote)
	print({"ok": True})


@files_app.command("get")
def files_get(id: str, remote: str, local: str):
	sbx = connect_fn(id)
	sbx.files.get(remote, local)
	print({"ok": True})


@files_app.command("ls")
def files_ls(id: str, path: str = "."):
	sbx = connect_fn(id)
	print([f.__dict__ for f in sbx.files.ls(path)])


@files_app.command("mkdir")
def files_mkdir(id: str, path: str, parents: bool = typer.Option(False)):
	sbx = connect_fn(id)
	sbx.files.mkdir(path, parents=parents)
	print({"ok": True})


@files_app.command("rm")
def files_rm(id: str, path: str, recursive: bool = typer.Option(False)):
	sbx = connect_fn(id)
	sbx.files.rm(path, recursive=recursive)
	print({"ok": True})


@templates_app.command("build")
def templates_build(path: str, name: Optional[str] = typer.Option(None)):
	img = Templates.build(path, name)
	print({"image": img})


@templates_app.command("ls")
def templates_ls():
	print(Templates.ls())


@templates_app.command("rm")
def templates_rm(name: str):
	Templates.rm(name)
	print({"ok": True})


if __name__ == "__main__":
	app()


