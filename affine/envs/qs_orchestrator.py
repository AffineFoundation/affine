from __future__ import annotations

import os
from typing import Dict, Tuple

import affine as af
from .remote_qs import RemoteEnvQS

import quixand as qs
import logging

# Specifies the FastAPI modules and internal ports for each environment
ENV_SPECS: Dict[str, Tuple[str, int]] = {
	"DED": ("agentenv_affine.ded_server:app", 8000),
	"HVM": ("agentenv_affine.hvm_server:app", 8000),
	"ABD": ("agentenv_affine.abd_server:app", 8000),
	"SAT": ("agentenv_affine.sat_server:app", 8000),
}

class QSandboxes:
	"""Manages a pool of Quixand sandboxes for the environments, and provides RemoteEnvQS clients.
	- image: name/tag of the image containing agentenv_affine and its dependencies
	- command per environment: uvicorn <module>:app --host 0.0.0.0 --port 8000
	"""
	def __init__(self, image: str, timeout_seconds: int = 24*3600):
		self.image = image
		self.timeout_seconds = timeout_seconds
		self.sandboxes: Dict[str, qs.Sandbox] = {}

	def start(self, name: str) -> qs.Sandbox:
		module, port = ENV_SPECS[name]
		if name in self.sandboxes:
			return self.sandboxes[name]
		# Override image ENTRYPOINT ["af"] with a shell to run uvicorn
		# Pass the uvicorn command as a single string argument to /bin/sh -lc
		cmd = f"/opt/venv/bin/python3 -m uvicorn {module} --host 0.0.0.0 --port {port} || python3 -m uvicorn {module} --host 0.0.0.0 --port {port}"
		sbx = qs.Sandbox(
			template=self.image,
			resources=qs.Resources(network="bridge"),
			env={"PATH": "/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
			entrypoint=["/bin/sh", "-lc"],
			command=[cmd],
			timeout=self.timeout_seconds,
		)
		self.sandboxes[name] = sbx
		try:
			st = sbx.status()
			logging.info(f"[QSandboxes] started {name}: sbx_id={sbx.id} container_id={sbx.container_id} state={st.state}")
			# Basic environment diagnostics inside sandbox
			res_py = sbx.run(["sh","-lc","/opt/venv/bin/python -V || python3 -V; which python || true; which python3 || true"]) ; logging.info(f"[QSandboxes] {name} python: {res_py.text.strip()}")
			res_uv = sbx.run(["sh","-lc","which uvicorn || true; /opt/venv/bin/python -m pip show uvicorn || python3 -m pip show uvicorn || pip3 show uvicorn || true"]) ; logging.info(f"[QSandboxes] {name} uvicorn: {res_uv.text.strip()}")
			res_pk = sbx.run(["sh","-lc","/opt/venv/bin/python3 -c 'import importlib.util as u;print(\"agentenv_affine:\", bool(u.find_spec(\"agentenv_affine\")));print(\"agentenv_alfworld:\", bool(u.find_spec(\"agentenv_alfworld\")))' || true"]) ; logging.info(f"[QSandboxes] {name} pkgs: {res_pk.text.strip()}")
			res_ps = sbx.run(["sh","-lc","ps -ef | sed -n '1,120p'"]) ; logging.info(f"[QSandboxes] {name} ps -ef:\n{res_ps.text}")
		except Exception as e:
			logging.warning(f"[QSandboxes] diagnostics failed for {name}: {e}")
			try:
				import docker  # type: ignore
				cli = docker.from_env()
				c = cli.containers.get(sbx.container_id)
				logs = c.logs(tail=200).decode(errors="ignore") if hasattr(c, "logs") else ""
				if logs:
					logging.error(f"[QSandboxes] {name} container logs (tail):\n{logs}")
			except Exception as e2:
				logging.warning(f"[QSandboxes] failed to fetch docker logs for {name}: {e2}")
		return sbx

	def client(self, name: str) -> RemoteEnvQS:
		return RemoteEnvQS(self.sandboxes[name])

	def stop(self, name: str) -> None:
		sbx = self.sandboxes.pop(name, None)
		if sbx:
			try:
				sbx.shutdown()
			except Exception:
				pass

	def stop_all(self) -> None:
		for k in list(self.sandboxes.keys()):
			self.stop(k)
