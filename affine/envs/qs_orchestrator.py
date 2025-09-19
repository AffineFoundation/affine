from __future__ import annotations

import os
from typing import Dict, Tuple

import affine as af
from .remote_qs import RemoteEnvQS

import quixand as qs

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
		sbx = qs.Sandbox(
			template=self.image,
			resources=qs.Resources(network="none"),  # no port exposed; we use proxy
			command=[
				"uvicorn", module, "--host", "0.0.0.0", "--port", str(port)
			],
			timeout=self.timeout_seconds,
		)
		self.sandboxes[name] = sbx
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
