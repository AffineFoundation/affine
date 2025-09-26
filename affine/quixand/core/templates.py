from __future__ import annotations

import hashlib
import json
import os
import subprocess
import fnmatch
from pathlib import Path
from typing import Optional, Dict, Any, List, Set

from ..config import Config


INDEX = Config.templates_dir() / "index.json"


class Templates:
	@staticmethod
	def build(path: str, name: Optional[str] = None, build_args: Optional[Dict[str, Any]] = None, build_path: Optional[str] = None) -> str:
		p = Path(path)
		if not p.exists():
			raise FileNotFoundError(path)
		dockerfile = p / "e2b.Dockerfile"
		if not dockerfile.exists():
			dockerfile = p / "Dockerfile"
		if not dockerfile.exists():
			raise FileNotFoundError("No e2b.Dockerfile or Dockerfile found")

		if build_path is None or build_path == "":
			build_path = p

		digest_content = _hash_dir(build_path)
		if build_args:
			# Add build args to hash to ensure unique images for different build args
			args_str = json.dumps(build_args, sort_keys=True)
			digest_content = hashlib.sha256(
				(digest_content + args_str).encode()
			).hexdigest()

		img_name = f"qs/{name or p.name}:{digest_content[:12]}"
		
		# Use runtime from Config
		cfg = Config()
		runtime = cfg.runtime or "docker"
		
		# Check if image already exists
		if _image_exists(img_name, runtime):
			print(f"Using cached image: {img_name}")
			# Still update the index for consistency
			idx = _load_index()
			idx[name or p.name] = {"image": img_name, "digest": digest_content}
			INDEX.write_text(json.dumps(idx, indent=2))
			return img_name
		
		# Build the image if it doesn't exist
		print(f"Building new image: {img_name}")
		cmd = [runtime, "build", "-f", str(dockerfile), "-t", img_name]
		
		# Add build arguments if provided
		if build_args:
			for key, value in build_args.items():
				cmd.extend(["--build-arg", f"{key}={value}"])
		cmd.append(str(build_path))
		subprocess.check_call(cmd)
		idx = _load_index()
		idx[name or p.name] = {"image": img_name, "digest": digest_content}
		INDEX.write_text(json.dumps(idx, indent=2))
		return img_name

	@staticmethod
	def ls() -> dict:
		return _load_index()

	@staticmethod
	def rm(name: str) -> None:
		idx = _load_index()
		idx.pop(name, None)
		INDEX.write_text(json.dumps(idx, indent=2))
	
	@staticmethod
	def agentgym(env_name: str) -> str:
		template_path = get_env_templates_dir("agentgym")
		base_image = "python:3.11-slim"
		tool_name = ""
		if env_name in ["webshop", "sciworld"]:
			base_image = "python:3.8-slim"
		elif env_name == "webarena":
			base_image = "python:3.10.13-slim"
		elif env_name == "webarena":
			base_image = "python:3.10.13-slim"
		elif env_name in ["academia", "movie", "sheet", "todo", "weather"]:
			base_image = "python:3.8.13-slim"
			tool_name = env_name
			env_name = "tool"
		elif env_name in ["maze", "wordle"]:
			base_image = "python:3.9.12-slim"
			tool_name = env_name
			env_name = "lmrlgym"
		elif env_name == "searchqa":
			base_image = "python:3.10-slim"
		return Templates.build(
			template_path,
			name=f"agentgym-{env_name}",
			build_args={"PREINSTALL_ENV": env_name, "TOOL_NAME": tool_name, "BASE_IMAGE": base_image}
		)

	@staticmethod
	def affine(env_name: str) -> str:
		"""Build the 'affine' env template image for a specific env name (abd/ded/sat)."""
		template_path = get_env_templates_dir("affine")
		return Templates.build(
			template_path,
			name=f"affine",
	    	build_path="."
		)

	@staticmethod
	def ridges(name="ridges") -> str:
		return Templates.build(
			get_env_templates_dir("ridges"),
			name=name,
		)


def _hash_dir(path: Path) -> str:
	h = hashlib.sha256()
	if isinstance(path, str):
		path = Path(path)
	
	ignore_patterns = _load_dockerignore(path)
	
	for root, dirs, files in os.walk(path):
		rel_root = Path(root).relative_to(path) if root != str(path) else Path(".")
		dirs[:] = [d for d in dirs if not _should_ignore(rel_root / d, ignore_patterns)]
		
		for f in sorted(files):
			rel_path = rel_root / f if str(rel_root) != "." else Path(f)
			if _should_ignore(rel_path, ignore_patterns) or f.startswith(".git"):
				continue
			
			full_path = Path(root) / f
			try:
				h.update(full_path.read_bytes())
			except (OSError, IOError):
				continue
	return h.hexdigest()


def _load_dockerignore(base_path: Path) -> Set[str]:
	dockerignore_path = base_path / ".dockerignore"
	if not dockerignore_path.exists():
		return set()
	patterns = set()
	try:
		with open(dockerignore_path, 'r') as f:
			for line in f:
				line = line.strip()
				if line and not line.startswith('#'):
					patterns.add(line)
	except (OSError, IOError):
		return set()
	return patterns


def _should_ignore(path: Path, patterns: Set[str]) -> bool:
	if not patterns:
		return False
	path_str = str(path).replace(os.sep, '/')
	for pattern in patterns:
		if pattern.startswith('!'):
			continue
		pattern = pattern.rstrip('/')
		if fnmatch.fnmatch(path_str, pattern):
			return True
		if '/' not in pattern:
			parts = path_str.split('/')
			for part in parts:
				if fnmatch.fnmatch(part, pattern):
					return True
	return False


def _load_index() -> dict:
	if not INDEX.exists():
		return {}
	try:
		return json.loads(INDEX.read_text())
	except Exception:
		return {}


def _image_exists(image_name: str, runtime: str = "docker") -> bool:
	"""Check if a Docker/Podman image exists locally."""
	try:
		result = subprocess.run(
			[runtime, "images", "-q", image_name],
			capture_output=True,
			text=True,
			check=False
		)
		return bool(result.stdout.strip())
	except Exception:
		return False


def get_env_templates_dir(env_template) -> str:
		try:
				current_dir = Path(__file__).resolve().parent
		except NameError:
				current_dir = Path(os.getcwd()).resolve()
		
		target_dir = current_dir.parent / "env_templates" / env_template
		return str(target_dir)