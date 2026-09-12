"""Keep our containers out of the datagen supervisor's reaper.

`rollouts.runners.verifiers.reap_containers` runs `docker rm -f` on every
container whose image name starts with one of the task-image namespaces
(mswebench/, alexgshaw/, terminal-lego/, ...) at the start of each of its
docker batches — it assumes it is the only user of those images on the pod.
A continuation that shares the pod would be killed mid-rollout (SandboxError,
exit 137).

The fix is local to the eval process that imports this plugin: before
`DockerRuntime.start` runs `docker run`, the task image is tagged under the
`recoverable.local/` namespace and the container is started from that tag,
so `docker ps` shows a name the reaper does not match. The supervisor's
`prune_images` (`docker rmi -f <original tag>`) then only untags its own
name; the layers stay referenced by ours. Nothing outside this process is
changed.
"""

from __future__ import annotations

import asyncio
import logging
import re

from verifiers.v1.runtimes.docker import DockerRuntime

log = logging.getLogger(__name__)

SHADOW_NS = "recoverable.local"
_original_start = DockerRuntime.start
_tag_locks: dict[str, asyncio.Lock] = {}


def shadow_tag(image: str) -> str:
    if image.startswith(SHADOW_NS + "/"):
        return image
    name, _, tag = image.rpartition(":")
    if not name or "/" in tag:
        name, tag = image, "latest"
    safe = re.sub(r"[^a-z0-9_.-]+", "_", name.lower())
    return f"{SHADOW_NS}/{safe}:{tag}"


async def _docker(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace")


async def ensure_shadow(image: str) -> str:
    target = shadow_tag(image)
    if target == image:
        return image
    lock = _tag_locks.setdefault(image, asyncio.Lock())
    async with lock:
        code, _ = await _docker("image", "inspect", target)
        if code == 0:
            return target
        code, _ = await _docker("image", "inspect", image)
        if code != 0:
            code, out = await _docker("pull", image)
            if code != 0:
                log.warning("shield: pull %s failed (%s); running the original name",
                            image, out.strip()[-300:])
                return image
        code, out = await _docker("tag", image, target)
        if code != 0:
            log.warning("shield: tag %s failed (%s)", image, out.strip()[-300:])
            return image
    return target


async def _start(self: DockerRuntime) -> None:
    shadow = await ensure_shadow(self.config.image)
    if shadow != self.config.image:
        self.config = self.config.model_copy(update={"image": shadow})
    await _original_start(self)


def install() -> None:
    if getattr(DockerRuntime.start, "_recoverable_shield", False):
        return
    _start._recoverable_shield = True  # type: ignore[attr-defined]
    DockerRuntime.start = _start  # type: ignore[method-assign]
