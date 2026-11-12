"""Shadow mode can not write live state or a live R2 prefix.

Terms:

* shadow state root: the one directory that a shadow process writes. Its path is
  in the environment variable ``PIT_SHADOW_STATE``. It is never a live directory.
* R2 prefix allowlist: the start of each object key that a shadow process can
  write. It is ``pit-shadow/``. Each live prefix is outside it.
* R2 output prefix: where one run writes, under the allowlist. The controller
  gives it as an argument or in ``PIT_R2_OUTPUT_PREFIX``, for example
  ``pit-shadow/batchctl/<run id>/``. A prefix outside the allowlist stops the start.

The guard is a hard deny in the process, not a convention:

* Files: an audit hook (PEP 578) sees each open-for-write, remove, rename,
  mkdir, rmdir, chmod, chown, link, symlink, truncate and utime of the Python
  process. A path outside the shadow state root and the scratch directory
  raises ``ShadowWriteError``. Symlinks are resolved first.
* R2: each S3 call and each presigned URL of botocore goes through one function.
  Read operations (Get, Head, List) are free. Each other operation needs every
  object key under the allowlist. A write with no object key (a bucket
  operation) is refused.

Limits: the hook does not see a child process (ssh, scp, lium) or a write that a
C extension does without Python. The rebound environment is the control there.
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import botocore.client

from ops.parallel_eval.live import pe_artifact_upload as AU

from .shadow import R2_PREFIX, SHADOW_R2_PREFIXES, STATE_ENV, ShadowWriteError, r2_key_problem
from .shadow import inside as _inside

# Host directories that the box pipeline writes (receipts, the pending index): the variable, and its place
# in the shadow state root. A value of the environment stays only if it is in the shadow state root.
HOST_OUTPUT_ENV = {'PIT_UPLOAD_ROOT': 'pit/uploads', 'PIT_FP32_PENDING_INDEX': 'pit/fp32-pending'}
# Environment variables that name live state. Shadow reads these places and never writes them.
LIVE_ROOT_ENV = (
    'PE_STATE', 'PE_ST', 'PE_HERE', 'PE_JOBS', 'PE_LIFE', 'LIFE_ROOT', 'PE_LIFE_STATE', 'HG_GATE_STATE',
    'PE_GATE_STATE', 'PE_GATE_EP', 'PE_HELDOUT',
)
READ_OPERATIONS = ('Get', 'Head', 'List')
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
# Audit event -> ((index of a path argument, index of its dir_fd argument or None), ...).
PATH_EVENTS: Dict[str, Tuple[Tuple[int, Optional[int]], ...]] = {
    'os.remove': ((0, 1),), 'os.rmdir': ((0, 1),), 'os.mkdir': ((0, 2),), 'os.chmod': ((0, 2),),
    'os.chown': ((0, 3),), 'os.truncate': ((0, None),), 'os.utime': ((0, 3),),
    'os.rename': ((0, 2), (1, 3)), 'os.link': ((1, 3),), 'os.symlink': ((1, 2),),
}
# These change the directory entry itself: a symlink in the last part of the path is not followed.
ENTRY_EVENTS = ('os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink')


class ShadowConfigError(RuntimeError):
    """Shadow mode has no safe state root."""


def _real(path: Any, dir_fd: Any = None, *, entry: bool = False) -> Optional[str]:
    """The real absolute path of a path argument. None: not a file path (a pipe, a socket)."""
    if isinstance(path, int):
        try:
            path = os.readlink('/proc/self/fd/%d' % path)
        except OSError:
            return None
        if not path.startswith('/'):
            return None
    text = os.fsdecode(path)
    if not os.path.isabs(text) and isinstance(dir_fd, int) and dir_fd >= 0:
        try:
            text = os.path.join(os.readlink('/proc/self/fd/%d' % dir_fd), text)
        except OSError:
            pass
    text = os.path.abspath(text)
    if entry:
        return os.path.join(os.path.realpath(os.path.dirname(text)), os.path.basename(text))
    return os.path.realpath(text)


@dataclass
class ShadowGuard:
    """The write policy of one shadow process."""

    state_root: Path
    r2_prefixes: Tuple[str, ...] = SHADOW_R2_PREFIXES
    scratch: Tuple[str, ...] = ()
    denied: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state_root = Path(os.path.realpath(str(self.state_root)))
        self.r2_prefixes = tuple(self.r2_prefixes)
        if not self.r2_prefixes or any(not p or not p.endswith('/') for p in self.r2_prefixes):
            raise ShadowConfigError('each R2 prefix of shadow must end with "/"')
        self._roots = (str(self.state_root),) + tuple(os.path.realpath(str(s)) for s in self.scratch) + ('/dev',)

    def _deny(self, text: str) -> None:
        self.denied.append(text)
        raise ShadowWriteError('shadow hard-deny: %s' % text)

    def check_path(self, path: Any, what: str, dir_fd: Any = None, *, entry: bool = False) -> None:
        real = _real(path, dir_fd, entry=entry)
        if real is None or any(_inside(real, root) for root in self._roots):
            return
        self._deny('%s of %s is outside the shadow state root %s' % (what, real, self.state_root))

    def check_s3(self, operation: str, params: Any) -> None:
        if operation.startswith(READ_OPERATIONS):
            return
        params = params if isinstance(params, dict) else {}
        keys = [params['Key']] if 'Key' in params else []
        delete = params.get('Delete')
        if isinstance(delete, dict):
            keys += [obj.get('Key') for obj in delete.get('Objects') or [] if isinstance(obj, dict)]
        if not keys:
            self._deny('S3 %s has no object key: shadow writes objects under %s only' % (operation, list(self.r2_prefixes)))
        for key in keys:
            problem = r2_key_problem(key, self.r2_prefixes)
            if problem:
                self._deny('S3 %s of key %r: %s' % (operation, key, problem))

    def activate(self) -> 'ShadowGuard':
        """Make this the policy of the process. The hooks go in one time and stay."""
        global _ACTIVE
        _install_hooks()
        _ACTIVE = self
        return self


_ACTIVE: Optional[ShadowGuard] = None
_HOOKED = False


def active() -> Optional[ShadowGuard]:
    return _ACTIVE


def leave() -> None:
    """End the policy (the end of the process, or the end of a test)."""
    global _ACTIVE
    _ACTIVE = None


def _audit(event: str, args: Tuple[Any, ...]) -> None:
    guard = _ACTIVE
    if guard is None:
        return
    if event == 'open':
        flags = args[2] if len(args) > 2 else None
        if isinstance(flags, int) and flags & WRITE_FLAGS:
            guard.check_path(args[0], 'a write')
        return
    spec = PATH_EVENTS.get(event)
    if spec is None:
        return
    for index, fd_index in spec:
        if index < len(args):
            dir_fd = args[fd_index] if fd_index is not None and fd_index < len(args) else None
            guard.check_path(args[index], event, dir_fd, entry=event in ENTRY_EVENTS)


def _install_hooks() -> None:
    global _HOOKED
    if _HOOKED:
        return
    emit = getattr(botocore.client.BaseClient, '_emit_api_params', None)
    if not callable(emit):
        # Fail closed: without this function the guard can not see the S3 calls of this botocore.
        raise ShadowConfigError('this botocore has no BaseClient._emit_api_params: the R2 guard can not go in')

    def guarded_emit(self: Any, api_params: Any, operation_model: Any, context: Any) -> Any:
        guard = _ACTIVE
        if guard is not None and self.meta.service_model.service_name == 's3':
            guard.check_s3(operation_model.name, api_params)
        return emit(self, api_params, operation_model, context)

    botocore.client.BaseClient._emit_api_params = guarded_emit
    sys.addaudithook(_audit)
    _HOOKED = True


def shadow_state_root(environ: Optional[MutableMapping[str, str]] = None) -> Optional[Path]:
    raw = (os.environ if environ is None else environ).get(STATE_ENV)
    return Path(raw) if raw else None


def shadow_r2_env(environ: Mapping[str, str], r2_prefix: Optional[str] = None) -> Dict[str, str]:
    """The R2 output prefix of a shadow process and its three write prefixes, as environment variables.

    The output prefix is ``r2_prefix``, or the variable ``PIT_R2_OUTPUT_PREFIX``, or ``pit-shadow/``. An
    output prefix that is not under the allowlist stops the start. The three write prefixes are always the
    directories under the output prefix, so a live prefix of the environment never stays.
    """
    base = r2_prefix if r2_prefix is not None else (environ.get(AU.OUTPUT_PREFIX_ENV) or R2_PREFIX)
    try:
        env = AU.output_prefix_env(base)
    except AU.ArtifactUploadError as exc:
        raise ShadowConfigError(str(exc)) from None
    problem = r2_key_problem(env[AU.OUTPUT_PREFIX_ENV] + 'x')
    if problem:
        raise ShadowConfigError('the R2 output prefix %s is not for shadow: %s' % (env[AU.OUTPUT_PREFIX_ENV], problem))
    return env


def enter_shadow(environ: Optional[MutableMapping[str, str]] = None, *, scratch: Optional[Sequence[str]] = None,
                 r2_prefix: Optional[str] = None) -> ShadowGuard:
    """Start the hard deny for this process. Shadow mode calls this before it does any work.

    ``PIT_SHADOW_STATE`` must name a directory that is not a live root, not inside one and not above one.
    The PIT state root, the host directories that the box pipeline writes and the R2 write prefixes of the
    process become shadow ones. ``r2_prefix`` is the output prefix of the run in R2 (see ``shadow_r2_env``).
    """
    environ = os.environ if environ is None else environ
    r2_env = shadow_r2_env(environ, r2_prefix)
    raw = environ.get(STATE_ENV)
    if not raw or not os.path.isabs(raw):
        raise ShadowConfigError('%s must be set to an absolute path: shadow mode needs its own state root' % STATE_ENV)
    root = os.path.realpath(raw)
    if root == '/':
        raise ShadowConfigError('%s must not be the root directory' % STATE_ENV)
    for name in LIVE_ROOT_ENV:
        if environ.get(name):
            live = os.path.realpath(environ[name])
            if _inside(root, live) or _inside(live, root):
                raise ShadowConfigError('%s=%s overlaps the live root %s=%s' % (STATE_ENV, root, name, live))
    environ['PIT_STATE'] = os.path.join(root, 'pit')
    for name, place in HOST_OUTPUT_ENV.items():
        if not environ.get(name) or not _inside(os.path.realpath(environ[name]), root):
            environ[name] = os.path.join(root, place)
    environ.update(r2_env)
    sys.dont_write_bytecode = True
    os.makedirs(root, exist_ok=True)
    guard = ShadowGuard(state_root=Path(root), scratch=tuple(scratch) if scratch is not None else (tempfile.gettempdir(),))
    return guard.activate()
