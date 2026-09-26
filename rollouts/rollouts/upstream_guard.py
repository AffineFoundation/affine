"""Block GitHub during the agent phase, and drop local git metadata.

The teacher reads the task's own fix from the sandbox network: curl of a
commit patch, git clone of upstream, pip download of a newer release. The
same images also keep remotes, tags, stashes and other branches, so
`git log --all` can show a stash named like the fix.

This module runs inside the verifiers eval process (see
loopguard_site/sitecustomize.py). It does two things when
ROLLOUTS_BLOCK_UPSTREAM=1:

* Wrap `resolve_runtime_config` so the agent-phase policy blocks
  `*.github.com` and `*.githubusercontent.com`. Allow stays `*`, so PyPI
  and other hosts keep working. Setup still runs with an open proxy:
  verifiers calls `prepare_execution(None)` before the task's setup, and
  that call is unrestricted. nl2lib already blocks every host; merging
  these names onto that policy leaves it framework-only.

* Wrap `DockerRuntime.prepare_execution`. When the agent phase starts
  (routes is not None), after setup has checked out the base commit, strip
  remotes, tags, stashes, reflog and branches other than HEAD, then
  `git gc --prune=now`. HEAD and its ancestors stay, because the grader
  checks out `$base` and that commit is HEAD after setup. A failure here
  is logged and the rollout continues.

PyPI stays open. mini-swe installs its harness from PyPI during the agent
phase, and a blanket block would stop the batch. Downloading the target
package from PyPI is still possible. affine_nl2lib already denies every
host on the agent phase, so that source does not have this hole.
"""

from __future__ import annotations

import sys

ENV = "ROLLOUTS_BLOCK_UPSTREAM"
# Groups whose tasks are a repo plus a hidden fix. Chat and math sources
# are not in this set. The runner turns the env var on for these groups
# when the runtime is docker.
GROUPS = frozenset({"coding", "terminal", "nl2repo"})
BLOCK_HOSTS = ("*.github.com", "*.githubusercontent.com")

# Best-effort. Exit 0 always. No-op when the workdir is not a git repo.
_STRIP = r"""
strip_repo() {
  dir="$1"
  [ -d "$dir" ] || return 0
  git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0
  top=$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null) || return 0
  case " $SEEN " in
    *" $top "*) return 0 ;;
  esac
  SEEN="$SEEN $top"
  cur=$(git -C "$top" symbolic-ref --short HEAD 2>/dev/null || true)
  git -C "$top" remote 2>/dev/null | while read -r name; do
    [ -n "$name" ] && git -C "$top" remote remove "$name" || true
  done
  git -C "$top" tag -l 2>/dev/null | while read -r tag; do
    [ -n "$tag" ] && git -C "$top" tag -d "$tag" || true
  done
  git -C "$top" stash clear >/dev/null 2>&1 || true
  git -C "$top" reflog expire --expire=now --all >/dev/null 2>&1 || true
  git -C "$top" for-each-ref --format='%(refname:short)' refs/heads 2>/dev/null | while read -r branch; do
    if [ -n "$branch" ] && [ "$branch" != "$cur" ]; then
      git -C "$top" branch -D "$branch" >/dev/null 2>&1 || true
    fi
  done
  git -C "$top" for-each-ref --format='%(refname)' refs/replace refs/notes refs/remotes 2>/dev/null | while read -r ref; do
    [ -n "$ref" ] && git -C "$top" update-ref -d "$ref" >/dev/null 2>&1 || true
  done
  git -C "$top" gc --prune=now >/dev/null 2>&1 || true
}
SEEN=""
strip_repo "$PWD"
for extra in /testbed /app /workspace /repo; do
  strip_repo "$extra"
done
if command -v find >/dev/null 2>&1; then
  find "$PWD" -maxdepth 3 -name .git -type d 2>/dev/null | while read -r gitdir; do
    strip_repo "$(dirname "$gitdir")"
  done
fi
exit 0
"""


def applies(source) -> bool:
    """Docker verifiers sources whose tasks ship a repo and a hidden fix."""
    return getattr(source, "runner", "") == "verifiers" and getattr(source, "group", "") in GROUPS


def install() -> bool:
    """Install both hooks. False when verifiers cannot be imported.

    Idempotent. A taskset that already blocks every host stays blocked.
    A pin without `with_task_network_policy` is left unchanged rather than
    failing the batch.
    """
    ok = _install_network()
    _install_git_strip()
    return ok


def _install_network() -> bool:
    try:
        from verifiers.v1.utils import compile as compile_mod
    except Exception:
        return False
    current = compile_mod.resolve_runtime_config
    if getattr(current, "_affine_upstream", False):
        return True

    def resolve_runtime_config(base, task, warned=None):
        config = current(base, task, warned)
        merge = getattr(config, "with_task_network_policy", None)
        if merge is None:
            return config
        try:
            return merge(list(config.allow), list(BLOCK_HOSTS))
        except Exception as exc:
            print(f"[upstream_guard] network block not merged ({exc!r})",
                  file=sys.stderr, flush=True)
            return config

    resolve_runtime_config._affine_upstream = True  # type: ignore[attr-defined]
    compile_mod.resolve_runtime_config = resolve_runtime_config
    return True


def _install_git_strip() -> bool:
    try:
        from verifiers.v1.runtimes.docker import DockerRuntime
    except Exception:
        return False
    current = DockerRuntime.prepare_execution
    if getattr(current, "_affine_upstream", False):
        return True

    async def prepare_execution(self, routes):
        if routes is not None:
            try:
                await self.run(["bash", "-lc", _STRIP], {})
            except Exception as exc:
                print(f"[upstream_guard] git strip skipped ({exc!r})",
                      file=sys.stderr, flush=True)
        return await current(self, routes)

    prepare_execution._affine_upstream = True  # type: ignore[attr-defined]
    DockerRuntime.prepare_execution = prepare_execution
    return True
