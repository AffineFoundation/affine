from __future__ import annotations

import atexit
import weakref
from threading import Lock
from typing import Optional, Dict, Any
import os

from ..config import Config
from .sandbox import Sandbox
from .playground import Playground
from .templates import Templates


# Global singleton instance
_global_manager: Optional["SandboxManager"] = None
_global_lock = Lock()


def _get_global_manager() -> "SandboxManager":
    """Get or create the global SandboxManager instance."""
    global _global_manager
    with _global_lock:
        if _global_manager is None:
            _global_manager = SandboxManager()
        return _global_manager


class SandboxWrapper:
    def __init__(self, sandbox: Sandbox, manager: "SandboxManager"):
        self._sandbox = sandbox
        self._manager = manager
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._sandbox, name)

    def shutdown(self):
        if not self._closed:
            try:
                self._sandbox.shutdown()
            except Exception:
                pass
            finally:
                self._manager._active_sandboxes.discard(self._sandbox)
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def __del__(self):
        self.shutdown()


class SandboxManager:
    """Manages sandbox instances with shared and pooled modes.

    Provides two modes:
    1. Shared mode: Global singleton sandbox instances by template
    2. Non-shared mode: Uses Playground for pooled sandbox optimization
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config()
        self._shared_sandboxes: Dict[str, Sandbox] = {}
        self._playgrounds: Dict[str, Playground] = {}
        self._sandbox_lock = Lock()
        self._playground_lock = Lock()
        self._active_sandboxes: weakref.WeakSet[Sandbox] = weakref.WeakSet()

        # Register cleanup on exit
        atexit.register(self.cleanup)

    def get_sandbox(
        self,
        template: str,
        shared: bool = False,
        pool_size: int = 1,
        timeout: int = 600,
        env: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Sandbox:
        """Get a sandbox instance.

        Args:
            template: Template name or Docker image
            shared: If True, returns a shared global instance
            pool_size: Size of playground pool (only for non-shared mode)
            timeout: Sandbox timeout in seconds
            env: Environment variables
            metadata: Sandbox metadata
            **kwargs: Additional sandbox configuration

        Returns:
            Sandbox instance
        """
        # Build template if it's a known template
        image, env = self._resolve_template(template, env)

        if shared:
            return self._get_shared_sandbox(image, timeout, env, metadata, **kwargs)
        else:
            return self._get_pooled_sandbox(
                image, pool_size, timeout, env, metadata, **kwargs
            )

    def _resolve_template(
        self, template: str, env: Optional[Dict[str, str]] = None
    ) -> tuple[str, Optional[Dict[str, str]]]:
        """Resolve template name to Docker image and update env with CHUTES_API_KEY if needed."""

        def ensure_chutes_api_key(
            env_dict: Optional[Dict[str, str]],
        ) -> Optional[Dict[str, str]]:
            api_key = os.getenv("CHUTES_API_KEY")
            assert api_key, "please set env CHUTES_API_KEY"

            if env_dict is None:
                return {"CHUTES_API_KEY": api_key}
            elif "CHUTES_API_KEY" not in env_dict:
                env_dict["CHUTES_API_KEY"] = api_key
            return env_dict

        if template.startswith("agentgym:"):
            env_name = template.split(":", 1)[1]
            updated_env = ensure_chutes_api_key(env)
            updated_env["TODO_KEY"] = os.environ.get("AGENTGYM_TOOL_TODO_KEY", "")
            updated_env["MOVIE_KEY"] = os.environ.get("AGENTGYM_TOOL_MOVIE_KEY", "")
            updated_env["SHEET_EMAIL"] = os.environ.get("AGENTGYM_TOOL_SHEET_EMAIL", "")
            return Templates.agentgym(env_name), updated_env

        # Affine-prefixed logical templates (abd/ded/sat...)
        if template.startswith("affine:"):
            env_name = template.split(":", 1)[1]
            updated_env = ensure_chutes_api_key(env)
            updated_env["ENV_NAME"] = env_name
            return Templates.affine(env_name), updated_env

        if template == "ridges" or template.startswith("ridges:"):
            name = template.split(":", 1)[-1] if ":" in template else "ridges"
            updated_env = ensure_chutes_api_key(env)
            return Templates.ridges(name), updated_env

        return template, env

    def _get_shared_sandbox(
        self,
        image: str,
        timeout: int,
        env: Optional[Dict[str, str]],
        metadata: Optional[Dict[str, Any]],
        **kwargs,
    ) -> Sandbox:
        """Get or create a shared sandbox instance."""
        with self._sandbox_lock:
            # Check if sandbox exists and is running
            if image in self._shared_sandboxes:
                sandbox = self._shared_sandboxes[image]
                try:
                    status = sandbox.status()
                    if status.state == "running":
                        # Refresh timeout to keep it alive
                        sandbox.refresh_timeout(timeout)
                        return sandbox
                except Exception:
                    # Sandbox is dead, remove it
                    del self._shared_sandboxes[image]

            # Create new shared sandbox
            restart_policy = kwargs.pop('restart_policy', 'unless-stopped')
            sandbox = Sandbox(
                template=image,
                timeout=timeout,
                env=env,
                metadata=metadata,
                restart_policy=restart_policy,
                **kwargs
            )
            self._shared_sandboxes[image] = sandbox
            self._active_sandboxes.add(sandbox)
            return sandbox

    def _get_pooled_sandbox(
        self,
        image: str,
        pool_size: int,
        timeout: int,
        env: Optional[Dict[str, str]],
        metadata: Optional[Dict[str, Any]],
        **kwargs,
    ) -> SandboxWrapper:
        """Get a sandbox from playground pool with automatic cleanup wrapper."""
        with self._playground_lock:
            # Create playground for this image if not exists
            if image not in self._playgrounds:
                config = Config(
                    image=image,
                    timeout=timeout,
                    env=env or {},
                    metadata=metadata or {},
                )
                playground = Playground(n=pool_size, config=config)
                playground.prewarm()
                self._playgrounds[image] = playground

            playground = self._playgrounds[image]
            sandbox = playground.create()

            self._active_sandboxes.add(sandbox)

            # Return wrapped sandbox for automatic cleanup
            return SandboxWrapper(sandbox, self)

    def cleanup_playgrounds(self) -> None:
        """Clean up all playgrounds."""
        with self._playground_lock:
            for playground in self._playgrounds.values():
                try:
                    playground.close()
                except Exception:
                    pass
            self._playgrounds.clear()

    def cleanup(self) -> None:
        self.cleanup_playgrounds()

        # Clean up any remaining active sandboxes
        for sandbox in list(self._active_sandboxes):
            try:
                sandbox.shutdown()
            except Exception:
                pass
        self._shared_sandboxes.clear()
        self._active_sandboxes.clear()

    def stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "shared_sandboxes": list(self._shared_sandboxes.keys()),
            "shared_count": len(self._shared_sandboxes),
            "playgrounds": list(self._playgrounds.keys()),
            "playground_count": len(self._playgrounds),
            "active_sandboxes": len(self._active_sandboxes),
        }

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass


# Global convenience function
def get_sandbox(template: str, shared: bool = False, **kwargs) -> Sandbox:
    """Get a sandbox using the global manager.

    This is the recommended way to get sandboxes. It ensures only one
    global SandboxManager exists in the application.

    Args:
        template: Template name or Docker image
        shared: If True, returns a shared global instance
        **kwargs: Additional sandbox configuration

    Returns:
        Sandbox instance
    """
    manager = _get_global_manager()
    return manager.get_sandbox(template, shared=shared, **kwargs)


def cleanup_all() -> None:
    """Clean up all resources managed by the global manager."""
    global _global_manager
    if _global_manager:
        _global_manager.cleanup()


def get_manager_stats() -> Dict[str, Any]:
    """Get statistics from the global manager."""
    manager = _get_global_manager()
    return manager.stats()
