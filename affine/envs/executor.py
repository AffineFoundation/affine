from __future__ import annotations

import os
import re
import atexit
from typing import Tuple, Optional, Dict, Any
import affine as af
from affine import quixand as qx


class SandboxExecutor:    
    # Shared playground for lightweight execution
    _playground_cache = {}
    
    @staticmethod
    def strip_fences(text: str) -> str:
        """Extract code from ```python ...``` or generic ``` ``` fences; fallback to raw text.
        Returns the last fenced block if multiple are present.
        """
        py_blocks = re.findall(r"```python\s*\n([\s\S]*?)```", text, re.DOTALL)
        if py_blocks:
            return py_blocks[-1].strip()
        code_blocks = re.findall(r"```(?:\w*)\s*\n([\s\S]*?)```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        return (m.group(1) if m else text).strip()
    
    @staticmethod
    def _get_playground(
        template: str = "python:3.11-slim",
        workdir: str = "/workspace",
        timeout: int = 600
    ) -> qx.Playground:
        """Get or create a playground for the given template with specified configuration.
        
        Args:
            template: Docker image to use
            workdir: Working directory in the container (default: /workspace)
            timeout: Execution timeout in seconds (default: 600)
            
        Returns:
            Configured Playground instance
        """
        # Create a unique cache key based on all parameters
        cache_key = f"{template}:{workdir}:{timeout}"
        
        if cache_key not in SandboxExecutor._playground_cache:
            config = qx.Config(timeout=timeout, image=template, workdir=workdir)
            playground = qx.Playground(n=2, config=config)
            SandboxExecutor._playground_cache[cache_key] = playground
            atexit.register(playground.close)
        return SandboxExecutor._playground_cache[cache_key]
    
    @staticmethod
    async def execute_code(
        program: str,
        stdin_text: str = "",
        *,
        template: str = "python:3.11-slim",
    ) -> Tuple[str, str]:
        """
        Execute Python code inside a Quixand sandbox with optional stdin.
        
        This is the lightweight mode - uses a pre-built image and creates/destroys
        containers on demand.
        
        Args:
            program: Python code to execute (supports code fences)
            stdin_text: Optional stdin input
            template: Docker image to use (default: python:3.11-slim)
            
        Returns:
            Tuple of (stdout, stderr) as text
        """
        src = SandboxExecutor.strip_fences(program)
        
        # Auto-run solve() when no __main__ guard exists
        if ("def solve" in src) and ("__name__" not in src):
            src = (
                src
                + "\n\nif __name__ == \"__main__\":\n"
                + "    res = solve()\n"
                + "    if res is not None:\n"
                + "        import sys\n"
                + "        if isinstance(res, (list, tuple)):\n"
                + "            print(*res)\n"
                + "        else:\n"
                + "            print(res)\n"
            )
        
        # Get or create playground
        playground = SandboxExecutor._get_playground(template)
        sandbox: Optional[qx.Sandbox] = playground.create()
        
        try:
            # Write main program
            sandbox.files.write("/workspace/main.py", src)
            
            # Handle stdin if provided
            redir = ""
            if stdin_text:
                if not stdin_text.endswith("\n"):
                    stdin_text = stdin_text + "\n"
                sandbox.files.write("/workspace/input.txt", stdin_text)
                redir = "< /workspace/input.txt"
            
            # Execute the program
            cmd = f"bash -lc 'python /workspace/main.py {redir} 1>/workspace/_out.txt 2>/workspace/_err.txt'"
            sandbox.run(cmd)
            
            # Read output
            try:
                out = sandbox.files.read("/workspace/_out.txt")
            except Exception:
                out = ""
            try:
                err = sandbox.files.read("/workspace/_err.txt")
            except Exception:
                err = ""
                
            return out, err
            
        finally:
            if sandbox is not None:
                try:
                    sandbox.shutdown()
                except Exception:
                    pass
    
    @staticmethod
    def compile_lean(
        lean_src: str,
        *,
        template: str = "leanprovercommunity/lean:latest",
        workdir: str = "/workspace",
        timeout: int = 600,
    ) -> Tuple[bool, str]:
        """
        Compile Lean source code in a containerized environment.
        
        This method writes the Lean source to a file and attempts compilation
        using the Lean compiler. It's useful for validating mathematical proofs.
        
        Args:
            lean_src: Lean source code to compile
            template: Docker image with Lean installed (default: leanprovercommunity/lean:latest)
            workdir: Working directory in the container (default: /workspace)
            timeout: Compilation timeout in seconds (default: 600)
            
        Returns:
            Tuple of (success: bool, compiler_output: str)
            - success: True if compilation succeeded (exit code 0)
            - compiler_output: Compiler stderr or stdout for debugging
        """
        # Use the shared _get_playground function with custom parameters
        playground = SandboxExecutor._get_playground(template, workdir, timeout)
        sandbox: Optional[qx.Sandbox] = playground.create()
        
        try:
            # Write Lean source file
            sandbox.files.write(f"{workdir}/Main.lean", lean_src)
            
            # Compile the Lean file
            cmd = f"bash -lc 'lean --make {workdir}/Main.lean 1>{workdir}/_out.txt 2>{workdir}/_err.txt || true'"
            result = sandbox.run(cmd)
            
            # Read compilation output
            out = ""
            err = ""
            try:
                out = sandbox.files.read(f"{workdir}/_out.txt")
            except Exception:
                pass
            try:
                err = sandbox.files.read(f"{workdir}/_err.txt")
            except Exception:
                pass
            
            # Check if compilation succeeded
            success = (result.exit_code == 0)
            return success, (err or out or "")
            
        finally:
            if sandbox is not None:
                try:
                    sandbox.shutdown()
                except Exception:
                    pass
    
    @staticmethod
    def ridges(
        problem_statement: str,
        timeout: int = 300,
        template_path: str = "affine/quixand/env_templates/ridges",
    ) -> Dict[str, Any]:
        """
        Run Ridges agent in a sandbox.
        
        This builds a custom image from the template and runs the agent.
        
        Args:
            problem_statement: The problem to solve
            timeout: Execution timeout in seconds
            template_path: Path to the Ridges template directory
            
        Returns:
            Dict containing the agent response
        """
        # Get API key from environment
        chutes_api_key = os.getenv("CHUTES_API_KEY", "")
        
        if not chutes_api_key:
            raise ValueError("CHUTES_API_KEY is required for Ridges execution")
        
        image = qx.Templates.build(template_path, name="ridges-agent")
        sandbox = qx.Sandbox(
            template=image,
            timeout=timeout,
            env={
                "CHUTES_API_KEY": chutes_api_key,
            },
        )
        try:
            sandbox.files.write("problem_statement.txt", problem_statement)
            response = sandbox.proxy.run(
                port=8000,
                path="/run",
                method="POST",
                timeout=timeout
            )
            return response
            
        finally:
            sandbox.shutdown()