import re
import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    prompt: Optional[str] = None
    extra: Dict[str, Any] = {}
    response: Optional[str] = None


class EvaluateResponse(BaseModel):
    score: float
    extra: Dict[str, Any] = {}


class CreateResponse(BaseModel):
    prompt: str
    extra: Dict[str, Any] = {}


def normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines())


def strip_fences(text: Optional[str]) -> str:
    if not text:
        return ""
    m = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else text


async def run_program(code: str, stdin: str, timeout: int = 5) -> Tuple[str, str]:
    import subprocess
    import tempfile

    loop = asyncio.get_running_loop()

    def _exec():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            p = subprocess.Popen(
                ["python", path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            try:
                out, err = p.communicate(stdin, timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                return "", "TIMEOUT"
            return out, err
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    return await loop.run_in_executor(None, _exec)


