from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Run Affine evaluator-only server (no env imports)")
    p.add_argument("--env", required=True)
    p.add_argument("--tool-name", default="")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=6999)
    a = p.parse_args()

    os.environ["ENV_NAME"] = a.env
    if a.tool_name:
        os.environ["TOOL_NAME"] = a.tool_name
    os.environ["QS_EVALUATOR_ONLY"] = "1"

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here / "env_templates" / "agentgym" / "agentenv"))

    import uvicorn
    uvicorn.run("affine.quixand.env_templates.agentgym.env:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()


