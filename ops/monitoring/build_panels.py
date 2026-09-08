#!/usr/bin/env python3
"""Generate self-contained Affine monitor cards from the shared HTML template."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "panels" / "monitor-template.html"
PANEL_IDS = (
    "validator", "weights", "duel", "scores", "dialects", "history", "eval",
    "bench-engine", "chat", "teacher", "router", "corpus", "corpus-fold",
    "datagen", "traces", "queue", "registrations", "benchmarks", "audits",
    "services", "fleet", "overview", "host",
)


def script_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True).replace("<", "\\u003c").replace(
        ">", "\\u003e"
    ).replace("&", "\\u0026")


def render(template: str, panel_id: str, data_dir: Path) -> str:
    for token in ("__PANEL_ID__", "__DATA_PATH__"):
        if template.count(token) != 1:
            raise ValueError(f"template must contain exactly one {token}")
    # Replace the path last so a directory name cannot introduce an ID token.
    return template.replace("__PANEL_ID__", script_string(panel_id)).replace(
        "__DATA_PATH__", script_string(str((data_dir / f"{panel_id}.json").resolve()))
    )


def build(output_dir: Path, data_dir: Path, *, check: bool = False) -> list[Path]:
    template = TEMPLATE.read_text(encoding="utf-8")
    rendered = [(output_dir / f"{panel_id}.html", render(template, panel_id, data_dir))
                for panel_id in PANEL_IDS]
    if check:
        mismatches = [str(path) for path, content in rendered
                      if not path.is_file() or path.read_text(encoding="utf-8") != content]
        if mismatches:
            raise ValueError("missing or out-of-date panels:\n" + "\n".join(mismatches))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for path, content in rendered:
            if path.is_file() and path.read_text(encoding="utf-8") == content:
                continue
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                                 dir=output_dir, suffix=".tmp", delete=False) as file:
                    temporary = Path(file.name)
                    file.write(content)
                temporary.chmod(0o644)
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return [path for path, _ in rendered]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "panels")
    parser.add_argument("--data-dir", type=Path,
                        help="collector payload directory (default: <output-dir>/data)")
    parser.add_argument("--check", action="store_true", help="verify generated files without writing")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    data_dir = (args.data_dir.expanduser() if args.data_dir else output_dir / "data").resolve()
    try:
        paths = build(output_dir, data_dir, check=args.check)
    except (OSError, ValueError) as error:
        parser.exit(1, f"build_panels: {error}\n")
    print(f"{'Verified' if args.check else 'Generated'} {len(paths)} monitoring panels:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
