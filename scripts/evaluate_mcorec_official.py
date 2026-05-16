#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common_mcorec import run_cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Call the official MCoRec baseline evaluator on a generated output folder.")
    ap.add_argument("--mcorec_baseline_repo", required=True, help="Path to cloned https://github.com/MCoRec/mcorec_baseline")
    ap.add_argument("--session_dir", required=True, help="Single session, split dir, or glob like data-bin/dev/*")
    ap.add_argument("--output_dir_name", required=True, help="e.g. output_qwen3_track or output_dolphin_qwen")
    ap.add_argument("--label_dir_name", default="labels")
    args = ap.parse_args()

    repo = Path(args.mcorec_baseline_repo).resolve()
    eval_py = repo / "script" / "evaluate.py"
    if not eval_py.exists():
        raise SystemExit(f"Could not find official evaluator at {eval_py}")

    cmd = [
        sys.executable,
        str(eval_py),
        "--session_dir", args.session_dir,
        "--output_dir_name", args.output_dir_name,
        "--label_dir_name", args.label_dir_name,
    ]
    run_cmd(cmd, cwd=repo)


if __name__ == "__main__":
    main()
