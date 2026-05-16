#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

from common_mcorec import write_json


def parse_evaluator_output(text: str) -> dict:
    sessions: list[dict] = []
    current: dict | None = None
    averages: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Evaluating session "):
            if current is not None:
                sessions.append(current)
            current = {"session_id": line.removeprefix("Evaluating session ").strip()}
            continue
        if current is None:
            continue
        if line.startswith("Conversation clustering F1 score: "):
            current["conversation_clustering_f1"] = float(line.split(": ", 1)[1])
        elif line.startswith("Speaker to WER: "):
            current["speaker_to_wer"] = ast.literal_eval(line.split(": ", 1)[1])
        elif line.startswith("Speaker clustering F1 score: "):
            current["speaker_clustering_f1"] = ast.literal_eval(line.split(": ", 1)[1])
        elif line.startswith("Joint ASR-Clustering Error Rate: "):
            current["joint_asr_clustering_error"] = ast.literal_eval(line.split(": ", 1)[1])
        elif line.startswith("Average Conversation Clustering F1 score: "):
            averages["average_conversation_clustering_f1"] = float(line.split(": ", 1)[1])
        elif line.startswith("Average Speaker WER: "):
            averages["average_speaker_wer"] = float(line.split(": ", 1)[1])
        elif line.startswith("Average Joint ASR-Clustering Error Rate: "):
            averages["average_joint_asr_clustering_error"] = float(line.split(": ", 1)[1])
    if current is not None:
        sessions.append(current)
    return {
        "sessions": sessions,
        "averages": averages,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Call the official MCoRec baseline evaluator on a generated output folder.")
    ap.add_argument("--mcorec_baseline_repo", required=True, help="Path to cloned https://github.com/MCoRec/mcorec_baseline")
    ap.add_argument("--session_dir", required=True, help="Single session, split dir, or glob like data-bin/dev/*")
    ap.add_argument("--output_dir_name", required=True, help="e.g. output_qwen3_track or output_dolphin_qwen")
    ap.add_argument("--label_dir_name", default="labels")
    ap.add_argument("--log_path", default="", help="Optional path to save raw evaluator stdout.")
    ap.add_argument("--summary_json", default="", help="Optional path to save parsed evaluator metrics as JSON.")
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
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(repo),
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    if args.summary_json:
        payload = parse_evaluator_output(proc.stdout)
        payload.update({
            "session_dir": args.session_dir,
            "output_dir_name": args.output_dir_name,
            "label_dir_name": args.label_dir_name,
            "returncode": proc.returncode,
        })
        write_json(Path(args.summary_json), payload)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
