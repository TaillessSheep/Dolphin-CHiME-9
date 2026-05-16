#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from mcorec_qwen_smoketest.common_mcorec import load_json, write_json


def metric(d: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def build_markdown(summary: Dict[str, Any]) -> str:
    m = summary["metrics"]
    lines = [
        f"# {summary['baseline_name']}",
        "",
        "## Core metrics",
        "",
        f"- Average Speaker WER: {m['average_speaker_wer']:.4f}",
        f"- Average Conversation Clustering F1: {m['average_conversation_clustering_f1']:.4f}",
        f"- Average JACER: {m['average_joint_asr_clustering_error']:.4f}",
        "",
        "## Latency",
        "",
        f"- Mean per-segment latency (ms): {m['mean_segment_latency_ms']:.2f}",
        f"- P50 per-segment latency (ms): {m['p50_segment_latency_ms']:.2f}",
        f"- P95 per-segment latency (ms): {m['p95_segment_latency_ms']:.2f}",
        f"- Overall RTF: {m['overall_rtf']:.4f}",
        f"- Prepare wall time (s): {m['prepare_wall_seconds']:.2f}",
        f"- Transcribe wall time (s): {m['transcribe_wall_seconds']:.2f}",
        f"- Total wall time (s): {m['total_wall_seconds']:.2f}",
        "",
        "## Coverage",
        "",
        f"- Total segments: {int(m['total_segments'])}",
        f"- Non-empty transcript ratio: {m['non_empty_ratio']:.4f}",
        "",
        "## Inputs",
        "",
        f"- Prepare stats: `{summary['inputs']['prepare_stats']}`",
        f"- Transcribe timing: `{summary['inputs']['transcribe_timing']}`",
        f"- Evaluation summary: `{summary['inputs']['evaluation_summary']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge prepare/transcribe/eval sidecars into one benchmark summary.")
    ap.add_argument("--baseline_name", required=True, help="Label for this benchmark, e.g. baseline_a_qwen or baseline_b_dolphin_qwen")
    ap.add_argument("--prepare_stats", required=True, help="Path to prepare_stats.json")
    ap.add_argument("--transcribe_timing", required=True, help="Path to transcribe timing JSON")
    ap.add_argument("--evaluation_summary", required=True, help="Path to parsed evaluator summary JSON")
    ap.add_argument("--out_json", default="", help="Optional output JSON path. Defaults to <evaluation_summary>.benchmark.json")
    ap.add_argument("--out_md", default="", help="Optional output Markdown path. Defaults to <evaluation_summary>.benchmark.md")
    args = ap.parse_args()

    prepare_stats = load_json(Path(args.prepare_stats), default={}) or {}
    transcribe_timing = load_json(Path(args.transcribe_timing), default={}) or {}
    evaluation_summary = load_json(Path(args.evaluation_summary), default={}) or {}

    out_json = Path(args.out_json) if args.out_json else Path(args.evaluation_summary).with_suffix(".benchmark.json")
    out_md = Path(args.out_md) if args.out_md else Path(args.evaluation_summary).with_suffix(".benchmark.md")

    summary = {
        "baseline_name": args.baseline_name,
        "metrics": {
            "average_speaker_wer": metric(evaluation_summary, "averages", "average_speaker_wer"),
            "average_conversation_clustering_f1": metric(evaluation_summary, "averages", "average_conversation_clustering_f1"),
            "average_joint_asr_clustering_error": metric(evaluation_summary, "averages", "average_joint_asr_clustering_error"),
            "mean_segment_latency_ms": metric(transcribe_timing, "per_segment_latency_ms", "mean"),
            "p50_segment_latency_ms": metric(transcribe_timing, "per_segment_latency_ms", "p50"),
            "p95_segment_latency_ms": metric(transcribe_timing, "per_segment_latency_ms", "p95"),
            "overall_rtf": metric(transcribe_timing, "overall_rtf"),
            "prepare_wall_seconds": metric(prepare_stats, "total_wall_seconds"),
            "transcribe_wall_seconds": metric(transcribe_timing, "total_wall_seconds"),
            "total_wall_seconds": metric(prepare_stats, "total_wall_seconds") + metric(transcribe_timing, "total_wall_seconds"),
            "total_segments": metric(transcribe_timing, "total_segments"),
            "non_empty_ratio": metric(transcribe_timing, "non_empty_ratio"),
        },
        "inputs": {
            "prepare_stats": args.prepare_stats,
            "transcribe_timing": args.transcribe_timing,
            "evaluation_summary": args.evaluation_summary,
        },
    }

    write_json(out_json, summary)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(summary), encoding="utf-8")
    print(f"Benchmark summary JSON: {out_json}")
    print(f"Benchmark summary Markdown: {out_md}")


if __name__ == "__main__":
    main()
