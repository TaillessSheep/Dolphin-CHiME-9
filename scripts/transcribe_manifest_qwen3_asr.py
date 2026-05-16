#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from common_mcorec import (
    make_cluster_mapping,
    read_jsonl,
    speaker_sort_key,
    write_json,
    write_vtt,
)


def batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def get_text(result: Any) -> str:
    if hasattr(result, "text"):
        return str(result.text or "")
    if isinstance(result, dict):
        return str(result.get("text") or result.get("transcript") or result.get("content") or "")
    return str(result or "")


def load_model(args: argparse.Namespace):
    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = getattr(torch, args.dtype)
    model = Qwen3ASRModel.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map=args.device_map,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Transcribe a segment manifest with Qwen3-ASR and write MCoRec-compatible .vtt outputs. "
            "Run this in an environment with qwen-asr installed."
        )
    )
    ap.add_argument("--manifest", required=True, help="JSONL manifest from prepare_qwen_track_manifest.py or prepare_dolphin_manifest.py")
    ap.add_argument("--model_id", default="Qwen/Qwen3-ASR-0.6B", help="Use Qwen/Qwen3-ASR-1.7B for the stronger baseline if you have VRAM.")
    ap.add_argument("--device_map", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_inference_batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--language", default="English", help="Set to auto for Qwen language detection.")
    ap.add_argument("--output_dir_name", default="output_qwen3", help="Created inside each session directory.")
    ap.add_argument("--transcript_jsonl", default="", help="Optional transcript log path. Defaults to <manifest>.transcripts.jsonl")
    ap.add_argument(
        "--cluster_mode",
        default="activity",
        choices=["activity", "singleton", "all_one", "copy_dev"],
        help="copy_dev is only for dev-set ASR ablations; do not use it for a fair challenge-style score.",
    )
    ap.add_argument("--cluster_bin_seconds", type=float, default=5.0)
    ap.add_argument("--cluster_threshold", type=float, default=0.10)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    records = read_jsonl(manifest_path)
    if not records:
        raise SystemExit(f"Manifest is empty: {manifest_path}")

    model = load_model(args)
    lang = None if args.language.lower() == "auto" else args.language

    # Transcribe in manifest order. Keeping the log separate makes it easy to debug bad segments.
    log_path = Path(args.transcript_jsonl) if args.transcript_jsonl else manifest_path.with_suffix(".transcripts.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    enriched: List[Dict[str, Any]] = []

    with log_path.open("w", encoding="utf-8") as log_f:
        for batch in batched(records, args.batch_size):
            audio_paths = [r["audio_path"] for r in batch]
            language_arg = None if lang is None else [lang] * len(audio_paths)
            print(f"Transcribing {len(audio_paths)} segments ...")
            results = model.transcribe(audio=audio_paths, language=language_arg)
            for rec, result in zip(batch, results):
                out = dict(rec)
                out["text"] = get_text(result).strip()
                if hasattr(result, "language"):
                    out["detected_language"] = str(result.language)
                enriched.append(out)
                log_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Group into MCoRec output folder layout.
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in enriched:
        by_session[rec["session_dir"]].append(rec)

    for session_dir_str, session_records in by_session.items():
        session_dir = Path(session_dir_str)
        out_dir = session_dir / args.output_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        speakers = sorted({r["speaker"] for r in session_records}, key=speaker_sort_key)
        by_spk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in session_records:
            by_spk[rec["speaker"]].append(rec)

        segments_by_speaker = {
            spk: [(float(r["start"]), float(r["end"])) for r in recs]
            for spk, recs in by_spk.items()
        }
        cluster_map = make_cluster_mapping(
            session_dir,
            speakers,
            segments_by_speaker,
            mode=args.cluster_mode,
            bin_seconds=args.cluster_bin_seconds,
            threshold=args.cluster_threshold,
        )
        write_json(out_dir / "speaker_to_cluster.json", cluster_map)

        for spk in speakers:
            cues = []
            for rec in sorted(by_spk[spk], key=lambda r: (float(r["start"]), float(r["end"]))):
                cues.append((float(rec["start"]), float(rec["end"]), rec.get("text", "")))
            write_vtt(out_dir / f"{spk}.vtt", cues)
        print(f"Wrote {out_dir}")

    print(f"Transcript log: {log_path}")


if __name__ == "__main__":
    main()
