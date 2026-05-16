#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

from mcorec_qwen_smoketest.common_mcorec import (
    CENTRAL_ASD_SEGMENTATION_DEFAULTS,
    SegmentRecord,
    compute_track_activity,
    crop_entries,
    extract_wav,
    list_sessions,
    list_speakers,
    require_ffmpeg,
    summarize_numeric_series,
    write_json,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare an audio-only Qwen3-ASR baseline manifest from MCoRec speaker crop tracks. "
            "Each segment keeps target-speaker timing/ID from MCoRec metadata, but the audio is not separated."
        )
    )
    ap.add_argument("--session_glob", required=True, help="A session dir, a split dir, or a glob like data-bin/dev/*")
    ap.add_argument("--out_root", default="runs/qwen_track", help="Directory for extracted wav segments and manifest.jsonl")
    ap.add_argument("--manifest_name", default="manifest.jsonl")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--asd_mode", default="official", choices=["official", "legacy"])
    ap.add_argument("--asd_threshold", type=float, default=0.5)
    ap.add_argument("--min_duration", type=float, default=0.35)
    ap.add_argument("--merge_gap", type=float, default=0.30)
    ap.add_argument("--pad", type=float, default=0.10)
    ap.add_argument("--asd_onset", type=float, default=CENTRAL_ASD_SEGMENTATION_DEFAULTS["onset"])
    ap.add_argument("--asd_offset", type=float, default=CENTRAL_ASD_SEGMENTATION_DEFAULTS["offset"])
    ap.add_argument("--asd_min_duration_on", type=float, default=CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_duration_on"])
    ap.add_argument("--asd_min_duration_off", type=float, default=CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_duration_off"])
    ap.add_argument("--asd_max_chunk_size", type=float, default=CENTRAL_ASD_SEGMENTATION_DEFAULTS["max_chunk_size"])
    ap.add_argument("--asd_min_chunk_size", type=float, default=CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_chunk_size"])
    ap.add_argument("--no_asd", action="store_true", help="Ignore track_xx_asd.json and use whole crop tracks.")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--max_segments_per_speaker", type=int, default=0, help="Debug limit; 0 means no limit.")
    args = ap.parse_args()

    require_ffmpeg()
    out_root = Path(args.out_root)
    wav_root = out_root / "wav"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / args.manifest_name
    stats_path = out_root / "prepare_stats.json"

    sessions = list_sessions(args.session_glob)
    if not sessions:
        raise SystemExit(f"No sessions found for {args.session_glob}")

    started_at = time.perf_counter()
    n_segments = 0
    track_latencies_ms: List[float] = []
    segment_durations: List[float] = []
    speaker_stats: Dict[str, int] = {}
    asd_reasons: Dict[str, int] = {}
    with manifest_path.open("w", encoding="utf-8") as mf:
        for session_dir in sessions:
            session_id = session_dir.name
            print(f"\n=== {session_id} ===")
            for speaker in list_speakers(session_dir):
                made_for_speaker = 0
                for entry in crop_entries(session_dir, speaker):
                    track_video = entry["video"]
                    track_started = time.perf_counter()
                    activity = compute_track_activity(
                        entry,
                        fps=args.fps,
                        asd_mode=args.asd_mode,
                        threshold=args.asd_threshold,
                        min_duration=args.min_duration,
                        merge_gap=args.merge_gap,
                        pad=args.pad,
                        use_asd=not args.no_asd,
                        onset=args.asd_onset,
                        offset=args.asd_offset,
                        min_duration_on=args.asd_min_duration_on,
                        min_duration_off=args.asd_min_duration_off,
                        max_chunk_size=args.asd_max_chunk_size,
                        min_chunk_size=args.asd_min_chunk_size,
                    )
                    windows = activity["windows"]
                    track_latency_ms = (time.perf_counter() - track_started) * 1000.0
                    track_latencies_ms.append(track_latency_ms)
                    asd_reason = str(activity["reason"])
                    asd_reasons[asd_reason] = asd_reasons.get(asd_reason, 0) + 1
                    for local_idx, (abs_s, abs_e, rel_s, rel_e) in enumerate(windows):
                        if args.max_segments_per_speaker and made_for_speaker >= args.max_segments_per_speaker:
                            break
                        track_stem = track_video.stem
                        out_wav = wav_root / session_id / speaker / f"{track_stem}_{local_idx:04d}.wav"
                        extract_wav(track_video, out_wav, start=rel_s, end=rel_e, sr=args.sr)
                        rec = SegmentRecord(
                            session_dir=str(session_dir),
                            session_id=session_id,
                            speaker=speaker,
                            track=str(track_video),
                            audio_path=str(out_wav),
                            start=abs_s,
                            end=abs_e,
                            rel_start=rel_s,
                            rel_end=rel_e,
                            source="qwen_track_audio_only",
                            asd_mode=activity["asd_mode"],
                            asd_reason=asd_reason,
                            track_duration=float(activity["track_duration"]),
                            prepare_latency_ms=track_latency_ms,
                        )
                        mf.write(rec.to_json() + "\n")
                        n_segments += 1
                        made_for_speaker += 1
                        segment_durations.append(max(0.0, rel_e - rel_s))
                speaker_stats[f"{session_id}/{speaker}"] = made_for_speaker
                print(f"{speaker}: {made_for_speaker} segments")

    total_wall = time.perf_counter() - started_at
    write_json(stats_path, {
        "manifest": str(manifest_path),
        "asd_mode": args.asd_mode,
        "total_segments": n_segments,
        "total_sessions": len(sessions),
        "total_wall_seconds": total_wall,
        "track_latency_ms": summarize_numeric_series(track_latencies_ms),
        "segment_duration_seconds": summarize_numeric_series(segment_durations),
        "speaker_segment_counts": speaker_stats,
        "asd_reasons": asd_reasons,
    })
    print(f"\nWrote {n_segments} segments to {manifest_path}")
    print(f"Prepare stats: {stats_path}")


if __name__ == "__main__":
    main()
