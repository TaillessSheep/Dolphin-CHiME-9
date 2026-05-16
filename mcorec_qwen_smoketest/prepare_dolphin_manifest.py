#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from mcorec_qwen_smoketest.common_mcorec import (
    CENTRAL_ASD_SEGMENTATION_DEFAULTS,
    SegmentRecord,
    compute_track_activity,
    crop_entries,
    extract_wav,
    ffprobe_duration,
    list_sessions,
    list_speakers,
    require_ffmpeg,
    run_cmd,
    summarize_numeric_series,
    write_json,
)

FACE_FAILURE_HINTS = (
    "First frame must detect at least",
    "Face landmarks not detected in initial frame",
)


def find_dolphin_inference(dolphin_repo: Path) -> Path:
    for name in ("Inference.py", "inference.py"):
        p = dolphin_repo / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find Inference.py or inference.py in Dolphin repo: {dolphin_repo}")


def trim_video_start(track_video: Path, out_video: Path, *, start: float) -> Path:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", str(track_video),
        "-c:v", "libx264",
        "-c:a", "aac",
        str(out_video),
    ]
    run_cmd(cmd)
    return out_video


def is_initial_face_failure(exc: Exception) -> bool:
    text = str(exc)
    return any(hint in text for hint in FACE_FAILURE_HINTS)


def run_dolphin_once(
    *,
    dolphin_repo: Path,
    input_video: Path,
    out_dir: Path,
    cuda_device: int,
    detect_every_n: int,
    face_scale: float,
    reuse: bool,
) -> Path:
    """Run Dolphin on one crop track and return speaker1_est.wav."""
    est = out_dir / "speaker1_est.wav"
    if reuse and est.exists() and est.stat().st_size > 0:
        print(f"[reuse] {est}")
        return est
    out_dir.mkdir(parents=True, exist_ok=True)
    inference_py = find_dolphin_inference(dolphin_repo)
    cmd = [
        sys.executable,
        str(inference_py),
        "--input", str(input_video),
        "--output", str(out_dir),
        "--speakers", "1",
        "--detect-every-n", str(detect_every_n),
        "--face-scale", str(face_scale),
        "--cuda-device", str(cuda_device),
    ]
    run_cmd(cmd, cwd=dolphin_repo)
    if not est.exists():
        # Some forks may name outputs differently. Try a forgiving search.
        candidates = sorted(out_dir.glob("*speaker*est*.wav")) + sorted(out_dir.glob("*_est.wav"))
        if candidates:
            est = candidates[0]
    if not est.exists():
        raise RuntimeError(f"Dolphin did not produce speaker1_est.wav in {out_dir}")
    return est


def run_dolphin_track(
    *,
    dolphin_repo: Path,
    track_video: Path,
    out_dir: Path,
    cuda_device: int,
    detect_every_n: int,
    face_scale: float,
    reuse: bool,
    trim_retry_seconds: float,
    max_trim_retries: int,
) -> dict:
    """Run Dolphin with optional trimmed-input retry for fragile first-frame detection."""
    attempts = max(0, max_trim_retries)
    last_exc: Exception | None = None
    for attempt_idx in range(attempts + 1):
        trim_offset = 0.0 if attempt_idx == 0 else trim_retry_seconds * attempt_idx
        attempt_dir = out_dir if trim_offset == 0 else out_dir / f"retry_trim_{attempt_idx:02d}"
        attempt_reuse = False
        input_video = track_video
        if trim_offset > 0:
            input_video = trim_video_start(
                track_video,
                out_dir / "retry_inputs" / f"{track_video.stem}_trim_{attempt_idx:02d}.mp4",
                start=trim_offset,
            )
            print(f"[retry] Dolphin face detection fallback with trim_offset={trim_offset:.3f}s for {track_video.name}")
        elif reuse:
            est = attempt_dir / "speaker1_est.wav"
            attempt_reuse = est.exists() and est.stat().st_size > 0
        try:
            est = run_dolphin_once(
                dolphin_repo=dolphin_repo,
                input_video=input_video,
                out_dir=attempt_dir,
                cuda_device=cuda_device,
                detect_every_n=detect_every_n,
                face_scale=face_scale,
                reuse=reuse and trim_offset == 0.0,
            )
            return {
                "separated_wav": est,
                "trim_offset": trim_offset,
                "status": "reused" if attempt_reuse else ("trim_retry_success" if trim_offset > 0 else "success"),
                "attempts": attempt_idx + 1,
                "input_video": str(input_video),
            }
        except Exception as exc:
            last_exc = exc
            if trim_offset == 0.0 and not is_initial_face_failure(exc):
                raise
            if trim_offset > 0.0 and not is_initial_face_failure(exc):
                raise
            if attempt_idx >= attempts or trim_retry_seconds <= 0:
                raise
    assert last_exc is not None
    raise last_exc


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run Dolphin as an audio-visual target speech extraction front-end on each MCoRec speaker crop track, "
            "then prepare a manifest of separated wav segments for Qwen3-ASR. Run this inside the Dolphin env."
        )
    )
    ap.add_argument("--session_glob", required=True, help="A session dir, a split dir, or a glob like data-bin/dev/*")
    ap.add_argument("--dolphin_repo", required=True, help="Path to cloned JusperLee/Dolphin repo")
    ap.add_argument("--out_root", default="runs/dolphin_qwen", help="Directory for Dolphin outputs, wav segments, and manifest.jsonl")
    ap.add_argument("--manifest_name", default="manifest.jsonl")
    ap.add_argument("--cuda_device", type=int, default=0)
    ap.add_argument("--detect_every_n", type=int, default=8)
    ap.add_argument("--face_scale", type=float, default=1.5)
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
    ap.add_argument("--reuse_dolphin", action="store_true", help="Reuse existing Dolphin speaker1_est.wav outputs.")
    ap.add_argument("--duration_tolerance", type=float, default=0.25, help="Max allowed |separated_duration - expected_duration| in seconds.")
    ap.add_argument("--trim_retry_seconds", type=float, default=0.40, help="Retry Dolphin with the first N seconds trimmed when first-frame face detection fails; set 0 to disable.")
    ap.add_argument("--max_trim_retries", type=int, default=1, help="How many trimmed-input retries to allow after a first-frame detection failure.")
    ap.add_argument("--max_tracks_per_speaker", type=int, default=0, help="Debug limit; 0 means no limit.")
    ap.add_argument("--max_segments_per_speaker", type=int, default=0, help="Debug limit; 0 means no limit.")
    args = ap.parse_args()

    require_ffmpeg()
    dolphin_repo = Path(args.dolphin_repo).resolve()
    find_dolphin_inference(dolphin_repo)

    out_root = Path(args.out_root)
    dolphin_root = out_root / "dolphin_tracks"
    wav_root = out_root / "wav"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / args.manifest_name
    stats_path = out_root / "prepare_stats.json"
    track_report_path = out_root / "dolphin_track_report.jsonl"
    failure_report_path = out_root / "dolphin_failures.jsonl"

    sessions = list_sessions(args.session_glob)
    if not sessions:
        raise SystemExit(f"No sessions found for {args.session_glob}")

    started_at = time.perf_counter()
    n_segments = 0
    track_latencies_ms: list[float] = []
    segment_durations: list[float] = []
    asd_reasons: dict[str, int] = {}
    dolphin_statuses: dict[str, int] = {}
    speaker_stats: dict[str, int] = {}
    track_reports: list[dict] = []
    failure_reports: list[dict] = []
    with manifest_path.open("w", encoding="utf-8") as mf:
        for session_dir in sessions:
            session_id = session_dir.name
            print(f"\n=== {session_id} ===")
            for speaker in list_speakers(session_dir):
                made_segments = 0
                processed_tracks = 0
                for entry in crop_entries(session_dir, speaker):
                    if args.max_tracks_per_speaker and processed_tracks >= args.max_tracks_per_speaker:
                        break
                    track_started = time.perf_counter()
                    track_video = entry["video"]
                    track_stem = track_video.stem
                    dolphin_out = dolphin_root / session_id / speaker / track_stem
                    try:
                        dolphin_result = run_dolphin_track(
                            dolphin_repo=dolphin_repo,
                            track_video=track_video,
                            out_dir=dolphin_out,
                            cuda_device=args.cuda_device,
                            detect_every_n=args.detect_every_n,
                            face_scale=args.face_scale,
                            reuse=args.reuse_dolphin,
                            trim_retry_seconds=args.trim_retry_seconds,
                            max_trim_retries=args.max_trim_retries,
                        )
                    except Exception as exc:
                        print(f"[WARN] Dolphin failed for {session_id}/{speaker}/{track_video.name}: {exc}")
                        failure_reports.append({
                            "session_id": session_id,
                            "speaker": speaker,
                            "track": str(track_video),
                            "reason": str(exc),
                        })
                        dolphin_statuses["failed"] = dolphin_statuses.get("failed", 0) + 1
                        continue
                    separated_wav = Path(dolphin_result["separated_wav"])
                    trim_offset = float(dolphin_result["trim_offset"])
                    track_duration = ffprobe_duration(track_video)
                    separated_duration = ffprobe_duration(separated_wav)
                    expected_duration = max(0.0, track_duration - trim_offset)
                    duration_delta = separated_duration - expected_duration
                    if abs(duration_delta) > args.duration_tolerance:
                        reason = (
                            f"Separated duration mismatch for {session_id}/{speaker}/{track_video.name}: "
                            f"expected {expected_duration:.3f}s, got {separated_duration:.3f}s"
                        )
                        print(f"[WARN] {reason}")
                        failure_reports.append({
                            "session_id": session_id,
                            "speaker": speaker,
                            "track": str(track_video),
                            "reason": reason,
                            "track_duration": track_duration,
                            "expected_duration": expected_duration,
                            "separated_duration": separated_duration,
                            "duration_delta": duration_delta,
                        })
                        dolphin_statuses["duration_mismatch"] = dolphin_statuses.get("duration_mismatch", 0) + 1
                        continue
                    processed_tracks += 1
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
                    asd_reason = str(activity["reason"])
                    asd_reasons[asd_reason] = asd_reasons.get(asd_reason, 0) + 1
                    emitted_for_track = 0
                    skipped_for_trim = 0
                    for local_idx, (abs_s, abs_e, rel_s, rel_e) in enumerate(windows):
                        if args.max_segments_per_speaker and made_segments >= args.max_segments_per_speaker:
                            break
                        effective_rel_s = max(rel_s, trim_offset)
                        extract_rel_s = max(0.0, effective_rel_s - trim_offset)
                        extract_rel_e = max(0.0, rel_e - trim_offset)
                        if extract_rel_e <= extract_rel_s:
                            skipped_for_trim += 1
                            continue
                        effective_abs_s = max(abs_s, float(activity["abs_start"]) + trim_offset)
                        out_wav = wav_root / session_id / speaker / f"{track_stem}_{local_idx:04d}.wav"
                        extract_wav(separated_wav, out_wav, start=extract_rel_s, end=extract_rel_e, sr=args.sr)
                        rec = SegmentRecord(
                            session_dir=str(session_dir),
                            session_id=session_id,
                            speaker=speaker,
                            track=str(track_video),
                            audio_path=str(out_wav),
                            start=effective_abs_s,
                            end=abs_e,
                            rel_start=effective_rel_s,
                            rel_end=rel_e,
                            source="dolphin_then_qwen",
                            asd_mode=activity["asd_mode"],
                            asd_reason=asd_reason,
                            track_duration=track_duration,
                            separated_duration=separated_duration,
                            duration_delta=duration_delta,
                            dolphin_status=str(dolphin_result["status"]),
                            trim_offset=trim_offset,
                        )
                        mf.write(rec.to_json() + "\n")
                        n_segments += 1
                        made_segments += 1
                        emitted_for_track += 1
                        segment_durations.append(max(0.0, rel_e - effective_rel_s))
                    track_latency_ms = (time.perf_counter() - track_started) * 1000.0
                    track_latencies_ms.append(track_latency_ms)
                    status = "success" if emitted_for_track > 0 else "no_segments"
                    dolphin_statuses[str(dolphin_result["status"])] = dolphin_statuses.get(str(dolphin_result["status"]), 0) + 1
                    track_reports.append({
                        "session_id": session_id,
                        "speaker": speaker,
                        "track": str(track_video),
                        "dolphin_output": str(separated_wav),
                        "dolphin_status": str(dolphin_result["status"]),
                        "attempts": int(dolphin_result["attempts"]),
                        "trim_offset": trim_offset,
                        "track_duration": track_duration,
                        "expected_duration": expected_duration,
                        "separated_duration": separated_duration,
                        "duration_delta": duration_delta,
                        "asd_reason": asd_reason,
                        "num_windows": len(windows),
                        "segments_emitted": emitted_for_track,
                        "segments_skipped_for_trim": skipped_for_trim,
                        "track_latency_ms": track_latency_ms,
                        "status": status,
                    })
                speaker_stats[f"{session_id}/{speaker}"] = made_segments
                print(f"{speaker}: {processed_tracks} Dolphin tracks, {made_segments} segments")

    total_wall = time.perf_counter() - started_at
    with track_report_path.open("w", encoding="utf-8") as f:
        for item in track_reports:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with failure_report_path.open("w", encoding="utf-8") as f:
        for item in failure_reports:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
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
        "dolphin_statuses": dolphin_statuses,
        "track_report": str(track_report_path),
        "failure_report": str(failure_report_path),
        "failure_count": len(failure_reports),
    })
    print(f"\nWrote {n_segments} separated segments to {manifest_path}")
    print(f"Track report: {track_report_path}")
    print(f"Failure report: {failure_report_path}")
    print(f"Prepare stats: {stats_path}")


if __name__ == "__main__":
    main()
