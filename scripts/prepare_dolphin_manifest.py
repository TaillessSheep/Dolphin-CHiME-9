#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from common_mcorec import (
    SegmentRecord,
    active_windows_for_track,
    crop_entries,
    extract_wav,
    list_sessions,
    list_speakers,
    require_ffmpeg,
    run_cmd,
)


def find_dolphin_inference(dolphin_repo: Path) -> Path:
    for name in ("Inference.py", "inference.py"):
        p = dolphin_repo / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find Inference.py or inference.py in Dolphin repo: {dolphin_repo}")


def run_dolphin_track(
        *,
        dolphin_repo: Path,
        track_video: Path,
        out_dir: Path,
        cuda_device: int,
        detect_every_n: int,
        face_scale: float,
        reuse: bool,
) -> Path:
    """Run Dolphin on a single MCoRec face-crop track and return separated audio."""
    # 使用绝对路径
    out_dir = out_dir.resolve()
    est = out_dir / "s1.mp4"

    if reuse and est.exists() and est.stat().st_size > 0:
        print(f"[reuse] {est}")
        return est

    out_dir.mkdir(parents=True, exist_ok=True)
    inference_py = find_dolphin_inference(dolphin_repo)
    cmd = [
        sys.executable,
        str(inference_py),
        "--input", str(track_video),
        "--output", str(out_dir),
        "--speakers", "1",
        "--detect-every-n", str(detect_every_n),
        "--face-scale", str(face_scale),
        "--cuda-device", str(cuda_device),
    ]
    run_cmd(cmd, cwd=dolphin_repo)

    # 等待文件写入完成（最多等待 30 秒）
    import time
    for _ in range(15):
        if est.exists():
            break
        time.sleep(1)

    if not est.exists():
        # 尝试找其他 mp4 文件
        mp4_files = list(out_dir.glob("*.mp4"))
        if mp4_files:
            est = mp4_files[0]
            print(f"[INFO] Found alternative mp4: {est}")
        else:
            raise RuntimeError(f"Dolphin did not produce s1.mp4 in {out_dir}")

    return est


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
    ap.add_argument("--asd_threshold", type=float, default=0.5)
    ap.add_argument("--min_duration", type=float, default=0.35)
    ap.add_argument("--merge_gap", type=float, default=0.30)
    ap.add_argument("--pad", type=float, default=0.10)
    ap.add_argument("--no_asd", action="store_true", help="Ignore track_xx_asd.json and use whole crop tracks.")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--reuse_dolphin", action="store_true", help="Reuse existing Dolphin speaker1_est.wav outputs.")
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

    sessions = list_sessions(args.session_glob)
    if not sessions:
        raise SystemExit(f"No sessions found for {args.session_glob}")

    n_segments = 0
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
                    track_video = entry["video"]
                    track_stem = track_video.stem
                    dolphin_out = dolphin_root / session_id / speaker / track_stem
                    try:
                        separated_wav = run_dolphin_track(
                            dolphin_repo=dolphin_repo,
                            track_video=track_video,
                            out_dir=dolphin_out,
                            cuda_device=args.cuda_device,
                            detect_every_n=args.detect_every_n,
                            face_scale=args.face_scale,
                            reuse=args.reuse_dolphin,
                        )
                    except Exception as exc:
                        print(f"[WARN] Dolphin failed for {session_id}/{speaker}/{track_video.name}: {exc}")
                        continue
                    processed_tracks += 1
                    windows = active_windows_for_track(
                        entry,
                        fps=args.fps,
                        threshold=args.asd_threshold,
                        min_duration=args.min_duration,
                        merge_gap=args.merge_gap,
                        pad=args.pad,
                        use_asd=not args.no_asd,
                    )
                    for local_idx, (abs_s, abs_e, rel_s, rel_e) in enumerate(windows):
                        if args.max_segments_per_speaker and made_segments >= args.max_segments_per_speaker:
                            break
                        out_wav = wav_root / session_id / speaker / f"{track_stem}_{local_idx:04d}.wav"
                        extract_wav(separated_wav, out_wav, start=rel_s, end=rel_e, sr=args.sr)
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
                            source="dolphin_then_qwen",
                        )
                        mf.write(rec.to_json() + "\n")
                        n_segments += 1
                        made_segments += 1
                print(f"{speaker}: {processed_tracks} Dolphin tracks, {made_segments} segments")

    print(f"\nWrote {n_segments} separated segments to {manifest_path}")


if __name__ == "__main__":
    main()
