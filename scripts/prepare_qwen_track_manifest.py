#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from common_mcorec import (
    SegmentRecord,
    active_windows_for_track,
    crop_entries,
    extract_wav,
    list_sessions,
    list_speakers,
    require_ffmpeg,
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
    ap.add_argument("--asd_threshold", type=float, default=0.5)
    ap.add_argument("--min_duration", type=float, default=0.35)
    ap.add_argument("--merge_gap", type=float, default=0.30)
    ap.add_argument("--pad", type=float, default=0.10)
    ap.add_argument("--no_asd", action="store_true", help="Ignore track_xx_asd.json and use whole crop tracks.")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--max_segments_per_speaker", type=int, default=0, help="Debug limit; 0 means no limit.")
    args = ap.parse_args()

    require_ffmpeg()
    out_root = Path(args.out_root)
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
                made_for_speaker = 0
                for entry in crop_entries(session_dir, speaker):
                    track_video = entry["video"]
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
                        )
                        mf.write(rec.to_json() + "\n")
                        n_segments += 1
                        made_for_speaker += 1
                print(f"{speaker}: {made_for_speaker} segments")

    print(f"\nWrote {n_segments} segments to {manifest_path}")


if __name__ == "__main__":
    main()
