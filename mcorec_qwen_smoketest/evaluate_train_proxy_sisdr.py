#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio

from mcorec_qwen_smoketest.common_mcorec import extract_wav, load_metadata, read_jsonl, require_ffmpeg


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    est = est.float().flatten()
    ref = ref.float().flatten()
    n = min(est.numel(), ref.numel())
    if n < 1600:
        return float("nan")
    est, ref = est[:n], ref[:n]
    est = est - est.mean()
    ref = ref - ref.mean()
    ref_energy = torch.sum(ref ** 2) + eps
    proj = torch.sum(est * ref) * ref / ref_energy
    noise = est - proj
    return float(10.0 * torch.log10((torch.sum(proj ** 2) + eps) / (torch.sum(noise ** 2) + eps)))


def ego_audio_for_segment(session_dir: Path, speaker: str, start: float, end: float, tmp_dir: Path) -> Optional[Path]:
    meta = load_metadata(session_dir)
    spk = meta.get(speaker, {})
    central = spk.get("central", {}) if isinstance(spk, dict) else {}
    ego = spk.get("ego", {}) if isinstance(spk, dict) else {}
    if not isinstance(ego, dict) or "video" not in ego or "uem" not in ego or "uem" not in central:
        return None
    ego_video = session_dir / ego["video"]
    if not ego_video.exists():
        return None
    c0 = float(central["uem"]["start"])
    e0 = float(ego["uem"]["start"])
    ego_start = max(0.0, e0 + (start - c0))
    ego_end = max(ego_start, e0 + (end - c0))
    out_wav = tmp_dir / f"{session_dir.name}_{speaker}_{start:.2f}_{end:.2f}_ego.wav"
    extract_wav(ego_video, out_wav, start=ego_start, end=ego_end, sr=16000)
    return out_wav


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Optional proxy separation evaluation for MCoRec train only. Compares manifest audio to close-talk ego/lapel audio "
            "using SI-SDR. This is not the official MCoRec metric and should be treated as a development sanity check."
        )
    )
    ap.add_argument("--manifest", required=True, help="Manifest whose audio_path is the enhanced/separated segment audio.")
    ap.add_argument("--out_csv", default="", help="Defaults to <manifest>.train_proxy_sisdr.csv")
    args = ap.parse_args()

    require_ffmpeg()
    manifest = Path(args.manifest)
    rows = read_jsonl(manifest)
    if not rows:
        raise SystemExit(f"Empty manifest: {manifest}")
    out_csv = Path(args.out_csv) if args.out_csv else manifest.with_suffix(".train_proxy_sisdr.csv")

    metrics = []
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for rec in rows:
            session_dir = Path(rec["session_dir"])
            speaker = rec["speaker"]
            enhanced = Path(rec["audio_path"])
            ref_wav = ego_audio_for_segment(session_dir, speaker, float(rec["start"]), float(rec["end"]), tmp_dir)
            if ref_wav is None:
                continue
            try:
                est, sr1 = torchaudio.load(enhanced)
                ref, sr2 = torchaudio.load(ref_wav)
                if sr1 != 16000:
                    est = torchaudio.functional.resample(est, sr1, 16000)
                if sr2 != 16000:
                    ref = torchaudio.functional.resample(ref, sr2, 16000)
                val = si_sdr(est.mean(dim=0), ref.mean(dim=0))
            except Exception as exc:
                print(f"[WARN] failed {enhanced}: {exc}")
                val = float("nan")
            metrics.append({
                "session_id": rec["session_id"],
                "speaker": speaker,
                "start": rec["start"],
                "end": rec["end"],
                "audio_path": str(enhanced),
                "proxy_si_sdr_db": val,
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "speaker", "start", "end", "audio_path", "proxy_si_sdr_db"])
        writer.writeheader()
        writer.writerows(metrics)
    vals = [m["proxy_si_sdr_db"] for m in metrics if m["proxy_si_sdr_db"] == m["proxy_si_sdr_db"]]
    if vals:
        print(f"Mean proxy SI-SDR: {sum(vals)/len(vals):.3f} dB over {len(vals)} segments")
    else:
        print("No proxy SI-SDR values computed. This script requires train sessions with ego/lapel audio.")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
