from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class SegmentRecord:
    session_dir: str
    session_id: str
    speaker: str
    track: str
    audio_path: str
    start: float
    end: float
    rel_start: float
    rel_end: float
    source: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


_TIME_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?")
_SPK_RE = re.compile(r"spk_(\d+)")


def run_cmd(cmd: Sequence[str], *, dry_run: bool = False, check: bool = True, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    printable = " ".join(str(x) for x in cmd)
    print(f"$ {printable}")
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None, check=check)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install it with conda install ffmpeg or apt/yum before running this script.")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH. Install ffmpeg, which usually provides ffprobe.")


def ffprobe_duration(media_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def extract_wav(media_path: Path, out_wav: Path, *, start: float = 0.0, end: Optional[float] = None, sr: int = 16000) -> None:
    """Extract a mono 16 kHz wav segment from a video/audio file."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    duration = None if end is None else max(0.0, end - start)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start and start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(media_path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", str(out_wav)]
    run_cmd(cmd)


def list_sessions(session_glob: str | Path) -> List[Path]:
    path = Path(session_glob)
    if any(ch in str(session_glob) for ch in "*?[]"):
        import glob
        sessions = [Path(p) for p in glob.glob(str(session_glob))]
    elif path.is_dir() and (path / "metadata.json").exists():
        sessions = [path]
    elif path.is_dir():
        sessions = [p for p in path.iterdir() if p.is_dir() and (p / "metadata.json").exists()]
    else:
        sessions = []
    return sorted(sessions, key=lambda p: p.name)


def speaker_sort_key(name: str) -> Tuple[int, str]:
    m = _SPK_RE.fullmatch(name)
    return (int(m.group(1)) if m else 10**9, name)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(session_dir: Path) -> Dict[str, Any]:
    meta = load_json(session_dir / "metadata.json", default={})
    if not isinstance(meta, dict):
        raise RuntimeError(f"metadata.json is not a JSON object: {session_dir / 'metadata.json'}")
    return meta


def list_speakers(session_dir: Path) -> List[str]:
    meta = load_metadata(session_dir)
    speakers = [k for k in meta.keys() if k.startswith("spk_")]
    if not speakers:
        spk_root = session_dir / "speakers"
        speakers = [p.name for p in spk_root.glob("spk_*") if p.is_dir()]
    return sorted(speakers, key=speaker_sort_key)


def crop_entries(session_dir: Path, speaker: str) -> List[Dict[str, Path]]:
    """Return MCoRec central crop entries for a speaker.

    Each returned dict contains at least video, crop_metadata, and asd paths.
    The official metadata normally has central.crops entries. If not, we fall back
    to globbing speakers/<spk>/central_crops/track_*.mp4.
    """
    meta = load_metadata(session_dir)
    entries: List[Dict[str, Path]] = []
    spk_obj = meta.get(speaker, {}) if isinstance(meta, dict) else {}
    central = spk_obj.get("central", {}) if isinstance(spk_obj, dict) else {}
    for item in central.get("crops", []) if isinstance(central, dict) else []:
        video = session_dir / item.get("video", "")
        crop_meta = session_dir / item.get("crop_metadata", "")
        bbox = session_dir / item.get("bbox", "")
        if video.exists():
            stem = video.with_suffix("")
            entries.append({
                "video": video,
                "crop_metadata": crop_meta,
                "bbox": bbox,
                "asd": stem.parent / f"{stem.name}_asd.json",
                "lip_video": stem.parent / f"{stem.name}_lip.av.mp4",
            })
    if not entries:
        cdir = session_dir / "speakers" / speaker / "central_crops"
        for video in sorted(cdir.glob("track_*.mp4")):
            if video.name.endswith("_lip.av.mp4"):
                continue
            stem = video.with_suffix("")
            entries.append({
                "video": video,
                "crop_metadata": video.with_suffix(".json"),
                "bbox": stem.parent / f"{stem.name}_bbox.json",
                "asd": stem.parent / f"{stem.name}_asd.json",
                "lip_video": stem.parent / f"{stem.name}_lip.av.mp4",
            })
    return entries


def track_times(entry: Dict[str, Path]) -> Tuple[float, float]:
    meta = load_json(entry["crop_metadata"], default={})
    if isinstance(meta, dict) and "start_time" in meta and "end_time" in meta:
        return float(meta["start_time"]), float(meta["end_time"])
    # Fallback: assume the track starts at 0 if metadata is missing.
    dur = ffprobe_duration(entry["video"])
    return 0.0, dur


def uem_for_speaker(session_dir: Path, speaker: str) -> Optional[Tuple[float, float]]:
    meta = load_metadata(session_dir)
    obj = meta.get(speaker, {}) if isinstance(meta, dict) else {}
    central = obj.get("central", {}) if isinstance(obj, dict) else {}
    uem = central.get("uem") if isinstance(central, dict) else None
    if isinstance(uem, dict) and "start" in uem and "end" in uem:
        return float(uem["start"]), float(uem["end"])
    return None


def _score_from_item(item: Any) -> Optional[float]:
    if isinstance(item, bool):
        return 1.0 if item else 0.0
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        for key in ("score", "prob", "probability", "active", "speech", "speaking", "is_speaking", "vad", "asd"):
            if key in item:
                val = item[key]
                if isinstance(val, bool):
                    return 1.0 if val else 0.0
                if isinstance(val, (int, float)):
                    return float(val)
        # Some ASD files have {'label': 1} or {'prediction': 0/1}
        for key in ("label", "pred", "prediction"):
            if key in item and isinstance(item[key], (int, float, bool)):
                return float(item[key])
    if isinstance(item, (list, tuple)) and item:
        # Common shapes: [frame, score] or [time, score]
        last = item[-1]
        if isinstance(last, (int, float, bool)):
            return float(last)
    return None


def parse_asd_scores(asd_path: Path) -> List[Tuple[int, float]]:
    """Parse a wide range of possible track_xx_asd.json shapes.

    Returns (frame_index, score) pairs. If no parseable values are found,
    returns an empty list.
    """
    data = load_json(asd_path, default=None)
    if data is None:
        return []
    pairs: List[Tuple[int, float]] = []
    if isinstance(data, list):
        for idx, item in enumerate(data):
            frame = idx
            if isinstance(item, dict):
                for key in ("frame", "frame_id", "idx", "index"):
                    if key in item and isinstance(item[key], (int, float, str)):
                        try:
                            frame = int(float(item[key]))
                        except ValueError:
                            frame = idx
                        break
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    frame = int(float(item[0]))
                except (TypeError, ValueError):
                    frame = idx
            score = _score_from_item(item)
            if score is not None:
                pairs.append((frame, score))
    elif isinstance(data, dict):
        # Dict may map frame ids to scores, or have a nested sequence under a common key.
        for key in ("frames", "scores", "asd", "predictions", "data"):
            if key in data and isinstance(data[key], (list, dict)):
                nested_path = asd_path.with_name(asd_path.name + f".{key}")
                # Recurse without writing temp files.
                nested = data[key]
                if isinstance(nested, list):
                    for idx, item in enumerate(nested):
                        score = _score_from_item(item)
                        if score is not None:
                            pairs.append((idx, score))
                elif isinstance(nested, dict):
                    for k, item in nested.items():
                        try:
                            frame = int(float(k))
                        except ValueError:
                            continue
                        score = _score_from_item(item)
                        if score is not None:
                            pairs.append((frame, score))
                if pairs:
                    return sorted(pairs)
        for k, item in data.items():
            try:
                frame = int(float(k))
            except (TypeError, ValueError):
                continue
            score = _score_from_item(item)
            if score is not None:
                pairs.append((frame, score))
    return sorted(pairs)


def merge_time_windows(windows: List[Tuple[float, float]], *, merge_gap: float, min_duration: float) -> List[Tuple[float, float]]:
    if not windows:
        return []
    windows = sorted((max(0.0, s), max(0.0, e)) for s, e in windows if e > s)
    merged: List[List[float]] = []
    for s, e in windows:
        if not merged or s - merged[-1][1] > merge_gap:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged if e - s >= min_duration]


def active_windows_for_track(
    entry: Dict[str, Path],
    *,
    fps: float = 25.0,
    threshold: float = 0.5,
    min_duration: float = 0.35,
    merge_gap: float = 0.30,
    pad: float = 0.10,
    use_asd: bool = True,
) -> List[Tuple[float, float, float, float]]:
    """Return windows as (absolute_start, absolute_end, rel_start, rel_end)."""
    abs_start, abs_end = track_times(entry)
    dur = max(0.0, abs_end - abs_start)
    if dur <= 0:
        dur = ffprobe_duration(entry["video"])
        abs_end = abs_start + dur
    windows_rel: List[Tuple[float, float]] = []
    scores = parse_asd_scores(entry["asd"]) if use_asd else []
    if scores:
        active_frames = [frame for frame, score in scores if score >= threshold]
        if active_frames:
            # Group contiguous active frames, allowing tiny holes by merging after conversion.
            tmp = []
            run_start = active_frames[0]
            prev = active_frames[0]
            for frame in active_frames[1:]:
                if frame <= prev + 1:
                    prev = frame
                else:
                    tmp.append((run_start / fps, (prev + 1) / fps))
                    run_start = prev = frame
            tmp.append((run_start / fps, (prev + 1) / fps))
            windows_rel = [(max(0.0, s - pad), min(dur, e + pad)) for s, e in tmp]
            windows_rel = merge_time_windows(windows_rel, merge_gap=merge_gap, min_duration=min_duration)
    if not windows_rel:
        windows_rel = [(0.0, dur)]
    out = []
    for rs, re in windows_rel:
        rs = max(0.0, min(rs, dur))
        re = max(rs, min(re, dur))
        if re - rs >= min_duration:
            out.append((abs_start + rs, abs_start + re, rs, re))
    return out


def vtt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - math.floor(seconds)) * 1000))
    total = int(math.floor(seconds))
    if ms == 1000:
        total += 1
        ms = 0
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def write_vtt(path: Path, cues: Iterable[Tuple[float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for start, end, text in cues:
            text = (text or "").strip()
            if not text:
                continue
            if end <= start:
                end = start + 0.10
            f.write(f"{vtt_timestamp(start)} --> {vtt_timestamp(end)}\n")
            f.write(text.replace("\n", " ").strip() + "\n\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def labels(self) -> Dict[str, int]:
        roots: Dict[str, int] = {}
        out: Dict[str, int] = {}
        for item in sorted(self.parent, key=speaker_sort_key):
            root = self.find(item)
            if root not in roots:
                roots[root] = len(roots)
            out[item] = roots[root]
        return out


def activity_cluster(
    speakers: Sequence[str],
    segments_by_speaker: Dict[str, List[Tuple[float, float]]],
    *,
    bin_seconds: float = 5.0,
    threshold: float = 0.10,
) -> Dict[str, int]:
    """Tiny unsupervised placeholder clusterer.

    It clusters speakers whose active-time bin Jaccard similarity exceeds threshold.
    This is intentionally simple; replace it with semantic or learned clustering for serious work.
    """
    speakers = list(speakers)
    if not speakers:
        return {}
    uf = UnionFind(speakers)
    active_bins: Dict[str, set[int]] = {}
    for spk in speakers:
        bins: set[int] = set()
        for s, e in segments_by_speaker.get(spk, []):
            b0 = int(s // bin_seconds)
            b1 = int(max(s, e - 1e-6) // bin_seconds)
            bins.update(range(b0, b1 + 1))
        active_bins[spk] = bins
    for i, a in enumerate(speakers):
        for b in speakers[i + 1:]:
            A, B = active_bins[a], active_bins[b]
            if not A or not B:
                continue
            sim = len(A & B) / max(1, len(A | B))
            if sim >= threshold:
                uf.union(a, b)
    return uf.labels()


def make_cluster_mapping(
    session_dir: Path,
    speakers: Sequence[str],
    segments_by_speaker: Dict[str, List[Tuple[float, float]]],
    *,
    mode: str = "activity",
    bin_seconds: float = 5.0,
    threshold: float = 0.10,
) -> Dict[str, int]:
    speakers = list(speakers)
    if mode == "singleton":
        return {spk: i for i, spk in enumerate(sorted(speakers, key=speaker_sort_key))}
    if mode == "all_one":
        return {spk: 0 for spk in sorted(speakers, key=speaker_sort_key)}
    if mode == "copy_dev":
        labels = load_json(session_dir / "labels" / "speaker_to_cluster.json", default=None)
        if isinstance(labels, dict):
            return {spk: int(labels.get(spk, i)) for i, spk in enumerate(sorted(speakers, key=speaker_sort_key))}
        print(f"[WARN] copy_dev requested but labels not found for {session_dir.name}; falling back to singleton clusters.")
        return {spk: i for i, spk in enumerate(sorted(speakers, key=speaker_sort_key))}
    if mode == "activity":
        return activity_cluster(speakers, segments_by_speaker, bin_seconds=bin_seconds, threshold=threshold)
    raise ValueError(f"Unknown cluster mode: {mode}")
