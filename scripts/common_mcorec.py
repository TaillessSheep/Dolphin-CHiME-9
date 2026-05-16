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
    asd_mode: Optional[str] = None
    asd_reason: Optional[str] = None
    track_duration: Optional[float] = None
    separated_duration: Optional[float] = None
    duration_delta: Optional[float] = None
    prepare_latency_ms: Optional[float] = None
    dolphin_status: Optional[str] = None
    trim_offset: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


_TIME_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?")
_SPK_RE = re.compile(r"spk_(\d+)")

CENTRAL_ASD_SEGMENTATION_DEFAULTS = {
    "onset": 1.0,
    "offset": 0.8,
    "min_duration_on": 1.0,
    "min_duration_off": 0.5,
    "max_chunk_size": 10.0,
    "min_chunk_size": 1.0,
}


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


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def summarize_numeric_series(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {
        "count": float(len(vals)),
        "mean": safe_mean(vals),
        "p50": percentile(vals, 0.50),
        "p95": percentile(vals, 0.95),
        "min": min(vals) if vals else 0.0,
        "max": max(vals) if vals else 0.0,
        "sum": float(sum(vals)),
    }


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


def segment_frames_by_asd(
    scores: Sequence[Tuple[int, float]],
    *,
    onset: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["onset"],
    offset: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["offset"],
    min_duration_on: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_duration_on"],
    min_duration_off: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_duration_off"],
    max_chunk_size: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["max_chunk_size"],
    min_chunk_size: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_chunk_size"],
    fps: float = 25.0,
) -> List[List[int]]:
    if not scores:
        return []
    frame_to_score = {int(frame): float(score) for frame, score in scores}
    frames = sorted(frame_to_score)
    min_frame = min(frames)
    min_duration_on_frames = int(min_duration_on * fps)
    min_duration_off_frames = int(min_duration_off * fps)
    max_chunk_frames = int(max_chunk_size * fps)
    min_chunk_frames = int(min_chunk_size * fps)

    speech_regions: List[List[int]] = []
    current_region: Optional[List[int]] = None
    is_active = False

    for frame in frames:
        score = frame_to_score.get(frame, -1.0)
        normalized_frame = frame - min_frame
        if not is_active:
            if score > onset:
                is_active = True
                current_region = [normalized_frame]
        else:
            if score < offset:
                is_active = False
                if current_region is not None:
                    speech_regions.append(current_region)
                    current_region = None
            else:
                assert current_region is not None
                current_region.append(normalized_frame)

    if current_region is not None:
        speech_regions.append(current_region)

    merged_regions: List[List[int]] = []
    if speech_regions:
        current_region = list(speech_regions[0])
        for next_region in speech_regions[1:]:
            gap = next_region[0] - current_region[-1] - 1
            if gap <= min_duration_off_frames:
                current_region.extend(next_region)
            else:
                merged_regions.append(current_region)
                current_region = list(next_region)
        merged_regions.append(current_region)

    final_segments: List[List[int]] = []
    for region in merged_regions:
        region_length = len(region)
        if region_length < min_duration_on_frames:
            continue
        if region_length > max_chunk_frames:
            num_chunks = math.ceil(region_length / max_chunk_frames)
            chunk_size = math.ceil(region_length / num_chunks)
            for idx in range(0, region_length, chunk_size):
                sub_segment = region[idx: idx + chunk_size]
                if len(sub_segment) >= min_chunk_frames:
                    final_segments.append(sub_segment)
        elif region_length >= min_chunk_frames:
            final_segments.append(region)

    return [[frame + min_frame for frame in segment] for segment in final_segments]


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


def _official_windows_rel(
    scores: Sequence[Tuple[int, float]],
    *,
    fps: float,
    onset: float,
    offset: float,
    min_duration_on: float,
    min_duration_off: float,
    max_chunk_size: float,
    min_chunk_size: float,
    dur: float,
) -> List[Tuple[float, float]]:
    min_frame = min(int(frame) for frame, _ in scores) if scores else 0
    segments = segment_frames_by_asd(
        scores,
        onset=onset,
        offset=offset,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        fps=fps,
    )
    windows_rel: List[Tuple[float, float]] = []
    for segment in segments:
        if not segment:
            continue
        rs = max(0.0, (segment[0] - min_frame) / fps)
        re = min(dur, ((segment[-1] - min_frame) + 1) / fps)
        if re > rs:
            windows_rel.append((rs, re))
    return windows_rel


def _legacy_windows_rel(
    scores: Sequence[Tuple[int, float]],
    *,
    fps: float,
    threshold: float,
    min_duration: float,
    merge_gap: float,
    pad: float,
    dur: float,
) -> List[Tuple[float, float]]:
    active_frames = [frame for frame, score in scores if score >= threshold]
    if not active_frames:
        return []
    tmp: List[Tuple[float, float]] = []
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
    return merge_time_windows(windows_rel, merge_gap=merge_gap, min_duration=min_duration)


def compute_track_activity(
    entry: Dict[str, Path],
    *,
    fps: float = 25.0,
    use_asd: bool = True,
    asd_mode: str = "official",
    threshold: float = 0.5,
    min_duration: float = 0.35,
    merge_gap: float = 0.30,
    pad: float = 0.10,
    onset: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["onset"],
    offset: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["offset"],
    min_duration_on: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_duration_on"],
    min_duration_off: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_duration_off"],
    max_chunk_size: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["max_chunk_size"],
    min_chunk_size: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_chunk_size"],
) -> Dict[str, Any]:
    abs_start, abs_end = track_times(entry)
    dur = max(0.0, abs_end - abs_start)
    if dur <= 0:
        dur = ffprobe_duration(entry["video"])
        abs_end = abs_start + dur

    reason = "no_asd_requested"
    scores: List[Tuple[int, float]] = []
    windows_rel: List[Tuple[float, float]] = []
    if use_asd:
        scores = parse_asd_scores(entry["asd"])
        if not scores:
            reason = "asd_missing_or_empty"
        else:
            if asd_mode == "official":
                windows_rel = _official_windows_rel(
                    scores,
                    fps=fps,
                    onset=onset,
                    offset=offset,
                    min_duration_on=min_duration_on,
                    min_duration_off=min_duration_off,
                    max_chunk_size=max_chunk_size,
                    min_chunk_size=min_chunk_size,
                    dur=dur,
                )
                reason = "asd_official_segments"
            elif asd_mode == "legacy":
                windows_rel = _legacy_windows_rel(
                    scores,
                    fps=fps,
                    threshold=threshold,
                    min_duration=min_duration,
                    merge_gap=merge_gap,
                    pad=pad,
                    dur=dur,
                )
                reason = "asd_legacy_segments"
            else:
                raise ValueError(f"Unknown ASD mode: {asd_mode}")
            if not windows_rel:
                reason = f"{reason}_empty"

    if not windows_rel:
        windows_rel = [(0.0, dur)] if dur > 0 else []
        if dur > 0:
            reason = f"{reason}_fallback_full_track"

    windows: List[Tuple[float, float, float, float]] = []
    keep_short_full_track = len(windows_rel) == 1 and windows_rel[0][0] == 0.0 and abs(windows_rel[0][1] - dur) < 1e-6
    for rs, re in windows_rel:
        rs = max(0.0, min(rs, dur))
        re = max(rs, min(re, dur))
        if re > rs and (re - rs >= min_duration or keep_short_full_track):
            windows.append((abs_start + rs, abs_start + re, rs, re))

    return {
        "windows": windows,
        "scores": scores,
        "reason": reason,
        "track_duration": dur,
        "abs_start": abs_start,
        "abs_end": abs_end,
        "asd_mode": asd_mode if use_asd else "disabled",
    }


def active_windows_for_track(
    entry: Dict[str, Path],
    *,
    fps: float = 25.0,
    asd_mode: str = "official",
    threshold: float = 0.5,
    min_duration: float = 0.35,
    merge_gap: float = 0.30,
    pad: float = 0.10,
    use_asd: bool = True,
    onset: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["onset"],
    offset: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["offset"],
    min_duration_on: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_duration_on"],
    min_duration_off: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_duration_off"],
    max_chunk_size: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["max_chunk_size"],
    min_chunk_size: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_chunk_size"],
) -> List[Tuple[float, float, float, float]]:
    """Return windows as (absolute_start, absolute_end, rel_start, rel_end)."""
    return compute_track_activity(
        entry,
        fps=fps,
        use_asd=use_asd,
        asd_mode=asd_mode,
        threshold=threshold,
        min_duration=min_duration,
        merge_gap=merge_gap,
        pad=pad,
        onset=onset,
        offset=offset,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
    )["windows"]


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


def calculate_overlap_duration(
    segments1: Sequence[Tuple[float, float]],
    segments2: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    total_overlap = 0.0
    total_duration1 = sum(max(0.0, end - start) for start, end in segments1)
    total_duration2 = sum(max(0.0, end - start) for start, end in segments2)
    for start1, end1 in segments1:
        for start2, end2 in segments2:
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            if overlap_end > overlap_start:
                total_overlap += overlap_end - overlap_start
    total_non_overlap = total_duration1 + total_duration2 - 2.0 * total_overlap
    return total_overlap, total_non_overlap


def calculate_conversation_scores(
    speaker_segments: Dict[str, List[Tuple[float, float]]],
    speakers: Sequence[str],
) -> Dict[Tuple[str, str], float]:
    scores: Dict[Tuple[str, str], float] = {}
    ordered = list(speakers)
    for idx, spk_a in enumerate(ordered):
        scores[(spk_a, spk_a)] = 1.0
        for spk_b in ordered[idx + 1:]:
            overlap, non_overlap = calculate_overlap_duration(
                speaker_segments.get(spk_a, []),
                speaker_segments.get(spk_b, []),
            )
            total = overlap + non_overlap
            score = 0.0 if total <= 0 else 1.0 - (overlap / total)
            scores[(spk_a, spk_b)] = score
            scores[(spk_b, spk_a)] = score
    return scores


def overlap_cluster(
    speakers: Sequence[str],
    speaker_segments: Dict[str, List[Tuple[float, float]]],
    *,
    threshold: float = 0.7,
) -> Dict[str, int]:
    speakers = list(sorted(speakers, key=speaker_sort_key))
    if not speakers:
        return {}
    scores = calculate_conversation_scores(speaker_segments, speakers)
    distance_limit = 1.0 - threshold
    clusters: List[List[str]] = [[speaker] for speaker in speakers]
    while True:
        best_pair: Optional[Tuple[int, int]] = None
        best_distance: Optional[float] = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = max(1.0 - scores.get((a, b), 0.0) for a in clusters[i] for b in clusters[j])
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)
        if best_pair is None or best_distance is None or best_distance > distance_limit:
            break
        i, j = best_pair
        clusters[i].extend(clusters[j])
        del clusters[j]

    out: Dict[str, int] = {}
    for cluster_id, cluster in enumerate(clusters):
        for speaker in sorted(cluster, key=speaker_sort_key):
            out[speaker] = cluster_id
    return out


def speaker_activity_segments_from_asd_paths(
    asd_paths: Sequence[Path],
    *,
    uem_start: float,
    uem_end: float,
    fps: float = 25.0,
    onset: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["onset"],
    offset: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["offset"],
    min_duration_on: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_duration_on"],
    min_duration_off: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_duration_off"],
    max_chunk_size: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["max_chunk_size"],
    min_chunk_size: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_chunk_size"],
) -> List[Tuple[float, float]]:
    all_scores: Dict[int, float] = {}
    for asd_path in sorted(asd_paths):
        for frame, score in parse_asd_scores(asd_path):
            all_scores[int(frame)] = float(score)
    merged_scores = sorted(all_scores.items())
    segments = segment_frames_by_asd(
        merged_scores,
        onset=onset,
        offset=offset,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        fps=fps,
    )
    aligned: List[Tuple[float, float]] = []
    for segment in segments:
        if not segment:
            continue
        seg_start = segment[0] / fps
        seg_end = segment[-1] / fps
        if seg_end < uem_start:
            continue
        if seg_start > uem_end:
            break
        aligned.append((seg_start - uem_start, seg_end - uem_start))
    return aligned


def speaker_activity_segments_for_session(
    session_dir: Path,
    speaker: str,
    *,
    fps: float = 25.0,
    onset: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["onset"],
    offset: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["offset"],
    min_duration_on: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_duration_on"],
    min_duration_off: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_duration_off"],
    max_chunk_size: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["max_chunk_size"],
    min_chunk_size: float = CENTRAL_ASD_SEGMENTATION_DEFAULTS["min_chunk_size"],
) -> List[Tuple[float, float]]:
    uem = uem_for_speaker(session_dir, speaker)
    if uem is None:
        return []
    entries = crop_entries(session_dir, speaker)
    asd_paths = [entry["asd"] for entry in entries if entry.get("asd")]
    return speaker_activity_segments_from_asd_paths(
        asd_paths,
        uem_start=uem[0],
        uem_end=uem[1],
        fps=fps,
        onset=onset,
        offset=offset,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
    )


def make_cluster_mapping(
    session_dir: Path,
    speakers: Sequence[str],
    segments_by_speaker: Dict[str, List[Tuple[float, float]]],
    *,
    mode: str = "overlap",
    bin_seconds: float = 5.0,
    threshold: float = 0.70,
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
    if mode in {"overlap", "official"}:
        speaker_segments = {
            spk: speaker_activity_segments_for_session(session_dir, spk) or segments_by_speaker.get(spk, [])
            for spk in speakers
        }
        return overlap_cluster(speakers, speaker_segments, threshold=threshold)
    if mode == "activity":
        return activity_cluster(speakers, segments_by_speaker, bin_seconds=bin_seconds, threshold=threshold)
    raise ValueError(f"Unknown cluster mode: {mode}")
