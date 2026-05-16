# MCoRec Qwen3-ASR and Dolphin baselines

This is a small runner package for two practical MCoRec baselines:

1. **Qwen3-ASR track-audio baseline**: use each MCoRec speaker crop track's audio directly, optionally gated by `track_xx_asd.json`, and transcribe with Qwen3-ASR.
2. **Dolphin → Qwen3-ASR baseline**: run Dolphin on each MCoRec speaker crop track as an audio-visual target speech extraction front end, then transcribe the separated audio with Qwen3-ASR.

Both systems write MCoRec-compatible output folders containing:

```text
session_xxx/output_name/
├── speaker_to_cluster.json
├── spk_0.vtt
├── spk_1.vtt
└── ...
```

The official MCoRec evaluator can then compute speaker WER, clustering F1, and joint ASR-clustering error.

## Security first

Do not hard-code Hugging Face tokens into scripts. Export a fresh read token in your shell:

```bash
export HF_TOKEN="hf_your_new_read_token_here"
```

If a token was pasted into chat, revoke it and create a new one before using these scripts.

## 0. Clone the official repos

```bash
git clone https://github.com/MCoRec/mcorec_baseline.git
git clone https://github.com/JusperLee/Dolphin.git
```

## 1. Download MCoRec data

You must first request access and accept the MCoRec Data Use Agreement on Hugging Face.

```bash
python -m venv .venv-download
source .venv-download/bin/activate
pip install huggingface_hub

python scripts/download_mcorec.py \
  --out_dir data-bin \
  --files dev_without_central_videos.zip \
  --extract
```

`dev_without_central_videos.zip` is enough for the crop-track based scripts here. Download `dev_only_central_videos.zip` too only if you plan to process the raw central 360-degree video.

## 2. Baseline A: Qwen3-ASR on MCoRec crop-track audio

Create an isolated Qwen ASR environment. The Qwen docs recommend Python 3.12, but recent Python 3.11 environments often work as well.

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
conda install ffmpeg -y
pip install -U qwen-asr torch torchaudio huggingface_hub
```

Prepare a segment manifest from the MCoRec crop tracks:

```bash
python scripts/prepare_qwen_track_manifest.py \
  --session_glob "data-bin/dev/*" \
  --out_root runs/qwen3_track \
  --asd_mode official
```

Transcribe it and write `output_qwen3_track` into each session folder:

```bash
python scripts/transcribe_manifest_qwen3_asr.py \
  --manifest runs/qwen3_track/manifest.jsonl \
  --model_id Qwen/Qwen3-ASR-0.6B \
  --output_dir_name output_qwen3_track \
  --language auto \
  --cluster_mode overlap
```

For a stronger but heavier ASR backend, switch to:

```bash
--model_id Qwen/Qwen3-ASR-1.7B
```

## 3. Baseline B: Dolphin front end, then Qwen3-ASR

Create a Dolphin environment and install the Dolphin repo dependencies:

```bash
conda create -n dolphin python=3.11 -y
conda activate dolphin
conda install ffmpeg -y
cd Dolphin
pip install torch torchvision torchaudio
pip install -r requirements.txt
cd ..
```

Run Dolphin on each target speaker crop track. Start with one session or small limits for a smoke test.

```bash
conda activate dolphin
python scripts/prepare_dolphin_manifest.py \
  --session_glob "data-bin/dev/session_132" \
  --dolphin_repo ./Dolphin \
  --out_root runs/dolphin_qwen_smoke \
  --cuda_device 0 \
  --asd_mode official \
  --reuse_dolphin \
  --max_tracks_per_speaker 1 \
  --max_segments_per_speaker 3
```

Full dev run:

```bash
python scripts/prepare_dolphin_manifest.py \
  --session_glob "data-bin/dev/*" \
  --dolphin_repo ./Dolphin \
  --out_root runs/dolphin_qwen \
  --cuda_device 0 \
  --asd_mode official \
  --reuse_dolphin
```

Then transcribe the Dolphin-separated segments in the Qwen environment:

```bash
conda activate qwen3-asr
python scripts/transcribe_manifest_qwen3_asr.py \
  --manifest runs/dolphin_qwen/manifest.jsonl \
  --model_id Qwen/Qwen3-ASR-0.6B \
  --output_dir_name output_dolphin_qwen \
  --language auto \
  --cluster_mode overlap
```

## 4. Official MCoRec evaluation

Install the official baseline repo according to its README, then call the evaluator:

```bash
conda activate mcorec   # or whatever env you use for mcorec_baseline
python scripts/evaluate_mcorec_official.py \
  --mcorec_baseline_repo ./mcorec_baseline \
  --session_dir "data-bin/dev/*" \
  --output_dir_name output_qwen3_track \
  --log_path runs/qwen3_track/eval.log \
  --summary_json runs/qwen3_track/eval_summary.json

python scripts/evaluate_mcorec_official.py \
  --mcorec_baseline_repo ./mcorec_baseline \
  --session_dir "data-bin/dev/*" \
  --output_dir_name output_dolphin_qwen \
  --log_path runs/dolphin_qwen/eval.log \
  --summary_json runs/dolphin_qwen/eval_summary.json
```

Turn the prepare/transcribe/eval sidecars into a compact benchmark summary:

```bash
python scripts/summarize_benchmarks.py \
  --baseline_name baseline_a_qwen3_track \
  --prepare_stats runs/qwen3_track/prepare_stats.json \
  --transcribe_timing runs/qwen3_track/manifest.timing.json \
  --evaluation_summary runs/qwen3_track/eval_summary.json

python scripts/summarize_benchmarks.py \
  --baseline_name baseline_b_dolphin_qwen \
  --prepare_stats runs/dolphin_qwen/prepare_stats.json \
  --transcribe_timing runs/dolphin_qwen/manifest.timing.json \
  --evaluation_summary runs/dolphin_qwen/eval_summary.json
```

The recommended `--cluster_mode overlap` follows the official baseline's ASD-overlap clustering logic much more closely than the old placeholder `activity` mode. For fair ASR-only ablations on dev, you can still use `--cluster_mode copy_dev`, but do not report that as a challenge-style end-to-end system because it copies ground-truth clustering.

## 5. Optional train-only proxy separation metric

MCoRec dev/eval do not provide clean per-speaker waveforms, so signal-level separation metrics are not official there. On train only, this script compares separated segments to the speaker's close-talk ego/lapel audio as a rough proxy SI-SDR sanity check:

```bash
python scripts/evaluate_train_proxy_sisdr.py \
  --manifest runs/dolphin_qwen_train/manifest.jsonl
```

Treat this as a proxy, not a publication-quality source-separation score.

## Notes

- The scripts are intentionally modular because Qwen3-ASR, Dolphin, and the official MCoRec baseline can have conflicting dependency requirements.
- `prepare_qwen_track_manifest.py` and `prepare_dolphin_manifest.py` now default to `--asd_mode official`, which uses hysteresis thresholds compatible with the official MCoRec baseline (`onset=1.0`, `offset=0.8`) instead of treating ASD scores as `[0, 1]` probabilities. Pass `--asd_mode legacy` only if you explicitly want the old single-threshold behavior.
- `--language auto` is recommended for Qwen3-ASR unless you know the dataset language in advance.
- `prepare_qwen_track_manifest.py` writes `prepare_stats.json`; `prepare_dolphin_manifest.py` also writes `dolphin_track_report.jsonl` and `dolphin_failures.jsonl`.
- `transcribe_manifest_qwen3_asr.py` writes `<manifest>.transcripts.jsonl` and `<manifest>.timing.json`.
- `prepare_dolphin_manifest.py` will retry Dolphin with a trimmed input if the upstream model fails with a first-frame face-detection error. The retry offset is recorded in the manifest and track report.
- If Dolphin-separated audio duration differs too much from the expected crop-track duration, the track is skipped and reported instead of silently producing misaligned segments.

## Suggested HPC smoke test

Before launching a full dev run on `hpc4`, validate the pipeline on one session and inspect the sidecars:

1. Run `prepare_dolphin_manifest.py` on one session such as `session_132`.
2. Inspect `runs/dolphin_qwen_smoke/prepare_stats.json`, `dolphin_track_report.jsonl`, and `dolphin_failures.jsonl`.
3. Run `transcribe_manifest_qwen3_asr.py` and inspect `manifest.transcripts.jsonl` plus `manifest.timing.json`.
4. Run `evaluate_mcorec_official.py --summary_json ...` and confirm WER/F1 are in a reasonable range before scaling to all dev sessions.
