# Dolphin CHiME-9 Baseline

For now, the goal is only to produce a baseline for [Dolphin](https://github.com/JusperLee/Dolphin) using [CHiME-9](https://www.chimechallenge.org/challenges/chime9/workshop). Main metric: latency and WER.

## Setup

This repo is meant to be place in the same path as the Dolphin repo, in parallet. 

```
.
├── Dolphin
└── Dolphin-CHiME-9
```

## Scope (For now)

- Run Dolphin latency evalutation on [CHiME-9 MCoRec](https://huggingface.co/datasets/MCoRecChallenge/MCoRec)
- Evaluate the WER for the resulting sound tracks
- No Dolphin training
- No Dolphin modification
- No full CHiME-9 benchmark reproduction
- No raw CHiME-9 data or output media stored in this repository

## Repository Structure

Dolphin-CHiME-9/
├── benchmark/
│   ├── benchmark_full_model_latency.py
│   └── benchmark_video_encoder_latency.py
├── scripts/
│   └── download_chime9_mcorec.sh
├── docs/
│   └── latency_benchmark.md
└── README.md
