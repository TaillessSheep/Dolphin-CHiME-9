# Dolphin Latency Benchmark Notes

Dolphin paper-reported latency should not be confused with full `Inference.py` runtime.

We report three types of timing:


| Timing type                        | Meaning                                         | Corresponds to paper? |
| ---------------------------------- | ----------------------------------------------- | --------------------- |
| Full-model forward latency         | audio tensor + visual tensor → estimated speech | Approx. Table 4       |
| VideoEncoder / DP-LipCoder latency | mouth ROI tensor → visual features              | Approx. Table 1       |


## Paper-reported Dolphin Latency

From Dolphin Table 4:


| Component               | CPU latency | GPU latency |
| ----------------------- | ----------- | ----------- |
| Full Dolphin AVSS model | 2117.96 ms  | 33.24 ms    |


From Dolphin Table 1:


| Component                              | CPU latency | GPU latency |
| -------------------------------------- | ----------- | ----------- |
| DP-LipCoder / pretrained video encoder | 2117.96 ms  | 23.24 ms    |


The paper reports latency on 1-second audio, but does not release the exact benchmark script or fully specify CPU thread settings, warmup/repeat counts, precision mode, and software versions.

## Benchmark 1: Full Dolphin Model Forward Latency

Script:

```bash
python benchmark/benchmark_full_model_latency.py \
  --gpu-only \
  --warmup 20 \
  --repeat 100
```

Input tensors:
audio:  [1, 16000]
visual: [1, 1, 25, 88, 88]

This corresponds to:

1-second audio at 16 kHz
1-second mouth ROI at 25 FPS
batch size = 1

Measured result:


| Device                | Mean latency | Std      | Notes                  |
| --------------------- | ------------ | -------- | ---------------------- |
| GPU H800 (SuperPod)   | 40.50 ms     | 1.16 ms  | 100 repeats, 20 warmup |
| CPU default threading | 1082.89 ms   | 18.06 ms | 20 repeats, 5 warmup   |


**Paper Table 4:**


| Device | Paper latency |
| ------ | ------------- |
| GPU    | 33.24 ms      |
| CPU    | 2117.96 ms    |


The GPU result is close to the paper-reported latency but not identical. The difference is expected because the paper does not specify all runtime details. The CPU result differs strongly from the paper, likely due to different CPU hardware and PyTorch thread settings.



## Benchmark 2: DP-LipCoder / VideoEncoder Forward Latency

Script:

```bash
python benchmark/benchmark_video_encoder_latency.py \
  --gpu-only \
  --warmup 20 \
  --repeat 100
```

Input tensor:

visual: [1, 1, 25, 88, 88]

Output tensor:

video feature: [1, 25, 3872]

This measures:

model.video_encoder(x)

It excludes:

pre_v1
video_blocks
modal fusion
separator
audio decoder
video decoding
face detection
tracking
mouth crop
file I/O


### Measured result

| Device / setting | Mean latency | Std |
|---|---:|---:|
| GPU H800 | 8.72 ms | 0.10 ms |
| CPU default threading | 347.33 ms | 11.06 ms |
| CPU 1 thread | 5899.02 ms | 94.77 ms |
| CPU 2 threads | 3123.39 ms | 63.12 ms |
| CPU 4 threads | 1680.76 ms | 15.75 ms |
| CPU 8 threads | 896.21 ms | 6.97 ms |
| CPU 16 threads | 621.93 ms | 7.55 ms |



---

## Summary

| Benchmark | Paper CPU | Paper GPU | Ours CPU | Ours GPU |
|---|---:|---:|---:|---:|
| Full Dolphin model, Table 4 approx. | 2117.96 ms | 33.24 ms | 1082.89 ms | 40.50 ms |
| DP-LipCoder / VideoEncoder, Table 1 approx. | 2117.96 ms | 23.24 ms | 347.33 ms default threads | 8.72 ms |

CPU latency varies heavily with thread settings. For VideoEncoder, measured CPU latency ranges from 5899 ms with 1 thread to 622 ms with 16 threads.