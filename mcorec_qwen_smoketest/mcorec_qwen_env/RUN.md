First, put Dolphin and mcorec_qwen_dolphin_baselines on the same directory.

After that, use conda to create 2 separate environment named dolphin_env (for Dolphin) and qwen_env (for mcorec_qwen_dolphin_baselines)

Replace `save2npz` function inside Dophin project to this one:
```
def save2npz(filename, data=None):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    try:
        payload = np.asarray(data)
    except ValueError:
        payload = np.array(data, dtype=object)
    np.savez_compressed(filename, data=payload)
```

Before running,
```
export HF_TOKEN=YOUR_HF_TOKEN
export TF_USE_LEGACY_KERAS=1
```

RUN!!!
```
python scripts/prepare_dolphin_manifest.py \
  --session_glob "/project/mmsense/MCoRec/data-bin/dev/session_132" \
  --dolphin_repo /home/jqiuar/capstone/Dolphin \
  --out_root runs/dolphin_qwen_smoke \
  --cuda_device 0 \
  --asd_mode official \
  --reuse_dolphin \
  --max_tracks_per_speaker 1 \
  --max_segments_per_speaker 3
```

The result appears inside `$Dolphin/runs/dolphin_qwen_smoke/

Enjoy



