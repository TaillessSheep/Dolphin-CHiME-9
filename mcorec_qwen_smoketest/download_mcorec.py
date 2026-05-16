#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    ap = argparse.ArgumentParser(description="Download gated MCoRec zip files after you have accepted the DUA on Hugging Face.")
    ap.add_argument("--out_dir", default="data-bin", help="Where zip files and extracted folders should go.")
    ap.add_argument(
        "--files",
        nargs="+",
        default=["dev_without_central_videos.zip"],
        help=(
            "MCoRec files to download. Examples: train_without_central_videos.zip, "
            "train_only_central_videos.zip, dev_without_central_videos.zip, dev_only_central_videos.zip"
        ),
    )
    ap.add_argument("--repo_id", default="MCoRecChallenge/MCoRec")
    ap.add_argument("--token_env", default="HF_TOKEN", help="Environment variable containing your new Hugging Face token.")
    ap.add_argument("--extract", action="store_true", help="Unzip after download.")
    args = ap.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"Set {args.token_env} to a valid *new* Hugging Face read token before running this script.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in args.files:
        print(f"Downloading {filename} ...")
        zip_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=filename,
            token=token,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
        zip_path = Path(zip_path)
        print(f"Saved: {zip_path}")
        if args.extract:
            print(f"Extracting {zip_path.name} to {out_dir} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)


if __name__ == "__main__":
    main()
