#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Download CHiME-9 Task 1 MCoRec
# For Dolphin inference baseline
# ==============================
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   bash download_chime9_mcorec.sh dev
#   bash download_chime9_mcorec.sh train
#   bash download_chime9_mcorec.sh eval
#   bash download_chime9_mcorec.sh all
#
# Recommended first:
#   bash download_chime9_mcorec.sh dev
#
# Notes:
# - This is a gated HuggingFace dataset.
# - You must request/accept access first:
#   https://huggingface.co/datasets/MCoRecChallenge/MCoRec
# - For Dolphin inference, central videos are required.

SPLIT="${1:-dev}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Unpacked dataset layout lives under repo ../data (override with EXTRACT_DIR).
EXTRACT_DIR="${EXTRACT_DIR:-${SCRIPT_DIR}/../data}"

DATA_DIR="${DATA_DIR:-data/chime9_mcorec}"
REPO_URL="https://huggingface.co/datasets/MCoRecChallenge/MCoRec/resolve/main"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "Run:"
  echo "  export HF_TOKEN=hf_xxx"
  exit 1
fi

mkdir -p "${DATA_DIR}"
mkdir -p "${EXTRACT_DIR}"
cd "${DATA_DIR}"

download_file() {
  local filename="$1"
  echo ""
  echo "Downloading ${filename} ..."
  wget --continue \
    --header="Authorization: Bearer ${HF_TOKEN}" \
    "${REPO_URL}/${filename}"
}

unzip_file() {
  local filename="$1"
  echo ""
  echo "Unzipping ${filename} into ${EXTRACT_DIR} ..."
  unzip -o "${filename}" -d "${EXTRACT_DIR}"
  rm -f "${filename}"
  echo "Removed ${filename}"
}

download_split() {
  local split="$1"

  case "${split}" in
    dev)
      # For Dolphin inference, you likely need central video + metadata/preprocessed files.
      download_file "dev_only_central_videos.zip"
      download_file "dev_without_central_videos.zip"

      unzip_file "dev_only_central_videos.zip"
      unzip_file "dev_without_central_videos.zip"
      ;;

    train)
      download_file "train_only_central_videos.zip"
      download_file "train_without_central_videos.zip"

      unzip_file "train_only_central_videos.zip"
      unzip_file "train_without_central_videos.zip"
      ;;

    eval)
      download_file "eval_only_central_videos.zip"
      download_file "eval_without_central_videos.zip"

      unzip_file "eval_only_central_videos.zip"
      unzip_file "eval_without_central_videos.zip"
      ;;

    all)
      download_split dev
      download_split train
      download_split eval
      ;;

    *)
      echo "ERROR: Unknown split: ${split}"
      echo "Allowed: dev | train | eval | all"
      exit 1
      ;;
  esac
}

download_split "${SPLIT}"

echo ""
echo "Download complete."
echo "Extracted to:"
echo "  ${EXTRACT_DIR}"
echo "Zip staging directory:"
echo "  $(pwd)"
echo ""
echo "Check sessions with:"
echo "  find $(pwd) -maxdepth 3 -name 'central_video.mp4' | head"