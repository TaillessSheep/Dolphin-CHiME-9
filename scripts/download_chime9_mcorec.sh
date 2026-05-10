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
# Zip downloads land here (override with DATA_DIR): repo-root tmp/ (not cwd).
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/../tmp}"

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
  local url="${REPO_URL}/${filename}"
  echo ""
  echo "Downloading ${filename} ..."
  if command -v wget >/dev/null 2>&1; then
    wget --continue \
      --header="Authorization: Bearer ${HF_TOKEN}" \
      "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --continue-at - \
      -H "Authorization: Bearer ${HF_TOKEN}" \
      -o "${filename}" \
      "${url}"
  else
    echo "ERROR: Need wget or curl to download."
    exit 1
  fi
}

unzip_file() {
  local filename="$1"
  echo ""
  echo "Unzipping ${filename}.zip into ${EXTRACT_DIR}/${filename}/ ..."
  unzip -o "${filename}.zip" -d "${EXTRACT_DIR}/${filename}"
  rm -f "${filename}.zip"
  echo "Removed ${filename}.zip"
}

download_split() {
  local split="$1"

  case "${split}" in
    dev)
      # For Dolphin inference, you likely need central video + metadata/preprocessed files.
      download_file "dev_only_central_videos.zip"
      download_file "dev_without_central_videos.zip"

      unzip_file "dev_only_central_videos"
      unzip_file "dev_without_central_videos"
      ;;

    train)
      download_file "train_only_central_videos.zip"
      download_file "train_without_central_videos.zip"

      unzip_file "train_only_central_videos"
      unzip_file "train_without_central_videos"
      ;;

    eval)
      download_file "eval_only_central_videos.zip"
      download_file "eval_without_central_videos.zip"

      unzip_file "eval_only_central_videos"
      unzip_file "eval_without_central_videos"
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
echo "Unpacked dataset lives under (one folder per zip):"
echo "  ${EXTRACT_DIR}/<name>/"
echo "Zip staging directory:"
echo "  ${DATA_DIR}"
echo ""
echo "Example — list central videos:"
echo "  find \"${EXTRACT_DIR}\" -name 'central_video.mp4' | head"