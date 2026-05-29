#!/usr/bin/env bash
set -euo pipefail

URL="${UCF_QNRF_URL:-https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip}"
ROOT_DIR="${1:-data}"
DOWNLOAD_DIR="${ROOT_DIR}/_downloads"
ZIP_PATH="${DOWNLOAD_DIR}/UCF-QNRF_ECCV18.zip"
EXTRACT_DIR="${DOWNLOAD_DIR}/UCF-QNRF_ECCV18_extract"
TARGET_DIR="${ROOT_DIR}/UCF-QNRF_ECCV18"

mkdir -p "${DOWNLOAD_DIR}" "${ROOT_DIR}"

echo "Download URL: ${URL}"
echo "Zip path: ${ZIP_PATH}"

if [ ! -s "${ZIP_PATH}" ]; then
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --continue=true \
      --max-connection-per-server=8 \
      --split=8 \
      --min-split-size=16M \
      --check-certificate=false \
      --dir="${DOWNLOAD_DIR}" \
      --out="$(basename "${ZIP_PATH}")" \
      "${URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl -k -L --fail --continue-at - --output "${ZIP_PATH}" "${URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget --no-check-certificate -c -O "${ZIP_PATH}" "${URL}"
  else
    echo "Install one downloader first: sudo apt-get install -y aria2 curl unzip"
    exit 1
  fi
else
  echo "Zip already exists; skipping download."
fi

if [ ! -s "${ZIP_PATH}" ]; then
  echo "Download failed or produced an empty file: ${ZIP_PATH}"
  exit 1
fi

rm -rf "${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}"
unzip -q "${ZIP_PATH}" -d "${EXTRACT_DIR}"

DATASET_ROOT=""
while IFS= read -r candidate; do
  if [ -d "${candidate}/Train" ] && [ -d "${candidate}/Test" ]; then
    DATASET_ROOT="${candidate}"
    break
  fi
done < <(find "${EXTRACT_DIR}" -type d)

if [ -z "${DATASET_ROOT}" ]; then
  echo "Could not find UCF-QNRF Train/Test folders after extraction."
  echo "Inspect: ${EXTRACT_DIR}"
  exit 1
fi

if [ "${DATASET_ROOT}" != "${TARGET_DIR}" ]; then
  rm -rf "${TARGET_DIR}"
  mkdir -p "$(dirname "${TARGET_DIR}")"
  mv "${DATASET_ROOT}" "${TARGET_DIR}"
fi

echo "Dataset ready: ${TARGET_DIR}"
echo
echo "Validate annotations:"
echo "  python scripts/check_qnrf_annotations.py --data_path ./${TARGET_DIR}"
echo
echo "Train PET on UCF-QNRF:"
echo "  CUDA_VISIBLE_DEVICES=0 python main.py \\"
echo "    --dataset_file QNRF \\"
echo "    --data_path ./${TARGET_DIR} \\"
echo "    --backbone vgg16_bn \\"
echo "    --output_dir vgg16_bn_qnrf_paper \\"
echo "    --epochs 1500 \\"
echo "    --eval_freq 5 \\"
echo "    --batch_size 8 \\"
echo "    --lr 1e-4 \\"
echo "    --lr_backbone 1e-5 \\"
echo "    --lr_scheduler step \\"
echo "    --pet_loss_variant paper \\"
echo "    --eval_max_size 1536 \\"
echo "    --score_threshold 0.5 \\"
echo "    --split_threshold 0.5 \\"
echo "    --seed 42"
