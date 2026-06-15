#!/usr/bin/env bash
#
# Upload the sft_vlm S3 demo dataset (PNG images + CSV) to your own bucket.
#
# example_data_s3.csv references images as s3://your-bucket/datasets/vlm_images/*.png.
# This script uploads the local PNGs in images/ to your bucket and writes a
# bucket-specific copy of the CSV (with "your-bucket" rewritten to your bucket
# name) to s3://<bucket>/datasets/vlm_train.csv -- the path the
# qwen2_5_vl_3b_lora_s3_file.yaml recipe already expects.
#
# Usage:
#   ./upload_example_to_s3.sh <bucket-name> [key-prefix]
#
# Examples:
#   ./upload_example_to_s3.sh my-vlm-bucket
#   ./upload_example_to_s3.sh my-vlm-bucket team-a/experiments
#
# Requires the AWS CLI with credentials resolvable by boto3/awscli
# (IAM role, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or aws configure).
set -euo pipefail

BUCKET="${1:?Usage: $0 <bucket-name> [key-prefix]}"
PREFIX="${2:-}"
# Normalize prefix: strip leading/trailing slashes, then add a trailing slash if non-empty.
PREFIX="${PREFIX#/}"; PREFIX="${PREFIX%/}"
[ -n "$PREFIX" ] && PREFIX="${PREFIX}/"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGES_DIR="${SCRIPT_DIR}/images"
SRC_CSV="${SCRIPT_DIR}/example_data_s3.csv"

IMAGES_DEST="s3://${BUCKET}/${PREFIX}datasets/vlm_images/"
CSV_DEST="s3://${BUCKET}/${PREFIX}datasets/vlm_train.csv"

echo "Uploading PNG images: ${IMAGES_DIR} -> ${IMAGES_DEST}"
aws s3 cp "${IMAGES_DIR}" "${IMAGES_DEST}" --recursive --exclude "*" --include "*.png"

# Rewrite the placeholder bucket/prefix in the CSV so its s3:// image URIs point
# at the just-uploaded objects, then upload the result as vlm_train.csv.
TMP_CSV="$(mktemp -t vlm_train.XXXXXX.csv)"
trap 'rm -f "${TMP_CSV}"' EXIT
sed "s#s3://your-bucket/datasets/vlm_images/#s3://${BUCKET}/${PREFIX}datasets/vlm_images/#g" \
    "${SRC_CSV}" > "${TMP_CSV}"

echo "Uploading dataset CSV: ${SRC_CSV} -> ${CSV_DEST}"
aws s3 cp "${TMP_CSV}" "${CSV_DEST}"

cat <<EOF

Done. Point the recipe at your uploaded dataset:

  data:
    datasets:
      - name: "${CSV_DEST}"
        image_column: "image_url"
        question_column: "question"
        response_column: "response"

The image_url column now holds s3:// URIs under:
  ${IMAGES_DEST}

If your bucket needs an explicit region or S3-compatible endpoint, also set:
  data:
    image_s3_region: "<your-region>"
    image_s3_endpoint_url: null   # e.g. http://minio:9000 for MinIO
EOF
