#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "${ROOT_DIR}/.env" ]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
fi

MINIO_ALIAS="localminio"
MINIO_URL="http://localhost:${MINIO_API_PORT:-9000}"
MINIO_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_PASS="${MINIO_ROOT_PASSWORD:-change_me_minio}"

if ! command -v mc >/dev/null 2>&1; then
  echo "MinIO client 'mc' not found."
  echo "Install it with: brew install minio/stable/mc"
  exit 1
fi

mc alias set "${MINIO_ALIAS}" "${MINIO_URL}" "${MINIO_USER}" "${MINIO_PASS}"
mc mb -p "${MINIO_ALIAS}/raw-videos" || true
mc mb -p "${MINIO_ALIAS}/processed-videos" || true
mc mb -p "${MINIO_ALIAS}/assets" || true

if [ -f "${ROOT_DIR}/assets/cta_like_subscribe.png" ]; then
  mc cp "${ROOT_DIR}/assets/cta_like_subscribe.png" "${MINIO_ALIAS}/assets/cta_like_subscribe.png" || true
fi
if [ -f "${ROOT_DIR}/assets/watermark.png" ]; then
  mc cp "${ROOT_DIR}/assets/watermark.png" "${MINIO_ALIAS}/assets/watermark.png" || true
fi

echo "MinIO buckets ready."
