import logging
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from .config import settings


logger = logging.getLogger(__name__)


def get_minio_client() -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


def ensure_bucket(client: Minio, bucket_name: str) -> None:
    if not client.bucket_exists(bucket_name):
        logger.info("Creating MinIO bucket '%s'", bucket_name)
        client.make_bucket(bucket_name)


def object_exists(bucket: str, object_name: str) -> bool:
    client = get_minio_client()
    ensure_bucket(client, bucket)
    try:
        client.stat_object(bucket, object_name)
        return True
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchVersion"}:
            return False
        raise RuntimeError(
            f"Failed to check object existence in MinIO for {bucket}/{object_name}: {exc}"
        ) from exc


def download_file(bucket: str, object_name: str, local_path: str) -> str:
    client = get_minio_client()
    ensure_bucket(client, bucket)
    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.fget_object(bucket, object_name, str(destination))
    except S3Error as exc:
        raise RuntimeError(
            f"Failed to download MinIO object {bucket}/{object_name} to {local_path}: {exc}"
        ) from exc
    return str(destination)


def upload_file(local_path: str, bucket: str, object_name: str) -> str:
    client = get_minio_client()
    ensure_bucket(client, bucket)
    try:
        client.fput_object(bucket, object_name, local_path)
    except S3Error as exc:
        raise RuntimeError(
            f"Failed to upload local file {local_path} to MinIO {bucket}/{object_name}: {exc}"
        ) from exc
    return object_name
