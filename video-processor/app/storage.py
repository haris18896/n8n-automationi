from minio import Minio
from .config import settings

def get_minio_client() -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )

def ensure_bucket(client: Minio, bucket_name: str) -> None:
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

def upload_file(local_path: str, bucket: str, object_name: str) -> str:
    client = get_minio_client()
    ensure_bucket(client, bucket)
    client.fput_object(bucket, object_name, local_path)
    return object_name
