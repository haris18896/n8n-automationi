import os

def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

class Settings:
    APP_PORT = int(os.getenv("APP_PORT", "8000"))
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_SECURE = _to_bool(os.getenv("MINIO_SECURE", "false"))
    MINIO_BUCKET_RAW = os.getenv("MINIO_BUCKET_RAW", "raw-videos")
    MINIO_BUCKET_OUTPUT = os.getenv("MINIO_BUCKET_OUTPUT", "processed-videos")
    MINIO_BUCKET_ASSETS = os.getenv("MINIO_BUCKET_ASSETS", "assets")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
    TRANSCRIBE_DEVICE = os.getenv("TRANSCRIBE_DEVICE", "cpu")
    TRANSCRIBE_COMPUTE_TYPE = os.getenv("TRANSCRIBE_COMPUTE_TYPE", "int8")
    DEFAULT_TARGET_RATIO = os.getenv("DEFAULT_TARGET_RATIO", "9:16")
    DEFAULT_CLIP_MIN_SECONDS = int(os.getenv("DEFAULT_CLIP_MIN_SECONDS", "20"))
    DEFAULT_CLIP_MAX_SECONDS = int(os.getenv("DEFAULT_CLIP_MAX_SECONDS", "45"))
    DEFAULT_TOP_K_SEGMENTS = int(os.getenv("DEFAULT_TOP_K_SEGMENTS", "5"))
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

settings = Settings()
