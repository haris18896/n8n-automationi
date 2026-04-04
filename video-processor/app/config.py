import json
import os
from dataclasses import dataclass
from typing import Mapping


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _to_csv_set(value: str | None, default: tuple[str, ...]) -> set[str]:
    if value is None or not value.strip():
        return {item.lower() for item in default}
    return {item.strip().lower() for item in value.split(",") if item.strip()}


@dataclass(frozen=True)
class RenderVideoConfig:
    width: int
    height: int
    mode: str
    codec: str
    preset: str
    crf: int
    pixel_format: str
    audio_codec: str
    audio_bitrate: str
    movflags: str


@dataclass(frozen=True)
class OverlayConfig:
    watermark_size: int
    watermark_margin: int
    cta_width: int
    cta_margin: int


@dataclass(frozen=True)
class SubtitleStyleConfig:
    font_name: str
    font_size: int
    primary_color: str
    highlight_color: str
    outline_color: str
    back_color: str
    bold: bool
    alignment: int
    margin_v: int
    margin_l: int
    margin_r: int
    outline: int
    shadow: int
    max_words_per_line: int
    max_lines: int
    max_chars_per_line: int
    highlighted_words: set[str]


DEFAULT_SUBTITLE_EMPHASIS_COLORS: dict[str, str] = {
    "action": "&H001890FF&",
    "emotional": "&H005564E6&",
    "dramatic": "&H00B469E6&",
    "warning": "&H0000D7FF&",
    "money": "&H0000E676&",
    "curiosity": "&H00FFB84D&",
}


@dataclass(frozen=True)
class SubtitleEmphasisConfig:
    llm_enabled: bool
    timeout_seconds: int
    min_words_for_llm: int
    min_chars_for_llm: int
    category_colors: Mapping[str, str]


def _load_subtitle_emphasis_colors() -> dict[str, str]:
    merged = dict(DEFAULT_SUBTITLE_EMPHASIS_COLORS)
    raw = os.getenv("SUBTITLE_EMPHASIS_CATEGORY_COLORS")
    if not raw or not raw.strip():
        return merged
    try:
        custom = json.loads(raw)
        if isinstance(custom, dict):
            for key, value in custom.items():
                k = str(key).strip().lower()
                if k in DEFAULT_SUBTITLE_EMPHASIS_COLORS and isinstance(value, str):
                    merged[k] = value.strip()
    except json.JSONDecodeError:
        pass
    return merged


class Settings:
    APP_PORT = _to_int(os.getenv("APP_PORT"), 8000)

    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_SECURE = _to_bool(os.getenv("MINIO_SECURE"), False)
    MINIO_BUCKET_RAW = os.getenv("MINIO_BUCKET_RAW", "raw-videos")
    MINIO_BUCKET_OUTPUT = os.getenv("MINIO_BUCKET_OUTPUT", "processed-videos")
    MINIO_BUCKET_ASSETS = os.getenv("MINIO_BUCKET_ASSETS", "assets")

    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
    TRANSCRIBE_DEVICE = os.getenv("TRANSCRIBE_DEVICE", "cpu")
    TRANSCRIBE_COMPUTE_TYPE = os.getenv("TRANSCRIBE_COMPUTE_TYPE", "int8")

    DEFAULT_TARGET_RATIO = os.getenv("DEFAULT_TARGET_RATIO", "9:16")
    DEFAULT_CLIP_MIN_SECONDS = _to_int(os.getenv("DEFAULT_CLIP_MIN_SECONDS"), 20)
    DEFAULT_CLIP_MAX_SECONDS = _to_int(os.getenv("DEFAULT_CLIP_MAX_SECONDS"), 45)
    DEFAULT_TOP_K_SEGMENTS = _to_int(os.getenv("DEFAULT_TOP_K_SEGMENTS"), 5)

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    OLLAMA_TIMEOUT_SECONDS = _to_int(os.getenv("OLLAMA_TIMEOUT_SECONDS"), 120)

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    render_video = RenderVideoConfig(
        width=_to_int(os.getenv("RENDER_WIDTH"), 1080),
        height=_to_int(os.getenv("RENDER_HEIGHT"), 1920),
        mode=os.getenv("RENDER_MODE", "cover_crop"),
        codec=os.getenv("VIDEO_CODEC", "libx264"),
        preset=os.getenv("VIDEO_PRESET", "medium"),
        crf=_to_int(os.getenv("VIDEO_CRF"), 19),
        pixel_format=os.getenv("VIDEO_PIXEL_FORMAT", "yuv420p"),
        audio_codec=os.getenv("AUDIO_CODEC", "aac"),
        audio_bitrate=os.getenv("AUDIO_BITRATE", "192k"),
        movflags=os.getenv("VIDEO_MOVFLAGS", "+faststart"),
    )

    render_overlay = OverlayConfig(
        watermark_size=_to_int(os.getenv("WATERMARK_SIZE"), 128),
        watermark_margin=_to_int(os.getenv("WATERMARK_MARGIN"), 18),
        cta_width=_to_int(os.getenv("CTA_WIDTH"), 200),
        cta_margin=_to_int(os.getenv("CTA_MARGIN"), 18),
    )

    subtitle_style = SubtitleStyleConfig(
        font_name=os.getenv("SUBTITLE_FONT_NAME", "DejaVu Sans"),
        font_size=_to_int(os.getenv("SUBTITLE_FONT_SIZE"), 54),
        primary_color=os.getenv("SUBTITLE_PRIMARY_COLOR", "&H00FFFFFF&"),
        highlight_color=os.getenv("SUBTITLE_HIGHLIGHT_COLOR", "&H0038D8FF&"),
        outline_color=os.getenv("SUBTITLE_OUTLINE_COLOR", "&H00000000&"),
        back_color=os.getenv("SUBTITLE_BACK_COLOR", "&H64000000&"),
        bold=_to_bool(os.getenv("SUBTITLE_BOLD"), True),
        alignment=_to_int(os.getenv("SUBTITLE_ALIGNMENT"), 2),
        margin_v=_to_int(os.getenv("SUBTITLE_MARGIN_V"), 120),
        margin_l=_to_int(os.getenv("SUBTITLE_MARGIN_L"), 80),
        margin_r=_to_int(os.getenv("SUBTITLE_MARGIN_R"), 80),
        outline=_to_int(os.getenv("SUBTITLE_OUTLINE"), 3),
        shadow=_to_int(os.getenv("SUBTITLE_SHADOW"), 0),
        max_words_per_line=_to_int(os.getenv("SUBTITLE_MAX_WORDS_PER_LINE"), 5),
        max_lines=_to_int(os.getenv("SUBTITLE_MAX_LINES"), 1),
        max_chars_per_line=_to_int(os.getenv("SUBTITLE_MAX_CHARS_PER_LINE"), 28),
        highlighted_words=_to_csv_set(
            os.getenv("SUBTITLE_HIGHLIGHT_WORDS"),
            (
                "amazing",
                "attack",
                "boom",
                "crazy",
                "danger",
                "destroy",
                "epic",
                "explodes",
                "fear",
                "fight",
                "finally",
                "hate",
                "huge",
                "immediately",
                "insane",
                "killer",
                "love",
                "massive",
                "must",
                "never",
                "no",
                "power",
                "proof",
                "secret",
                "shock",
                "shocking",
                "stop",
                "truth",
                "urgent",
                "warning",
                "wild",
                "win",
                "worst",
                "yes",
            ),
        ),
    )

    subtitle_emphasis = SubtitleEmphasisConfig(
        llm_enabled=_to_bool(os.getenv("SUBTITLE_LLM_EMPHASIS_ENABLED"), False),
        timeout_seconds=_to_int(os.getenv("SUBTITLE_EMPHASIS_TIMEOUT_SECONDS"), 25),
        min_words_for_llm=_to_int(os.getenv("SUBTITLE_EMPHASIS_MIN_WORDS"), 3),
        min_chars_for_llm=_to_int(os.getenv("SUBTITLE_EMPHASIS_MIN_CHARS"), 10),
        category_colors=_load_subtitle_emphasis_colors(),
    )


settings = Settings()
