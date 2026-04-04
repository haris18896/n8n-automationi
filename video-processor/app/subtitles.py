import re
from pathlib import Path
from typing import Iterable

from faster_whisper import WhisperModel

from .config import SubtitleStyleConfig, settings


_model: WhisperModel | None = None


def get_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _model


def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def format_ass_timestamp(seconds: float) -> str:
    safe_seconds = max(0.0, seconds)
    hours = int(safe_seconds // 3600)
    minutes = int((safe_seconds % 3600) // 60)
    secs = int(safe_seconds % 60)
    centis = int(round((safe_seconds - int(safe_seconds)) * 100))
    if centis == 100:
        secs += 1
        centis = 0
    if secs == 60:
        minutes += 1
        secs = 0
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"


def transcribe_segments(
    media_path: str,
    model_name: str,
    device: str,
    compute_type: str,
    language: str = "en",
):
    model = get_model(model_name, device, compute_type)
    segments, info = model.transcribe(
        media_path,
        language=language,
        vad_filter=True,
        word_timestamps=True,
    )
    return list(segments), info


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _ass_escape(value: str) -> str:
    escaped = value.replace("\\", r"\\")
    escaped = escaped.replace("{", r"\{").replace("}", r"\}")
    return escaped.replace("\n", r"\N")


def _tokenize_with_punctuation(value: str) -> list[str]:
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", value, flags=re.UNICODE)


def _normalize_match_token(value: str) -> str:
    return re.sub(r"^[^\w']+|[^\w']+$", "", value.lower())


def style_ass_text(text: str, style: SubtitleStyleConfig) -> str:
    styled_lines: list[str] = []
    for raw_line in text.split(r"\N"):
        tokens = _tokenize_with_punctuation(raw_line)
        if not tokens:
            continue

        styled_parts: list[str] = []
        for token in tokens:
            escaped = _ass_escape(token)
            normalized = _normalize_match_token(token)
            if normalized and normalized in style.highlighted_words:
                escaped = f"{{\\c{style.highlight_color}}}{escaped}{{\\c{style.primary_color}}}"

            if styled_parts and re.match(r"[^\w']", token):
                styled_parts[-1] = f"{styled_parts[-1]}{escaped}"
            else:
                styled_parts.append(escaped)
        styled_lines.append(" ".join(styled_parts))
    return r"\N".join(styled_lines)


def wrap_subtitle_text(text: str, style: SubtitleStyleConfig) -> str:
    words = text.split()
    if not words:
        return ""

    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
        candidate_words = current_line + [word]
        candidate_text = " ".join(candidate_words)
        if current_line and (
            len(candidate_words) > style.max_words_per_line
            or len(candidate_text) > style.max_chars_per_line
        ):
            lines.append(" ".join(current_line))
            current_line = [word]
            if len(lines) >= style.max_lines:
                break
        else:
            current_line = candidate_words

    if current_line and len(lines) < style.max_lines:
        lines.append(" ".join(current_line))

    if not lines:
        lines = [" ".join(words[: style.max_words_per_line])]

    return r"\N".join(lines[: style.max_lines])


def _style_to_ass_line(style: SubtitleStyleConfig) -> str:
    bold_value = -1 if style.bold else 0
    return (
        "Style: Default,"
        f"{style.font_name},{style.font_size},{style.primary_color},"
        f"{style.highlight_color},{style.outline_color},{style.back_color},"
        f"{bold_value},0,0,0,100,100,0,0,1,{style.outline},{style.shadow},"
        f"{style.alignment},{style.margin_l},{style.margin_r},{style.margin_v},1"
    )


def _iter_segment_words(segment) -> list[tuple[float, float, str]]:
    words = getattr(segment, "words", None) or []
    timed_words: list[tuple[float, float, str]] = []
    for word in words:
        text = _clean_text(getattr(word, "word", ""))
        if not text:
            continue
        start = float(getattr(word, "start", getattr(segment, "start", 0.0)))
        end = float(getattr(word, "end", getattr(segment, "end", start)))
        if end <= start:
            end = start + 0.2
        timed_words.append((start, end, text))
    return timed_words


def _chunk_words_for_ass(
    timed_words: list[tuple[float, float, str]], style: SubtitleStyleConfig
) -> list[tuple[float, float, str]]:
    if not timed_words:
        return []

    chunks: list[tuple[float, float, str]] = []
    current_words: list[str] = []
    current_start = timed_words[0][0]
    current_end = timed_words[0][1]

    for start, end, word in timed_words:
        candidate_words = current_words + [word]
        candidate_text = " ".join(candidate_words)
        needs_wrap = current_words and (
            len(candidate_words) > style.max_words_per_line * style.max_lines
            or len(candidate_text) > style.max_chars_per_line * style.max_lines
        )
        if needs_wrap:
            chunks.append((current_start, current_end, " ".join(current_words)))
            current_words = [word]
            current_start = start
            current_end = end
        else:
            current_words = candidate_words
            current_end = end

    if current_words:
        chunks.append((current_start, current_end, " ".join(current_words)))
    return chunks


def write_srt_from_segments(segments: Iterable, srt_output_path: str) -> str:
    transcript_lines: list[str] = []
    with open(srt_output_path, "w", encoding="utf-8") as handle:
        for index, segment in enumerate(segments, start=1):
            text = _clean_text(getattr(segment, "text", ""))
            if not text:
                continue
            transcript_lines.append(text)
            handle.write(f"{index}\n")
            handle.write(
                f"{format_srt_time(float(segment.start))} --> {format_srt_time(float(segment.end))}\n"
            )
            handle.write(f"{text}\n\n")
    return " ".join(transcript_lines)


def write_ass_from_segments(
    segments: Iterable,
    ass_output_path: str,
    style: SubtitleStyleConfig | None = None,
) -> str:
    subtitle_style = style or settings.subtitle_style
    transcript_lines: list[str] = []
    output_path = Path(ass_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events: list[str] = []
    for segment in segments:
        timed_words = _iter_segment_words(segment)
        if timed_words:
            chunks = _chunk_words_for_ass(timed_words, subtitle_style)
        else:
            text = _clean_text(getattr(segment, "text", ""))
            if not text:
                continue
            chunks = [(float(segment.start), float(segment.end), text)]

        for start, end, text in chunks:
            cleaned = _clean_text(text)
            if not cleaned:
                continue
            transcript_lines.append(cleaned)
            wrapped = wrap_subtitle_text(cleaned, subtitle_style)
            styled_text = style_ass_text(wrapped, subtitle_style)
            events.append(
                "Dialogue: 0,"
                f"{format_ass_timestamp(start)},{format_ass_timestamp(end)},Default,,0,0,0,,{styled_text}"
            )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("[Script Info]\n")
        handle.write("ScriptType: v4.00+\n")
        handle.write("ScaledBorderAndShadow: yes\n")
        handle.write("WrapStyle: 0\n")
        handle.write("YCbCr Matrix: TV.709\n\n")

        handle.write("[V4+ Styles]\n")
        handle.write(
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
            "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
            "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        )
        handle.write(f"{_style_to_ass_line(subtitle_style)}\n\n")

        handle.write("[Events]\n")
        handle.write("Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
        for event in events:
            handle.write(f"{event}\n")

    return " ".join(transcript_lines)
