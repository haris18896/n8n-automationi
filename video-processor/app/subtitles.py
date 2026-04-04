from faster_whisper import WhisperModel

_model = None

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

def transcribe_segments(media_path: str, model_name: str, device: str, compute_type: str, language: str = "en"):
    model = get_model(model_name, device, compute_type)
    segments, info = model.transcribe(
        media_path,
        language=language,
        vad_filter=True,
        word_timestamps=True,
    )
    return list(segments), info

def write_srt_from_segments(segments, srt_output_path: str) -> str:
    transcript_lines = []
    with open(srt_output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_srt_time(seg.start)
            end = format_srt_time(seg.end)
            text = (seg.text or "").strip()
            if not text:
                continue
            transcript_lines.append(text)
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
    return " ".join(transcript_lines)
