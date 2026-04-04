from pathlib import Path
from typing import List
from .config import settings
from .ffmpeg_utils import burn_subtitles_and_overlays, detect_scene_cuts, download_video, trim_video
from .models import CandidateSegment
from .ollama_ranker import rank_segments_with_ollama
from .storage import upload_file
from .subtitles import transcribe_segments, write_srt_from_segments

def seconds_to_hhmmss(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"

def segment_to_text(segment) -> str:
    return (segment.text or "").strip()

def heuristic_score(text: str, duration: float, start: float) -> float:
    text_lower = text.lower()
    hook_words = ["why","how","secret","mistake","never","truth","crazy","worst","best","nobody","everyone","stop","don't","imagine","here's","listen","problem","million","money","warning","important","shocking","insane"]
    emotional_words = ["love","hate","angry","scared","afraid","embarrassed","surprised","excited","sad","pain","failure","success"]
    score = 0.0
    word_count = len(text.split())
    if 25 <= word_count <= 90:
        score += 1.2
    elif 15 <= word_count <= 120:
        score += 0.6
    if duration <= 45:
        score += 0.8
    if duration <= 35:
        score += 0.4
    if start < 120:
        score += 0.4
    if "?" in text:
        score += 0.5
    for w in hook_words:
        if w in text_lower:
            score += 0.18
    for w in emotional_words:
        if w in text_lower:
            score += 0.12
    if any(p in text_lower for p in ["let me show you", "here's why", "the reason is", "what happened was"]):
        score += 0.8
    if any(p in text_lower for p in ["um ", "uh ", "you know", "like i said"]):
        score -= 0.4
    return round(score, 3)

def build_candidate_segments(transcript_segments, min_seconds: int, max_seconds: int):
    candidates = []
    n = len(transcript_segments)
    for i in range(n):
        current_texts = []
        start = transcript_segments[i].start
        end = transcript_segments[i].end
        for j in range(i, n):
            seg = transcript_segments[j]
            current_texts.append(segment_to_text(seg))
            end = seg.end
            duration = end - start
            if duration < min_seconds:
                continue
            if duration > max_seconds:
                break
            text = " ".join([t for t in current_texts if t]).strip()
            if not text:
                continue
            score = heuristic_score(text=text, duration=duration, start=start)
            candidates.append({"start": float(start), "end": float(end), "duration": float(duration), "score": float(score), "text": text, "reason": "heuristic"})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

def snap_to_nearest_scene(start: float, end: float, scene_cuts: List[float]):
    if not scene_cuts:
        return start, end
    def nearest_before(value):
        previous = [x for x in scene_cuts if x <= value]
        return previous[-1] if previous else value
    def nearest_after(value):
        following = [x for x in scene_cuts if x >= value]
        return following[0] if following else value
    new_start = nearest_before(start)
    new_end = nearest_after(end)
    if new_end <= new_start:
        return start, end
    return new_start, new_end

def choose_best_segment(candidates):
    if not candidates:
        raise RuntimeError("No candidate segments could be generated from the transcript.")
    top_candidates = candidates[: settings.DEFAULT_TOP_K_SEGMENTS]
    llm_ranking = rank_segments_with_ollama(top_candidates)
    llm_map = {}
    for item in llm_ranking:
        seg_id = item.get("id")
        if isinstance(seg_id, int) and 1 <= seg_id <= len(top_candidates):
            llm_map[seg_id - 1] = {"llm_score": float(item.get("llm_score", 0)), "reason": str(item.get("reason", "")).strip() or "llm"}
    rescored = []
    for idx, c in enumerate(top_candidates):
        llm = llm_map.get(idx)
        final_score = c["score"]
        reason = c.get("reason", "heuristic")
        if llm:
            final_score += llm["llm_score"] * 0.35
            reason = llm["reason"]
        merged = dict(c)
        merged["score"] = round(final_score, 3)
        merged["reason"] = reason
        rescored.append(merged)
    rescored.sort(key=lambda x: x["score"], reverse=True)
    return rescored[0], rescored

def hhmmss_to_seconds(value: str) -> float:
    parts = [int(x) for x in value.split(":")]
    if len(parts) != 3:
        raise RuntimeError(f"Invalid time format: {value}")
    return float(parts[0] * 3600 + parts[1] * 60 + parts[2])

def process_clip_job(payload) -> dict:
    job_id = payload.job_id
    workdir = Path(f"/tmp/video-processor/{job_id}")
    workdir.mkdir(parents=True, exist_ok=True)
    source_video = workdir / "source.mp4"
    clip_video = workdir / "clip.mp4"
    final_video = workdir / "final.mp4"
    source_srt = workdir / "source_subtitles.srt"
    clip_srt = workdir / "clip_subtitles.srt"
    cta_asset = "/app/assets/cta_like_subscribe.png"
    watermark_asset = "/app/assets/watermark.png"

    download_video(payload.youtube_url, str(source_video))
    upload_file(str(source_video), settings.MINIO_BUCKET_RAW, f"{job_id}/source.mp4")

    selected_start = None
    selected_end = None
    candidate_segments = []
    transcript = None

    if payload.mode == "manual":
        if not payload.start_time or not payload.end_time:
            raise RuntimeError("Manual mode requires both start_time and end_time.")
        trim_video(str(source_video), str(clip_video), payload.start_time, payload.end_time)
        selected_start = hhmmss_to_seconds(payload.start_time)
        selected_end = hhmmss_to_seconds(payload.end_time)
    else:
        transcript_segments, _ = transcribe_segments(
            str(source_video),
            model_name=settings.WHISPER_MODEL,
            device=settings.TRANSCRIBE_DEVICE,
            compute_type=settings.TRANSCRIBE_COMPUTE_TYPE,
            language=payload.language,
        )
        transcript = write_srt_from_segments(transcript_segments, str(source_srt))
        min_seconds = max(10, min(settings.DEFAULT_CLIP_MIN_SECONDS, int(payload.desired_duration or 30)))
        max_seconds = max(min_seconds, min(settings.DEFAULT_CLIP_MAX_SECONDS, int((payload.desired_duration or 30) + 15)))
        candidates = build_candidate_segments(transcript_segments, min_seconds=min_seconds, max_seconds=max_seconds)
        best, top_ranked = choose_best_segment(candidates)
        scene_cuts = detect_scene_cuts(str(source_video), threshold=0.35)
        snapped_start, snapped_end = snap_to_nearest_scene(best["start"], best["end"], scene_cuts)
        selected_start = float(snapped_start)
        selected_end = float(snapped_end)
        trim_video(str(source_video), str(clip_video), seconds_to_hhmmss(selected_start), seconds_to_hhmmss(selected_end))
        candidate_segments = [
            CandidateSegment(start=float(item["start"]), end=float(item["end"]), duration=float(item["duration"]), score=float(item["score"]), text=str(item["text"]), reason=str(item.get("reason", "")))
            for item in top_ranked
        ]

    clip_segments, _ = transcribe_segments(
        str(clip_video),
        model_name=settings.WHISPER_MODEL,
        device=settings.TRANSCRIBE_DEVICE,
        compute_type=settings.TRANSCRIBE_COMPUTE_TYPE,
        language=payload.language,
    )
    clip_transcript = write_srt_from_segments(clip_segments, str(clip_srt))
    burn_subtitles_and_overlays(
        input_path=str(clip_video),
        output_path=str(final_video),
        srt_path=str(clip_srt) if payload.subtitle_enabled else None,
        cta_path=cta_asset if payload.cta_enabled and Path(cta_asset).exists() else None,
        watermark_path=watermark_asset if payload.watermark_enabled and Path(watermark_asset).exists() else None,
    )
    object_name = f"{job_id}/final.mp4"
    upload_file(str(final_video), settings.MINIO_BUCKET_OUTPUT, object_name)
    return {
        "success": True,
        "job_id": job_id,
        "status": "done",
        "output_object": object_name,
        "transcript": clip_transcript,
        "error": None,
        "selected_start": selected_start,
        "selected_end": selected_end,
        "candidate_segments": candidate_segments,
    }
