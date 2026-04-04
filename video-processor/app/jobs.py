import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from .config import settings
from .clip_completion import extend_clip_end_for_complete_sentence
from .ffmpeg_utils import (
    burn_subtitles_and_overlays,
    detect_scene_cuts,
    download_video,
    get_media_duration_seconds,
    get_render_transform_mode,
    trim_video,
)
from .models import CandidateSegment
from .ollama_ranker import rank_segments_with_ollama
from .reframing import CropPlan, analyze_subject_aware_crop
from .storage import download_file, object_exists, upload_file
from .subtitles import transcribe_segments, write_ass_from_segments


logger = logging.getLogger(__name__)


@contextmanager
def log_timing(stage: str, job_id: str):
    started_at = time.perf_counter()
    logger.info("Job %s started stage '%s'", job_id, stage)
    try:
        yield
    finally:
        logger.info(
            "Job %s finished stage '%s' in %.2fs",
            job_id,
            stage,
            time.perf_counter() - started_at,
        )


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
    hook_words = [
        "why",
        "how",
        "secret",
        "mistake",
        "never",
        "truth",
        "crazy",
        "worst",
        "best",
        "nobody",
        "everyone",
        "stop",
        "don't",
        "imagine",
        "here's",
        "listen",
        "problem",
        "million",
        "money",
        "warning",
        "important",
        "shocking",
        "insane",
    ]
    emotional_words = [
        "love",
        "hate",
        "angry",
        "scared",
        "afraid",
        "embarrassed",
        "surprised",
        "excited",
        "sad",
        "pain",
        "failure",
        "success",
    ]
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
    for word in hook_words:
        if word in text_lower:
            score += 0.18
    for word in emotional_words:
        if word in text_lower:
            score += 0.12
    if any(
        phrase in text_lower
        for phrase in ["let me show you", "here's why", "the reason is", "what happened was"]
    ):
        score += 0.8
    if any(phrase in text_lower for phrase in ["um ", "uh ", "you know", "like i said"]):
        score -= 0.4
    return round(score, 3)


def build_candidate_segments(transcript_segments, min_seconds: int, max_seconds: int):
    candidates = []
    segment_count = len(transcript_segments)
    for i in range(segment_count):
        current_texts = []
        start = transcript_segments[i].start
        end = transcript_segments[i].end
        for j in range(i, segment_count):
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
            candidates.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "duration": float(duration),
                    "score": float(score),
                    "text": text,
                    "reason": "heuristic",
                }
            )
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def snap_to_nearest_scene(start: float, end: float, scene_cuts: list[float]):
    if not scene_cuts:
        return start, end

    def nearest_before(value: float) -> float:
        previous = [cut for cut in scene_cuts if cut <= value]
        return previous[-1] if previous else value

    def nearest_after(value: float) -> float:
        following = [cut for cut in scene_cuts if cut >= value]
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
            llm_map[seg_id - 1] = {
                "llm_score": float(item.get("llm_score", 0)),
                "reason": str(item.get("reason", "")).strip() or "llm",
            }
    rescored = []
    for index, candidate in enumerate(top_candidates):
        llm = llm_map.get(index)
        final_score = candidate["score"]
        reason = candidate.get("reason", "heuristic")
        if llm:
            final_score += llm["llm_score"] * 0.35
            reason = llm["reason"]
        merged = dict(candidate)
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


def _candidate_models(items: Iterable[dict]) -> list[CandidateSegment]:
    return [
        CandidateSegment(
            start=float(item["start"]),
            end=float(item["end"]),
            duration=float(item["duration"]),
            score=float(item["score"]),
            text=str(item["text"]),
            reason=str(item.get("reason", "")),
        )
        for item in items
    ]


def _resolve_optional_asset(path: str, enabled: bool, label: str) -> str | None:
    asset_path = Path(path)
    if not enabled:
        return None
    if asset_path.exists():
        return str(asset_path)
    logger.warning("Optional %s asset not found at %s", label, path)
    return None


def _raw_video_object_name(job_id: str) -> str:
    return f"{job_id}/source.mp4"


def _final_video_object_name(job_id: str) -> str:
    return f"{job_id}/final.mp4"


def _ensure_source_video(job_id: str, youtube_url: str, source_video: Path) -> str:
    raw_object = _raw_video_object_name(job_id)
    if object_exists(settings.MINIO_BUCKET_RAW, raw_object):
        logger.info(
            "Job %s reusing existing raw video from MinIO: %s/%s",
            job_id,
            settings.MINIO_BUCKET_RAW,
            raw_object,
        )
        download_file(settings.MINIO_BUCKET_RAW, raw_object, str(source_video))
        return "minio"

    logger.info("Job %s downloading source video from YouTube: %s", job_id, youtube_url)
    download_video(youtube_url, str(source_video))
    logger.info(
        "Job %s uploading newly downloaded raw video to MinIO: %s/%s",
        job_id,
        settings.MINIO_BUCKET_RAW,
        raw_object,
    )
    upload_file(str(source_video), settings.MINIO_BUCKET_RAW, raw_object)
    return "youtube"


def _build_crop_plan(clip_video: Path, render_mode: str, job_id: str) -> CropPlan | None:
    if render_mode != "cover_crop":
        logger.info(
            "Job %s skipping subject-aware crop because render_mode=%s",
            job_id,
            render_mode,
        )
        return None

    if not settings.subject_aware_crop.enabled:
        logger.info("Job %s subject-aware crop disabled by config", job_id)
        return None

    try:
        with log_timing("subject_aware_crop", job_id):
            crop_plan = analyze_subject_aware_crop(str(clip_video))
        logger.info(
            "Job %s subject-aware crop ready with %s keyframes and methods=%s",
            job_id,
            len(crop_plan.keyframes),
            crop_plan.method_counts,
        )
        return crop_plan
    except Exception as exc:
        logger.warning(
            "Job %s subject-aware crop failed, falling back to static center crop: %s",
            job_id,
            exc,
        )
        return None


def process_clip_job(payload) -> dict:
    job_id = payload.job_id
    workdir = Path(f"/tmp/video-processor/{job_id}")
    workdir.mkdir(parents=True, exist_ok=True)

    source_video = workdir / "source.mp4"
    clip_video = workdir / "clip.mp4"
    final_video = workdir / "final.mp4"
    clip_ass = workdir / "clip_subtitles.ass"

    cta_asset = "/app/assets/cta_like_subscribe.png"
    watermark_asset = "/app/assets/watermark.png"

    render_mode = get_render_transform_mode(getattr(payload, "render_mode", None))
    logger.info("Job %s started with mode=%s render_mode=%s", job_id, payload.mode, render_mode)

    with log_timing("source_prepare", job_id):
        source_origin = _ensure_source_video(job_id, payload.youtube_url, source_video)

    selected_start = None
    selected_end = None
    candidate_segments: list[CandidateSegment] = []

    if payload.mode == "manual":
        if not payload.start_time or not payload.end_time:
            raise RuntimeError("Manual mode requires both start_time and end_time.")
        selected_start = hhmmss_to_seconds(payload.start_time)
        selected_end = hhmmss_to_seconds(payload.end_time)
        logger.info(
            "Job %s selected manual clip range start=%s end=%s",
            job_id,
            payload.start_time,
            payload.end_time,
        )
        with log_timing("clip_trim_manual", job_id):
            trim_video(
                str(source_video),
                str(clip_video),
                payload.start_time,
                payload.end_time,
            )
    else:
        with log_timing("transcribe_source", job_id):
            transcript_segments, _ = transcribe_segments(
                str(source_video),
                model_name=settings.WHISPER_MODEL,
                device=settings.TRANSCRIBE_DEVICE,
                compute_type=settings.TRANSCRIBE_COMPUTE_TYPE,
                language=payload.language,
            )

        min_seconds = max(10, min(settings.DEFAULT_CLIP_MIN_SECONDS, int(payload.desired_duration or 30)))
        max_seconds = max(
            min_seconds,
            min(settings.DEFAULT_CLIP_MAX_SECONDS, int((payload.desired_duration or 30) + 15)),
        )
        logger.info(
            "Job %s auto clip search min_seconds=%s max_seconds=%s",
            job_id,
            min_seconds,
            max_seconds,
        )
        candidates = build_candidate_segments(
            transcript_segments,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
        )
        best, top_ranked = choose_best_segment(candidates)
        with log_timing("scene_detection", job_id):
            scene_cuts = detect_scene_cuts(str(source_video), threshold=0.35)
        snapped_start, snapped_end = snap_to_nearest_scene(best["start"], best["end"], scene_cuts)
        selected_start = float(snapped_start)
        selected_end = float(snapped_end)
        if settings.AUTO_CLIP_SENTENCE_COMPLETION_ENABLED:
            probe_duration = get_media_duration_seconds(str(source_video))
            last_speech = max(
                (float(getattr(s, "end", 0.0)) for s in transcript_segments),
                default=selected_end,
            )
            hard_cap = probe_duration if probe_duration is not None else last_speech + 5.0
            hard_cap = max(hard_cap, selected_end)
            extended_end, completion_reason = extend_clip_end_for_complete_sentence(
                transcript_segments,
                selected_start,
                selected_end,
                max_extra_seconds=float(settings.AUTO_CLIP_COMPLETION_MAX_EXTRA_SECONDS),
                hard_cap_end=hard_cap,
            )
            if extended_end > selected_end + 0.05:
                logger.info(
                    "Job %s extended clip end for complete sentence: %s -> %s (+%.2fs) [%s]",
                    job_id,
                    seconds_to_hhmmss(selected_end),
                    seconds_to_hhmmss(extended_end),
                    extended_end - selected_end,
                    completion_reason,
                )
            else:
                logger.info(
                    "Job %s clip end unchanged after sentence pass [%s]",
                    job_id,
                    completion_reason,
                )
            selected_end = extended_end
        logger.info(
            "Job %s selected auto clip range start=%s end=%s score=%.3f reason=%s",
            job_id,
            seconds_to_hhmmss(selected_start),
            seconds_to_hhmmss(selected_end),
            best["score"],
            best.get("reason", "heuristic"),
        )
        with log_timing("clip_trim_auto", job_id):
            trim_video(
                str(source_video),
                str(clip_video),
                seconds_to_hhmmss(selected_start),
                seconds_to_hhmmss(selected_end),
            )
        candidate_segments = _candidate_models(top_ranked)

    clip_transcript = None
    subtitle_path: str | None = None
    with log_timing("transcribe_clip", job_id):
        clip_segments, _ = transcribe_segments(
            str(clip_video),
            model_name=settings.WHISPER_MODEL,
            device=settings.TRANSCRIBE_DEVICE,
            compute_type=settings.TRANSCRIBE_COMPUTE_TYPE,
            language=payload.language,
        )
    if payload.subtitle_enabled:
        with log_timing("subtitle_ass_generate", job_id):
            clip_transcript = write_ass_from_segments(clip_segments, str(clip_ass))
        subtitle_path = str(clip_ass)
        logger.info("Job %s generated subtitle file at %s", job_id, subtitle_path)
    else:
        clip_transcript = " ".join(segment_to_text(segment) for segment in clip_segments).strip()

    crop_plan = _build_crop_plan(clip_video, render_mode, job_id)

    with log_timing("final_render", job_id):
        burn_subtitles_and_overlays(
            input_path=str(clip_video),
            output_path=str(final_video),
            ass_path=subtitle_path,
            cta_path=_resolve_optional_asset(cta_asset, payload.cta_enabled, "cta"),
            watermark_path=_resolve_optional_asset(
                watermark_asset,
                payload.watermark_enabled,
                "watermark",
            ),
            render_mode=render_mode,
            crop_plan=crop_plan,
        )

    object_name = _final_video_object_name(job_id)
    with log_timing("upload_output", job_id):
        upload_file(str(final_video), settings.MINIO_BUCKET_OUTPUT, object_name)
    logger.info(
        "Job %s completed; source_origin=%s final_output=%s/%s",
        job_id,
        source_origin,
        settings.MINIO_BUCKET_OUTPUT,
        object_name,
    )

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
