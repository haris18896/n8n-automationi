"""
Extend auto-mode clip end times so the included speech ends on a complete sentence
when possible (may exceed desired_duration).
"""

import re
from typing import Any

_SENTENCE_TAIL = re.compile(r'[.!?…](?:["\'\)\]]+)?\s*$', re.UNICODE)


def _segment_text(segment: Any) -> str:
    return (getattr(segment, "text", None) or "").strip()


def _ordered_segments(segments: list[Any]) -> list[Any]:
    return sorted(segments, key=lambda s: float(getattr(s, "start", 0.0)))


def transcribed_text_in_window(
    segments: list[Any], t_start: float, t_end: float
) -> str:
    parts: list[str] = []
    for seg in _ordered_segments(segments):
        s = float(getattr(seg, "start", 0.0))
        e = float(getattr(seg, "end", s))
        if e <= t_start or s >= t_end:
            continue
        t = _segment_text(seg)
        if t:
            parts.append(t)
    return " ".join(parts).strip()


def _ends_with_sentence_punctuation(text: str) -> bool:
    collapsed = " ".join(text.split()).strip()
    if not collapsed:
        return True
    return bool(_SENTENCE_TAIL.search(collapsed))


def _extend_to_whisper_segment_boundary(segments: list[Any], end: float) -> float:
    """If ``end`` cuts through a Whisper segment, move it to that segment's end."""
    new_end = float(end)
    for seg in _ordered_segments(segments):
        s = float(getattr(seg, "start", 0.0))
        e = float(getattr(seg, "end", s))
        if s < new_end < e - 1e-3:
            new_end = max(new_end, e)
    return new_end


def _smallest_segment_end_strictly_after(segments: list[Any], end: float) -> float | None:
    candidates: list[float] = []
    for seg in _ordered_segments(segments):
        e = float(getattr(seg, "end", 0.0))
        if e > end + 1e-3:
            candidates.append(e)
    return min(candidates) if candidates else None


def extend_clip_end_for_complete_sentence(
    segments: list[Any],
    clip_start: float,
    clip_end: float,
    *,
    max_extra_seconds: float,
    hard_cap_end: float,
) -> tuple[float, str]:
    """
    Push ``clip_end`` forward (never before ``clip_end``) until transcript text in
    [clip_start, end] ends with . ? ! or …, or limits hit.

    Returns (new_end, reason).
    """
    if not segments:
        return float(clip_end), "no_segments"

    start_f = float(clip_start)
    end_f = float(clip_end)
    original = end_f

    end_f = _extend_to_whisper_segment_boundary(segments, end_f)
    boundary_note = ""
    if end_f > original + 0.05:
        boundary_note = "whisper_segment_boundary"

    abs_max = min(float(hard_cap_end), end_f + float(max_extra_seconds))
    abs_max = max(abs_max, end_f)

    while end_f < abs_max - 1e-3:
        text = transcribed_text_in_window(segments, start_f, end_f)
        if _ends_with_sentence_punctuation(text):
            reason = "sentence_terminal"
            if boundary_note:
                reason = f"{reason}+{boundary_note}"
            return end_f, reason

        nxt = _smallest_segment_end_strictly_after(segments, end_f)
        if nxt is None:
            reason = "no_more_segments"
            if boundary_note:
                reason = f"{reason}+{boundary_note}"
            return end_f, reason

        end_f = min(nxt, abs_max)

    text = transcribed_text_in_window(segments, start_f, end_f)
    if _ends_with_sentence_punctuation(text):
        reason = "sentence_terminal_at_cap"
    else:
        reason = "max_extra_or_duration_cap"
    if boundary_note:
        reason = f"{reason}+{boundary_note}"
    return end_f, reason
