import json
import logging
import re
from typing import Any

import requests

from .config import settings

logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = frozenset(
    {"action", "emotional", "dramatic", "warning", "money", "curiosity"}
)

_SYSTEM = (
    "You label emphasis words for short vertical-video subtitles. "
    "Reply with strict JSON only: no markdown fences, no commentary."
)

_TASK = """
Task: Pick up to 3 words from that line worth emphasizing.

Rules:
- At most 3 words in "highlights".
- Each "word" must appear exactly in the subtitle line (same spelling; case-insensitive match).
- If no word is strong enough, return "highlights": [].
- Each "category" must be exactly one of: action, emotional, dramatic, warning, money, curiosity.
  - action: verbs, movement, imperative force
  - emotional: feelings, reactions
  - dramatic: stakes, reveals, intensity
  - warning: danger, risk, alerts
  - money: price, wealth, cost
  - curiosity: hooks, questions, mystery
- Return strict JSON only with exactly this structure:
{"highlights":[{"word":"string","category":"string"}]}
""".strip()


def _normalize_word(value: str) -> str:
    return re.sub(r"^[^\w']+|[^\w']+$", "", (value or "").lower())


def line_qualifies_for_llm(line: str) -> bool:
    cfg = settings.subtitle_emphasis
    cleaned = " ".join((line or "").split())
    if not cleaned:
        return False
    words = len(cleaned.split())
    return words >= cfg.min_words_for_llm and len(cleaned) >= cfg.min_chars_for_llm


def _whole_word_in_line(line_lower: str, word_lower: str) -> bool:
    if not word_lower:
        return False
    pattern = r"(?<![\w'])" + re.escape(word_lower) + r"(?![\w'])"
    return re.search(pattern, line_lower) is not None


def _parse_ollama_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed: Any = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _validate_highlights(line: str, highlights: Any) -> list[dict[str, str]]:
    if not isinstance(highlights, list):
        logger.warning(
            "subtitle emphasis: invalid JSON shape (highlights not a list): %r",
            highlights,
        )
        return []

    line_lower = line.lower()
    result: list[dict[str, str]] = []
    seen_norm: set[str] = set()

    for item in highlights:
        if len(result) >= 3:
            break
        if not isinstance(item, dict):
            continue
        word = item.get("word")
        category = item.get("category")
        if not isinstance(word, str) or not isinstance(category, str):
            continue
        word = word.strip()
        category = category.strip().lower()
        if not word or category not in ALLOWED_CATEGORIES:
            if word and category not in ALLOWED_CATEGORIES:
                logger.info(
                    "subtitle emphasis: dropped invalid category %r for word %r",
                    category,
                    word,
                )
            continue
        if not _whole_word_in_line(line_lower, word.lower()):
            logger.info(
                "subtitle emphasis: dropped word not found verbatim in line: %r",
                word,
            )
            continue
        norm = _normalize_word(word)
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        result.append({"word": word, "category": category})

    return result


def extract_emphasis_for_line(line: str) -> list[dict[str, str]]:
    """
    Returns up to 3 validated {word, category} dicts for one subtitle line.
    On any failure, returns [] (caller uses heuristic styling).
    """
    cfg = settings.subtitle_emphasis
    if not cfg.llm_enabled:
        return []

    cleaned = " ".join((line or "").split())
    if not cleaned:
        return []

    if not line_qualifies_for_llm(cleaned):
        logger.debug(
            "subtitle emphasis: skip LLM (short line) words=%s chars=%s",
            len(cleaned.split()),
            len(cleaned),
        )
        return []

    prompt = (
        "Subtitle line (verbatim): "
        + json.dumps(cleaned, ensure_ascii=False)
        + "\n\n"
        + _TASK
    )

    try:
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "system": _SYSTEM,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=cfg.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        raw = data.get("response", "")
        if not isinstance(raw, str):
            logger.warning(
                "subtitle emphasis: invalid Ollama response type for 'response' field"
            )
            return []
        parsed = _parse_ollama_json(raw)
        if parsed is None:
            logger.warning(
                "subtitle emphasis: invalid Ollama JSON (parse failed): %.200s",
                raw,
            )
            return []
        validated = _validate_highlights(cleaned, parsed.get("highlights"))
        if validated:
            logger.info(
                "subtitle emphasis: success (%s highlights) for line %.80s%s",
                len(validated),
                cleaned,
                "…" if len(cleaned) > 80 else "",
            )
        else:
            logger.info(
                "subtitle emphasis: model returned no usable highlights for line %.80s%s",
                cleaned,
                "…" if len(cleaned) > 80 else "",
            )
        return validated
    except requests.Timeout:
        logger.warning("subtitle emphasis: Ollama timeout; heuristic fallback")
        return []
    except requests.RequestException as exc:
        logger.warning(
            "subtitle emphasis: Ollama request error %s; heuristic fallback", exc
        )
        return []
    except Exception as exc:
        logger.warning(
            "subtitle emphasis: unexpected error %s; heuristic fallback", exc
        )
        return []
