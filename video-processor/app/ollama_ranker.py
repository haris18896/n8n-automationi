import json
import requests
from .config import settings

SYSTEM_PROMPT = (
    "You are ranking short-form video transcript segments for virality. "
    "Prefer segments with a strong hook, curiosity, emotional punch, clear standalone meaning, and clean payoff. "
    "Return strict JSON only."
)

def rank_segments_with_ollama(candidates):
    payload_candidates = []
    for idx, c in enumerate(candidates, start=1):
        payload_candidates.append(
            {
                "id": idx,
                "start": round(c["start"], 2),
                "end": round(c["end"], 2),
                "duration": round(c["duration"], 2),
                "text": c["text"],
                "heuristic_score": round(c["score"], 3),
            }
        )

    prompt = {
        "task": "Rank the strongest short-video segments.",
        "rules": [
            "Prefer standalone clarity.",
            "Prefer strong opening lines.",
            "Prefer emotional, surprising, educational, or controversial moments.",
            "Avoid weak setup with no payoff.",
            "Keep top 3 only."
        ],
        "candidates": payload_candidates,
        "response_format": {
            "top_segments": [
                {"id": "integer", "llm_score": "number from 0 to 10", "reason": "short text"}
            ]
        }
    }

    try:
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "system": SYSTEM_PROMPT,
                "prompt": json.dumps(prompt, ensure_ascii=False),
                "stream": False,
                "format": "json",
            },
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "{}")
        parsed = json.loads(text)
        return parsed.get("top_segments", [])
    except Exception:
        return []
