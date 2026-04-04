import logging

from fastapi import FastAPI

from .config import settings
from .jobs import process_clip_job
from .models import ClipJobRequest, ClipJobResponse

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="YouTube Video Clipper", version="1.1.0")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/process", response_model=ClipJobResponse)
def process(job: ClipJobRequest):
    try:
        result = process_clip_job(job)
        return ClipJobResponse(**result)
    except Exception as e:
        return ClipJobResponse(
            success=False,
            job_id=job.job_id,
            status="failed",
            output_object=None,
            transcript=None,
            error=str(e),
            selected_start=None,
            selected_end=None,
            candidate_segments=[],
        )
