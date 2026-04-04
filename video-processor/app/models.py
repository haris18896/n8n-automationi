from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class ClipJobRequest(BaseModel):
    job_id: str = Field(..., description="Unique job id from Google Sheet")
    mode: Literal["manual", "auto"] = "auto"
    youtube_url: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    desired_duration: Optional[int] = 30
    language: str = "en"
    subtitle_enabled: bool = True
    cta_enabled: bool = True
    watermark_enabled: bool = True
    watermark_text: Optional[str] = None

class CandidateSegment(BaseModel):
    start: float
    end: float
    duration: float
    score: float
    text: str
    reason: Optional[str] = None

class ClipJobResponse(BaseModel):
    success: bool
    job_id: str
    status: str
    output_object: Optional[str] = None
    transcript: Optional[str] = None
    error: Optional[str] = None
    selected_start: Optional[float] = None
    selected_end: Optional[float] = None
    candidate_segments: List[CandidateSegment] = []
