# AI Pipeline - YouTube Video Clipper

Local-first YouTube clipper for macOS Apple Silicon with:

- Google Sheets as the control panel
- n8n for orchestration
- Postgres + Redis for durable n8n queue mode
- MinIO for storage
- a local video-processor service for download, clipping, subtitles, overlays, watermarking, and optional auto-clip ranking using Ollama running natively on macOS

## Quick start

1. Install Docker Desktop
2. Install Ollama natively on macOS and pull a model:
   `ollama pull qwen2.5:3b`
3. Install MinIO client:
   `brew install minio/stable/mc`
4. Copy environment file:
   `cp .env.example .env`
5. Put overlay assets into:
   - `assets/cta_like_subscribe.png`
   - `assets/watermark.png`
6. Start:
   `make init`
7. Open:
   - n8n: http://localhost:5678
   - MinIO Console: http://localhost:9001

## Google Sheet columns

job_id
mode
youtube_url
start_time
end_time
desired_duration
language
subtitle_enabled
cta_enabled
watermark_enabled
watermark_text
status
output_object
transcript
selected_start
selected_end
candidate_segments_json
error
created_at
updated_at
