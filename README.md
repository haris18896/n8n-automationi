# AI pipeline (n8n, Ollama, MinIO, Kokoro TTS, host ComfyUI)

This repo orchestrates a small **local AI pipeline** with Docker Compose:

| Service | Role |
|--------|------|
| **n8n** | Workflow automation, webhooks, integrations |
| **Ollama** | Local LLM inference API |
| **MinIO** | S3-compatible object storage |
| **Kokoro (fastkoko)** | Text-to-speech API ([remsky/kokoro-fastapi](https://github.com/remsky/Kokoro-FastAPI) image) |
| **ComfyUI** | **Runs on your host**, not in Compose—containers reach it via `host.docker.internal` |

Data for **n8n**, **Ollama**, and **MinIO** persists in **Docker named volumes** (see below).

---

## Prerequisites

- **Docker Desktop** (or Docker Engine + Compose v2) on macOS, Linux, or Windows with WSL2
- **ComfyUI** installed and running on the host when workflows need it (default: `http://localhost:8188`)
- Optional: enough disk/RAM for Ollama models and Kokoro warmup (first start can take a minute)

---

## One-time setup

### 1. Clone and enter the repo

```bash
cd ai-pipeline
```

### 2. Configure environment

Create a `.env` file in the project root (Compose loads it automatically).  
If you already have one, adjust values as needed.

**Minimum variables to set or review:**

| Variable | Purpose |
|----------|---------|
| `COMPOSE_PROJECT_NAME` | Prefix for container names (e.g. `ai_pipeline`) |
| `N8N_PORT`, `OLLAMA_PORT`, `FASTKOKO_PORT`, `MINIO_*_PORT` | Host ports (avoid conflicts) |
| `N8N_BASIC_AUTH_*`, `N8N_ENCRYPTION_KEY` | n8n login and credential encryption |
| `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` | MinIO admin |
| `COMFYUI_BASE_URL` | URL **as seen from inside containers**—typically `http://host.docker.internal:8188` on Mac/Win Docker Desktop |
| `FASTKOKO_IMAGE` | e.g. `ghcr.io/remsky/kokoro-fastapi-cpu:latest` (CPU). For NVIDIA GPU, use the GPU image and extra runtime config separately |

> **Security:** Do not commit `.env`. It is listed in `.gitignore`. Use strong passwords and a proper `N8N_ENCRYPTION_KEY` in any shared or production-like setup.

### 3. Start ComfyUI on the host (if you use it)

Start your local ComfyUI so it listens on the port referenced by `COMFYUI_BASE_URL` (commonly **8188**).

---

## Start and stop

**Recommended (Makefile):**

```bash
make up      # docker compose up -d --build
make down    # docker compose down
make logs    # follow logs
make ps      # container status
make config  # print merged compose (sanity check)
```

**Or with Docker Compose directly:**

```bash
docker compose up -d --build
docker compose down
```

---

## URLs (default ports from `.env`)

Replace ports if you changed them in `.env`.

| Service | URL | Notes |
|---------|-----|--------|
| **n8n** | `http://localhost:<N8N_PORT>` (default **5678**) | Basic auth if enabled in `.env` |
| **MinIO Console** | `http://localhost:<MINIO_CONSOLE_PORT>` (default **9001**) | Web UI for buckets |
| **MinIO S3 API** | `http://localhost:<MINIO_API_PORT>` (default **9000**) | For SDKs / n8n S3 nodes, not a browser homepage |
| **Ollama** | `http://localhost:<OLLAMA_PORT>` (default **11434**) | HTTP API; pull models with `docker exec -it <ollama_container> ollama pull <model>` |
| **Kokoro (TTS)** | `http://localhost:<FASTKOKO_PORT>` (default **5005**) | Maps to app port **8880** inside the image |
| **Kokoro API docs** | `http://localhost:<FASTKOKO_PORT>/docs` | Swagger UI |
| **Kokoro web player** | `http://localhost:<FASTKOKO_PORT>/web/` | Beta UI (if exposed by the image) |
| **ComfyUI (host)** | `http://localhost:8188` | Open in the browser on the **host**—not the same as `COMFYUI_BASE_URL` inside Docker |

**Internal URLs (from other containers):** Ollama `http://ollama:11434`, MinIO `http://minio:9000`. Use `COMFYUI_BASE_URL` for host ComfyUI from containers.

---

## Persistence

Docker **named volumes** (not wiped by `docker compose down`):

| Volume | Data |
|--------|------|
| `n8n_data` | n8n workflows, credentials, SQLite DB under `/home/node/.n8n` |
| `ollama_data` | Downloaded models |
| `minio_data` | MinIO buckets and objects |

To remove volumes as well (destructive):

```bash
docker compose down -v
```

---

## Troubleshooting

### Port already in use

- **5000 on macOS** is often taken by **AirPlay Receiver** (Control Center). This project maps Kokoro to **`FASTKOKO_PORT`** (e.g. **5005**), not 5000.
- Change conflicting variables in `.env`, then `docker compose up -d`.

### Kokoro / `FASTKOKO_PORT` not reachable

- The **remsky** image serves HTTP on **8880** inside the container; Compose maps **`FASTKOKO_PORT:8880`**. If you change images, confirm the internal port and update `docker-compose.yml` accordingly.
- On first start, the model **warms up**; wait until logs show the server is ready, then open `/docs` or `/web/`.

### ComfyUI from n8n or other containers fails

- Ensure ComfyUI is running on the host and listening on the expected port.
- Keep `COMFYUI_BASE_URL=http://host.docker.internal:<port>` for Docker Desktop (Mac/Windows). On Linux you may need `extra_hosts` or the host gateway IP; adjust if your setup differs.

### Validate compose file

```bash
make config
# or
docker compose config
```

---

## Project layout

```
ai-pipeline/
├── docker-compose.yml   # Services, networks, volumes
├── Makefile             # Shortcuts for compose commands
├── .env                 # Local secrets (create yourself, not committed)
├── fastkoko/            # Legacy Dockerfile (optional); stack uses FASTKOKO_IMAGE by default
└── README.md            # This file
```

---

## License

Add your license here if applicable.
