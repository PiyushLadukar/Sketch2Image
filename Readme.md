<div align="center">

# 🎨 SketchVerse AI
### Software Design Document

**Transform Human Sketches into High-Quality AI Generated Images**
*using Computer Vision, RAG, Diffusion Models, and Local LLMs*

---

**Document Type:** Software Design Document (SDD)
**Version:** 1.0
**Deployment Model:** 100% Local Inference — No Third-Party AI APIs

</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Features](#4-features)
5. [Technology Stack](#5-technology-stack)
6. [System Architecture](#6-system-architecture)
7. [Application Workflow](#7-application-workflow)
8. [AI Pipeline](#8-ai-pipeline)
9. [Algorithms & Models Used](#9-algorithms--models-used)
10. [Folder Structure](#10-folder-structure)
11. [Database Overview](#11-database-overview)
12. [API Overview](#12-api-overview)
13. [Installation](#13-installation)
14. [Future Scope](#14-future-scope)
15. [Conclusion](#15-conclusion)

---

## 1. Project Overview

SketchVerse AI is a **self-hosted, privacy-first generative AI platform** that converts freehand sketches into high-quality images. Users draw on a professional canvas, describe their intent in natural language, and the system combines **computer vision, retrieval-augmented generation (RAG), a local LLM, and ControlNet-guided Stable Diffusion** to produce a structurally faithful, semantically rich final image.

**Key differentiator:** the entire pipeline — vision, retrieval, reasoning, and image generation — runs on **local, open-weight models**. No OpenAI, Stability AI, Replicate, or HuggingFace Inference API is used at any stage.

The product experience combines the feel of Photoshop/Paint (canvas), Canva AI (prompt-driven generation), and ChatGPT (conversational refinement) into a single self-hosted application.

**Why it matters:**

- Sketches communicate spatial intent (layout, proportion, composition) that text prompts alone cannot express precisely.
- A local-only inference stack removes recurring API costs, eliminates data-privacy exposure, and works in air-gapped or compliance-sensitive environments.
- Combining retrieval with generation grounds AI-filled details (materials, lighting conventions, style vocabulary) in curated knowledge rather than model hallucination.

---

## 2. Problem Statement

Existing sketch-to-image tools suffer from recurring limitations:

| Problem | Impact |
|---|---|
| Reliance on third-party inference APIs | Data privacy risk, unpredictable cost, vendor lock-in |
| No retrieval/grounding layer | Generic, detail-poor prompts and outputs |
| Weak structural fidelity (plain img2img) | Sketch geometry is often ignored by the model |
| Single-shot generation | No conversational refinement across iterations |
| Minimal canvas tooling | Most demos offer only one brush, no layers/undo |

**Goal:** Given a sketch `S` and description `T`, generate an image `I` that respects the sketch's structure, satisfies the described intent, and is enriched with retrieved domain knowledge — entirely offline.

This requires solving three distinct sub-problems simultaneously: **structural preservation** (the output must honor the sketch's geometry), **semantic alignment** (the output must match the described intent), and **knowledge grounding** (details the user did not specify must be filled in coherently, not hallucinated). No single off-the-shelf model solves all three — hence the multi-stage pipeline described in Section 8.

---

## 3. Objectives

| Category | Objective |
|---|---|
| **Short-Term** | End-to-end pipeline (canvas → generation) with <8s latency at 768×768 on an RTX 4090-class GPU |
| **Short-Term** | Professional canvas: layers, undo/redo, shapes, selection, fill |
| **Short-Term** | Local RAG pipeline (ChromaDB + BGE) over a curated knowledge base |
| **Short-Term** | Local LLM integration (Llama/Mistral via Ollama) for prompt synthesis |
| **Long-Term** | Region-specific inpainting, multi-image batch generation |
| **Long-Term** | Collaborative multiplayer canvas; animation/video/3D export |
| **Research** | Quantify RAG's impact on prompt-adherence and detail accuracy |
| **Industrial** | Deployable reference architecture for EdTech, game studios, architecture firms, and civic-tech teams with strict data-locality requirements |

---

## 4. Features

| Tier | Features | Notes |
|---|---|---|
| **Core** | Pressure-sensitive brush canvas, layers, undo/redo, prompt panel, one-click generation, per-project generation history | Baseline functionality required for MVP release |
| **Advanced** | Region-specific regeneration (inpainting), style presets, multi-seed batch generation, reference-image style transfer (IP-Adapter) | Enables iterative, non-destructive refinement |
| **AI** | Sketch captioning (VLM), RAG-grounded prompt enrichment, LLM prompt optimization, auto negative-prompt suggestion | Core differentiator vs. plain ControlNet+SD demos |
| **Future** | Voice prompting, multi-view/3D generation, real-time collaboration, mobile app | See Section 14 |
| **Enterprise** | RBAC (Admin/Creator/Viewer), audit logging, air-gapped deployment, SSO (SAML/OIDC) | Targets compliance-sensitive deployments |

- The **Paint Application** supports brush, pencil, eraser, shapes, selection (marquee/lasso), fill (flood-fill), text, color picker, zoom/pan, and keyboard shortcuts — comparable in scope to a lightweight Photoshop/Photopea subset.
- The **Prompt Assistant** maintains conversation memory per project, so refinement instructions ("more dramatic lighting") resolve correctly against prior context instead of being interpreted in isolation.
- **Autosave** persists canvas state every 2 seconds of idle time, minimizing data loss risk during long editing sessions.

---

## 5. Technology Stack

| Layer | Technology | Why Chosen |
|---|---|---|
| **Frontend** | Next.js 14, React 18, TypeScript, Tailwind CSS, Konva.js, Zustand, TanStack Query | SSR for fast first paint, type-safe cross-cutting data shapes, performant canvas scene-graph |
| **Backend** | FastAPI, Pydantic, SQLAlchemy 2.0 (async), Alembic, Celery, structlog | Native async for I/O/GPU-bound workloads, auto OpenAPI schema, strict validation |
| **AI / ML** | PyTorch, `diffusers`, `transformers`, Ollama, `sentence-transformers`, OpenCV, xformers | Mature, open-source, fully local-inference-capable |
| **Data** | PostgreSQL 16, ChromaDB, Redis, MinIO/local FS | Relational integrity + embedded vector search + zero-ops caching/queue |
| **DevOps** | Docker, Docker Compose, NGINX, GitHub Actions, k6 | Reproducible, containerized, CI-tested deployment |
| **Core Models** | SDXL, ControlNet, FLUX (optional), BLIP-2, CLIP/SigLIP, Llama 3.1 / Qwen / Mistral, BGE | Best-in-class open-weight models per pipeline stage (Section 9) |

- **Frontend/backend split** allows the canvas client to be replaced (e.g., a future mobile app) without touching the AI pipeline.
- **Async-first backend** ensures GPU-bound inference never blocks unrelated API traffic.
- **All components are open-source and self-hostable**, consistent with the zero-external-API constraint.

---

## 6. System Architecture

```mermaid
graph TB
    subgraph Client
        UI[Next.js Web App]
        Canvas[Canvas Engine - Konva]
    end
    subgraph Gateway
        NGINX[NGINX Reverse Proxy]
        WS[WebSocket Gateway]
    end
    subgraph Backend[FastAPI Backend]
        API[REST API]
        Auth[Auth Service - JWT]
        Orchestrator[Generation Orchestrator]
    end
    subgraph AIEngine[AI Engine]
        CV[OpenCV Preprocessing]
        VLM[Vision-Language Model]
        Retriever[RAG Retriever]
        LLM[Local LLM]
        Diffusion[SDXL + ControlNet]
    end
    subgraph Storage
        PG[(PostgreSQL)]
        VDB[(ChromaDB)]
        OBJ[(Object Storage)]
        Redis[(Redis)]
    end

    UI --> NGINX --> API
    Canvas --> UI
    NGINX --> WS
    API --> Auth
    API --> Orchestrator --> CV --> VLM --> Retriever --> VDB
    Retriever --> LLM --> Diffusion --> OBJ
    Orchestrator --> Redis
    API --> PG
    Orchestrator --> WS
```

**Design rationale:** Stateless web tier (FastAPI replicas) is decoupled from stateful GPU workers (AI Engine) via a Redis/Celery queue — allowing independent scaling and natural backpressure when GPU demand exceeds capacity.

| Layer | Responsibility |
|---|---|
| Frontend | Canvas rendering, prompt UI, live progress via WebSocket |
| Gateway | TLS termination, load balancing, WebSocket proxying |
| Backend | Auth, validation, request orchestration, persistence |
| AI Engine | CV preprocessing, captioning, retrieval, LLM reasoning, diffusion |
| Storage | Relational data (Postgres), vectors (Chroma), files (object store), cache/queue (Redis) |

**Deployment topology notes:**

- FastAPI replicas are horizontally scalable on standard (CPU-only) instances since they do not hold GPU state.
- A fixed pool of GPU-bound AI Engine workers is scaled independently, sized to available hardware.
- Redis + Celery provide the queue between the two tiers, giving natural backpressure and fair scheduling when generation demand exceeds GPU capacity, instead of failing requests outright.
- NGINX terminates TLS and proxies both REST and WebSocket traffic to backend replicas.

---

## 7. Application Workflow

```mermaid
sequenceDiagram
    actor User
    participant UI
    participant API
    participant AI as AI Engine
    participant DB as Storage

    User->>UI: Open app / login (JWT)
    User->>UI: Draw sketch + write prompt
    UI->>API: POST /generate (sketch + prompt)
    API->>AI: Preprocess → Caption → Retrieve → Build Prompt → Generate
    AI-->>API: Generated image + metadata
    API->>DB: Persist image, prompt, retrieved docs
    API-->>UI: WebSocket progress + final result
    User->>UI: Request regeneration (delta instruction)
    UI->>API: POST /regenerate
    API->>AI: Merge conversation memory + re-generate
    AI-->>API: Updated image
    API-->>UI: Updated result
```

| Stage | Description |
|---|---|
| Authentication | JWT login; refresh token via httpOnly cookie |
| Canvas | Vector stroke capture, rasterized to PNG at generation time |
| Preprocessing | Denoise, threshold, edge-extract sketch |
| Vision Analysis | VLM captions sketch content |
| RAG | Retrieve relevant knowledge-base chunks |
| Prompt Builder | LLM synthesizes final positive/negative prompt |
| Generation | ControlNet + SDXL produce the image |
| Post-Processing | Optional upscaling/face-restoration |
| Saving & History | Full metadata persisted; browsable generation gallery |
| Regeneration | Delta instructions merged with conversation memory |

---

## 8. AI Pipeline

```mermaid
flowchart LR
    A[Sketch PNG] --> B[OpenCV Preprocessing]
    B --> C[Edge/Feature Extraction]
    C --> D[Vision-Language Captioning]
    D --> E[RAG Retrieval]
    E --> F[LLM Prompt Synthesis]
    F --> G[ControlNet Conditioning]
    G --> H[Stable Diffusion XL]
    H --> I[Post-Processing]
    I --> J[Final Image]
```

**Computer Vision Preprocessing** (OpenCV): median blur (denoise) → Gaussian blur → Otsu thresholding → Canny edge detection → contour detection → skeletonization → resize/letterbox to model resolution. This produces a clean structural edge map for ControlNet conditioning.

**Vision-Language Captioning:** BLIP-2 generates a structural caption of the sketch (subject, composition), disambiguating what the strokes represent and feeding both the LLM and RAG query.

**RAG Retrieval:** Query (user prompt + caption) is embedded via BGE and matched against a ChromaDB knowledge base (style guides, lighting, materials, common artifacts) using cosine similarity + optional cross-encoder reranking.

**Prompt Synthesis:** A local LLM merges (user prompt, caption, retrieved knowledge, style preset, conversation memory) into a structured positive prompt and an auto-generated negative prompt.

**Generation:** ControlNet-conditioned SDXL performs iterative denoising, guided by both the text prompt (via cross-attention) and the sketch edge map (via the ControlNet branch), producing the final image.

**Post-Processing:** Optional Real-ESRGAN upscaling and GFPGAN face restoration (for portrait sketches) are applied, followed by format conversion (PNG for storage, WebP for delivery).

**Key tunable parameters:**

| Parameter | Default | Effect |
|---|---|---|
| CFG Scale | 6.5–7.5 | Strength of prompt adherence |
| ControlNet Conditioning Scale | 0.6–1.0 | Strength of structural fidelity to sketch |
| Inference Steps | 25–30 | Quality vs. latency trade-off |
| Scheduler | DPM++ 2M Karras | Fewer steps for comparable quality vs. Euler |

---

## 9. Algorithms & Models Used

Each model below was selected specifically for its role in the pipeline; only open-weight, locally-runnable checkpoints are used — no hosted inference APIs.

**OpenCV — Classical Preprocessing (Canny, Otsu, Morphology).** Raw sketches are denoised (median/Gaussian blur), binarized (Otsu's automatic thresholding), and reduced to a clean structural edge map via Canny edge detection and skeletonization. This edge map is the exact conditioning signal ControlNet consumes, so its quality directly determines how faithfully the final image respects the user's drawn geometry.

**CLIP.** CLIP is the native text encoder embedded inside SDXL — it converts the final synthesized prompt into the embeddings the diffusion UNet cross-attends to at every denoising step. Standalone, CLIP is also used to compute a prompt-image similarity score for automated quality monitoring of generated outputs.

**SigLIP.** A faster CLIP-style dual encoder used for the "similar sketches" search feature and to speed up the sketch-captioning stage. It was chosen over plain CLIP for this sub-task because it offers comparable accuracy with lower inference latency, which matters since it runs synchronously in the UI.

**BLIP-2.** Generates the structural caption of a sketch during the Vision Analysis stage (e.g., "a two-story house with a triangular roof"). It connects a frozen vision encoder to a frozen LLM through a lightweight Q-Former, keeping inference cost low enough to run on every generation request without materially adding to latency.

**ControlNet.** The mechanism that makes the sketch's geometry actually influence the output. A duplicated, trainable copy of the SDXL UNet's encoder processes the sketch edge map and injects spatially-aligned feature offsets into the frozen base model via zero-initialized convolutions — steering structure without degrading the base model's generative quality or prompt-following ability.

**Stable Diffusion XL (SDXL).** The primary image-synthesis engine. A latent diffusion model with dual text encoders (CLIP + OpenCLIP) that denoises within a compressed latent space rather than raw pixels, making 1024×1024 generation tractable on consumer/workstation GPUs. Chosen over SD 1.5 for stronger prompt adherence and image quality, and over FLUX as the *default* due to its more mature ControlNet/LoRA ecosystem and lower VRAM requirement.

**FLUX (optional).** A newer Diffusion-Transformer model using rectified-flow matching, offered as an opt-in "Pro Quality" mode for users with 24GB+ VRAM who want higher fidelity on complex, multi-subject prompts at the cost of slower generation.

**Llama 3.1 / Qwen / Mistral (served via Ollama).** Local LLMs handle prompt synthesis, negative-prompt generation, and multi-turn conversational refinement (e.g., "make the lighting warmer"). Llama 3.1 8B Instruct is the default for its strong instruction-following at a VRAM footprint that fits alongside the diffusion model; Qwen is offered for multilingual prompt workflows; Mistral 7B serves as a lightweight fallback for lower-VRAM deployments.

**Sentence-Transformers + BGE Embeddings.** BGE (`bge-large-en-v1.5`) embeds both knowledge-base documents and user queries into a shared vector space for RAG retrieval. It was selected as the strongest fully-local, open-weight embedding model available — avoiding any dependency on a hosted embedding API (e.g., OpenAI or Cohere embeddings).

**Vector Similarity Search (Cosine Similarity + HNSW).** RAG retrieval ranks knowledge-base chunks by cosine similarity between query and document embeddings. ChromaDB indexes these embeddings using HNSW (Hierarchical Navigable Small World) graphs, giving approximate nearest-neighbor search in logarithmic rather than linear time as the knowledge base scales.

**Latent Diffusion & Classifier-Free Guidance (CFG).** The core generative process: a VAE compresses images into a compact latent space, and a UNet iteratively denoises a latent through a learned reverse process, conditioned jointly on the text prompt (cross-attention) and sketch structure (ControlNet). At each step, Classifier-Free Guidance blends conditioned and unconditioned noise predictions to strengthen prompt adherence; SketchVerse AI defaults to a CFG scale of 6.5–7.5, balancing fidelity against oversaturation artifacts seen at higher values.

---

## 10. Folder Structure

```
sketchverse-ai/
├── frontend/          # Next.js app (canvas, prompt UI, history)
│   ├── app/            # Routes (auth, dashboard, canvas)
│   ├── components/      # Canvas, prompt, history, shared UI
│   ├── hooks/ store/ lib/
├── backend/           # FastAPI app
│   ├── app/routers/      # auth, projects, generation, history
│   ├── app/services/     # Business logic layer
│   ├── app/models/ schemas/ core/
│   ├── alembic/          # DB migrations
├── ai_engine/          # Framework-agnostic AI package
│   ├── cv/ vision/ rag/ llm/ diffusion/
│   ├── orchestrator.py  celery_worker.py
├── knowledge_base/      # RAG source docs + ingestion script
├── infra/              # Docker, NGINX, k8s configs
├── scripts/             # Model download, migration, reindex
├── docker-compose.yml
└── README.md
```

**Design rationale:** `ai_engine/` is decoupled from FastAPI so GPU workers scale independently of the web tier (Celery-based); `frontend/` communicates only via the documented API contract, keeping it swappable/replaceable.

---

## 11. Database Overview

```mermaid
erDiagram
    USERS ||--o{ PROJECTS : owns
    PROJECTS ||--o{ CANVAS_STATES : contains
    PROJECTS ||--o{ GENERATIONS : contains
    GENERATIONS ||--o{ GENERATED_IMAGES : produces
    GENERATIONS }o--|| PROMPTS : uses
    GENERATIONS ||--o{ RETRIEVED_DOCS : cites

    USERS { uuid id PK, string email, string role }
    PROJECTS { uuid id PK, uuid user_id FK, jsonb canvas_metadata }
    CANVAS_STATES { uuid id PK, uuid project_id FK, jsonb layer_data, int version }
    PROMPTS { uuid id PK, text final_prompt, text negative_prompt, jsonb conversation_memory }
    GENERATIONS { uuid id PK, uuid project_id FK, int seed, float cfg_scale, string status }
    GENERATED_IMAGES { uuid id PK, uuid generation_id FK, string object_store_path }
    RETRIEVED_DOCS { uuid id PK, uuid generation_id FK, string chunk_id, float similarity_score }
```

**PostgreSQL** is the system of record (JSONB support for flexible layer/prompt metadata, strong full-text search). **ChromaDB** stores the RAG vector index separately. Every generation is fully traceable: sketch → prompt lineage → retrieved documents → model parameters → output image — supporting the project's audit/reproducibility requirement.

| Table | Purpose |
|---|---|
| `users` | Credentials (bcrypt-hashed), role, account metadata |
| `projects` | Top-level workspace container per user |
| `canvas_states` | Versioned layer/stroke snapshots (autosave) |
| `prompts` | Raw input, LLM-optimized final prompt, negative prompt, conversation memory |
| `generations` | Seed, CFG scale, ControlNet scale, model checkpoint, status |
| `generated_images` | Object-storage path and image metadata |
| `retrieved_docs` | Knowledge-base chunks cited per generation, with similarity scores |
| `sessions` | Hashed refresh tokens, enabling server-side revocation |

---

## 12. API Overview

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/auth/register` | POST | Create user account |
| `/api/v1/auth/login` | POST | Authenticate, issue JWT + refresh cookie |
| `/api/v1/auth/refresh` | POST | Rotate access token |
| `/api/v1/projects` | POST | Create project |
| `/api/v1/projects/{id}` | GET | Fetch project + canvas state |
| `/api/v1/projects/{id}/canvas` | PUT | Autosave canvas (optimistic concurrency) |
| `/api/v1/projects/{id}/generate` | POST | Trigger AI generation (async, WebSocket progress) |
| `/api/v1/projects/{id}/regenerate` | POST | Regenerate with delta instruction |
| `/api/v1/generations/{id}` | GET | Fetch generation + prompt lineage |
| `/api/v1/projects/{id}/history` | GET | Paginated generation history |

All responses use a consistent error envelope (`error.code`, `error.message`, `error.request_id`). Authenticated routes require `Authorization: Bearer <jwt>`. Generation is asynchronous — the POST returns `202 Accepted` and results stream via WebSocket.

**Security controls:** JWT (15-min access tokens), RBAC (Admin/Creator/Viewer), Redis-backed rate limiting, strict input validation (Pydantic), prompt-injection heuristics on LLM-bound text, CORS allowlisting, and file/MIME validation on sketch uploads.

**Sample — Generation Request/Response:**

```json
// POST /api/v1/projects/{id}/generate
{
  "sketch_png_base64": "...",
  "prompt": "a knight in medieval armor, dramatic lighting",
  "style_id": "photorealistic",
  "controlnet_scale": 0.8
}

// 202 Accepted
{ "generation_id": "gen_8f2a1c", "status": "queued" }
```

**Standard error envelope:**

```json
{ "error": { "code": "PROMPT_TOO_LONG", "message": "...", "request_id": "req_8f2a1c" } }
```

---

## 13. Installation

**Prerequisites:** NVIDIA GPU (≥12GB VRAM recommended), Docker + Docker Compose, NVIDIA Container Toolkit, ~40GB disk for model weights.

```bash
git clone https://github.com/your-org/sketchverse-ai.git
cd sketchverse-ai
cp .env.example .env              # configure secrets
./scripts/download_models.sh      # downloads SDXL, ControlNet, LLM, BGE weights
docker compose up --build
```

- Frontend: `http://localhost:3000` | API: `http://localhost:8000`
- Native (non-Docker) setup: create a Python 3.11 venv for `backend/` and `ai_engine/`, run `npm install` for `frontend/`, and install Ollama for local LLM serving.
- Windows: Docker Desktop with WSL2 backend recommended for GPU passthrough.
- macOS: diffusion inference runs via PyTorch's MPS backend on Apple Silicon (lower throughput than CUDA).

**Key environment variables:**

```
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/sketchverse
REDIS_URL=redis://redis:6379/0
JWT_SECRET=<generate a long random secret>
LLM_BACKEND=ollama
LLM_MODEL_NAME=llama3.1:8b-instruct
CUDA_VISIBLE_DEVICES=0
```

**Verify installation:**

```bash
curl http://localhost:8000/api/v1/health
# {"status": "ok", "gpu_available": true, "models_loaded": [...]}
```

---

## 14. Future Scope

| Roadmap Item | Description |
|---|---|
| In-canvas inpainting/outpainting | Selective regeneration directly on generated results, not just the original sketch |
| Multi-image batch generation | Coherent image sets (e.g., multiple angles of the same design) via shared seeds |
| Animation & video generation | AnimateDiff-style motion modules for short looping clips from a still |
| Image-to-3D export | Convert a generated image into a usable 3D mesh for game/product pipelines |
| Real-time collaborative canvas | CRDT-based (Yjs) multiplayer sketching with live presence |
| Voice-driven prompting | Local Whisper.cpp speech-to-text dictation |
| Native mobile app | React Native client reusing the existing REST/WebSocket contract |
| Agentic generation mode | LLM autonomously adjusts CFG/ControlNet scale and retries toward a stated goal |

---

## 15. Conclusion

SketchVerse AI shows that a production-grade generative AI application — canvas editor, computer vision, RAG, local LLM reasoning, and ControlNet-guided diffusion — can be built and operated **entirely on self-controlled infrastructure**, with no dependency on third-party inference APIs. Every design choice balances technical soundness, operational simplicity, and full generation traceability. The result is a reference architecture suitable for teams that need modern generative AI capabilities without surrendering data control.

---

*End of Document*