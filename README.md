# Multimodal RAG (CLIP + Qdrant Hybrid Search)

A Retrieval-Augmented Generation pipeline that answers questions over a PDF using **both text and images** (charts, figures) as retrievable context — powered by CLIP embeddings, Qdrant hybrid search, NVIDIA reranking, and a vision-capable LLM for final answer generation.

## Architecture

```
PDF
 │
 ├─► Docling (parser) ──► output.md (text) + extracted images (fitz)
 │
 ├─► Chunking (HybridChunker, CLIP tokenizer, 77-token limit)
 │
 ├─► Embedding (CLIP: text + image → shared 512-dim space)
 │
 ├─► Storage (Qdrant: dense CLIP vectors + sparse BM25 vectors)
 │
 ├─► Retrieval (separate text/image dense search, merged)
 │
 ├─► Reranking (NVIDIA llama-nemotron-rerank-vl-1b-v2, text candidates only)
 │
 └─► Answer generation (Ollama qwen2.5vl — reads real chart pixels, not just paths)
```

## Why this design

- **CLIP** embeds text and images into the *same* vector space, so a text query can retrieve relevant charts directly via cosine similarity — no separate image-captioning step required.
- **CLIP's modality gap** means text-to-text similarity scores are structurally higher than text-to-image similarity, even for genuinely relevant images. To prevent images from being drowned out, retrieval runs **two separate searches** (text-only, image-only) and merges the results, rather than one combined ranked search.
- **Reranking is text-only.** The NVIDIA reranker used here scores via `page_content`, so image documents (empty text) are excluded from reranking and passed through untouched instead.
- **Answer generation uses a real vision-language model** (`qwen2.5vl` via Ollama). Earlier versions of this pipeline passed images to the LLM as a file path string — the model could never actually "see" the chart. The current version base64-encodes retrieved images and attaches them as real image content in the prompt, so questions like *"what was the Q3 revenue comparison?"* can be answered by reading bar heights directly off the chart.

## Project structure

```
RAG_CLIP/
├── data/                      # source PDFs
├── output/
│   ├── output.md              # Docling markdown export
│   └── images/                # extracted images (fitz)
├── src/
│   ├── parse.py                # PDF → markdown (Docling)
│   ├── extract_image.py        # PDF → images (PyMuPDF/fitz)
│   ├── chunking.py             # markdown/PDF → chunks (HybridChunker)
│   ├── embedd_and_store.py     # CLIP embeddings + Qdrant storage
│   ├── hybrid_search.py        # retrieval + rerank + vision-LLM answer
│   ├── main.py                 # FastAPI backend
│   └── frontend/
│       └── static/
│           └── index.html      # web UI
├── .env                        # API keys (not committed)
└── pyproject.toml
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management and running scripts.

### 1. Install dependencies

If you already have a `pyproject.toml`, sync the environment:

```bash
uv sync
```

Or, if starting fresh, add dependencies directly:

```bash
uv add torch pillow transformers qdrant-client fastembed python-dotenv pymupdf docling docling-core langchain-core langchain-qdrant langchain-nvidia-ai-endpoints langchain-ollama fastapi uvicorn streamlit
```

> `streamlit` is optional — kept only for the earlier, simpler UI (`app.py`); the primary UI is served via FastAPI (`main.py`).

### 2. Environment variables

Create a `.env` file in the project root:

```
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
NVIDIA_API_KEY=your-nvidia-api-key
```

> **Note:** rotate any API key that has ever been pasted into a screenshot, chat log, or committed to version control.

### 3. Pull the vision-language model (Ollama)

```bash
ollama pull qwen2.5vl:latest
```

Ollama must be running locally for answer generation to work.

## Usage

All scripts are run with `uv run` so they execute inside the project's managed virtual environment.

### Step 1 — Parse the PDF

```bash
uv run src/parse.py
```
Produces `output/output.md`.

### Step 2 — Extract images

```bash
uv run src/extract_image.py
```
Produces images under `output/images/`.

### Step 3 — Chunk, embed, and store

```bash
uv run src/embedd_and_store.py
```
This creates the Qdrant collection (`multimodal_rag`), chunks the document, embeds text and images via CLIP, and upserts everything with hybrid (dense + sparse) vectors.

> Re-running this script deletes and rebuilds the collection from scratch — safe to re-run after any pipeline change.

### Step 4 — Run the API + web UI

```bash
cd src
uv run uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** in a browser, ask a question, and see the answer plus the retrieved text/image sources.

### Optional — Streamlit UI

An earlier, simpler UI is also available:

```bash
uv run streamlit run app.py
```

## API reference

### `POST /ask`

**Request:**
```json
{ "question": "What was the Q3 revenue comparison?" }
```

**Response:**
```json
{
  "answer": "In Q3, FY2025 revenue was approximately $18.5M compared to $13.0M in FY2024.",
  "sources": [
    { "content_type": "text", "text": "..." },
    { "content_type": "image", "image_url": "/images/ACME_Corp_..._img1.png" }
  ]
}
```

### `GET /images/{filename}`

Serves extracted images directly (used by the frontend to render retrieved charts).

## Known limitations

- **Page numbers on text chunks require chunking directly from the PDF**, not the markdown export — markdown has no positional metadata (`prov`), so page numbers will show as `None` if chunking is run against `output.md`.
- **Only 3-image-scale collections have been tested** with the "separate text/image retrieval" approach; at larger scale, image retrieval quality and count (`IMAGE_CANDIDATES`) may need tuning.
- **Sparse (BM25) search is currently only used at storage time**, not in the manual `retrieve_multimodal` retrieval path — dense CLIP similarity is used for both text and image branches at query time.
- **Reranking cannot score images** with the current LangChain `NVIDIARerank` wrapper (it only forwards `page_content`), so images bypass reranking entirely and are always included alongside reranked text.

## Possible next steps

- Route sparse (BM25) search back into the text branch of retrieval, fused via RRF with dense CLIP scores.
- Use NVIDIA's raw REST API (bypassing the LangChain wrapper) to enable genuine vision-aware reranking of image candidates.
- Chunk directly from the source PDF to restore accurate page-number metadata.
- Add multi-document support (currently scoped to a single `doc_id` per run).
