# Savarkar GPT — Historical Explorer

A Retrieval-Augmented Generation (RAG) application that answers questions about **Vinayak Damodar Savarkar** and related Indian history, grounded strictly in published biographies and historical records.

Built with a **modular, plug-and-play architecture** — every component (LLM, embedding model, vector database, chunking strategy) can be swapped by changing environment variables alone, with zero code changes.

### Demo Video

[Watch the demo on Google Drive]([https://drive.google.com/file/d/1Tg3d4LqzVDOMPmUyky1DxwaZm8nPY6Wa/view?usp=sharing](https://drive.google.com/file/d/1oEuuwq0uo5RQnBjz7RQumTcsqwchxaE8/view?usp=sharing))

---

## Features

- **RAG Pipeline** — retrieves relevant passages from a vector database and generates grounded, citation-backed answers
- **Historian Persona** — responses read like a knowledgeable, neutral historian explaining facts, not a chatbot
- **6 Source Books** — covers works by Dhananjay Keer, Vikram Sampath, and Savarkar's own writings
- **7,742 Vector Embeddings** — full knowledge base indexed in Pinecone for fast semantic search
- **Web UI** — clean, dark-themed "museum-grade" chat interface with sample question chips and collapsible source citations
- **CLI Mode** — interactive terminal-based Q&A for quick testing
- **Token Usage Monitoring** — every request's input/output tokens and latency are logged to a JSONL file, viewable via CLI or API
- **Fully Configurable** — all models, API keys, and parameters managed via `.env`

---

## Tech Stack

| Component | Technology |
|---|---|
| **Framework** | LangChain |
| **LLM** | Google Gemini 2.5 Flash (configurable) |
| **Embeddings** | `paraphrase-multilingual-mpnet-base-v2` (Sentence Transformers, runs locally) |
| **Vector Database** | Pinecone (serverless) |
| **Web Server** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML / CSS / JavaScript |
| **PDF Processing** | PyMuPDF |

---

## Project Structure

```
Savarkar GPT/
├── app/                          # Backend application modules
│   ├── api/
│   │   └── server.py             # FastAPI server, REST API endpoints
│   ├── chunking/
│   │   └── chunker.py            # Document loading and text splitting
│   ├── config/
│   │   └── settings.py           # Centralized configuration from .env
│   ├── embeddings/
│   │   └── embedder.py           # Sentence-transformer embedding wrapper
│   ├── llm/
│   │   └── gemini.py             # Gemini LLM wrapper with historian prompt
│   ├── pipeline/
│   │   └── ingest.py             # Data ingestion pipeline (chunk → embed → upsert)
│   ├── monitoring/
│   │   └── token_logger.py       # Token usage logging to JSONL (input/output/latency)
│   ├── rag/
│   │   └── chain.py              # RAG orchestration (retrieve → augment → generate)
│   └── vectorstore/
│       └── pinecone_store.py     # Pinecone index management and querying
│
├── frontend/                     # Web UI (served by FastAPI)
│   ├── index.html                # Main page layout
│   ├── css/
│   │   └── style.css             # Dark scholarly theme styling
│   └── js/
│       └── app.js                # Chat logic, API calls, UI interactions
│
├── data/                         # Source PDF books (6 files)
├── json_output/                  # Preprocessed JSON from PDFs (6 books + index)
├── logs/                         # Token usage logs (auto-created, gitignored)
│   └── token_usage.jsonl         # One JSON line per request
├── main.py                       # CLI entry point (serve, ingest, query, stats, usage, config)
├── pdf_to_json.py                # PDF-to-structured-JSON conversion script
├── requirements.txt              # Python dependencies
└── .env                          # API keys and configuration (not committed)
```

---

## Setup

### 1. Clone and create virtual environment

```bash
cd "Savarkar GPT"
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# --- Pinecone Vector Database ---
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=savarkar-gpt
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_METRIC=cosine

# --- Google Gemini LLM ---
GOOGLE_API_KEY=your_google_api_key
LLM_MODEL_NAME=gemini-2.5-flash
LLM_TEMPERATURE=0.1
LLM_MAX_OUTPUT_TOKENS=2048

# --- Embedding Model (runs locally, no API key needed) ---
EMBEDDING_MODEL_NAME=paraphrase-multilingual-mpnet-base-v2
EMBEDDING_DIMENSION=768
EMBEDDING_BATCH_SIZE=64

# --- Chunking ---
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CHUNK_SEPARATORS=\n\n,\n,. , ,

# --- RAG Retrieval ---
TOP_K_RESULTS=5
PINECONE_METADATA_TEXT_LIMIT=1000
PINECONE_UPSERT_BATCH_SIZE=100

# --- Data Path ---
DATA_DIR=json_output
```

### 4. Ingest data (run once)

Convert PDFs to JSON (if not already done):

```bash
python3 pdf_to_json.py
```

Embed and upsert into Pinecone:

```bash
python3 main.py ingest
```

### 5. Launch the web UI

```bash
python3 main.py serve
```

Open **http://localhost:8000** in your browser.

---

## Usage

### Web UI

```bash
python3 main.py serve
```

Visit `http://localhost:8000` — ask questions, click sample chips, and view source citations.

### CLI (Interactive)

```bash
python3 main.py query
```

Type questions in the terminal. Type `quit` to exit.

### Token Usage Monitoring

```bash
python3 main.py usage
```

Shows total requests, token counts (input/output), averages, and the last 10 queries — all logged automatically in the background.

### Other Commands

| Command | Description |
|---|---|
| `python3 main.py serve` | Start web server (FastAPI + frontend) |
| `python3 main.py ingest` | Run data ingestion pipeline |
| `python3 main.py query` | Interactive CLI Q&A |
| `python3 main.py stats` | Show Pinecone index statistics |
| `python3 main.py usage` | Show token usage statistics |
| `python3 main.py config` | Print current configuration |

---

## Plug-and-Play Architecture

Every component is configurable via `.env` — no code changes needed:

| What to change | Environment variable | Example |
|---|---|---|
| Switch LLM | `LLM_MODEL_NAME` | `gemini-2.0-flash`, `gemini-1.5-pro` |
| Change embedding model | `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` |
| Adjust chunk size | `CHUNK_SIZE`, `CHUNK_OVERLAP` | `500`, `100` |
| Change retrieval depth | `TOP_K_RESULTS` | `10` |
| Tune creativity | `LLM_TEMPERATURE` | `0.3` |
| Switch Pinecone index | `PINECONE_INDEX_NAME` | `my-other-index` |

---

## Knowledge Base (Source Books)

| # | Title | Author |
|---|---|---|
| 1 | Six Glorious Epochs of Indian History | V.D. Savarkar |
| 2 | Hindupadpadshahi (English) | V.D. Savarkar |
| 3 | Hindurashtra Darshan | V.D. Savarkar |
| 4 | Savarkar: Echoes from a Forgotten Past, 1883–1924 | Vikram Sampath |
| 5 | Savarkar: A Contested Legacy, 1924–1966 | Vikram Sampath |
| 6 | Veer Savarkar | Dhananjay Keer |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/query` | Send a question, get a grounded answer with sources |
| `GET` | `/api/health` | Health check (model, index, readiness) |
| `GET` | `/api/usage` | Token usage statistics (total, averages, recent requests) |
| `GET` | `/` | Serve the web UI |

### Example API call

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who was Veer Savarkar?"}'
```

Response:

```json
{
  "question": "Who was Veer Savarkar?",
  "answer": "Vinayak Damodar Savarkar, often referred to as...",
  "sources": [
    {
      "title": "Veer Savarkar",
      "author": "Dhananjay Keer",
      "chapter": "Old Age",
      "relevance_score": 0.77
    }
  ],
  "num_passages_retrieved": 5
}
```

---

## License

This project is for educational and research purposes.
