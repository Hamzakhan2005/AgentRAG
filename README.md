<div align="center">

# 🧠 Agentic RagBot

### A self-correcting, agentic RAG chatbot built with LangGraph

Chat with your documents — powered by a multi-step agent that **routes**, **retrieves**, **grades its own retrieval**, **rewrites bad queries**, **falls back to live web search**, and **checks itself for hallucinations** before answering.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-1C3C3C?style=flat-square)](https://www.langchain.com/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?style=flat-square)](https://groq.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B6B?style=flat-square)](https://www.trychroma.com/)

[Live Demo](https://agentic-rag-bot-beryl.vercel.app) · [Report Bug](../../issues) · [Request Feature](../../issues)

</div>

---

## 📖 Table of Contents

- [What Makes This Different](#-what-makes-this-different)
- [Architecture](#-architecture)
- [How the Agent Thinks](#-how-the-agent-thinks)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Known Limitations](#-known-limitations--roadmap)
- [License](#-license)

---

## 🚀 What Makes This Different

Most RAG chatbots follow one fixed path: **retrieve → generate**, with zero self-checking. If the retriever pulls irrelevant chunks, the model confidently hallucinates an answer anyway.

**Agentic RagBot doesn't trust its first attempt.** It's built as a graph of specialized nodes, each responsible for one decision, so the pipeline can *detect* a bad retrieval and *correct* it before the user ever sees a wrong answer:

| Basic RAG | Agentic RagBot |
|---|---|
| Retrieve once, generate | Retrieves, **grades relevance**, retries with a rewritten query if the grade fails |
| No fallback if retrieval fails | Falls back to **live web search** after failed retrieval attempts |
| No idea if the answer is grounded | **Checks its own output** for hallucinations against the source documents |
| One-size-fits-all query handling | **Routes** each query to the right strategy — documents, history, or the web |

---

## 🏗️ Architecture

```
                              ┌──────────────┐
                              │   User Query  │
                              └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │  Router Node  │   Classifies query as:
                              │   (LLM call)  │   rag / direct / websearch
                              └──────┬───────┘
                     ┌───────────────┼────────────────┐
                     │               │                │
              ┌──────▼──────┐ ┌──────▼──────┐  ┌───────▼────────┐
              │  Retriever   │ │Direct Answer│  │   Web Search    │
              │ (ChromaDB)   │ │ (history/   │  │    (Tavily)     │
              │              │ │  general    │  │                 │
              └──────┬──────┘ │  knowledge) │  └───────┬────────┘
                     │        └──────┬──────┘          │
              ┌──────▼──────┐        │                 │
              │    Grader    │        │                 │
              │ (per-chunk   │        │                 │
              │  relevance)  │        │                 │
              └──────┬──────┘        │                 │
                     │                │                 │
         ┌───────────┴───────────┐    │                 │
         │                       │    │                 │
   docs relevant?          no relevant docs              │
         │                (retry < 2x)                   │
         │                       │                       │
         │                ┌──────▼──────┐                │
         │                │  Rewriter    │                │
         │                │ (reformulate │                │
         │                │  query) ─────┼──► back to Retriever
         │                └─────────────┘                │
         │                (after 2 retries, fall through) │
         │                       │                       │
         └───────────────┬───────┴───────────────────────┘
                          │
                   ┌──────▼──────┐
                   │  Generator   │  Produces the answer
                   │              │  from gathered context
                   └──────┬──────┘
                          │
                   ┌──────▼──────────┐
                   │Hallucination Chk │  Flags unsupported
                   │  (LLM cross-     │  claims before
                   │   check vs docs) │  returning the answer
                   └──────┬──────────┘
                          │
                   ┌──────▼──────┐
                   │ Final Answer │
                   └─────────────┘
```

## 🧩 How the Agent Thinks

| Node | Job | Detail |
|---|---|---|
| **Router** | Decide the strategy | One LLM call classifies the query as `rag`, `direct`, or `websearch`. Follow-up questions ("tell me more about that") are forced to `direct` so conversation history is reused instead of re-retrieving. |
| **Retriever** | Pull relevant chunks | Embeds the query with a local sentence-transformer and searches the persistent ChromaDB collection for the top matches. |
| **Grader** | Quality-check retrieval | Asks the LLM a yes/no relevance question **per chunk**. Chunks that fail are dropped before they ever reach the generator. |
| **Rewriter** | Recover from bad retrieval | If nothing passes grading, reformulates the query and sends it back through the retriever — capped at 2 retries to avoid infinite loops. |
| **Web Search** | Fallback for anything document retrieval can't answer | Uses Tavily to pull live, LLM-ready search results when documents don't have the answer or retries are exhausted. |
| **Generator** | Write the answer | Synthesizes a response from whatever grounded context survived the pipeline (documents or search results). |
| **Hallucination Checker** | Final safety net | Cross-checks the generated answer against the source chunks and flags it if it looks unsupported. |

---

## 🛠️ Tech Stack

<table>
<tr>
<td valign="top" width="50%">

**Backend**
- **FastAPI** — async Python API framework
- **LangGraph** — stateful, cyclic agent orchestration
- **LangChain** — document loading & text splitting
- **Groq (Llama 3.3 70B)** — LLM inference, chosen for speed since the pipeline makes several sequential model calls per query
- **HuggingFace (`all-MiniLM-L6-v2`)** — local, free embeddings
- **ChromaDB** — persistent local vector store
- **Tavily** — web search API for the fallback path
- **pdfplumber / python-docx** — document parsing (PDF, DOCX, TXT)

</td>
<td valign="top" width="50%">

**Frontend**
- **React 19** — UI library
- **Vite** — dev server & build tool
- **Axios** — API communication
- **Framer Motion** — animations
- Plain CSS — no external UI framework

</td>
</tr>
</table>

**Why this stack?** Groq keeps the multi-hop pipeline (router → grader → generator → hallucination check) fast despite chaining several LLM calls per query. Local HuggingFace embeddings and embedded ChromaDB mean the project runs with zero paid infrastructure beyond the two LLM API keys — ideal for a self-funded project, with the known tradeoff that local disk persistence doesn't survive redeploys on most ephemeral hosts.

---

## 📁 Project Structure

```
Agentic-RagBot/
├── backend/
│   ├── app/
│   │   ├── agent/
│   │   │   ├── graph.py              # LangGraph wiring — nodes & edges
│   │   │   ├── state.py              # Shared agent state (TypedDict)
│   │   │   ├── memory.py
│   │   │   └── nodes/
│   │   │       ├── router.py         # Query classification
│   │   │       ├── retriever.py      # Vector search
│   │   │       ├── grader.py         # Per-chunk relevance grading
│   │   │       ├── rewriter.py       # Query reformulation
│   │   │       ├── web_search.py     # Tavily fallback
│   │   │       ├── generator.py      # Answer synthesis
│   │   │       └── hallucination_checker.py
│   │   ├── routes/
│   │   │   ├── chat.py               # /api/chat, /api/sessions/*
│   │   │   └── upload.py             # /api/upload, /api/documents
│   │   ├── services/
│   │   │   ├── llm.py                # Groq client setup
│   │   │   ├── embeddings.py         # HuggingFace embedding model
│   │   │   ├── vectorstore.py        # ChromaDB operations
│   │   │   ├── parser.py             # PDF/DOCX/TXT text extraction
│   │   │   └── rag.py / retriever.py
│   │   ├── config.py                 # Env var loading
│   │   └── main.py                   # FastAPI app entrypoint
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── ChatWindow.jsx
    │   │   ├── FileUpload.jsx
    │   │   ├── InputBar.jsx
    │   │   ├── IntroScreen.jsx
    │   │   ├── MessageBubble.jsx
    │   │   ├── Sidebar.jsx
    │   │   └── Badge.jsx
    │   ├── api.jsx                   # Axios API layer
    │   └── App.jsx
    ├── package.json
    └── vite.config.js
```

---

## ⚡ Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- API keys: [Groq](https://console.groq.com/) (required) and [Tavily](https://tavily.com/) (required for web search fallback)

### 1. Clone the repo

```bash
git clone https://github.com/Hamzakhan2005/Agentic-RagBot.git
cd Agentic-RagBot
```

### 2. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file inside `backend/`:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
HF_TOKEN=your_huggingface_token_here
CHROMA_PERSIST_DIR=./chroma_db
```

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be live at `http://localhost:8000` — visit `http://localhost:8000/docs` for interactive FastAPI docs.

### 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The app will be live at `http://localhost:5173`.

> ⚠️ Make sure the frontend's API base URL (in `src/api.jsx`) points to your backend address.

---

## 🔌 API Reference

### `POST /api/chat`
Send a message to the agent.

**Request body:**
```json
{
  "query": "What does the uploaded contract say about termination?",
  "session_id": "optional-existing-session-id",
  "stream": false
}
```

**Response:**
```json
{
  "answer": "...",
  "sources": ["contract.pdf"],
  "chunks_used": 3,
  "route_taken": "rag",
  "web_search_used": false,
  "rewritten_query": null,
  "session_id": "generated-or-reused-session-id"
}
```

### `POST /api/upload`
Upload a document (`.pdf`, `.txt`, `.docx`) for indexing.

```json
{
  "message": "File uploaded and indexed successfully",
  "filename": "contract.pdf",
  "chunks_created": 12
}
```

### `GET /api/sessions/{session_id}/history`
Retrieve the full chat history for a session.

### `DELETE /api/sessions/{session_id}`
Clear a session's chat history.

### `GET /api/documents/count`
Get the number of chunks currently in the vector store.

### `DELETE /api/documents`
Clear **all** documents from the vector store.

> 🔓 **Note:** none of these endpoints currently require authentication — see [Known Limitations](#-known-limitations--roadmap).

---

## ⚙️ Configuration

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | LLM inference for every agent node |
| `TAVILY_API_KEY` | ✅ | Powers the web search fallback node |
| `HF_TOKEN` | ⚠️ | Needed if pulling gated HuggingFace models |
| `CHROMA_PERSIST_DIR` | ❌ (defaults to `./chroma_db`) | Where the vector store persists to disk |

---

## 🚧 Known Limitations & Roadmap

This project is a working demonstration of an agentic RAG architecture, not a production-hardened service. Being upfront about the gaps:

- [ ] **Sessions are in-memory** — a server restart clears all chat history. Needs Redis/Postgres.
- [ ] **ChromaDB persists to local disk** — doesn't survive redeploys on ephemeral hosts; needs a managed vector DB for real deployment.
- [ ] **No authentication or rate limiting** — all endpoints are currently open.
- [ ] **No automated tests** — planned: unit tests for graph routing logic, integration tests for the API.
- [ ] **`stream` flag exists but isn't wired up** — the agent runs synchronously; streaming tokens to the UI is a planned improvement.
- [ ] **Grader calls run sequentially** — parallelizing them with `asyncio.gather` would meaningfully cut latency.

Contributions and suggestions welcome — open an issue or PR.

---

## 📄 License

This project currently has no license file. 

---

<div align="center">

Built by [Hamza Khan](https://github.com/Hamzakhan2005)

</div>
