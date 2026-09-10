<div align="center">

# 🧠 Agentic RagBot

### A self-correcting, agentic RAG chatbot built with LangGraph

Chat with your documents — powered by a multi-step agent that **routes**, **retrieves**, **grades its own retrieval**, **rewrites bad queries**, **escalates to full-corpus coverage when needed**, **falls back to live web search**, and **checks itself for hallucinations** before answering.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-1C3C3C?style=flat-square)](https://www.langchain.com/langgraph)
[![Groq](https://img.shields.io/badge/Groq-gpt--oss--120b-F55036?style=flat-square)](https://groq.com/)
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
- [Performance & Scaling](#-performance--scaling)
- [Notable Engineering Decisions](#-notable-engineering-decisions)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Known Limitations](#-known-limitations--roadmap)
- [License](#-license)

---

## 🚀 What Makes This Different

Most RAG chatbots follow one fixed path: **retrieve → generate**, with zero self-checking. If the retriever pulls irrelevant chunks — or a small document loses a similarity search to a bigger, more talkative one — the model confidently hallucinates an answer anyway, and nobody notices until it's wrong.

**Agentic RagBot doesn't trust its first attempt, and doesn't trust a single document's odds against the whole collection.** It's built as a graph of specialized nodes, each responsible for one decision, so the pipeline can *detect* a bad or incomplete retrieval and *correct* it before the user ever sees a wrong answer:

| Basic RAG | Agentic RagBot |
|---|---|
| Retrieve once, generate | Retrieves, **grades relevance per document**, retries with a rewritten query if the grade fails |
| Fixed top-k across the whole collection | **Detects multi-document questions** ("compare X and Y", "all of my invoices") and guarantees every uploaded file gets checked, not just whichever chunks win a similarity search |
| No fallback if retrieval fails twice | Escalates from rewrite → **full-corpus fallback** → live web search, in that order |
| No idea if the answer is grounded | **Checks its own output** for hallucinations against the full source content, and regenerates once if it fails |
| One-size-fits-all query handling | **Routes** each query to the right strategy — documents, history, or the web — with a fast path that skips the router LLM call entirely when a filename is named explicitly |
| Re-initializes its LLM/embedding clients per call | **Singleton-cached** clients — response time stays flat as document count grows instead of degrading |

---

## 🏗️ Architecture

```
                              ┌──────────────┐
                              │   User Query  │
                              └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │  Router Node  │   Classifies query as:
                              │              │   rag / direct / websearch
                              │  Fast-path:   │   Skips the LLM call entirely
                              │  named file?  │   if the query names an
                              └──────┬───────┘   uploaded file directly
                     ┌───────────────┼────────────────┐
                     │               │                │
              ┌──────▼──────┐ ┌──────▼──────┐  ┌───────▼────────┐
              │  Retriever   │ │Direct Answer│  │   Web Search    │
              │              │ │ (history/   │  │    (Tavily)     │
              │  3 modes:    │ │  general    │  │                 │
              │  • named-file │ │  knowledge) │  └───────┬────────┘
              │    scoped    │ └──────┬──────┘          │
              │  • broad     │        │                 │
              │    (multi-doc │        │                 │
              │    query →   │        │                 │
              │    fetch all │        │                 │
              │    sources)  │        │                 │
              │  • top-k     │        │                 │
              │    default   │        │                 │
              └──────┬──────┘        │                 │
                     │                │                 │
              ┌──────▼──────┐        │                 │
              │    Grader    │        │                 │
              │ single batched│        │                 │
              │ LLM call,     │        │                 │
              │ grades every  │        │                 │
              │ candidate     │        │                 │
              │ DOCUMENT at   │        │                 │
              │ once (not per │        │                 │
              │ chunk)        │        │                 │
              └──────┬──────┘        │                 │
                     │                │                 │
         ┌───────────┴───────────┐    │                 │
         │                       │    │                 │
   docs relevant?          no relevant docs              │
         │                       │                       │
         │           ┌───────────┴───────────┐           │
         │      attempt 1: Rewriter    attempt 2: Broad   │
         │      (reformulate query,    Fallback (guarantee │
         │       retry Retriever)      coverage of every  │
         │           │                 uploaded document) │
         │           └──────►Retriever        │           │
         │                                     │           │
         │              (still nothing after both) ────────┤
         │                                                 │
         └───────────────────────┬─────────────────────────┘
                                  │
                           ┌──────▼──────┐
                           │  Generator   │  Answers from graded context.
                           │              │  Prior chat history is clearly
                           │              │  labeled "context only" so it
                           │              │  doesn't bleed facts from an
                           │              │  earlier answer about a
                           │              │  different document.
                           └──────┬──────┘
                                  │
                           ┌──────▼──────────┐
                           │Hallucination Chk │  Checks the answer against
                           │                  │  FULL source content (not
                           │  fail → regenerate│  truncated), regenerates
                           │  once, stricter   │  once with a stricter
                           │  prompt           │  prompt if unsupported
                           └──────┬──────────┘
                                  │
                           ┌──────▼──────┐
                           │ Final Answer │
                           └─────────────┘
```

## 🧩 How the Agent Thinks

| Node | Job | Detail |
|---|---|---|
| **Router** | Decide the strategy | One LLM call classifies the query as `rag`, `direct`, or `websearch`. Follow-ups reuse history via `direct`. **Fast path:** if the query names an uploaded file directly, the router LLM call is skipped entirely — retrieval scopes straight to that file. |
| **Retriever** | Pull relevant chunks | Three modes: **named-file** (scoped ChromaDB metadata filter), **broad** (query implies multiple/all documents — bypasses top-k entirely and pulls a guaranteed sample from *every* uploaded source so no document can lose a similarity competition), **default** (top-k similarity search, k scaled to document count). |
| **Grader** | Quality-check retrieval | Groups retrieved chunks **by source document** and grades all candidates in a **single batched LLM call** returning structured JSON — not one call per chunk. Explicitly instructed that a document only needs to answer *part* of a multi-part question to count as relevant, so compound questions about two different documents don't get rejected for "not fully answering" the whole query. |
| **Rewriter** | Recover from bad wording | First-attempt recovery: reformulates the query, explicitly forbidden from inventing unstated facts (dates, numbers) that weren't in the original question. |
| **Broad Fallback** | Recover from bad retrieval | Second-attempt recovery: if a rewritten query *still* finds nothing, the problem usually isn't wording — it's a small or generically-worded document losing a similarity search against a larger collection. This node skips straight to full-corpus coverage instead of trying a third narrow search. |
| **Web Search** | Last resort | Tavily search when documents genuinely don't contain the answer or all retrieval attempts are exhausted. |
| **Generator** | Write the answer | Synthesizes from graded context. Chat history is passed as clearly-labeled background, not blended into the same message as the authoritative retrieved context — prevents the model from echoing a previous answer about a *different* document. |
| **Hallucination Checker** | Final safety net | Cross-checks the answer against **full, untruncated** source content and regenerates once with a stricter "only use what's explicitly there" prompt if the check fails. |

---

## ⚡ Performance & Scaling

| Scenario | Response time | What's happening |
|---|---|---|
| Cold start (first query after boot) | ~16s | One-time LLM client + embedding model warmup |
| Named-file fast path | ~12s | Router LLM call skipped |
| 2 documents, compound query | ~20s | |
| 5 documents, broad (multi-doc) query | ~18s compute | Flat vs. the 2-doc case |
| 10 documents, 27 chunks graded | ~20s compute | Still a **single** grading call regardless of chunk count |

**Response time holds flat from 2 to 10 documents.** Before the batched-grading and singleton-caching work, grading alone would have meant one sequential LLM call per retrieved document — at 10 documents that's 10 grading calls stacked on top of everything else. Regrouping to one batched JSON-graded call per query turned that from O(n) into O(1).

**Current target: 15 concurrent documents**, matching the app's `MAX_DOCUMENTS_PER_SESSION` cap. Verified reliable through 10 documents with both single-document and cross-document queries answering correctly. The main variable at higher document counts isn't correctness — it's external Groq API rate limits (429s), which add unpredictable backoff delay as per-query prompt token volume grows with more documents in context.

---

## 🔍 Notable Engineering Decisions

A few things worth knowing if you're reading the code (or asking about it):

- **Singleton-cached LLM & embedding clients.** These were originally re-instantiated on *every single node call* — router, retriever, each grading call, rewriter, generator, hallucination checker — adding multi-second overhead per call. Caching them at module level cut per-query latency by roughly 60% without touching any retrieval logic.
- **Batched, per-document grading instead of per-chunk grading.** Grading isolated 500-character fragments in total isolation meant the grader could reject a chunk containing a departure *time* just because the *date* happened to land in a different fragment. Grouping chunks by source and grading each document holistically — in one batched call — fixed both the correctness bug and the latency cost at once.
- **Broad-query detection is opt-in, not default.** Guaranteeing full-corpus coverage on every single query would be wasteful for simple single-document lookups. Detecting compound/multi-document intent (keywords like "all", "compare", " and ") and only escalating retrieval then keeps the common case fast.
- **Two-stage retrieval escalation, not one.** A failed retrieval first gets a query rewrite (handles genuine wording ambiguity — the fix is cheap and often works). If that *still* fails, the second attempt skips straight to full-corpus coverage instead of trying a third narrow rewrite — because by that point the real problem is usually that a small or vaguely-worded document lost a similarity race, not that the query needs better phrasing.

---

## 🛠️ Tech Stack

<table>
<tr>
<td valign="top" width="50%">

**Backend**
- **FastAPI** — async Python API framework
- **LangGraph** — stateful, cyclic agent orchestration
- **LangChain** — document loading & text splitting
- **Groq (gpt-oss-120b)** — LLM inference, chosen for speed since the pipeline makes several sequential model calls per query
- **HuggingFace (`all-MiniLM-L6-v2`)** — local, free embeddings, singleton-cached at process startup
- **ChromaDB** — persistent local vector store, also singleton-cached
- **Tavily** — web search API for the fallback path
- **pdfplumber / python-docx** — document parsing with table extraction; PyMuPDF + Tesseract OCR fallback for scanned PDF pages

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

**Why this stack?** Groq keeps the multi-hop pipeline (router → grader → generator → hallucination check) fast despite chaining several LLM calls per query. Local HuggingFace embeddings and embedded ChromaDB mean the project runs with zero paid infrastructure beyond the two LLM/search API keys — ideal for a self-funded project, with the known tradeoff that local disk persistence doesn't survive redeploys on most ephemeral hosts.

---

## 📁 Project Structure

```
Agentic-RagBot/
├── backend/
│   ├── app/
│   │   ├── agent/
│   │   │   ├── graph.py              # LangGraph wiring — nodes & conditional edges
│   │   │   ├── state.py              # Shared agent state (TypedDict)
│   │   │   ├── memory.py
│   │   │   └── nodes/
│   │   │       ├── router.py         # Query classification + named-file fast path
│   │   │       ├── retriever.py      # Node wrapper around services/retriever.py
│   │   │       ├── grader.py         # Batched per-document relevance grading
│   │   │       ├── rewriter.py       # Query reformulation (no invented facts)
│   │   │       ├── web_search.py     # Tavily fallback
│   │   │       ├── generator.py      # Answer synthesis, history/context separation
│   │   │       └── hallucination_checker.py
│   │   ├── routes/
│   │   │   ├── chat.py               # /api/chat, /api/sessions/*
│   │   │   └── upload.py             # /api/upload, /api/documents
│   │   ├── services/
│   │   │   ├── llm.py                # Groq client — singleton-cached
│   │   │   ├── embeddings.py         # HuggingFace embedding model — singleton-cached
│   │   │   ├── vectorstore.py        # ChromaDB ops, broad-query detection, named-file detection
│   │   │   ├── parser.py             # PDF/DOCX/TXT extraction + OCR fallback
│   │   │   └── retriever.py          # 3-mode retrieval logic (named/broad/default)
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
Upload a document (`.pdf`, `.txt`, `.docx`) for indexing. Rejected once the session hits `MAX_DOCUMENTS_PER_SESSION` (default 15).

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
Clear **all** documents from the vector store and reset session document tracking.

> 🔓 **Note:** none of these endpoints currently require authentication — see [Known Limitations](#-known-limitations--roadmap).

---

## ⚙️ Configuration

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | LLM inference for every agent node |
| `TAVILY_API_KEY` | ✅ | Powers the web search fallback node |
| `HF_TOKEN` | ⚠️ | Needed if pulling gated HuggingFace models |
| `CHROMA_PERSIST_DIR` | ❌ (defaults to `./chroma_db`) | Where the vector store persists to disk |
| `MAX_DOCUMENTS_PER_SESSION` | ❌ (defaults to `15`) | Hard cap on concurrent documents per session |

---

## 🚧 Known Limitations & Roadmap

This project is a working demonstration of an agentic RAG architecture, not a production-hardened service. Being upfront about the gaps:

- [ ] **Sessions are in-memory** — a server restart clears all chat history. Needs Redis/Postgres.
- [ ] **ChromaDB persists to local disk** — doesn't survive redeploys on ephemeral hosts; needs a managed vector DB for real deployment.
- [ ] **No authentication or rate limiting** — all endpoints are currently open.
- [ ] **No automated tests** — planned: unit tests for graph routing logic, integration tests for the API.
- [ ] **`stream` flag exists but isn't wired up** — the agent runs synchronously; streaming tokens to the UI is a planned improvement.
- [ ] **Groq rate limits (429s) add unpredictable backoff delay** at higher document counts and prompt volumes — external constraint, worth budgeting for at scale or moving to a higher API tier.
- [ ] **No reranking layer** — relies on cosine similarity + LLM grading rather than a dedicated reranker; would help precision further at larger document counts.
- [ ] **Independent LLM calls (router / grader / hallucination-check) run sequentially** — some could be parallelized with `asyncio.gather` for further latency reduction.

Contributions and suggestions welcome — open an issue or PR.

---

## 📄 License

This project currently has no license file.

---

<div align="center">

Built by [Hamza Khan](https://github.com/Hamzakhan2005)

</div>
