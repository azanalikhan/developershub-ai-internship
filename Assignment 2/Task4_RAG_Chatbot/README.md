# Task 4: Context-Aware Chatbot Using LangChain / RAG

## Objective
Build a conversational chatbot that (a) remembers conversational history/context
and (b) retrieves grounded answers from a vectorized custom knowledge base
(Retrieval-Augmented Generation), deployed with Streamlit.

---

## Files in this folder — what each one does, and the order to use them

| Order | File | What it does | Run directly? |
|---|---|---|---|
| 1 | `requirements.txt` | LangChain + FAISS + embeddings + LLM backend packages + Streamlit | Once, before anything else |
| 2 | `data/corpus/company_handbook.txt` | Sample knowledge base (a fake internal handbook: remote work, leave policy, onboarding, expenses, IT support). Drop your own `.txt`/`.md` files here to use a different knowledge base | N/A — just data |
| 3 | `ingest.py` | Reads everything in `data/corpus/`, splits it into chunks, embeds them, builds a FAISS vector index, saves it to `vector_store/` | Yes — run first, before chatting |
| 4 | `chatbot.py` | Core RAG logic: retrieval + conversation memory via `ConversationalRetrievalChain`. Has a CLI chat loop built in | Yes — for terminal-based chat |
| 5 | `app.py` | Streamlit chat UI wrapping `chatbot.py` | Yes — for the web UI version |

**Run order: `ingest.py` -> `chatbot.py` (or `app.py`)**

---

## Step-by-step: how to actually run it

### 0. Install dependencies (once)
```powershell
python -m pip install -r requirements.txt
```
LangChain pulls in a lot of sub-dependencies — this can take a few minutes.

### 1. Set your LLM backend and API key
This project supports three LLM backends, chosen via the `LLM_BACKEND`
environment variable:

| Backend | `LLM_BACKEND` value | API key env var | Model used |
|---|---|---|---|
| Groq | `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| Anthropic | `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |

Set both env vars **in the same terminal window** you'll run the scripts
from — they don't persist across new terminal sessions unless set as
permanent Windows environment variables.

```powershell
$env:LLM_BACKEND = "groq"
$env:GROQ_API_KEY = "gsk_..."
```

### 2. Build the vector index
```powershell
python ingest.py
```
Downloads a small local embedding model
(`sentence-transformers/all-MiniLM-L6-v2`, ~90MB, from Hugging Face — needs
normal internet access) on first run, then chunks and embeds everything in
`data/corpus/`, saving the FAISS index to `vector_store/`.

### 3. Chat from the terminal
```powershell
python chatbot.py
```
Type a question, get a grounded answer, ask a follow-up — the bot remembers
context across turns. Type `exit` to quit.

### 4. Or launch the Streamlit web UI
```powershell
streamlit run app.py
```

---

## Troubleshooting notes (things that came up while setting this up)
- **`LangChainDeprecationWarning` about `ConversationBufferMemory`** —
  cosmetic. LangChain is nudging toward its newer agent/checkpointing APIs,
  but the classic memory + chain approach used here still works correctly
  and is simpler for this use case. Safe to ignore.
- **`ModuleNotFoundError: No module named 'langchain.chains'`** — newer
  LangChain versions (1.x) moved the classic chain implementations
  (`ConversationalRetrievalChain`, `ConversationBufferMemory`) into the
  `langchain_classic` package. Already fixed in `chatbot.py` — it imports
  from `langchain_classic` directly.
- **`anthropic.AuthenticationError: invalid x-api-key`** — this is not a
  code bug, it means the API key itself is wrong, missing, expired, or set
  in a different terminal window than the one running the script. Check with
  `echo $env:ANTHROPIC_API_KEY` (or the equivalent for your chosen backend)
  before assuming anything else is broken.
- **Don't have an Anthropic/OpenAI key, but have a Groq key?** — set
  `LLM_BACKEND=groq` and `GROQ_API_KEY=...` instead; `chatbot.py` supports
  Groq's `llama-3.3-70b-versatile` model as a third backend option.

---

## Methodology / Approach
1. **Corpus** — `data/corpus/` holds the custom knowledge base (a sample
   internal handbook is included; drop any `.txt`/`.md` files here to use
   your own).
2. **Ingestion (`ingest.py`)** — documents are chunked with LangChain's
   `RecursiveCharacterTextSplitter` (chunk size 500, overlap 75), embedded,
   and stored in a local **FAISS** vector store (`vector_store/`) for fast
   similarity search.
3. **Embeddings** — local `sentence-transformers/all-MiniLM-L6-v2`
   embeddings by default (free, no API key, runs on CPU); configurable to
   OpenAI embeddings via the `EMBEDDING_BACKEND` env var.
4. **Retrieval + memory (`chatbot.py`)** — a LangChain
   `ConversationalRetrievalChain` combines:
   - `ConversationBufferMemory` for multi-turn context, so follow-up
     questions like "what about for interns?" resolve correctly
   - A FAISS retriever (top-4 similarity search) that pulls relevant chunks
     for each question before the LLM answers, grounding responses in the
     corpus instead of relying on the model's general knowledge
5. **LLM backend** — configurable via `LLM_BACKEND` env var: Groq, Anthropic,
   or OpenAI (see table above).
6. **Deployment (`app.py`)** — a Streamlit chat UI wrapping `chatbot.py`.

## Key Results / Observations
Real transcript from a terminal run (`python chatbot.py`), using the Groq
backend (`llama-3.3-70b-versatile`):

```
You: What's the remote work policy?
Bot: The remote work policy at DevelopersHub Corporation allows employees to
work remotely up to 3 days per week, subject to manager approval.
Additionally, there are core collaboration hours from 10:00 AM to 3:00 PM in
the employee's local timezone, and remote employees are required to attend
the weekly all-hands meeting on Mondays at 9:30 AM.
  (sources: company_handbook.txt)

You: And what are the core hours?
Bot: The core collaboration hours for remote work at DevelopersHub
Corporation are 10:00 AM to 3:00 PM in the employee's local timezone.
  (sources: company_handbook.txt)

You: How many leave days do I get?
Bot: Full-time employees at DevelopersHub Corporation accrue 18 paid leave
days per year, plus 10 public holidays.
  (sources: company_handbook.txt)
```

What this demonstrates:
- **Grounded retrieval works correctly** — every answer is accurate against
  the source `company_handbook.txt` content, and every response is properly
  attributed to its source chunk. No hallucinated policy details.
- **Conversational memory resolves context correctly** — the second question,
  *"And what are the core hours?"*, contains no explicit mention of "remote
  work." The bot correctly inferred from the buffered conversation history
  that "core hours" referred to the remote work policy just discussed, rather
  than treating it as an ambiguous or unrelated question. This is exactly
  what `ConversationBufferMemory` is meant to enable.
- **Topic switching also works cleanly** — the third question moved to an
  unrelated topic (leave policy), and the bot correctly retrieved fresh,
  relevant context rather than getting anchored on the previous remote-work
  topic.
- Retrieval quality (not just LLM quality) is the main lever on answer
  accuracy in this architecture — chunk size, chunk overlap, and `k` (number
  of retrieved chunks) in `ingest.py`/`chatbot.py` are the first things to
  tune if answers ever feel incomplete or off-topic on a larger/messier
  knowledge base.

## Files
- `ingest.py` — loads `data/corpus/`, chunks, embeds, and builds the FAISS index
- `chatbot.py` — core `ConversationalRetrievalChain` logic + CLI chat loop, supports Groq/Anthropic/OpenAI backends
- `app.py` — Streamlit chat UI
- `data/corpus/` — sample custom knowledge base (replace with your own docs)
- `vector_store/` — generated FAISS index (created by `ingest.py`)
