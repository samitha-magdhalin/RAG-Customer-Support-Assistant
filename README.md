# Solstice Outdoor Co. — Customer Support RAG Agent

A Streamlit web app that answers customer questions about a fictional outdoor
gear retailer, **Solstice Outdoor Co.**, using Retrieval-Augmented Generation
(RAG) with cited sources, a transparency panel showing the retrieved FAQ
snippets, and conversational memory.

## Screenshot

![Solstice Support Assistant dashboard](screenshots/dashboard.png)

*Chat interface showing a grounded answer with cited source line and the
retrieved-snippet transparency panel expanded.*

## Architecture

```
User (browser)
    │
    ▼
Streamlit chat UI (app.py)
    │
    ▼
LangChain retrieval chain (rag.py)
  create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    │            │
    ▼            ▼
FAISS vector   chat_history list
store (local)   (HumanMessage/AIMessage, tracked in app.py)
    │
    ▼
data/solstice_faq.txt  →  split into line-level chunks
```

1. **Knowledge base**: `data/solstice_faq.txt` contains mock FAQ content about
   Solstice's shipping, returns, business hours, and membership tiers. Each
   `Line N:` entry is parsed into its own `Document` so the exact line can be
   cited.
2. **Vector store**: On first run, `rag.py` embeds each line with a free local
   HuggingFace embedding model (`all-MiniLM-L6-v2`) and stores the vectors in a
   local FAISS index (`faiss_index/`), reused on subsequent runs.
3. **LLM & retrieval chain**: Llama 3.3 70B via Groq (`langchain-groq`)
   generates the final answer. The chain is built with LangChain's current,
   stable API — `create_retrieval_chain` + `create_stuff_documents_chain` —
   rather than the older `ConversationalRetrievalChain`/
   `ConversationBufferMemory` classes, which have been restructured across
   several LangChain releases and are more prone to import breakage.
4. **Conversational memory**: handled explicitly in `app.py` as a plain list
   of `HumanMessage`/`AIMessage` objects (`st.session_state.lc_history`),
   passed into the chain on every turn via a `MessagesPlaceholder` in the
   prompt. This is what lets "How much does it cost to ship there?" resolve
   correctly after "Do you ship to India?".
5. **Citations**: citations come **purely from the retriever's metadata**
   (`format_sources` / `format_snippets` in `rag.py`) — the LLM is explicitly
   told not to write its own "Sources:" line, so there's exactly one citation
   mechanism, not two competing ones.
6. **Transparency panel**: each answer has an expandable "🔍 Show retrieved
   FAQ snippet(s)" section showing the exact FAQ text the agent grounded its
   answer in — lets a reviewer verify the RAG pipeline is really retrieving,
   not hallucinating.
7. **UI**: `app.py` is a Streamlit chat interface with clickable example
   questions for first-time users, plus a custom dark/amber theme
   (`.streamlit/config.toml`).

## Setup (local)

```bash
git clone <your-repo-url>
cd assignment1-rag-support
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GROQ_API_KEY

streamlit run app.py
```

The first run downloads the embedding model and builds the FAISS index (a
few seconds); later runs reuse the cached index in `faiss_index/`.

> **Note on Python version**: this project depends on numpy/faiss/torch,
> which may not yet have prebuilt wheels for very new Python releases (e.g.
> 3.14) on Windows. Python 3.11 or 3.12 is recommended for a smooth,
> no-compiler-needed install.

> **Note on LangChain version**: `requirements.txt` pins `langchain<1.0`.
> LangChain 1.0 (released late 2025) removed `langchain.chains` from the core
> package and moved it into a separate `langchain-classic` package, which
> would break the imports in `rag.py`. Staying on the 0.3.x line keeps the
> code in this repo working as-is.

## Deploying to Streamlit Community Cloud (free)

1. Push this folder to a public GitHub repo (include the `.streamlit/`
   folder — it holds the theme, not secrets).
2. Go to https://share.streamlit.io → "New app" → select your repo/branch and
   set the main file path to `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
4. Deploy. The app builds the FAISS index automatically on first boot.

## Example conversation

```
User: Do you ship to India?
Agent: Yes, Solstice ships to India as part of its international shipping.
       Cited sources: solstice_faq.txt (Line 1)

User: How much does it cost to ship there?
Agent: International shipping to India costs a flat rate of $22.99 and takes
       8-15 business days.
       Cited sources: solstice_faq.txt (Line 3)
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit chat UI: chat, example questions, snippet transparency panel, manual conversation history |
| `rag.py` | RAG pipeline: document loading, FAISS index, retrieval chain, citations |
| `data/solstice_faq.txt` | Mock knowledge base |
| `.streamlit/config.toml` | Native Streamlit theme |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for required environment variables |