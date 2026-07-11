"""
rag.py
Core RAG (Retrieval-Augmented Generation) logic for the Solstice Outdoor Co.
Customer Support Agent.

Built on LangChain's current, stable retrieval-chain API
(create_retrieval_chain + create_stuff_documents_chain) rather than the older
ConversationalRetrievalChain/ConversationBufferMemory classes, which have been
restructured across LangChain releases and are more prone to import breakage.
Conversation history is managed explicitly as a list of messages instead.

- Loads the mock FAQ file (data/solstice_faq.txt)
- Splits it into line-tagged chunks so we can cite exact source lines
- Builds/loads a local FAISS vector store using free HuggingFace embeddings
- Retrieves relevant chunks + generates a grounded answer, given prior chat history
"""

import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

DATA_PATH = Path(__file__).parent / "data" / "solstice_faq.txt"
INDEX_PATH = Path(__file__).parent / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_faq_documents() -> list[Document]:
    """Load the FAQ file and turn each 'Line N: ...' entry into its own Document
    so the retriever can cite an exact line number as the source."""
    text = DATA_PATH.read_text(encoding="utf-8")
    docs = []
    current_section = "General"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("SECTION"):
            current_section = line
            continue
        if line.startswith("Line "):
            try:
                line_no, content = line.split(":", 1)
            except ValueError:
                continue
            docs.append(
                Document(
                    page_content=content.strip(),
                    metadata={
                        "source": "solstice_faq.txt",
                        "line": line_no.strip(),
                        "section": current_section,
                    },
                )
            )
    return docs


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_or_load_vectorstore() -> FAISS:
    """Build the FAISS index on first run, then reuse the cached index on later runs."""
    embeddings = get_embeddings()
    if INDEX_PATH.exists():
        return FAISS.load_local(
            str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )
    docs = load_faq_documents()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(INDEX_PATH))
    return vectorstore


SYSTEM_PROMPT = """You are Solstice Outdoor Co.'s helpful customer support assistant.
Use ONLY the context below (retrieved from Solstice's internal FAQ) to answer the customer's question.
If the answer is not contained in the context, say you don't have that information and suggest they contact support.
Do not include a "Sources" line yourself — citations are added separately by the application.

Context:
{context}"""


def build_rag_chain(groq_api_key: str | None = None):
    """Builds a retrieval chain (retriever + LLM) using LangChain's current
    stable API. Chat history is NOT stored inside this chain — the caller
    passes it in on every call (see ask())."""
    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=groq_api_key or os.environ.get("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


def ask(chain, question: str, chat_history_messages: list) -> dict:
    """Run one turn. chat_history_messages is a list of langchain_core
    HumanMessage/AIMessage objects representing prior turns (conversational
    memory), maintained by the caller (app.py) across the session."""
    result = chain.invoke({"input": question, "chat_history": chat_history_messages})
    return {
        "answer": result["answer"],
        "source_documents": result.get("context", []),
    }


def format_sources(source_documents) -> str:
    """Turn retrieved Document metadata into a single human-readable citation line."""
    seen = set()
    lines = []
    for doc in source_documents:
        key = (doc.metadata.get("source"), doc.metadata.get("line"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{doc.metadata.get('source')} ({doc.metadata.get('line')})")
    return ", ".join(lines) if lines else "No direct source found."


def format_snippets(source_documents) -> list[dict]:
    """Return the actual retrieved FAQ text + metadata, for a UI panel that
    shows the customer exactly what the agent looked up (transparency)."""
    seen = set()
    snippets = []
    for doc in source_documents:
        key = (doc.metadata.get("source"), doc.metadata.get("line"))
        if key in seen:
            continue
        seen.add(key)
        snippets.append(
            {
                "source": doc.metadata.get("source"),
                "line": doc.metadata.get("line"),
                "section": doc.metadata.get("section", "").replace("SECTION", "Section").strip(),
                "text": doc.page_content,
            }
        )
    return snippets
