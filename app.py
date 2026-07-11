"""
app.py
Streamlit chat UI for the Solstice Outdoor Co. Customer Support RAG Agent.

Run locally:
    streamlit run app.py

Requires GROQ_API_KEY to be set as an environment variable or Streamlit secret.
"""

import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag import build_rag_chain, ask, format_sources, format_snippets

st.set_page_config(page_title="Solstice Support Assistant", page_icon="🏔️", layout="centered")

st.title("🏔️ Solstice Outdoor Co. Support Assistant")
st.caption(
    "Ask about shipping, returns, business hours, or membership tiers. "
    "Answers are grounded in Solstice's FAQ with cited sources."
)

# --- API key handling: env var, Streamlit secrets, or manual input ---
api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
if not api_key:
    api_key = st.text_input("Enter your Groq API key to begin:", type="password")
    if not api_key:
        st.info("An API key is required to run the assistant.")
        st.stop()

# --- Build (and cache) the RAG chain for this session ---
if "chain" not in st.session_state:
    with st.spinner("Loading knowledge base and starting the assistant..."):
        st.session_state.chain = build_rag_chain(groq_api_key=api_key)
    st.session_state.messages = []          # for display: [{"role":..., "content":..., ...}]
    st.session_state.lc_history = []        # for the model: [HumanMessage, AIMessage, ...]

EXAMPLE_QUESTIONS = [
    "Do you ship to India?",
    "What's your return policy for hiking boots?",
    "What's included in the Summit membership?",
]


def ask_question(question: str):
    """Shared handler so both typed input and example-question buttons run
    through the exact same RAG + memory pipeline."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(st.session_state.chain, question, st.session_state.lc_history)
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            sources_line = format_sources(source_docs)
            snippets = format_snippets(source_docs)

            st.markdown(answer)
            st.caption(f"**Cited sources:** {sources_line}")

            with st.expander("🔍 Show retrieved FAQ snippet(s)"):
                if snippets:
                    for s in snippets:
                        st.markdown(f"**{s['source']} — {s['line']}** · _{s['section']}_")
                        st.markdown(f"> {s['text']}")
                else:
                    st.markdown("No snippets retrieved for this answer.")

    # Update both the display history and the LLM-facing conversational memory
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources_line": sources_line,
            "snippets": snippets,
        }
    )
    st.session_state.lc_history.append(HumanMessage(content=question))
    st.session_state.lc_history.append(AIMessage(content=answer))


# --- Render chat history (including snippet expanders for past turns) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources_line"):
            st.caption(f"**Cited sources:** {msg['sources_line']}")
            with st.expander("🔍 Show retrieved FAQ snippet(s)"):
                for s in msg.get("snippets", []):
                    st.markdown(f"**{s['source']} — {s['line']}** · _{s['section']}_")
                    st.markdown(f"> {s['text']}")

# --- Clickable example questions (only shown before the first message) ---
if not st.session_state.messages:
    st.write("**Try asking:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            ask_question(q)
            st.rerun()

# --- Handle new typed input ---
user_input = st.chat_input("Ask a question, e.g. 'Do you ship to India?'")
if user_input:
    ask_question(user_input)

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant retrieves answers from a local FAISS vector store "
        "built from Solstice's mock FAQ (`data/solstice_faq.txt`) and uses "
        "Llama 3.3 70B (via Groq) to generate grounded, cited responses. "
        "Conversation memory is preserved for the current session, so "
        "follow-up questions (e.g. 'how much does it cost?') work correctly. "
        "Each answer also shows the exact FAQ line(s) it was grounded in, "
        "so you can verify the response yourself."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.session_state.pop("chain", None)
        st.rerun()
