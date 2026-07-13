"""
Streamlit UI for the context-aware RAG chatbot.

Usage:
    streamlit run app.py
"""
import streamlit as st

from chatbot import build_chain, VECTOR_STORE_DIR

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 Context-Aware RAG Chatbot")
st.caption("Ask about the knowledge base in `data/corpus/` — answers are grounded "
           "in retrieved documents, and the bot remembers the conversation.")

if not VECTOR_STORE_DIR.exists():
    st.error(
        f"No vector store found at `{VECTOR_STORE_DIR}`. "
        "Run `python ingest.py` first to build the index from `data/corpus/`."
    )
    st.stop()


@st.cache_resource
def get_chain():
    return build_chain()


if "chain" not in st.session_state:
    with st.spinner("Loading vector store and LLM..."):
        st.session_state.chain = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke({"question": prompt})
            answer = result["answer"]
            sources = sorted({doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])})

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"- {s}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
