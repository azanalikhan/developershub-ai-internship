"""
Core RAG chatbot: context-aware conversation via ConversationBufferMemory +
grounded retrieval via a FAISS vector store.

CLI usage:
    python chatbot.py
"""
import os
from pathlib import Path

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

from ingest import get_embeddings

VECTOR_STORE_DIR = Path(__file__).parent / "vector_store"


def get_llm():
    """Returns a chat LLM based on the LLM_BACKEND env var (anthropic|openai|groq)."""
    backend = os.environ.get("LLM_BACKEND", "anthropic")
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    elif backend == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2)


def load_vector_store():
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"{VECTOR_STORE_DIR} not found. Run `python ingest.py` first to build the index."
        )
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(VECTOR_STORE_DIR), embeddings, allow_dangerous_deserialization=True
    )


def build_chain(k=4):
    """Builds a ConversationalRetrievalChain with buffer memory + FAISS retrieval."""
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    llm = get_llm()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return chain


def run_cli():
    print("RAG chatbot ready. Type 'exit' to quit.\n")
    chain = build_chain()

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        result = chain.invoke({"question": query})
        print(f"\nBot: {result['answer']}\n")

        sources = {doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])}
        if sources:
            print(f"  (sources: {', '.join(sources)})\n")


if __name__ == "__main__":
    run_cli()
