"""
Loads all documents from data/corpus/, splits them into chunks, embeds them,
and persists a FAISS vector store to vector_store/.

Usage:
    python ingest.py
"""
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_DIR = Path(__file__).parent / "data" / "corpus"
VECTOR_STORE_DIR = Path(__file__).parent / "vector_store"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
CORPUS_EXTENSIONS = {".txt", ".md"}


def load_corpus_documents():
    """Reads every .txt/.md file under CORPUS_DIR into a langchain Document."""
    docs = []
    for path in sorted(CORPUS_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in CORPUS_EXTENSIONS:
            text = path.read_text(encoding="utf-8")
            docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def get_embeddings():
    """Returns an embeddings object based on the EMBEDDING_BACKEND env var."""
    backend = os.environ.get("EMBEDDING_BACKEND", "huggingface")
    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def main():
    if not CORPUS_DIR.exists() or not any(CORPUS_DIR.glob("*")):
        raise FileNotFoundError(
            f"No documents found in {CORPUS_DIR}. Add .txt/.md files there first."
        )

    print(f"Loading documents from {CORPUS_DIR} ...")
    documents = load_corpus_documents()
    print(f"Loaded {len(documents)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print("Embedding chunks and building FAISS index...")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    print(f"Saved FAISS index to {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
