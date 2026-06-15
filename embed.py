"""
embed.py — Embed all chunks and store them in a local ChromaDB collection.

Embedding model : paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers)
Vector store    : ChromaDB (persistent, local)
Collection name : heeseung_guide

Usage:
    python embed.py              # embed chunks.json → chroma_db/
    python embed.py --reset      # drop and recreate the collection first
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── configuration ─────────────────────────────────────────────────────────────
CHUNKS_FILE = Path("chunks.json")
CHROMA_DIR  = Path("chroma_db")
COLLECTION  = "heeseung_guide"
MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE  = 64          # number of texts to embed per model.encode() call


def main(reset: bool = False) -> None:
    # ── load chunks ───────────────────────────────────────────────────────────
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} not found — run `python ingest.py` first"
        )
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}\n")

    # ── load embedding model ──────────────────────────────────────────────────
    print(f"Loading embedding model: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)
    # The default max_seq_length for this model is 128, matching its training
    # regime, but the underlying architecture supports up to 512 tokens.
    # Raising it ensures that facts appearing late in longer chunks (e.g. the
    # Trivia section at ~162 tok, or TMI groups at ~226 tok) are not silently
    # truncated during embedding and therefore missed during retrieval.
    model.max_seq_length = 512
    print(f"  max_seq_length = {model.max_seq_length}  (raised from default 128)")

    # ── connect to ChromaDB ───────────────────────────────────────────────────
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"\nDropped existing collection '{COLLECTION}'")
        except Exception:
            pass   # collection did not exist — that's fine

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    print(f"Collection '{COLLECTION}'  ({collection.count()} existing vectors)\n")

    # ── prepare data ──────────────────────────────────────────────────────────
    texts     = [c["text"] for c in chunks]
    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source_file": c["source_file"],
            "language":    c["language"],
            "doc_type":    c["doc_type"],
            "section":     c["section"],
            "chunk_index": c["chunk_index"],
            "token_count": c["token_count"],
        }
        for c in chunks
    ]

    # ── embed in batches and upsert ───────────────────────────────────────────
    print(f"Embedding {len(texts)} chunks (batch size {BATCH_SIZE}) …")
    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))

        embeddings = model.encode(
            texts[start:end],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings,
            metadatas=metadatas[start:end],
        )
        print(f"  {end:>4} / {len(texts)}")

    print(
        f"\n✓  {len(chunks)} chunks stored in ChromaDB collection '{COLLECTION}'\n"
        f"   Path: {CHROMA_DIR.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks and store in ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the ChromaDB collection before embedding",
    )
    args = parser.parse_args()
    main(reset=args.reset)
