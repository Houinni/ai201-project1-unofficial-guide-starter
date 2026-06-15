"""
retrieve.py — Query the ChromaDB vector store for relevant chunks.

API:
    from retrieve import query_chunks
    results = query_chunks("What score did Heeseung get on I-LAND?", k=5, language="en")

CLI:
    python retrieve.py "your question"
    python retrieve.py "your question" --k 5 --lang en
    python retrieve.py --eval                # run all 5 evaluation questions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── configuration ─────────────────────────────────────────────────────────────
CHROMA_DIR = Path("chroma_db")
COLLECTION = "heeseung_guide"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Lazy-loaded singletons — avoid re-loading on every query_chunks() call
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
        # Match the max_seq_length used in embed.py so query embeddings live
        # in the same vector space as the stored chunk embeddings.
        _model.max_seq_length = 512
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB directory '{CHROMA_DIR}' not found — "
                "run `python embed.py` first"
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION)
    return _collection


def query_chunks(
    question: str,
    k: int = 5,
    language: str | None = None,
) -> list[dict]:
    """
    Retrieve the top-k chunks most semantically similar to *question*.

    Args:
        question : Natural-language query string (EN or ZH).
        k        : Number of results to return (default 5).
        language : Optional language filter — "en" or "zh".
                   Filters to one language so EN and ZH versions of the same
                   fact do not both appear in the results (planning.md § Challenge 1).

    Returns:
        List of result dicts, each containing:
            text         — chunk text
            source_file  — originating document filename
            language     — "en" or "zh"
            doc_type     — "list", "interview", or "wiki"
            section      — section or date heading from the source document
            chunk_index  — integer position within its source file
            token_count  — approximate BPE token count
            score        — cosine *distance* (lower = more similar, 0 = identical)
    """
    model      = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([question], convert_to_numpy=True).tolist()

    where = {"language": language} if language else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text":  doc,
                "score": round(float(dist), 4),
                **meta,
            }
        )

    return hits


# ── CLI ───────────────────────────────────────────────────────────────────────
EVAL_QUERIES = [
    (
        "Q1: Fake Love I-LAND score",
        "What score did Heeseung receive for his Fake Love performance on I-LAND, "
        "and what was notable about it?",
    ),
    (
        "Q2: Big Hit scouting story",
        "How did Heeseung get discovered by Big Hit Entertainment?",
    ),
    (
        "Q3: Convenience store order",
        "What is Heeseung's go-to convenience store order?",
    ),
    (
        "Q4: MTV News work vs self-investment",
        "What did Heeseung say in his MTV News interview about balancing work "
        "and personal investment?",
    ),
    (
        "Q5: Favorite fruit and Gong Cha drink",
        "What is Heeseung's favorite fruit, and what drink does he order at Gong Cha?",
    ),
]


def _print_results(label: str, question: str, results: list[dict]) -> None:
    print(f"\n{'─'*68}")
    print(f"{label}")
    print(f"Query : {question!r}")
    for rank, r in enumerate(results, 1):
        print(
            f"  [{rank}] score={r['score']:.4f}  "
            f"{r['source_file']}  chunk {r['chunk_index']}  "
            f"lang={r['language']}  ({r['token_count']} tok)"
        )
        preview = r["text"][:200].replace("\n", " ")
        if len(r["text"]) > 200:
            preview += "…"
        print(f"       {preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Heeseung guide vector store.")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--k",    type=int,  default=5,    help="Number of results (default 5)")
    parser.add_argument("--lang", default=None,            help="Language filter: en or zh")
    parser.add_argument("--eval", action="store_true",     help="Run all 5 evaluation questions")
    args = parser.parse_args()

    if args.eval:
        print("── Running evaluation queries ────────────────────────────────────────")
        for label, q in EVAL_QUERIES:
            results = query_chunks(q, k=args.k, language=args.lang)
            _print_results(label, q, results)
        print(f"\n{'─'*68}")
        print("Done.")

    elif args.question:
        results = query_chunks(args.question, k=args.k, language=args.lang)
        _print_results("Ad-hoc query", args.question, results)
        print()

    else:
        parser.print_help()
        sys.exit(1)
