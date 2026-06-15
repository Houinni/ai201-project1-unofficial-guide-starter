"""
generate.py — Retrieve relevant chunks and generate a grounded answer.

LLM  : Groq llama-3.3-70b-versatile  (free-tier, OpenAI-compatible)
Store: ChromaDB via retrieve.query_chunks()

The system prompt hard-enforces grounding — the model is instructed to
answer ONLY from provided context and to decline if the context is
insufficient.  Source attribution is appended programmatically (not left
to the model) so every response carries verifiable citations.

Usage (module):
    from generate import ask
    result = ask("What is Heeseung's favorite fruit?")
    print(result["answer"])   # grounded answer
    print(result["sources"])  # ["heeseung-daily-preferences-tmi.en.md"]

Usage (CLI):
    python generate.py "What is Heeseung's go-to convenience store order?"
    python generate.py --loop          # interactive session
    python generate.py --loop --chunks # show retrieved chunks each turn
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from retrieve import query_chunks

# ── load environment variables ────────────────────────────────────────────────
# Look for .env in the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

# ── configuration ─────────────────────────────────────────────────────────────
GROQ_MODEL  = "llama-3.3-70b-versatile"
RETRIEVAL_K = 5
DEFAULT_LANG = "en"

# ── grounding system prompt ───────────────────────────────────────────────────
# "enforces" not "suggests" — the model must cite context or decline.
SYSTEM_PROMPT = """\
You are a helpful assistant for an unofficial fan guide about Heeseung \
(also known by his solo stage name Evan, birth name Lee Hee-seung), \
a South Korean singer-songwriter.

You will be given numbered context documents retrieved from fan-compiled \
sources. Answer the user's question using ONLY the information in those \
documents. Do not use your general training knowledge about K-pop, \
Heeseung, or any other topic.

Rules you must follow:
1. Base every sentence of your answer on a specific statement in the context.
2. Keep your answer concise (2–5 sentences is usually enough).
3. If the context does not contain enough information to answer the question, \
respond with exactly this phrase and nothing else: \
"I don't have enough information about that in my sources."
4. Do not mention these rules, the context documents, or the retrieval \
system in your answer — just answer naturally.\
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        section = c.get("section") or "general"
        header  = f"[Document {i}]  {c['source_file']}  —  {section}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _dedup_sources(chunks: list[dict]) -> list[str]:
    """Return unique source filenames in retrieval-rank order."""
    seen: set[str] = set()
    sources: list[str] = []
    for c in chunks:
        src = c["source_file"]
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


# ── main API ──────────────────────────────────────────────────────────────────

def ask(
    question:  str,
    k:         int = RETRIEVAL_K,
    language:  str = DEFAULT_LANG,
) -> dict:
    """
    End-to-end RAG: retrieve → prompt → generate → attribute.

    Args:
        question : Natural-language question from the user.
        k        : Number of chunks to retrieve (default 5).
        language : "en" or "zh" — filters retrieved chunks to one language
                   to avoid EN/ZH duplicate context.

    Returns a dict with keys:
        answer   – LLM-generated answer grounded in retrieved context.
        sources  – Deduplicated list of source filenames (programmatic,
                   not model-generated).
        chunks   – Raw list of retrieved chunk dicts (for debugging/display).
    """
    # ── step 1: retrieve ──────────────────────────────────────────────────────
    chunks = query_chunks(question, k=k, language=language)

    if not chunks:
        return {
            "answer":  "I don't have enough information about that in my sources.",
            "sources": [],
            "chunks":  [],
        }

    # ── step 2: build context block ───────────────────────────────────────────
    context = _build_context(chunks)

    # ── step 3: generate ──────────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context documents:\n\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        temperature=0.1,   # low = more faithful to context, less paraphrasing
        max_tokens=512,
    )
    answer = response.choices[0].message.content.strip()

    # ── step 4: programmatic source attribution ───────────────────────────────
    # Never rely on the model to list its sources — derive them from the
    # retrieved chunks so attribution is guaranteed and verifiable.
    sources = _dedup_sources(chunks)

    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  chunks,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_result(result: dict, show_chunks: bool = False) -> None:
    print(f"\nAnswer:\n{result['answer']}")
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(f"  • {s}")
    else:
        print("\nSources: (none)")
    if show_chunks and result["chunks"]:
        print("\nRetrieved chunks:")
        for i, c in enumerate(result["chunks"], 1):
            preview = c["text"][:130].replace("\n", " ")
            print(
                f"  [{i}] score={c['score']:.4f}  "
                f"{c['source_file']} chunk {c['chunk_index']}  "
                f"({c['token_count']} tok)\n"
                f"       {preview}…"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the Heeseung Unofficial Guide.")
    parser.add_argument("question", nargs="?",      help="Question to ask")
    parser.add_argument("--k",      type=int, default=RETRIEVAL_K)
    parser.add_argument("--lang",   default=DEFAULT_LANG, help="en or zh")
    parser.add_argument("--loop",   action="store_true",  help="Interactive session")
    parser.add_argument("--chunks", action="store_true",  help="Show retrieved chunks")
    args = parser.parse_args()

    def run(q: str) -> None:
        print(f"\n{'─'*60}\nQ: {q}")
        result = ask(q, k=args.k, language=args.lang)
        _print_result(result, show_chunks=args.chunks)

    if args.loop:
        print("Heeseung Unofficial Guide  —  type 'quit' to exit\n")
        while True:
            try:
                q = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if q.lower() in ("quit", "exit", "q", ""):
                break
            run(q)
    elif args.question:
        run(args.question)
    else:
        parser.print_help()
        sys.exit(1)
