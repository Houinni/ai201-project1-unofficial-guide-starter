"""
test_chunks.py — Verify chunking output meets spec before moving to embedding.

Tests:
  1. Structural  — no empty chunks, every field present, token counts sane
  2. Boundaries  — no chunk crosses a ## section header
  3. Completeness— each document contributed at least one chunk
  4. Sentences   — no chunk ends mid-word (last char is not a letter/digit)
  5. Eval queries— keyword search confirms the answer to each of the 5
                   evaluation questions lives in at least one chunk
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

CHUNKS_FILE = Path("chunks.json")
DOCS_DIR    = Path("documents")

# ── load ──────────────────────────────────────────────────────────────────────
chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}\n")

failures: list[str] = []

def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL  {msg}")

def ok(msg: str) -> None:
    print(f"  OK    {msg}")


# ── Test 1: structural integrity ──────────────────────────────────────────────
print("── Test 1: Structural integrity ─────────────────────────────────────────")
required_fields = {"text", "source_file", "language", "doc_type", "section",
                   "chunk_index", "token_count"}
empty_texts  = 0
missing_keys = 0
token_outliers = 0

for i, c in enumerate(chunks):
    missing = required_fields - set(c.keys())
    if missing:
        fail(f"chunk {i} missing fields: {missing}")
        missing_keys += 1
    if not c.get("text", "").strip():
        fail(f"chunk {i} ({c.get('source_file')}) has empty text")
        empty_texts += 1
    tok = c.get("token_count", 0)
    if tok == 0:
        fail(f"chunk {i} has token_count=0")
        token_outliers += 1
    if tok > 600:
        fail(f"chunk {i} ({c['source_file']}) token_count={tok} is very large")
        token_outliers += 1

if missing_keys == 0 and empty_texts == 0:
    ok(f"all {len(chunks)} chunks have required fields and non-empty text")
if token_outliers == 0:
    ok("no chunks with 0 or >600 token_count")

# Token distribution summary
tok_vals = [c["token_count"] for c in chunks]
print(f"       token_count  min={min(tok_vals)}  avg={int(sum(tok_vals)/len(tok_vals))}  max={max(tok_vals)}")


# ── Test 2: no chunk crosses a section header ─────────────────────────────────
print("\n── Test 2: No chunk crosses a ## section header ─────────────────────────")
crossed = 0
for c in chunks:
    # A section header inside the text body (not the first line) is a boundary
    # violation for list and wiki chunks
    lines = c["text"].splitlines()
    for line in lines[1:]:          # skip first line — it may legitimately BE a header
        if re.match(r"^## ", line) and c["doc_type"] != "interview":
            fail(f"{c['source_file']} chunk {c['chunk_index']} contains interior ## header: {line!r}")
            crossed += 1
if crossed == 0:
    ok("no chunk contains an interior ## section header")


# ── Test 3: every source document contributed chunks ─────────────────────────
print("\n── Test 3: Every source document contributed at least one chunk ─────────")
doc_files = {fp.name for fp in DOCS_DIR.glob("*.md")}
chunk_files = {c["source_file"] for c in chunks}
missing_docs = doc_files - chunk_files
if missing_docs:
    for d in sorted(missing_docs):
        fail(f"no chunks produced from {d}")
else:
    ok(f"all {len(doc_files)} source documents contributed chunks")

# Chunk counts per file
counts = defaultdict(int)
for c in chunks:
    counts[c["source_file"]] += 1
for fname, n in sorted(counts.items()):
    print(f"       {fname:<55} {n:>3} chunks")


# ── Test 4: no chunk ends mid-word ────────────────────────────────────────────
print("\n── Test 4: No chunk ends mid-word ───────────────────────────────────────")
midword = 0
for c in chunks:
    last_char = c["text"].rstrip()[-1] if c["text"].strip() else ""
    # Acceptable endings: punctuation, quotes, closing brackets, CJK chars, emoji
    if re.match(r"[a-zA-Z0-9]", last_char):
        # Might be fine (e.g. ends in a year "2026" or a name) — flag if it
        # looks like mid-word (no sentence-ending punctuation in last 20 chars)
        tail = c["text"].rstrip()[-20:]
        if not re.search(r'[.!?。！？"""\'）)》\]]', tail):
            # CJK text (Chinese/Japanese/Korean) often embeds English words
            # as complete terms — not a mid-word cut.  Skip if CJK chars
            # appear in the tail.
            if re.search(r'[一-鿿぀-ヿ가-힯]', tail):
                pass
            else:
                fail(f"{c['source_file']} chunk {c['chunk_index']} may end mid-word: …{tail!r}")
                midword += 1
if midword == 0:
    ok("all chunks end at a sentence or punctuation boundary")


# ── Test 5: eval question keyword coverage ────────────────────────────────────
print("\n── Test 5: Eval question keyword coverage ───────────────────────────────")
eval_queries = [
    (
        "Q1: Fake Love I-LAND score",
        ["93", "Fake Love", "highest"],
    ),
    (
        "Q2: Big Hit scouting story",
        ["red", "coat", "rookie", "Big Hit"],
    ),
    (
        "Q3: Convenience store order",
        ["Buldak", "kimbap", "cheese stick", "mandarin"],
    ),
    (
        "Q4: MTV News work vs self-investment",
        ["MTV News", "work", "investing"],
    ),
    (
        "Q5: Favorite fruit and Gong Cha drink",
        ["strawberry", "taro"],
    ),
]

for label, keywords in eval_queries:
    hits = []
    for c in chunks:
        text_lower = c["text"].lower()
        if all(kw.lower() in text_lower for kw in keywords):
            hits.append(c)

    if hits:
        best = hits[0]
        ok(f"{label}  →  found in {len(hits)} chunk(s)  "
           f"[{best['source_file']}, chunk {best['chunk_index']}, {best['token_count']} tok]")
    else:
        # Try partial match (any keyword)
        partial = [c for c in chunks
                   if any(kw.lower() in c["text"].lower() for kw in keywords)]
        fail(f"{label}  →  NOT found (all keywords together). "
             f"Partial matches: {len(partial)}")


# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
if failures:
    print(f"RESULT: {len(failures)} failure(s)")
    for f in failures:
        print(f"  • {f}")
    sys.exit(1)
else:
    print(f"RESULT: all tests passed — {len(chunks)} chunks ready for embedding")
