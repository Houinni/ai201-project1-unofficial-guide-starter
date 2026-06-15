"""
ingest.py — Load, clean, and chunk source documents.

Chunking rules (planning.md § Chunking Strategy):

  list      (*-tmi.md, *-little-facts.md)
            Group 5–8 numbered items per chunk; never cross a ## section header.
            No overlap — list items are independent facts.

  interview (*-interviews-*.md, *-reflections.md)
            One Q&A exchange per chunk.  Date header + optional sub-section
            header are prepended to every chunk for context.
            50-token overlap when a single exchange exceeds the chunk ceiling.

  wiki      (evan-born-2001-kpop-wiki*.md)
            Split at ## section headers.  Sections > 400 tokens are further
            split at paragraph breaks with 50-token overlap.

Run:
    python ingest.py                  # chunks documents/ and saves chunks.json
    python ingest.py --sample         # also prints one sample chunk per doc type
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 300   # target token ceiling per chunk
OVERLAP         = 50    # overlap in tokens (wiki + interview narrative only)
LIST_GROUP_SIZE = 6     # default numbered items per list chunk  (5–8 range)

# Question-line patterns — match both ASCII colon (:) and full-width colon (：)
# used in Chinese-translated interview files
_Q_PATTERNS = re.compile(
    r'^(Q[：:]|MTV News[：:]|GQ[：:])\s*',
    re.IGNORECASE,
)
# Answer-line patterns (same dual-colon handling)
_A_PATTERNS = re.compile(
    r'^(A[：:]|🦌[：:]|Heeseung[：:])\s*',
)
# Date section header  e.g.  ## 221126
_DATE_HEADER = re.compile(r'^## (\d{6})\s*$')
# Sub-section header   e.g.  ### NETWORK
_SUB_HEADER  = re.compile(r'^###+ (.+)')
# Any ##-level section header
_ANY_HEADER  = re.compile(r'^##+ ')
# Numbered list item
_LIST_ITEM   = re.compile(r'^\d+\.')


# ── token counting ─────────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    """
    Approximate token count for mixed EN / ZH / KO text.
      · CJK characters  ≈ 1 token each (each char is usually one token in BPE)
      · Latin words      × 1.3  (BPE inflation factor for subword splitting)
    No external dependency — good enough for chunk-size gating.
    """
    cjk   = len(re.findall(r'[一-鿿぀-ヿ가-힯]', text))
    rest  = re.sub(r'[一-鿿぀-ヿ가-힯]', ' ', text)
    latin = len(re.findall(r'\b\w+\b', rest))
    return cjk + int(latin * 1.3)


# ── metadata helpers ───────────────────────────────────────────────────────────
def detect_language(filename: str) -> str:
    """'zh' for *.zh.md, 'en' for everything else."""
    return 'zh' if filename.endswith('.zh.md') else 'en'


def detect_doc_type(filename: str) -> str:
    """
    Classify by filename:
      'interview'  → *-interviews-*.md  or  *-reflections.md
      'wiki'       → *-wiki*.md
      'list'       → everything else (TMI files, 50-little-facts, etc.)
    """
    name = Path(filename).name
    if 'interview' in name or 'reflection' in name:
        return 'interview'
    if 'wiki' in name:
        return 'wiki'
    return 'list'


# ── cleaning ───────────────────────────────────────────────────────────────────

_HTML_ENTITIES = {
    '&amp;': '&', '&nbsp;': ' ', '&lt;': '<',
    '&gt;': '>', '&quot;': '"', '&#39;': "'",
}

def clean_document(text: str) -> str:
    """
    General cleaning applied to every document before chunking:
      - Remove HTML comments  <!-- ... -->
      - Decode common HTML entities  (&amp; &nbsp; &lt; etc.)
      - Strip markdown link URLs, keep display text  ([label](url) → label)
      - Remove metadata blockquotes  (> Note: / > Source: lines)
      - Normalise line endings and trailing whitespace
      - Remove bare horizontal-rule lines  (--- on its own line)
      - Collapse 3+ blank lines to 2
    """
    # HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # HTML entities (named)
    for entity, char in _HTML_ENTITIES.items():
        text = text.replace(entity, char)
    # HTML entities (numeric)
    text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
    # Markdown links → keep display text only
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    # Metadata blockquotes (> Note: ..., > Source: ..., > Warning: ...)
    text = re.sub(r'^> (?:Note|Source|Warning|Disclaimer):.*$', '', text, flags=re.MULTILINE)
    # Normalise line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Strip trailing whitespace per line
    text = '\n'.join(line.rstrip() for line in text.splitlines())
    # Remove bare horizontal rules
    text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# Sections in the wiki file that are page metadata, not biographical content.
# Removing these prevents low-signal chunks from polluting the vector store.
_WIKI_NOISE_SECTIONS = {
    'Page Identity',
    'Categories',
    'Language Versions Listed',
    'Related Wikis Listed',
    'Page Contents Outline',
    'Gallery',
    'References Listed On The Page',
    'Official Links Listed',
    'Related Navigation On The Evan Page',
}
# File-level header lines in the wiki document that are metadata, not content
_WIKI_META_RE = re.compile(r'^(Source page:|Accessed:|Note: This is an English)')


def clean_wiki(text: str) -> str:
    """
    Wiki-specific cleaning applied after clean_document():
      - Drop non-content sections (Categories, Language Versions, References,
        Official Links, Related Navigation, Gallery, Page Contents Outline,
        Page Identity)
      - Drop file-level metadata lines (Source page:, Accessed:, disclaimer)
    Anything under a kept ## header is preserved as-is.
    """
    lines = text.splitlines()
    result: list[str] = []
    skipping   = False
    skip_level = 0

    for line in lines:
        m = re.match(r'^(#+) ', line)
        if m:
            level = len(m.group(1))
            title = line[level + 1:].strip()

            if title in _WIKI_NOISE_SECTIONS:
                # Start skipping this section and all its sub-sections
                skipping   = True
                skip_level = level
            elif skipping and level > skip_level:
                # Sub-section of a noise section — keep skipping
                pass
            else:
                # New peer/parent section — stop skipping, keep this header
                skipping = False
                result.append(line)
        elif not skipping and not _WIKI_META_RE.match(line):
            result.append(line)

    joined = '\n'.join(result)
    # Re-collapse blank lines introduced by section removal
    joined = re.sub(r'\n{3,}', '\n\n', joined)
    return joined.strip()


# ── chunk dataclass ────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    text:        str
    source_file: str
    language:    str
    doc_type:    str
    section:     str
    chunk_index: int
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_count = count_tokens(self.text)


# ── list chunker ───────────────────────────────────────────────────────────────
def chunk_list(text: str, src: str, lang: str) -> list[Chunk]:
    """
    Groups 5–8 numbered items per chunk using only natural paragraph boundaries:
      · Flush at every ## section header  (never cross section boundaries)
      · Flush after LIST_GROUP_SIZE items have been collected
    Token count is NOT used as a hard cutoff — every item is always emitted
    complete so no sentence is ever split mid-way.
    No overlap.
    """
    chunks: list[Chunk] = []
    idx     = 0
    section = ''
    items:  list[str] = []

    def flush(section: str, items: list[str], idx: int) -> int:
        if not items:
            return idx
        prefix = f'{section}\n\n' if section else ''
        body   = (prefix + '\n'.join(items)).strip()
        chunks.append(Chunk(body, src, lang, 'list', section, idx))
        return idx + 1

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Section header → flush current batch before moving on
        if _ANY_HEADER.match(line):
            idx     = flush(section, items, idx)
            items   = []
            section = re.sub(r'^##+ ', '', line).strip()
            i += 1
            continue

        # Numbered item — collect continuation lines (indented or blank-free)
        if _LIST_ITEM.match(line):
            item_parts = [line]
            j = i + 1
            while (
                j < len(lines)
                and lines[j]
                and not _LIST_ITEM.match(lines[j])
                and not _ANY_HEADER.match(lines[j])
            ):
                item_parts.append(lines[j].strip())
                j += 1
            items.append(' '.join(item_parts))
            i = j

            # Flush only when group size is reached — never on token count
            if len(items) >= LIST_GROUP_SIZE:
                idx   = flush(section, items, idx)
                items = []
            continue

        i += 1

    flush(section, items, idx)  # trailing items
    return chunks


# ── interview chunker ──────────────────────────────────────────────────────────
def chunk_interview(text: str, src: str, lang: str) -> list[Chunk]:
    """
    Emits one chunk per Q&A exchange (or per standalone paragraph/message).
    Exchange boundary = the next question-starter line or date/sub-section header.
    Date header (## XXXXXX) and optional sub-section (### NAME) are
    prepended to every chunk for context.
    Chunks are NEVER split mid-exchange — every chunk is a semantically
    complete paragraph regardless of token count.
    """
    chunks: list[Chunk] = []
    idx       = 0
    date      = ''
    subsect   = ''
    exchange: list[str] = []

    def make_prefix(date: str, sub: str) -> str:
        parts = [p for p in (date, sub) if p]
        return '\n'.join(parts)

    def flush_exchange(date: str, sub: str, exchange: list[str], idx: int) -> int:
        # Drop lines that are purely document-level headings (h1 / h2 titles)
        content_lines = [l for l in exchange if not re.match(r'^# ', l)]
        body = '\n'.join(content_lines).strip()
        if not body or count_tokens(body) < 10:
            return idx
        prefix = make_prefix(date, sub)
        full   = (f'{prefix}\n\n{body}' if prefix else body).strip()
        # Always emit the full exchange — never split mid-paragraph
        chunks.append(Chunk(full, src, lang, 'interview', sub or date, idx))
        return idx + 1

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Date header  →  flush previous exchange, reset sub-section
        m_date = _DATE_HEADER.match(line)
        if m_date:
            idx     = flush_exchange(date, subsect, exchange, idx)
            date    = m_date.group(1)
            subsect = ''
            exchange = []
            i += 1
            continue

        # Sub-section header  →  flush previous exchange, keep date
        m_sub = _SUB_HEADER.match(line)
        if m_sub:
            idx     = flush_exchange(date, subsect, exchange, idx)
            subsect = m_sub.group(1).strip()
            exchange = []
            i += 1
            continue

        # New question line  →  flush previous exchange, start new one
        if _Q_PATTERNS.match(line):
            idx      = flush_exchange(date, subsect, exchange, idx)
            exchange = [line]
            i += 1
            continue

        # Everything else (answer lines, body paragraphs) — append to exchange
        if line or exchange:
            exchange.append(line)
        i += 1

    flush_exchange(date, subsect, exchange, idx)
    return chunks


# ── wiki chunker ───────────────────────────────────────────────────────────────
def chunk_wiki(text: str, src: str, lang: str) -> list[Chunk]:
    """
    Splits the document at every ## section header.
    Sections ≤ CHUNK_SIZE tokens → emitted as one chunk.
    Sections > CHUNK_SIZE tokens → split at paragraph breaks with OVERLAP
    token carry-over; the section header is re-prepended to every sub-chunk.
    """
    chunks: list[Chunk] = []
    idx = 0

    header_re = re.compile(r'^(##+ .+)$', re.MULTILINE)
    parts = header_re.split(text)

    # parts alternates:  [pre-content, header, body, header, body, ...]
    sections: list[tuple[str, str]] = []
    i = 0
    while i < len(parts):
        if re.match(r'^##+ ', parts[i].strip()):
            header = parts[i].strip()
            body   = parts[i + 1].strip() if i + 1 < len(parts) else ''
            sections.append((header, body))
            i += 2
        else:
            i += 1

    for header, body in sections:
        section_name = re.sub(r'^##+ ', '', header).strip()
        if not body.strip():
            # Skip header-only sections (body is empty because sub-sections
            # follow immediately, e.g. ## Discography → ### Digital Singles)
            continue
        full = f'{header}\n\n{body}'.strip()

        if count_tokens(full) <= CHUNK_SIZE:
            chunks.append(Chunk(full, src, lang, 'wiki', section_name, idx))
            idx += 1
            continue

        # Split at paragraph breaks; re-prepend header to every sub-chunk
        paragraphs = re.split(r'\n{2,}', body)
        buf: list[str] = [header]
        buf_toks = count_tokens(header)

        for para in paragraphs:
            pt = count_tokens(para)
            if buf_toks + pt > CHUNK_SIZE and len(buf) > 1:
                chunks.append(Chunk('\n\n'.join(buf).strip(), src, lang,
                                    'wiki', section_name, idx))
                idx += 1
                # Carry-over overlap: last paragraph + re-anchor to header
                tail     = buf[-1] if count_tokens(buf[-1]) <= OVERLAP else ''
                buf      = [header] + ([tail] if tail else []) + [para]
                buf_toks = count_tokens('\n\n'.join(buf))
            else:
                buf.append(para)
                buf_toks += pt

        if buf:
            chunks.append(Chunk('\n\n'.join(buf).strip(), src, lang,
                                'wiki', section_name, idx))
            idx += 1

    return chunks


# ── main ───────────────────────────────────────────────────────────────────────
def load_and_chunk(documents_dir: str = 'documents') -> list[Chunk]:
    """
    Load every *.md file in documents_dir, clean it, classify it, and chunk it.
    Returns a flat list of Chunk objects.
    """
    path  = Path(documents_dir)
    files = sorted(path.glob('*.md'))

    if not files:
        print(f'No .md files found in {path.resolve()}')
        return []

    all_chunks: list[Chunk] = []

    col = 55
    print(f"{'File':<{col}} {'Type':<10} {'Chunks':>6}  {'Avg tok':>7}  {'Max tok':>7}")
    print('─' * (col + 35))

    for fp in files:
        raw     = fp.read_text(encoding='utf-8')
        cleaned = clean_document(raw)
        lang    = detect_language(fp.name)
        dtype   = detect_doc_type(fp.name)

        if dtype == 'wiki':
            cleaned = clean_wiki(cleaned)

        if dtype == 'list':
            file_chunks = chunk_list(cleaned, fp.name, lang)
        elif dtype == 'interview':
            file_chunks = chunk_interview(cleaned, fp.name, lang)
        else:
            file_chunks = chunk_wiki(cleaned, fp.name, lang)

        all_chunks.extend(file_chunks)

        if file_chunks:
            avg_tok = int(sum(c.token_count for c in file_chunks) / len(file_chunks))
            max_tok = max(c.token_count for c in file_chunks)
        else:
            avg_tok = max_tok = 0

        print(f'{fp.name:<{col}} {dtype:<10} {len(file_chunks):>6}  {avg_tok:>7}  {max_tok:>7}')

    print('─' * (col + 35))
    print(f"{'TOTAL':<{col}} {'':<10} {len(all_chunks):>6}")
    return all_chunks


def print_samples(chunks: list[Chunk]) -> None:
    """Print one representative chunk per doc type."""
    print('\n── Sample chunk per doc type ───────────────────────────────────────────')
    seen: set[str] = set()
    for c in chunks:
        if c.doc_type not in seen:
            seen.add(c.doc_type)
            print(f'\n[{c.doc_type.upper()}]  {c.source_file}')
            print(f'section: {c.section!r}   tokens: {c.token_count}')
            print('─' * 60)
            # Truncate at the last complete line within 600 chars
            if len(c.text) > 600:
                cutoff = c.text[:600].rfind('\n')
                preview = c.text[:cutoff] if cutoff > 0 else c.text[:600]
                print(preview + '\n…')
            else:
                print(c.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load, clean, and chunk source documents.')
    parser.add_argument('--dir',    default='documents', help='Path to documents folder')
    parser.add_argument('--out',    default='chunks.json', help='Output JSON file')
    parser.add_argument('--sample', action='store_true',   help='Print sample chunks')
    args = parser.parse_args()

    print('Loading and chunking documents…\n')
    chunks = load_and_chunk(args.dir)

    if args.sample:
        print_samples(chunks)

    out = Path(args.out)
    with out.open('w', encoding='utf-8') as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)

    print(f'\n✓  {len(chunks)} chunks written to {out.resolve()}')
