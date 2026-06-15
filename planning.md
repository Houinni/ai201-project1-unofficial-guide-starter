# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

This project builds an unofficial guide to Heeseung (Lee Hee-seung), a South Korean singer-songwriter who debuted with the boy group ENHYPEN in 2020 and launched a solo career in 2026 under the stage name Evan. The domain is K-pop artist personal and biographical knowledge — covering personality, daily preferences, childhood, trainee history, creative identity, and self-reflections from interviews — available in both English and Chinese.

This knowledge is valuable because fans actively seek the personal, human side of artists that official channels never surface. Agency websites, press releases, and promotional accounts only publish career content. The richer material — personality quirks, food preferences, growth philosophies, candid interview moments — is scattered across fan forums, Chinese fan community posts, translated interview compilations, and social media threads. For bilingual (EN/ZH) fans and researchers there is no single consolidated source. This project aggregates that distributed, community-generated knowledge into a searchable RAG-powered guide.

---

## Documents

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Kpop Wiki — Evan (born 2001) (EN) | Official wiki page digest: profile, career timeline, discography, filmography, writing credits, awards, and trivia | https://kpop.fandom.com/wiki/Evan_%28born_2001%29 |
| 2 | 50 Little Facts About Heeseung (EN) | Fan-compiled list of 49 personal facts covering childhood, family, trainee days, and personal quirks | documents/heeseung-50-little-facts.en.md | http://xhslink.com/o/1KENFxFLxWJ |
| 3 | 50 Little Facts About Heeseung (ZH) | Chinese translation of source #2 | documents/heeseung-50-little-facts.zh.md | http://xhslink.com/o/1KENFxFLxWJ |
| 4 | Childhood and Trainee TMI (EN) | Fan-compiled facts about his school life, casting story, and pre-debut journey at Big Hit | documents/heeseung-childhood-trainee-tmi.en.md | http://xhslink.com/o/8Hox5W2nBa9 |
| 5 | Childhood and Trainee TMI (ZH) | Chinese translation of source #4 | documents/heeseung-childhood-trainee-tmi.zh.md | http://xhslink.com/o/8Hox5W2nBa9 |
| 6 | Daily Life and Preferences TMI (EN) | Fan-compiled facts covering sleep habits, food and drink favorites, daily routines, and personal tastes | documents/heeseung-daily-preferences-tmi.en.md | http://xhslink.com/o/A3RLKz5kr3k |
| 7 | Daily Life and Preferences TMI (ZH) | Chinese translation of source #6 | documents/heeseung-daily-preferences-tmi.zh.md | http://xhslink.com/o/A3RLKz5kr3k |
| 8 | Interview Excerpts — Fan Connection and Growth (EN) | Dated Q&A excerpts from 2022–2024 media appearances on MBTI change, connecting with fans, and self-perception | documents/heeseung-fan-connection-growth-reflections.en.md | http://xhslink.com/o/6N3bwEsCP7K |
| 9 | Interview Excerpts — Fan Connection and Growth (ZH) | Chinese translation of source #8 | documents/heeseung-fan-connection-growth-reflections.zh.md | http://xhslink.com/o/6N3bwEsCP7K |
| 10 | Hobbies and Sensibility TMI (EN) | Fan-compiled facts about games, travel, aesthetics, food preferences, and sensory habits | documents/heeseung-hobbies-sensibility-tmi.en.md |
| 11 | Hobbies and Sensibility TMI (ZH) | Chinese translation of source #10 | documents/heeseung-hobbies-sensibility-tmi.zh.md |
| 12 | Interview Excerpts — Growth and Leadership (EN) | Dated Q&A excerpts from 2020–2021 media appearances (GQ, MTV News) on leadership, work ethic, and self-understanding | documents/heeseung-interviews-growth-leadership.en.md |  http://xhslink.com/o/6N3bwEsCP7K |
| 13 | Interview Excerpts — Growth and Leadership (ZH) | Chinese translation of source #12 | documents/heeseung-interviews-growth-leadership.zh.md |  http://xhslink.com/o/6N3bwEsCP7K |
| 14 | General TMI (EN) | Fan-compiled miscellaneous facts covering travel, food, notable personal moments, and recommendations | documents/heeseung-tmi.en.md | http://xhslink.com/o/AXELINqrtBQ |
| 15 | General TMI (ZH) | Chinese translation of source #14 | documents/heeseung-tmi.zh.md | http://xhslink.com/o/AXELINqrtBQ |

---

## Chunking Strategy

**Chunk size:** 300 tokens

**Overlap:** 50 tokens

**Reasoning:**

The corpus has three distinct structures that each warrant a different split rule, all targeting the same 300-token ceiling:

- **TMI / numbered-list files** (`*-tmi.md`, `50-little-facts.md`): Each numbered item is 20–60 tokens — too short to embed meaningfully on its own. Group **3 consecutive items** per chunk, keeping items within the same section heading together (e.g., all "Food and Drink" items stay in one chunk). This was initially set to 5–8 items, but empirical testing showed that with 6 items per chunk a specific fact is only 1/6 of the chunk's semantic content and gets diluted in the embedding — especially in flat lists like `50-little-facts` that have no section headers to provide topical separation. With 3 items, each fact is 1/3 of the chunk and retrieves more reliably against specific queries.
- **Interview Q&A files** (`*-interviews-*.md`, `*-reflections.md`): Each date-headed Q&A exchange (question + full answer) is a natural unit of meaning at roughly 80–200 tokens. One exchange = one chunk. The date header is prepended to every chunk as metadata context.
- **Wiki page** (`evan-born-2001-kpop-wiki.en.md`): Split at `##` section headers. Sections under 300 tokens stay whole; sections over 400 tokens are split further at paragraph breaks.

50-token overlap is applied only at narrative paragraph boundaries (wiki and interview files). List-based chunks use no overlap since numbered items are independent facts.

---

## Retrieval Approach

**Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` via `sentence-transformers`

**Top-k:** 5

**Production tradeoff reflection:**

The corpus is bilingual (EN + ZH), so a multilingual model is required. `paraphrase-multilingual-MiniLM-L12-v2` supports 50+ languages, is lightweight (118 MB), and runs fast on CPU — appropriate for a class project with no GPU.

If deploying for real users with no cost constraint, the tradeoffs to weigh would be:

- **Accuracy on domain-specific text**: K-pop fan language includes many Korean proper nouns, stage names, and idol-specific vocabulary. A general multilingual model may under-perform on these. A fine-tuned model on K-pop or entertainment corpora (if one existed) would improve retrieval precision. Alternatively, OpenAI's `text-embedding-3-large` or Cohere's `embed-multilingual-v3.0` offer stronger cross-lingual semantic alignment at higher cost.
- **Context length**: Most chunks are under 300 tokens, well within any model's limit. This is not a concern for this corpus.
- **Multilingual query matching**: If a user queries in English about a fact that only appears in a Chinese chunk (or vice versa), cross-lingual retrieval quality depends heavily on the model. `paraphrase-multilingual-MiniLM-L12-v2` handles this reasonably; production use would warrant evaluation with mixed-language test queries.
- **Latency**: For a chatbot serving many concurrent users, a hosted embedding API (Cohere, OpenAI) would offload inference cost but introduce network latency and per-token pricing.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What score did Heeseung receive for his Fake Love performance on I-LAND, and what was notable about it? | He received a score of 93 — the highest individual mark given by the producers on the show. |
| 2 | How did Heeseung get discovered by Big Hit Entertainment? | He failed his arts high school audition because he was introverted and did not know how to show himself. As he walked out, Big Hit's rookie development team noticed him for his looks and signed him on the spot. He was wearing a long red padded coat. |
| 3 | What is Heeseung's go-to convenience store order? | Buldak noodles, a spicy-sauce triangle kimbap, a cheese stick, and mandarin jelly. |
| 4 | What did Heeseung say in his MTV News interview about balancing work and personal investment? | He said that focusing on work is also part of investing in himself, and that he does not think of the two as different things. (MTV News interview, April 2021) |
| 5 | What is Heeseung's favorite fruit, and what drink does he order at Gong Cha? | His favorite fruit is strawberry. At Gong Cha, he orders taro milk tea. |

---

## Anticipated Challenges

1. **Duplicate retrieval across EN and ZH versions.** The English and Chinese documents contain identical facts. When a user queries in English, the retriever may return both the EN and ZH chunks for the same fact, filling the context window with redundant content and causing the generator to repeat itself. This is a real risk because the embeddings for a fact and its translation will be very close in vector space. Mitigation: attach a `language` metadata field to every chunk at ingestion time, and filter retrieved results to one language (defaulting to the query language) before passing to the generator.

2. **Mixed-topic chunks producing weak retrieval for specific queries.** The TMI files mix facts from different categories within the same numbered list (e.g., a food preference followed by a childhood memory followed by a travel note). If grouping by raw item count rather than by section header, a chunk could contain five unrelated facts, and its embedding will be a semantic average that ranks poorly against a specific query like "what does Heeseung eat at rest stops." Mitigation: split at the document's existing section headers first (e.g., "Food and Drink TMI" subsections), then group items within each section — never group across section boundaries.

---

## Architecture

```
                          BUILD TIME
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌─────────────────┐     ┌─────────────────┐     ┌───────────────┐  │
│  │  1. Ingestion   │────▶│  2. Chunking    │────▶│ 3. Embedding  │  │
│  │                 │     │                 │     │  + Storage    │  │
│  │ pathlib /       │     │ custom          │     │               │  │
│  │ glob.glob       │     │ chunk_text()    │     │ sentence-     │  │
│  │                 │     │ 300 tok chunks  │     │ transformers  │  │
│  │ reads all .md   │     │ 50 tok overlap  │     │ (multilingual │  │
│  │ from documents/ │     │ per-type rules  │     │ -MiniLM-L12)  │  │
│  └─────────────────┘     └─────────────────┘     │ + ChromaDB    │  │
│                                                   └───────────────┘  │
└──────────────────────────────────────────────────────────────────────┘

                          QUERY TIME
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  User query                                                          │
│      │                                                               │
│      ▼                                                               │
│  ┌─────────────────┐     ┌─────────────────┐     ┌───────────────┐  │
│  │  4. Retrieval   │────▶│ 5. Generation   │────▶│   Response    │  │
│  │                 │     │                 │     │               │  │
│  │ embed query     │     │ Claude API      │     │ answer with   │  │
│  │ cosine sim      │     │ (claude-3-5-    │     │ source refs   │  │
│  │ top-k = 5       │     │  haiku or       │     │               │  │
│  │ ChromaDB        │     │  sonnet)        │     │               │  │
│  └─────────────────┘     └─────────────────┘     └───────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**
- **Tool:** Claude (claude.ai chat)
- **Input:** The Chunking Strategy section of this file + the folder structure of `documents/` + one example document pasted in full
- **Expected output:** `ingest.py` containing a `load_documents(path)` function that reads all `.md` files and a `chunk_text(text, doc_type, chunk_size=300, overlap=50)` function that applies the per-type rules (list grouping vs. header splitting vs. paragraph splitting)
- **Verification:** Run `python ingest.py` and print the first 3 chunks from each document type; confirm that no chunk crosses a section header boundary and that list-based chunks contain 5–8 items

**Milestone 4 — Embedding and retrieval:**
- **Tool:** GitHub Copilot (for ChromaDB boilerplate) + Claude for the retrieval query logic
- **Input:** The Retrieval Approach section of this file + the output of `ingest.py` + ChromaDB quickstart docs
- **Expected output:** `embed.py` that loads chunks, embeds them with `paraphrase-multilingual-MiniLM-L12-v2`, and stores them in a local ChromaDB collection with metadata (`source_file`, `language`, `doc_type`); `retrieve.py` with a `query_chunks(question, k=5, language=None)` function that returns top-k chunks with scores
- **Verification:** Run all 5 evaluation questions from the Evaluation Plan through `retrieve.py` and confirm the chunk containing the correct answer appears in the top-5 results for each

**Milestone 5 — Generation and interface:**
- **Tool:** Claude Code
- **Input:** The Architecture diagram + Retrieval Approach section + `retrieve.py` interface + Claude API docs
- **Expected output:** `generate.py` with an `answer_question(question)` function that calls `query_chunks()`, builds a prompt with the retrieved chunks as context, and calls the Claude API (claude-3-5-haiku) to produce an answer; a simple CLI loop (`python generate.py`) that accepts user questions and prints answers with source file references
- **Verification:** Run all 5 evaluation questions end-to-end through the full pipeline and compare the generated answers against the expected answers in the Evaluation Plan; flag any question where the answer is factually wrong or the correct source chunk was not retrieved
