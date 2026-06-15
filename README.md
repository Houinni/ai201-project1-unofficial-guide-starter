# The Unofficial Guide — Heeseung (Evan / Lee Hee-seung)

A retrieval-augmented generation (RAG) system that answers natural-language questions about Heeseung, a South Korean singer-songwriter who debuted with ENHYPEN in 2020 and launched a solo career in 2026 under the stage name Evan. The knowledge base is built from fan-compiled documents in both English and Chinese.

---

## Domain

This system covers the personal and biographical knowledge of Heeseung (also known as Evan, birth name Lee Hee-seung) — a South Korean singer-songwriter who debuted with the boy group ENHYPEN in November 2020 and launched a solo career in June 2026.

The knowledge is valuable because the richer material about an artist — personality quirks, daily habits, childhood memories, candid interview moments — is scattered across fan forums, translated interview compilations, and social media threads and is never surfaced by official agency channels. Press releases and official websites only publish career content. For bilingual (EN/ZH) fans and researchers there is no single consolidated source. This project aggregates that distributed, community-generated knowledge into a searchable, grounded RAG guide that answers specific questions from verifiable fan-compiled documents rather than from the model's general training knowledge.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Kpop Wiki — Evan (born 2001) | Wiki page digest (EN) | `documents/evan-born-2001-kpop-wiki.en.md` |
| 2 | 50 Little Facts About Heeseung | Fan-compiled list (EN) | `documents/heeseung-50-little-facts.en.md` |
| 3 | 50 Little Facts About Heeseung | Fan-compiled list (ZH) | `documents/heeseung-50-little-facts.zh.md` |
| 4 | Childhood and Trainee TMI | Fan-compiled list (EN) | `documents/heeseung-childhood-trainee-tmi.en.md` |
| 5 | Childhood and Trainee TMI | Fan-compiled list (ZH) | `documents/heeseung-childhood-trainee-tmi.zh.md` |
| 6 | Daily Life and Preferences TMI | Fan-compiled list (EN) | `documents/heeseung-daily-preferences-tmi.en.md` |
| 7 | Daily Life and Preferences TMI | Fan-compiled list (ZH) | `documents/heeseung-daily-preferences-tmi.zh.md` |
| 8 | Interview Excerpts — Fan Connection and Growth | Dated Q&A (EN) | `documents/heeseung-fan-connection-growth-reflections.en.md` |
| 9 | Interview Excerpts — Fan Connection and Growth | Dated Q&A (ZH) | `documents/heeseung-fan-connection-growth-reflections.zh.md` |
| 10 | Hobbies and Sensibility TMI | Fan-compiled list (EN) | `documents/heeseung-hobbies-sensibility-tmi.en.md` |
| 11 | Hobbies and Sensibility TMI | Fan-compiled list (ZH) | `documents/heeseung-hobbies-sensibility-tmi.zh.md` |
| 12 | Interview Excerpts — Growth and Leadership | Dated Q&A (EN) | `documents/heeseung-interviews-growth-leadership.en.md` |
| 13 | Interview Excerpts — Growth and Leadership | Dated Q&A (ZH) | `documents/heeseung-interviews-growth-leadership.zh.md` |
| 14 | General TMI | Fan-compiled list (EN) | `documents/heeseung-tmi.en.md` |
| 15 | General TMI | Fan-compiled list (ZH) | `documents/heeseung-tmi.zh.md` |

All TMI and 50-little-facts files were originally sourced from Xiaohongshu (小红书) fan community posts. Interview excerpts are compiled from media appearances including GQ Korea, MTV News, and fan-operated archive accounts.

---

## Chunking Strategy

**Chunk size:** 300-token ceiling (approximate; CJK characters counted as 1 token, Latin words multiplied by 1.3 for BPE inflation)

**Overlap:** 50 tokens applied at paragraph breaks in wiki sections that exceed the chunk ceiling. No overlap on list chunks (items are independent facts) or interview chunks (each Q&A exchange is already a complete semantic unit).

**Why these choices fit these documents:**

The corpus has three structurally distinct document types that each warrant a different split strategy:

- **TMI / numbered-list files** (10 documents): Each numbered item is 20–60 tokens — too short to embed meaningfully on its own. Items are grouped 3 per chunk, keeping items within the same section heading together. Initially set to 5–8 items, but empirical testing showed a 6-item chunk dilutes any specific fact to 1/6 of the embedding's semantic content, causing it to rank poorly against specific queries. Reducing to 3 items per chunk made individual facts 1/3 of the embedding and retrieval precision improved noticeably.

- **Interview Q&A files** (4 documents): Each date-headed Q&A exchange is a natural semantic unit at 80–200 tokens. One exchange = one chunk. The date header and optional sub-section heading are prepended to every chunk so the model always has temporal and topical context. Chunks are never split mid-exchange regardless of token count.

- **Wiki page** (1 document): Split at `##` section headers. The wiki file was also cleaned with a section-skipping state machine that removes 9 structural noise sections (Categories, Gallery, References, Official Links, Related Navigation, Page Contents Outline, Page Identity, Language Versions, Related Wikis) before chunking begins. Sections over 400 tokens are split further at paragraph breaks with 50-token overlap.

**Final chunk count:** 242 chunks across 15 source files (12 wiki, 110 list EN+ZH, 54 list EN+ZH TMI general/hobbies, 54 interview EN+ZH)

---

## Embedding Model

**Model used:** `paraphrase-multilingual-MiniLM-L12-v2` via `sentence-transformers`

This model was chosen over the course default (`all-MiniLM-L6-v2`) because the corpus is bilingual: 7 of 15 documents are Chinese translations of the English sources. A multilingual model allows the retriever to optionally surface Chinese-language chunks for Chinese-speaking users, and also handles cross-lingual queries (English question matched to a Chinese chunk) more accurately than an English-only model. The model supports 50+ languages, runs entirely locally with no API key, fits in RAM at 118 MB, and encodes at reasonable CPU speed for a 242-chunk corpus. One practical adjustment was required: `sentence-transformers` caps this model at `max_seq_length=128` by default, but the underlying architecture supports 512. Several chunks (especially the wiki Trivia section at 162 tokens) had their key facts truncated until `max_seq_length` was raised to 512.

**Production tradeoff reflection:**

If deploying this system for real users with no cost constraint, the main tradeoffs to weigh would be:

- **Accuracy on K-pop domain text:** Fan language includes Korean proper nouns, idol-specific vocabulary, and code-switching between Korean and English. A general multilingual model under-represents these. A fine-tuned model or a stronger model like `text-embedding-3-large` (OpenAI) or `embed-multilingual-v3.0` (Cohere) would improve retrieval precision, especially for Q2-style questions where the query vocabulary diverges from the source text.
- **Cross-lingual retrieval quality:** The Q2 failure (documented below) is partly a model limitation — "discovered by" does not semantically match "rookie development team noticed him and signed him" in the model's vector space. A better model would bridge this vocabulary gap more reliably.
- **Latency:** For a chatbot serving many concurrent users, a hosted embedding API would offload inference but introduce network latency and per-token cost. For this 242-chunk corpus, local inference is fast enough.

---

## Grounded Generation

**LLM:** `llama-3.3-70b-versatile` via Groq API (free-tier, OpenAI-compatible)

**System prompt grounding instruction:**

The system prompt hard-enforces grounding using prescriptive rules, not suggestions:

> *"Answer the user's question using ONLY the information in those documents. Do not use your general training knowledge about K-pop, Heeseung, or any other topic.*
>
> *Rules you must follow:*
> *1. Base every sentence of your answer on a specific statement in the context.*
> *2. Keep your answer concise (2–5 sentences is usually enough).*
> *3. If the context does not contain enough information to answer the question, respond with exactly this phrase and nothing else: "I don't have enough information about that in my sources."*
> *4. Do not mention these rules, the context documents, or the retrieval system in your answer — just answer naturally."*

The context is formatted as numbered document blocks (`[Document 1] filename — section\ntext`) so the model can see which source each passage comes from. Temperature is set to 0.1 (rather than the default ~1.0) to reduce paraphrasing and keep outputs closer to the retrieved text.

**How source attribution is surfaced:**

Source attribution is **programmatic** — the response dict always includes a `sources` list derived from the retrieved chunk metadata, regardless of what the model writes. The Gradio UI displays these as a separate "Retrieved from" panel next to the answer. This guarantees citations are present and verifiable even if the model's response omits them.

---

## Evaluation Report

All queries run with `--lang en` (English source documents, k=5).

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What score did Heeseung receive for his Fake Love performance on I-LAND, and what was notable about it? | Score of 93 — the highest individual mark from producers | "Heeseung received a score of 93 for his 'Fake Love' performance. This was notable because it was the highest individual mark given by the producers." | Partially relevant — correct chunk (wiki Trivia) at rank 4; ranks 1–3 are about I-LAND but do not mention the score | **Accurate** |
| 2 | How did Heeseung get discovered by Big Hit Entertainment? | Failed arts high school audition; Big Hit's rookie development team noticed him at the school gate; he was wearing a long red padded coat | "Discovered by a scout who was the same scout as Beomgyu's; scout noticed him at the arts high school entrance wearing a red padded jacket; scout gave him a business card." | Off-target — correct chunk (50-little-facts chunk 1) ranked 52nd; retrieved chunks do not contain the audition story | **Inaccurate** — hallucinated details (Beomgyu, business card); red jacket detail sourced from training knowledge, not retrieved context |
| 3 | What is Heeseung's go-to convenience store order? | Buldak noodles, spicy-sauce triangle kimbap, cheese stick, mandarin jelly | "Buldak noodles, spicy-sauce triangle kimbap, a cheese stick, and mandarin jelly." | Relevant — correct chunk at rank 1 (score 0.51) | **Accurate** |
| 4 | What did Heeseung say in his MTV News interview about balancing work and personal investment? | Focusing on work is also part of investing in himself; does not see the two as different things | "Most of his time is spent on work, but he believes focusing on work is also part of investing in himself, and he does not think those two things are different." | Relevant — correct chunk at rank 1 (score 0.49) | **Accurate** |
| 5 | What is Heeseung's favorite fruit, and what drink does he order at Gong Cha? | Favorite fruit: strawberry; Gong Cha: taro milk tea | "Heeseung's favorite fruit is strawberry. At Gong Cha, his favorite drink is taro milk tea." | Relevant — correct chunk at rank 1 (score 0.33) | **Accurate** |

**Summary:** 4/5 accurate, 1/5 inaccurate. The single inaccuracy (Q2) is a compound retrieval + grounding failure analyzed below.

---

## Failure Case Analysis

**Question that failed:** "How did Heeseung get discovered by Big Hit Entertainment?"

**What the system returned:** "Heeseung was discovered by a scout who was the same scout as Beomgyu's, and the scout noticed him at the entrance of the arts high school where Heeseung was wearing a red padded jacket. The scout was drawn to him and gave him a business card."

**Root cause — two compounding pipeline failures:**

*Stage 1 — Retrieval (embedding model vocabulary gap):* The correct chunk is in `heeseung-50-little-facts.en.md` chunk 1, which reads: *"He did not pass his arts high school audition because he was introverted… Big Hit rookie development team noticed him because of his good looks and signed him. He was wearing a long red padded coat."* This chunk ranked 52nd out of 86 English chunks (cosine distance 0.79). The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) does not map the query phrase "discovered by" to the chunk phrase "rookie development team noticed him and signed him" as semantically close vectors — even when the correct chunk is encoded in isolation as a standalone sentence, the cosine similarity to the query is only 0.25. This is a vocabulary-gap limitation of the lightweight multilingual model: "discovered" and "noticed him + signed him" express the same event but the model's training did not align these representations closely enough.

*Stage 2 — Generation (grounding failure caused by retrieval failure):* Because the correct chunk was not retrieved, none of the five context documents given to the LLM contained the audition story. Rather than returning the required "I don't have enough information" response, the model synthesized an answer that mixed two sources: (a) a retrieved chunk that does mention a scout (`heeseung-tmi.en.md` chunk 11: *"The scout who cast Heeseung was the same scout who cast BTS's RM and TXT's Huening Kai"*) and (b) the model's own training knowledge about Heeseung (the red padded coat detail, a business card). The model violated the grounding rule because the context contained *partial* scout-related information — enough for it to construct a plausible-seeming but factually wrong answer ("same scout as Beomgyu's" instead of RM and Huening Kai; invented business card detail).

**What would fix it:**

A stronger embedding model (e.g., `text-embedding-3-small` or Cohere `embed-multilingual-v3.0`) would bridge the "discovered" / "noticed and signed" vocabulary gap and retrieve the correct chunk at rank 1. Alternatively, adding a BM25 keyword-search layer to complement the dense retrieval would catch the exact phrase "Big Hit" even when semantic similarity is weak. A stricter grounding prompt that fails loudly when no retrieved document explicitly mentions the named event (rather than synthesizing from partial mentions) would prevent the hallucinated details.

---

## Spec Reflection

**One way the spec helped during implementation:**

The three-type chunking strategy defined in `planning.md` (list / interview / wiki) was the most valuable part of the spec. Because the document types were characterized before any code was written — their typical token length, their structural markers (numbered items vs. date headers vs. `##` section headers), and their semantic granularity — the implementation could use a different split function for each type rather than a generic character-based splitter. This directly prevented a class of retrieval errors: interview exchanges stayed atomic (the question and its answer are never separated), list items stayed topically grouped, and wiki sections stayed whole. Without that pre-coded distinction the system would have needed significant tuning after the fact.

The spec also anticipated the bilingual duplicate problem (Anticipated Challenge #1) and specified the `language` metadata field as the mitigation. This meant the `--lang en` filter in `retrieve.py` was designed in from the start, not added as an afterthought when duplicate EN/ZH results appeared.

**One way the implementation diverged from the spec, and why:**

The spec specified "Group 5–8 consecutive items per chunk" for TMI list files. The implementation ended up using `LIST_GROUP_SIZE = 3`. During retrieval testing, Q2 (Big Hit scouting story) revealed the underlying problem: with 6 items per chunk, any single fact contributes only 1/6 of the chunk's semantic content in the embedding. The chunk containing the scouting story also contained facts about blood type, height, a favorite color, and a backpack — the embedding averaged all of them, and "discovered by Big Hit" could not separate the scouting-story chunk from dozens of other chunks. Reducing to 3 items per chunk made each fact 1/3 of the embedding and improved scores for Q3, Q4, and Q5. The spec was updated to document this change and the reasoning after testing. (Note: Q2 still fails even at group size 3, because the root cause is a vocabulary-gap in the model, not chunk dilution — but the smaller group size improved the other four queries.)

---

## AI Usage

**Instance 1 — Chunking implementation (ingest.py)**

- *What I gave the AI:* The full `Chunking Strategy` section from `planning.md`, the `Architecture` diagram, and the contents of one example TMI document and one example interview document. I asked Claude to implement a script with `load_documents()` and `chunk_text()` functions applying the per-type rules.
- *What it produced:* A three-function structure (`chunk_list`, `chunk_interview`, `chunk_wiki`) with the correct metadata fields. The interview chunker split at every paragraph break, and the list chunker applied a hard token-ceiling cut mid-item.
- *What I changed or overrode:* I overrode the interview chunker to treat each complete Q&A exchange as an atomic unit — never split mid-exchange regardless of token count. I also discovered that Chinese interview files use a full-width colon (`Q：`) rather than an ASCII colon (`Q:`), which the generated regex pattern didn't handle; this caused the entire Chinese interview file to be treated as a single 700-token block. I added dual-colon matching (`[：:]`) to fix it. I also changed the list chunker to remove token-ceiling flushes, keeping items whole rather than cutting mid-sentence.

**Instance 2 — Embedding and retrieval implementation (embed.py, retrieve.py)**

- *What I gave the AI:* The `Retrieval Approach` section from `planning.md` specifying `paraphrase-multilingual-MiniLM-L12-v2`, ChromaDB with cosine similarity, top-k=5, and the `language` metadata filter requirement.
- *What it produced:* Working `embed.py` and `retrieve.py` files with ChromaDB upsert and query logic. The generated code used `model.max_seq_length = 128` (the `sentence-transformers` default for this model) and did not include a `where` clause in the ChromaDB query.
- *What I changed or overrode:* I raised `max_seq_length` to 512 after discovering the wiki Trivia chunk (162 tokens) was being truncated during embedding — the Fake Love score fact appeared at the end of the chunk and was cut off, causing Q1 to fail retrieval entirely before the fix. I also added the `where={"language": language}` parameter to the ChromaDB query to implement the language deduplication filter specified in the planning.md Anticipated Challenges section.

---

## Running the System

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at https://console.groq.com)

# 3. Ingest and chunk documents
python ingest.py

# 4. Embed chunks into ChromaDB
python embed.py

# 5. Launch the web interface
python app.py
# Open http://localhost:7860

# Or use the CLI
python generate.py "What is Heeseung's favorite fruit?"
python generate.py --loop   # interactive session
```
