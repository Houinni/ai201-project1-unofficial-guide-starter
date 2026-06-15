"""
app.py — Gradio web interface for the Heeseung Unofficial Guide.

Run:
    python app.py
    open http://localhost:7860

The interface accepts a natural-language question, retrieves the top-5
relevant chunks from ChromaDB, sends them to Groq's LLM with a grounding
prompt, and displays the answer together with source attribution.
"""

from __future__ import annotations

import gradio as gr

from generate import ask, DEFAULT_LANG

# ── query handler ──────────────────────────────────────────────────────────────

def handle_query(question: str, language: str) -> tuple[str, str, str]:
    """
    Returns (answer, sources_text, debug_chunks_text).
    Called by every Gradio submit / button-click event.
    """
    question = question.strip()
    if not question:
        return "", "", ""

    result = ask(question, language=language)

    answer = result["answer"]

    sources = (
        "\n".join(f"• {s}" for s in result["sources"])
        if result["sources"]
        else "—"
    )

    # Debug view: show each retrieved chunk with its score and a text preview
    debug_lines: list[str] = []
    for i, c in enumerate(result["chunks"], 1):
        preview = c["text"][:160].replace("\n", " ")
        debug_lines.append(
            f"[{i}] score={c['score']:.4f}  "
            f"{c['source_file']}  chunk {c['chunk_index']}  "
            f"({c['token_count']} tok)\n"
            f"     {preview}…"
        )
    debug = "\n\n".join(debug_lines) if debug_lines else "—"

    return answer, sources, debug


# ── example questions ──────────────────────────────────────────────────────────

EXAMPLES = [
    "What is Heeseung's favorite fruit, and what does he order at Gong Cha?",
    "What score did Heeseung receive for his Fake Love performance on I-LAND?",
    "What is Heeseung's go-to convenience store order?",
    "What did Heeseung say in his MTV News interview about work and self-investment?",
    "What are Heeseung's favorite games?",
    "What did Heeseung say about his MBTI change from I to E?",
    "What is something your documents don't cover?",   # grounding test
]

# ── Gradio layout ──────────────────────────────────────────────────────────────

with gr.Blocks(title="Heeseung Unofficial Guide") as demo:

    gr.Markdown(
        "# 🦌 Heeseung Unofficial Guide\n"
        "Ask anything about Heeseung (Evan / Lee Hee-seung). "
        "Answers are grounded in fan-compiled source documents only — "
        "the system will say so if your question isn't covered."
    )

    with gr.Row():
        with gr.Column(scale=4):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="What is Heeseung's favorite fruit?",
                lines=2,
            )
        with gr.Column(scale=1, min_width=140):
            lang_radio = gr.Radio(
                choices=["en", "zh"],
                value=DEFAULT_LANG,
                label="Source language",
            )

    ask_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        answer_box = gr.Textbox(
            label="Answer",
            lines=7,
            interactive=False,
        )
        sources_box = gr.Textbox(
            label="Retrieved from",
            lines=7,
            interactive=False,
        )

    with gr.Accordion("Retrieved chunks (debug)", open=False):
        chunks_box = gr.Textbox(
            label="Top-5 chunks sent to the LLM",
            lines=14,
            interactive=False,
        )

    gr.Examples(
        examples=[[q, "en"] for q in EXAMPLES],
        inputs=[question_box, lang_radio],
        label="Try an example",
    )

    # wire up both button click and Enter-key submit
    ask_btn.click(
        handle_query,
        inputs=[question_box, lang_radio],
        outputs=[answer_box, sources_box, chunks_box],
    )
    question_box.submit(
        handle_query,
        inputs=[question_box, lang_radio],
        outputs=[answer_box, sources_box, chunks_box],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
