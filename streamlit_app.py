from __future__ import annotations

from pathlib import Path

import streamlit as st

from classifier.justifiers import JustificationError
from classifier.ui import classify_one, get_artifacts_status


ARTIFACTS_DIR = "artifacts"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_resource
def _load_model_and_metadata_cached(
    artifacts_dir: str,
    model_mtime: float,
    metadata_mtime: float,
):
    # Imported here so the app can render even if optional deps fail to import.
    from classifier.artifacts import load_metadata, load_model

    return load_model(artifacts_dir), load_metadata(artifacts_dir)


@st.cache_resource
def _load_retriever_cached(
    artifacts_dir: str,
    rag_fingerprint: tuple[float, float, float],
):
    from classifier.artifacts import load_rag_index

    return load_rag_index(artifacts_dir)


def main() -> None:
    st.set_page_config(page_title="Ticket Classifier", layout="wide")

    st.title("IT Service Ticket Classifier")
    st.caption("ML (TF-IDF + LinearSVC) classification with linear or optional LLM+RAG justifications.")

    status = get_artifacts_status(ARTIFACTS_DIR)
    if not (status.has_model and status.has_metadata):
        st.error(
            "Model artifacts not found.\n\n"
            f"- Expected: `{status.model_path}` and `{status.metadata_path}`\n"
            "- Train first: `uv run train dataset.csv`"
        )
        st.stop()

    with st.sidebar:
        st.header("Settings")
        justification_mode = st.radio(
            "Justification",
            options=["linear", "llm"],
            horizontal=True,
            help="Linear is deterministic; LLM uses RAG evidence + an OpenAI-compatible API.",
        )

        k_similar = 5
        reasoning_effort = None
        if justification_mode == "llm":
            if not status.has_rag:
                st.warning(
                    f"RAG index not found at `{status.rag_dir}`. "
                    "Train once with LLM/RAG enabled to build it."
                )
            k_similar = st.slider(
                "k similar tickets (RAG)",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
            )
            reasoning_effort = st.selectbox(
                "Reasoning effort (optional)",
                options=[None, "low", "medium", "high"],
                index=0,
            )

        show_details = st.checkbox("Show details", value=True)
        show_prompts = False
        if justification_mode == "llm":
            show_prompts = st.checkbox(
                "Show prompts",
                value=False,
                disabled=not show_details,
            )

    ticket = st.text_area(
        "Ticket text",
        height=220,
        placeholder="Paste the ticket text here...",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        run = st.button("Classify", type="primary")
    with col2:
        st.write("")

    if run:
        ticket_text = (ticket or "").strip()
        if not ticket_text:
            st.error("Ticket text is required.")
            return

        try:
            classifier, metadata = _load_model_and_metadata_cached(
                str(status.artifacts_dir),
                _mtime(status.model_path),
                _mtime(status.metadata_path),
            )

            retriever = None
            if justification_mode == "llm":
                if not status.has_rag:
                    st.error("RAG artifacts not found. Train once with LLM/RAG enabled to build the index.")
                    return
                rag_dir = status.rag_dir
                retriever = _load_retriever_cached(
                    str(status.artifacts_dir),
                    (
                        _mtime(rag_dir / "embeddings.npy"),
                        _mtime(rag_dir / "tickets.csv.gz"),
                        _mtime(rag_dir / "meta.json"),
                    ),
                )

            output, details = classify_one(
                ticket=ticket_text,
                classifier=classifier,
                justification_mode=justification_mode,
                k_similar=k_similar,
                reasoning_effort=reasoning_effort,
                retriever=retriever,
            )
        except JustificationError as e:
            st.error(f"Justification failed: {e.reason}")
            if e.raw_response:
                with st.expander("Raw response"):
                    st.code(e.raw_response)
            return
        except Exception as e:
            st.error(str(e))
            return

        st.subheader("Output")
        st.json(output)

        if show_details:
            st.subheader("Details")
            st.write(f"- predicted class: `{details.predicted_class}`")
            st.write(f"- justification source: `{details.justification_source}`")
            st.write(f"- retries: `{details.retries}`")
            st.write(
                "- token usage: "
                f"`prompt={details.token_usage.prompt_tokens}`, "
                f"`completion={details.token_usage.completion_tokens}`, "
                f"`total={details.token_usage.total_tokens}`"
            )

            if details.evidence_terms:
                st.write("Evidence terms")
                st.write(", ".join(f"`{t}`" for t in details.evidence_terms))

            if details.similar_tickets:
                with st.expander("Similar tickets (RAG evidence)", expanded=False):
                    rows = []
                    for t in details.similar_tickets:
                        text = (t.get("text") or "").strip().replace("\n", " ")
                        if len(text) > 240:
                            text = text[:240] + "..."
                        rows.append(
                            {
                                "class": t.get("class"),
                                "score": float(t.get("score") or 0.0),
                                "text": text,
                            }
                        )
                    st.dataframe(rows, use_container_width=True, hide_index=True)

            if show_prompts:
                if details.system_prompt:
                    with st.expander("System prompt", expanded=False):
                        st.code(details.system_prompt)
                if details.user_prompt:
                    with st.expander("User prompt", expanded=False):
                        st.code(details.user_prompt)
                if details.reasoning:
                    with st.expander("Reasoning", expanded=False):
                        st.code(details.reasoning)

            st.divider()
            st.caption(f"Available classes: {', '.join(metadata.classes)}")


if __name__ == "__main__":
    main()
