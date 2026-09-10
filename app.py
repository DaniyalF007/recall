import os
import tempfile

import streamlit as st

from pipeline.ingestion import (
    load_pdf,
    chunk_text,
    is_scanned_pdf,
)

from pipeline.audio import transcribe_audio

from pipeline.embeddings import build_index

from pipeline.retrieval import (
    build_bm25_index,
    dense_search,
    bm25_search,
    reciprocal_rank_fusion,
)

from pipeline.generation import generate_answer


@st.cache_resource
def load_pipeline(file):
    from pipeline.ocr import extract_text_from_scanned_pdf

    if is_scanned_pdf(file):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            text, ocr_latency = extract_text_from_scanned_pdf(
                tmp_path
            )

            st.info(
                f"Scanned PDF detected — OCR latency: "
                f"{ocr_latency:.4f} seconds"
            )

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    else:
        text = load_pdf(file)

    chunks = chunk_text(text)

    model, index = build_index(chunks)

    # Build BM25 once when the document is processed.
    # The index is reused for every query.
    bm25 = build_bm25_index(chunks)

    return chunks, model, index, bm25


st.set_page_config(
    page_title="Recall",
    page_icon="📚"
)

st.title("📚 Recall")
st.write("A local RAG-powered study assistant")


upload_type = st.radio(
    "Select input type",
    ["PDF", "Audio"]
)


if upload_type == "PDF":
    upload_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )
    audio_file = None

else:
    audio_file = st.file_uploader(
        "Upload an audio file",
        type=["mp3", "wav", "m4a"]
    )
    upload_file = None


retrieval_mode = st.selectbox(
    "Select retrieval strategy",
    ["Dense", "BM25", "Hybrid (RRF)"]
)


if "clear_count" not in st.session_state:
    st.session_state.clear_count = 0


question_input = st.text_input(
    "Ask a question about the document",
    key=f"question_{st.session_state.clear_count}"
)


col1, col2 = st.columns(2)

with col1:
    search = st.button("Search")

with col2:
    clear = st.button("Clear")


if clear:
    st.session_state.clear_count += 1
    st.rerun()


# =========================================================
# PDF PIPELINE
# =========================================================

if upload_file:

    st.success("File uploaded successfully")

    chunks, model, index, bm25 = load_pipeline(
        upload_file
    )

    if search and question_input:

        with st.spinner("Searching document..."):

            if retrieval_mode == "Dense":

                results, latency = dense_search(
                    model,
                    index,
                    chunks,
                    question_input
                )

                retrieved_chunks = [
                    chunk for chunk, _ in results
                ]

                scores = [
                    score for _, score in results
                ]

            elif retrieval_mode == "BM25":

                results, latency = bm25_search(
                    bm25,
                    chunks,
                    question_input
                )

                retrieved_chunks = [
                    chunk for chunk, _ in results
                ]

                scores = [
                    score for _, score in results
                ]

            else:

                dense_results, dense_latency = dense_search(
                    model,
                    index,
                    chunks,
                    question_input
                )

                bm25_results, bm25_latency = bm25_search(
                    bm25,
                    chunks,
                    question_input
                )

                results, rrf_latency = reciprocal_rank_fusion(
                    dense_results,
                    bm25_results
                )

                latency = (
                    dense_latency
                    + bm25_latency
                    + rrf_latency
                )

                retrieved_chunks = [
                    chunk for chunk, _ in results
                ]

                scores = [
                    score for _, score in results
                ]

            context = "\n\n".join(
                retrieved_chunks
            )

            answer = generate_answer(
                context,
                question_input
            )

        st.success(
            "Answer generated successfully"
        )

        st.info(
            f"Retrieval latency: {latency:.4f} seconds"
        )

        st.write("### Answer")
        st.write(answer)

        with st.expander("Retrieved Sources"):

            for rank, chunk in enumerate(
                retrieved_chunks
            ):

                st.write(
                    f"### Chunk {rank + 1}"
                )

                st.write(
                    f"Score: {scores[rank]:.4f}"
                )

                st.write(chunk)

                st.write("---")


# =========================================================
# AUDIO PIPELINE
# =========================================================

if audio_file:

    st.success(
        "Audio file uploaded successfully"
    )

    with st.spinner("Transcribing audio..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        ) as tmp:

            tmp.write(
                audio_file.read()
            )

            tmp_path = tmp.name

        try:

            transcript, latency = transcribe_audio(
                tmp_path
            )

        finally:

            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    st.success(
        "Transcription complete"
    )

    st.info(
        f"Transcription latency: "
        f"{latency:.4f} seconds"
    )

    st.write("### Transcript")
    st.write(transcript)
