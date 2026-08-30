import streamlit as st
from pipeline.ingestion import load_pdf, chunk_text
from pipeline.embeddings import build_index
from pipeline.retrieval import dense_search, bm25_search, reciprocal_rank_fusion
from pipeline.generation import generate_answer


@st.cache_resource
def load_pipeline(file):
    text = load_pdf(file)
    chunks = chunk_text(text)
    model, index = build_index(chunks)
    return chunks, model, index


st.set_page_config(page_title="Recall", page_icon="📚")
st.title("📚 Recall")
st.write("A local RAG-powered study assistant")

upload_file = st.file_uploader("Upload a PDF", type=["pdf"])

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

if upload_file:
    st.success("File uploaded successfully")
    chunks, model, index = load_pipeline(upload_file)

    if search and question_input:
        with st.spinner("Searching document..."):

            if retrieval_mode == "Dense":
                results = dense_search(model, index, chunks, question_input)
                retrieved_chunks = [chunk for chunk, _ in results]
                scores = [score for _, score in results]

            elif retrieval_mode == "BM25":
                results = bm25_search(chunks, question_input)
                retrieved_chunks = [chunk for chunk, _ in results]
                scores = [score for _, score in results]

            else:
                dense_results = dense_search(
                    model, index, chunks, question_input)
                bm25_results = bm25_search(chunks, question_input)
                results = reciprocal_rank_fusion(dense_results, bm25_results)
                retrieved_chunks = [chunk for chunk, _ in results]
                scores = [score for _, score in results]

            context = "\n\n".join(retrieved_chunks)
            answer = generate_answer(context, question_input)

        st.success("Answer generated successfully")
        st.write("### Answer")
        st.write(answer)

        with st.expander("Retrieved Sources"):
            for rank, chunk in enumerate(retrieved_chunks):
                st.write(f"### Chunk {rank + 1}")
                st.write(f"Score: {scores[rank]:.4f}")
                st.write(chunk)
                st.write("---")
