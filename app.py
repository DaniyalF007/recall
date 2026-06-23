import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama


@st.cache_resource
def load_pipeline(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return chunks, model, index


st.set_page_config(page_title="Recall", page_icon="📚")
st.title("📚 Recall")
st.write("A local RAG-powered study assistant")

upload_file = st.file_uploader("Upload a PDF", type=["pdf"])

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
            query_embedding = model.encode([question_input])
            D, I = index.search(np.array(query_embedding), k=3)
            retrieved_chunks = [chunks[i] for i in I[0]]

            context = "\n\n".join(retrieved_chunks)
            prompt = f"""You are Recall, a document-grounded study assistant.
Your task is to answer questions using ONLY the provided context.

Rules:
1. Use only information contained in the context.
2. Do not use outside knowledge.
3. Do not make assumptions or guesses.
4. If the answer is not in the context, say you cannot find it in the document.
5. Provide concise but complete answers.
6. When possible, answer in 1-3 sentences.
7. If the context contains a specific number, requirement, definition, or fact, include it directly in the answer.
8. Do not mention the context, chunks, or retrieval process.

Context:
{context}

Question: {question_input}
Answer: """

            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response["message"]["content"]

        st.success("Answer generated successfully")
        st.write("### Answer")
        st.write(answer)

        with st.expander("Retrieved Sources"):
            for rank, chunk_idx in enumerate(I[0]):
                st.write(f"### Chunk {rank + 1}")
                st.write(f"Distance Score: {D[0][rank]:.4f}")
                st.write(chunks[chunk_idx])
                st.write("---")
