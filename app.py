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
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return chunks, model, index


st.title("Recall")
st.write("My local study assistant")

upload_file = st.file_uploader("Upload your file here")
if "clear_count" not in st.session_state:
    st.session_state.clear_count = 0

question_input = st.text_input(
    "Type your question here",
    key=f"question_{st.session_state.clear_count}"
)

if st.button("Clear"):
    st.session_state.clear_count += 1
    st.rerun()


if upload_file:
    st.write("File uploaded successfully")
    chunks, model, index = load_pipeline(upload_file)

    submit = st.button("Search")
    if submit and question_input:
        query_embedding = model.encode([question_input])
        D, I = index.search(np.array(query_embedding), k=3)
        retrieved_chunks = [chunks[i] for i in I[0]]

        context = "\n\n".join(retrieved_chunks)
        prompt = f"""You are a study assistant. Answer ONLY using the context below.
If the answer is not in the context, say 'I cannot find that in the document.'

Context:
{context}

Question: {question_input}
Answer:"""

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response["message"]["content"]

        st.write("### Answer")
        st.write(answer)

        st.write("### Source passages")
        for chunk in retrieved_chunks:
            st.write(chunk)
            st.write("---")
