import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

st.title("Recall")
st.write("My local study assistant")

upload_file = st.file_uploader("Upload your file here")
question_input = st.text_input("Type your question here")

if upload_file:
    st.write("File uploaded successfully")
    reader = PdfReader(upload_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # --- chunk ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # --- embed ---
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    # --- store in FAISS ---
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # --- retrieve and generate ---
    submit = st.button("Search")
    if submit and question_input:
        query_embedding = model.encode([question_input])
        D, I = index.search(np.array(query_embedding), k=3)
        retrieved_chunks = [chunks[i] for i in I[0]]

        # --- build prompt ---
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""You are a study assistant. Answer ONLY using the context below.
If the answer is not in the context, say 'I cannot find that in the document.'

Context:
{context}

Question: {question_input}
Answer:"""

        # --- generate ---
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
