import ollama
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- ingest ---
reader = PdfReader("FinalProjectTemplates.pdf")
text = ""
for page in reader.pages:
    text = text + page.extract_text()

# --- chunk ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# --- embed all chunks ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# --- store in FAISS ---
dimension = embeddings.shape[1]          # 384
index = faiss.IndexFlatL2(dimension)     # create the searchable index
index.add(np.array(embeddings))          # add all chunk embeddings

print(f"Stored {index.ntotal} chunks in FAISS.")

# --- search ---
question = "What is needed for the Orchestrating AI template mditerm?"
question_embedding = model.encode([question])
distances, positions = index.search(np.array(question_embedding), k=3)

print("\nTop 3 most relevant chunks:\n")
for pos in positions[0]:
    print("----")
    print(chunks[pos])

# --- build the context from retrieved chunks ---
retrieved_chunks = [chunks[pos] for pos in positions[0]]
context = "\n\n".join(retrieved_chunks)

# --- build the prompt ---
prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I cannot find that in the document."

Context:
{context}

Question: {question}

Answer:"""

# --- ask the local LLM ---
response = ollama.chat(model="llama3", messages=[
    {"role": "user", "content": prompt}
])

print("\n========== ANSWER ==========\n")
print(response["message"]["content"])
