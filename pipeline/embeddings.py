from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def build_index(chunks: list[str]):
    """Build FAISS index from text chunks."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return model, index


def encode_query(model, query: str) -> np.ndarray:
    """Encode a query into a vector."""
    return model.encode([query])
