from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def build_index(chunks: list[str]):
    """Build FAISS index from text chunks."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.asarray(embeddings, dtype="float32"))
    return model, index


def encode_query(model, query: str) -> np.ndarray:
    """Encode a query into a normalised vector."""
    return model.encode([query], normalize_embeddings=True)
