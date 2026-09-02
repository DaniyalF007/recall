import time
import numpy as np
from rank_bm25 import BM25Okapi


def dense_search(model, index, chunks: list[str], query: str, k: int = 3):
    """Dense retrieval using FAISS similarity search."""
    start = time.time()
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=k)
    results = [(chunks[i], D[0][rank]) for rank, i in enumerate(I[0])]
    latency = time.time() - start
    return results, latency


def bm25_search(chunks: list[str], query: str, k: int = 3):
    """Lexical retrieval using BM25."""
    start = time.time()
    tokenised_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenised_chunks)
    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)
    top_k_indices = np.argsort(scores)[::-1][:k]
    results = [(chunks[i], scores[i]) for i in top_k_indices]
    latency = time.time() - start
    return results, latency


def reciprocal_rank_fusion(dense_results, bm25_results, k: int = 60):
    """Combine dense and BM25 results using RRF."""
    start = time.time()
    scores = {}

    for rank, (chunk, _) in enumerate(dense_results):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)

    for rank, (chunk, _) in enumerate(bm25_results):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)

    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    latency = time.time() - start
    return sorted_chunks[:3], latency
