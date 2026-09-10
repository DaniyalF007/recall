import time

import numpy as np
from rank_bm25 import BM25Okapi


def build_bm25_index(chunks: list[str]):
    """Build a BM25 index from document chunks."""
    tokenised_chunks = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenised_chunks)


def dense_search(
    model,
    index,
    chunks: list[str],
    query: str,
    k: int = 3
):
    """Retrieve the top-k chunks using dense semantic retrieval."""
    start = time.perf_counter()

    query_embedding = model.encode([query])
    distances, positions = index.search(
        np.asarray(query_embedding),
        k
    )

    results = [
        (chunks[position], float(distances[0][rank]))
        for rank, position in enumerate(positions[0])
    ]

    latency = time.perf_counter() - start

    return results, latency


def bm25_search(
    bm25,
    chunks: list[str],
    query: str,
    k: int = 3
):
    """Retrieve the top-k chunks using a pre-built BM25 index."""
    start = time.perf_counter()

    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)

    top_k_indices = np.argsort(scores)[::-1][:k]

    results = [
        (chunks[index], float(scores[index]))
        for index in top_k_indices
    ]

    latency = time.perf_counter() - start

    return results, latency


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k: int = 60,
    top_k: int = 3
):
    """Combine dense and BM25 rankings using Reciprocal Rank Fusion."""
    start = time.perf_counter()

    scores = {}

    for rank, (chunk, _) in enumerate(dense_results):
        scores[chunk] = scores.get(chunk, 0.0) + (
            1 / (k + rank + 1)
        )

    for rank, (chunk, _) in enumerate(bm25_results):
        scores[chunk] = scores.get(chunk, 0.0) + (
            1 / (k + rank + 1)
        )

    sorted_chunks = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    latency = time.perf_counter() - start

    return sorted_chunks[:top_k], latency
