import json
import time
from pipeline.ingestion import load_pdf, chunk_text
from pipeline.embeddings import build_index
from pipeline.retrieval import (
    build_bm25_index,
    dense_search,
    bm25_search,
    reciprocal_rank_fusion,
)
from pipeline.generation import generate_answer


def load_document(pdf_path: str):
    """Load and index a PDF document."""
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    model, index = build_index(chunks)
    bm25 = build_bm25_index(chunks)
    return chunks, model, index, bm25


def evaluate_question(question: str, ground_truth: str, chunks, model, index, bm25, retrieval_mode: str):
    """Run a single question through the pipeline and return results."""

    if retrieval_mode == "Dense":
        results, latency = dense_search(model, index, chunks, question)
        retrieved_chunks = [chunk for chunk, _ in results]
        scores = [float(score) for _, score in results]

    elif retrieval_mode == "BM25":
        results, latency = bm25_search(bm25, chunks, question)
        retrieved_chunks = [chunk for chunk, _ in results]
        scores = [float(score) for _, score in results]

    else:
        dense_results, dense_latency = dense_search(
            model, index, chunks, question, k=20)
        bm25_results, bm25_latency = bm25_search(
            bm25, chunks, question, k=20)
        results, rrf_latency = reciprocal_rank_fusion(
            dense_results, bm25_results)
        latency = dense_latency + bm25_latency + rrf_latency
        retrieved_chunks = [chunk for chunk, _ in results]
        scores = [float(score) for _, score in results]

    context = "\n\n".join(retrieved_chunks)
    answer = generate_answer(context, question)

    return {
        "retrieval_mode": retrieval_mode,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "scores": scores,
        "latency": latency
    }


def run_evaluation(benchmark_path: str, document_paths: dict, output_path: str):
    """Run full evaluation across all questions and retrieval strategies."""

    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)

    print("Loading documents...")
    documents = {}
    for doc in benchmark["documents"]:
        doc_id = doc["id"]
        if doc_id in document_paths:
            print(f"Loading {doc['name']}...")
            chunks, model, index, bm25 = load_document(document_paths[doc_id])
            documents[doc_id] = {
                "chunks": chunks,
                "model": model,
                "index": index,
                "bm25": bm25
            }

    results = []
    total = len(benchmark["questions"])

    for i, q in enumerate(benchmark["questions"]):
        doc_id = q["document"]
        if doc_id not in documents:
            print(f"Skipping {q['id']} — document not loaded")
            continue

        print(f"Running {q['id']} ({i+1}/{total}): {q['question'][:50]}...")

        chunks = documents[doc_id]["chunks"]
        model = documents[doc_id]["model"]
        index = documents[doc_id]["index"]
        bm25 = documents[doc_id]["bm25"]

        question_result = {
            "id": q["id"],
            "document": doc_id,
            "type": q["type"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "results": {}
        }

        for mode in ["Dense", "BM25", "Hybrid (RRF)"]:
            try:
                result = evaluate_question(
                    q["question"],
                    q["ground_truth"],
                    chunks, model, index, bm25, mode
                )
                question_result["results"][mode] = result
                print(f"  {mode}: {result['latency']:.4f}s")
            except Exception as e:
                print(f"  {mode}: ERROR — {e}")
                question_result["results"][mode] = {"error": str(e)}

        results.append(question_result)

    output = {
        "benchmark": benchmark_path,
        "total_questions": len(results),
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nEvaluation complete. Results saved to {output_path}")
    return output


if __name__ == "__main__":
    document_paths = {
        "doc1": "data/FinalProjectTemplates.pdf",
        "doc2": "data/lewis2020rag.pdf",
        "doc3": "data/preliminary_report.pdf"
    }

    run_evaluation(
        benchmark_path="data/benchmark.json",
        document_paths=document_paths,
        output_path="data/results.json"
    )
