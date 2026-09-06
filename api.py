import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.ingestion import load_pdf, chunk_text, is_scanned_pdf
from pipeline.embeddings import build_index
from pipeline.retrieval import dense_search, bm25_search, reciprocal_rank_fusion
from pipeline.generation import generate_answer

app = FastAPI(title="Recall API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipeline_store = {}


class QueryRequest(BaseModel):
    question: str
    retrieval_mode: str = "Dense"


@app.get("/")
def root():
    return {"message": "Recall API is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if is_scanned_pdf(tmp_path):
            from pipeline.ocr import extract_text_from_scanned_pdf
            text, ocr_latency = extract_text_from_scanned_pdf(tmp_path)
        else:
            text = load_pdf(tmp_path)
            ocr_latency = None

        chunks = chunk_text(text)
        model, index = build_index(chunks)

        pipeline_store["chunks"] = chunks
        pipeline_store["model"] = model
        pipeline_store["index"] = index

        return {
            "message": "Document processed successfully",
            "chunks": len(chunks),
            "ocr_latency": ocr_latency
        }
    finally:
        os.unlink(tmp_path)


@app.post("/query")
def query_document(request: QueryRequest):
    """Query the processed document."""
    if "chunks" not in pipeline_store:
        return {"error": "No document uploaded yet"}

    chunks = pipeline_store["chunks"]
    model = pipeline_store["model"]
    index = pipeline_store["index"]

    if request.retrieval_mode == "Dense":
        results, latency = dense_search(model, index, chunks, request.question)
        retrieved_chunks = [chunk for chunk, _ in results]
        scores = [float(score) for _, score in results]

    elif request.retrieval_mode == "BM25":
        results, latency = bm25_search(chunks, request.question)
        retrieved_chunks = [chunk for chunk, _ in results]
        scores = [float(score) for _, score in results]

    else:
        dense_results, dense_latency = dense_search(
            model, index, chunks, request.question)
        bm25_results, bm25_latency = bm25_search(chunks, request.question)
        results, rrf_latency = reciprocal_rank_fusion(
            dense_results, bm25_results)
        latency = dense_latency + bm25_latency + rrf_latency
        retrieved_chunks = [chunk for chunk, _ in results]
        scores = [float(score) for _, score in results]

    answer = generate_answer("\n\n".join(retrieved_chunks), request.question)

    return {
        "answer": answer,
        "retrieval_mode": request.retrieval_mode,
        "latency": latency,
        "sources": [
            {"chunk": chunk, "score": score}
            for chunk, score in zip(retrieved_chunks, scores)
        ]
    }
