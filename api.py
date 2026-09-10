from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.ingestion import (
    is_scanned_pdf,
    load_pdf,
    chunk_text,
)

from pipeline.ocr import extract_text_from_scanned_pdf

from pipeline.embeddings import build_index

from pipeline.retrieval import (
    build_bm25_index,
    dense_search,
    bm25_search,
    reciprocal_rank_fusion,
)

from pipeline.generation import generate_answer


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pipeline_store = {}


class QueryRequest(BaseModel):
    question: str
    retrieval_mode: str = "Dense"


@app.get("/")
def root():
    return {
        "message": "Recall API is running"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    contents = await file.read()

    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        scanned = is_scanned_pdf(temp_path)

        if scanned:
            text, ocr_latency = extract_text_from_scanned_pdf(
                temp_path
            )
        else:
            text = load_pdf(temp_path)
            ocr_latency = 0.0

        chunks = chunk_text(text)

        model, index = build_index(chunks)

        # Build BM25 index once when the document is uploaded.
        # It is then reused for every query.
        bm25 = build_bm25_index(chunks)

        pipeline_store["chunks"] = chunks
        pipeline_store["model"] = model
        pipeline_store["index"] = index
        pipeline_store["bm25"] = bm25

        pipeline_store["filename"] = file.filename

        return {
            "message": "Document uploaded successfully.",
            "filename": file.filename,
            "chunks": len(chunks),
            "scanned": scanned,
            "ocr_latency": ocr_latency,
        }

    finally:
        import os

        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/query")
def query_document(request: QueryRequest):

    if not pipeline_store:
        raise HTTPException(
            status_code=400,
            detail="Please upload a document first."
        )

    chunks = pipeline_store["chunks"]
    model = pipeline_store["model"]
    index = pipeline_store["index"]
    bm25 = pipeline_store["bm25"]

    question = request.question
    retrieval_mode = request.retrieval_mode

    if retrieval_mode == "Dense":

        results, latency = dense_search(
            model,
            index,
            chunks,
            question,
        )

    elif retrieval_mode == "BM25":

        results, latency = bm25_search(
            bm25,
            chunks,
            question,
        )

    elif retrieval_mode == "Hybrid (RRF)":

        dense_results, dense_latency = dense_search(
            model,
            index,
            chunks,
            question,
        )

        bm25_results, bm25_latency = bm25_search(
            bm25,
            chunks,
            question,
        )

        results, rrf_latency = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
        )

        latency = (
            dense_latency
            + bm25_latency
            + rrf_latency
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid retrieval mode."
        )

    context = "\n\n".join(
        chunk for chunk, _ in results
    )

    answer = generate_answer(
        question,
        context,
    )

    sources = [
        {
            "text": chunk,
            "score": score,
        }
        for chunk, score in results
    ]

    return {
        "answer": answer,
        "retrieval_mode": retrieval_mode,
        "latency": latency,
        "sources": sources,
    }
