# Recall — A Transparent Local RAG Study Assistant

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) study assistant developed for the CM3070 Final Project under the CM3020 Artificial Intelligence (Orchestrating Models) template.

Users upload study materials, ask questions in natural language, and receive answers grounded in retrieved passages from those documents. Source passages and similarity scores are displayed alongside every answer to support transparency and verification.

## Research Question

How does the choice of retrieval strategy influence retrieval effectiveness and answer quality within a transparent, fully local Retrieval-Augmented Generation system for document question answering?

## Project Aim

Recall investigates how different retrieval strategies affect answer quality in a document-grounded question-answering system. It implements and evaluates BM25, dense semantic retrieval, and hybrid retrieval under identical conditions so that observed differences are attributable to the retrieval strategy alone.

## Architecture

Recall orchestrates five pre-trained models within a single unified pipeline:

- all-MiniLM-L6-v2 — semantic text embeddings
- Llama 3 via Ollama — local language model generation
- Whisper (OpenAI) — audio transcription
- TrOCR (Microsoft) — scanned PDF OCR
- BM25 (rank-bm25) — lexical retrieval

## Features

- Digital PDF ingestion and chunking
- Dense semantic retrieval (FAISS + all-MiniLM-L6-v2, cosine similarity)
- BM25 lexical retrieval
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Whisper audio transcription
- TrOCR scanned PDF support with automatic detection
- Grounded answer generation using Llama 3
- Source citation display with similarity scores
- Grounding guardrail (refuses out-of-scope questions)
- Retrieval latency measurement per strategy
- Streamlit interface and React frontend, both with retrieval mode selector
- FastAPI backend with REST endpoints
- 50-question benchmark across three document types
- Reproducible scoring and retrieval metrics (Recall@k, MRR)

## Retrieval Strategies

BM25 — keyword-based retrieval using term frequency and inverse document frequency.

Dense Retrieval — semantic retrieval using all-MiniLM-L6-v2 embeddings indexed with FAISS.

Hybrid Retrieval — combination of BM25 and dense retrieval using Reciprocal Rank Fusion (RRF).

## Evaluation

All three strategies were evaluated under identical conditions on a 50-question benchmark spanning factual, paraphrased, inferential, and negative questions across three document types.

Metrics measured: answer accuracy by question type, Recall@1 and Recall@3, Mean Reciprocal Rank (MRR), negative-question abstention rate, and retrieval latency per strategy.

RAGAS was scoped in the proposal but replaced with manual ground-truth scoring plus Recall@k/MRR, due to a dependency conflict under a fully-local configuration; this is discussed in the report. Evaluation code (pipeline/evaluation.py), scoring (score.py), retrieval metrics (metrics.py), the benchmark, gold labels, and raw results are all included under data/ for reproducibility.

## Technology Stack

- Backend API: FastAPI + Uvicorn
- Frontend: React and Streamlit
- PDF parsing: PyPDF
- Text chunking: LangChain RecursiveCharacterTextSplitter
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector index: FAISS
- Lexical retrieval: rank-bm25
- Language model: Llama 3 via Ollama
- Audio transcription: OpenAI Whisper
- OCR: Microsoft TrOCR

## How to Run

Requires Python 3.11 and Ollama installed. Audio transcription also requires ffmpeg (brew install ffmpeg on macOS).

1. Clone

    git clone https://github.com/DaniyalF007/recall.git
    cd recall

2. Virtual environment

    python3.11 -m venv venv
    source venv/bin/activate

3. Install dependencies

    pip install -r requirements.txt

4. Pull and start the local model

    ollama pull llama3
    ollama serve

5. Streamlit interface

    streamlit run app.py

6. FastAPI backend

    uvicorn api:app --reload

7. React frontend

    cd frontend
    npm install
    npm start

## Repository Structure

- app.py — Streamlit UI
- api.py — FastAPI backend
- metrics.py — Recall@k and MRR
- score.py — answer-accuracy scoring
- pipeline/ — ingestion, embeddings, retrieval, generation, audio, ocr, evaluation
- frontend/ — React frontend
- tests/ — development tests
- data/ — benchmark.json, gold.json, results.json
