# Recall — A Transparent Local RAG Study Assistant

## Overview

Recall is a fully local, privacy-preserving retrieval-augmented generation (RAG)
study assistant. Users upload their own documents, ask questions, and receive
answers grounded strictly in those documents, with the exact source passage shown
for verification.

This project is submitted for CM3070 Final Project at the University of London,
using the CM3020 AI "Orchestrating AI models" template.

## Research Question

How much does retrieval strategy influence answer quality in a local RAG study
assistant operating on student learning materials?

## System Overview

The system compares three retrieval strategies under identical conditions:

- **BM25** — keyword-based sparse retrieval
- **Dense** — semantic retrieval using sentence-transformers + FAISS
- **Hybrid** — Reciprocal Rank Fusion combining both

## Models Used

- `all-MiniLM-L6-v2` (sentence-transformers) — text embeddings
- `Llama 3` via Ollama — local answer generation
- `Whisper` (OpenAI) — audio transcription (final system)

## Tech Stack

Python, Streamlit, FAISS, LangChain, sentence-transformers, Ollama, rank-bm25, RAGAS

## Project Status

- [x] Dense retrieval pipeline
- [x] Streamlit UI with file upload and grounded Q&A
- [x] Grounding guardrail (refuses out-of-document questions)
- [ ] BM25 retrieval
- [ ] Hybrid retrieval (RRF)
- [ ] Whisper audio ingestion
- [ ] Full comparative evaluation (Recall@k, RAGAS, latency)

## How to Run

## How to Run

1. Clone the repo: `git clone https://github.com/DaniyalF007/recall.git`
2. Create and activate virtual environment: `cd recall && python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Install and start Ollama with Llama 3: `ollama run llama3`
5. Run the app: `streamlit run app.py`

## Author

Daniyal Farooqui — University of London BSc Computer Science
