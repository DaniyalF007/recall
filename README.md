cat > ~/Desktop/recall/README.md << 'EOF'

# Recall — A Transparent Local RAG Study Assistant

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) study assistant developed for the CM3070 Final Project under the CM3020 Artificial Intelligence (Orchestrating Models) template.

Users upload study materials, ask questions in natural language, and receive answers grounded in retrieved passages from those documents. Source passages and similarity scores are displayed alongside every answer to support transparency and verification.

---

## Research Question

How does the choice of retrieval strategy influence retrieval effectiveness and answer quality within a transparent, fully local Retrieval-Augmented Generation system for document question answering?

---

## Project Aim

Recall investigates how different retrieval strategies affect answer quality in a document-grounded question-answering system. It implements and evaluates BM25, dense semantic retrieval, and hybrid retrieval under identical conditions — same document collection, chunking configuration, embedding model, and language model — so that observed differences are attributable to the retrieval strategy alone.

---

## Why Local and Transparent?

Existing AI study tools provide varying levels of source transparency, but retrieval and ranking decisions remain largely hidden. Recall runs entirely on the user's machine with no data leaving the device, and displays the exact source passages alongside every answer so responses can be verified against the original document.

---

## Architecture

Recall orchestrates five pre-trained models within a single unified pipeline:

- **all-MiniLM-L6-v2** — semantic text embeddings
- **Llama 3 via Ollama** — local language model generation
- **Whisper (OpenAI)** — audio transcription
- **TrOCR (Microsoft)** — scanned PDF OCR
- **BM25 (rank-bm25)** — lexical retrieval

---

## Features (Complete)

- Digital PDF ingestion and chunking
- Dense semantic retrieval (FAISS + all-MiniLM-L6-v2)
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

---

## Retrieval Strategies

Three retrieval methods are compared under identical experimental conditions:

**BM25** — keyword-based retrieval using term frequency and inverse document frequency.

**Dense Retrieval** — semantic retrieval using all-MiniLM-L6-v2 embeddings indexed with FAISS (cosine similarity).

**Hybrid Retrieval** — combination of BM25 and dense retrieval using Reciprocal Rank Fusion (RRF), prioritising passages ranked highly by both methods.

---

## Evaluation

All three strategies were evaluated under identical conditions on a 50-question benchmark spanning factual, paraphrased, inferential, and negative questions across three document types.

**Metrics measured:**

- Answer accuracy (manual ground-truth scoring, by question type)
- Recall@1 and Recall@3
- Mean Reciprocal Rank (MRR)
- Negative-question abstention rate
- Retrieval latency per strategy

RAGAS was scoped in the proposal but replaced with manual ground-truth scoring plus Recall@k/MRR, due to a dependency conflict under a fully-local configuration; this is discussed in the report. Evaluation code (`pipeline/evaluation.py`), scoring (`score.py`), retrieval metrics (`metrics.py`), the benchmark, gold labels, and raw results are all included under `data/` for reproducibility.

---

## Technology Stack

| Component           | Technology                               |
| ------------------- | ---------------------------------------- |
| Backend API         | FastAPI + Uvicorn                        |
| Frontend            | React and Streamlit                      |
| PDF parsing         | PyPDF                                    |
| Text chunking       | LangChain RecursiveCharacterTextSplitter |
| Embeddings          | sentence-transformers (all-MiniLM-L6-v2) |
| Vector index        | FAISS                                    |
| Lexical retrieval   | rank-bm25                                |
| Language model      | Llama 3 via Ollama                       |
| Audio transcription | OpenAI Whisper                           |
| OCR                 | Microsoft TrOCR                          |

---

## How to Run

Requires Python 3.11 and Ollama installed. Audio transcription also requires ffmpeg (`brew install ffmpeg` on macOS).

**1. Clone**

```bash
git clone https://github.com/DaniyalF007/recall.git
cd recall
```

**2. Virtual environment**

```bash
python3.11 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Pull and start the local model**

```bash
ollama pull llama3
ollama serve
```

**5. Streamlit interface**

```bash
streamlit run app.py
```

Opens at http://localhost:8501

**6. FastAPI backend**

```bash
uvicorn api:app --reload
```

Opens at http://127.0.0.1:8000

**7. React frontend**

```bash
cd frontend
npm install
npm start
```

Opens at http://localhost:3000

---

## Repository Structure
