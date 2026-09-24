import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

# --- local judge: Llama 3 via Ollama, deterministic ---
judge_llm = LangchainLLMWrapper(ChatOllama(model="llama3", temperature=0))
judge_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="llama3"))

MODE = "Hybrid (RRF)"   # which retrieval mode to score
LIMIT = 5               # small test run first; raise to 50 once it works

results = json.load(open("data/results.json"))["results"]

rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
for r in results[:LIMIT]:
    res = r["results"][MODE]
    rows["question"].append(r["question"])
    rows["answer"].append(res["answer"])
    rows["contexts"].append(res["retrieved_chunks"])
    rows["ground_truth"].append(r["ground_truth"])

dataset = Dataset.from_dict(rows)

print(f"Scoring {LIMIT} questions for {MODE} with local Llama 3 (this is slow)...")
scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=judge_llm,
    embeddings=judge_emb,
)
print(scores)