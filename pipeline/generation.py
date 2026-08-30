import ollama


def generate_answer(context: str, question: str) -> str:
    """Generate a grounded answer using Llama 3."""
    prompt = f"""You are Recall, a document-grounded study assistant.
Your task is to answer questions using ONLY the provided context.

Rules:
1. Use only information contained in the context.
2. Do not use outside knowledge.
3. Do not make assumptions or guesses.
4. If the answer is not in the context, say you cannot find it in the document.
5. Provide concise but complete answers.
6. When possible, answer in 1-3 sentences.
7. If the context contains a specific number, requirement, definition, or fact, include it directly in the answer.
8. Do not mention the context, chunks, or retrieval process.

Context:
{context}

Question: {question}
Answer: """

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
