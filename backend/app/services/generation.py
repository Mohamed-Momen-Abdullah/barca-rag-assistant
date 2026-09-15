"""
Generation service.

Builds the same citation-grounded prompt template used in the notebook
(section 2.4) and calls the local Ollama model to produce an answer.
"""

import logging

import ollama

from app.core.config import get_vector_store_config
from app.services.retrieval import retrieval_service

logger = logging.getLogger(__name__)


def build_prompt(question: str, hits: list[dict]) -> str:
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        source = f"{h['metadata']['source_group']}/{h['metadata']['filename']}"
        context_blocks.append(f"[Source {i}: {source}]\n{h['text']}")
    context = "\n\n".join(context_blocks)

    return f"""You are a knowledgeable FC Barcelona history assistant. Answer the question ONLY using the context below. \
If the context does not contain the answer, say you don't have enough information -- do not use outside knowledge.
After your answer, list which Source numbers you used.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, k: int | None = None) -> dict:
    """Retrieve context, build the prompt, call Ollama, return answer + sources."""
    hits = retrieval_service.retrieve(question, k=k)
    prompt = build_prompt(question, hits)

    vector_config = get_vector_store_config()
    model_name = vector_config["ollama_model"]

    logger.info("Calling Ollama model '%s' for question: %s", model_name, question)
    response = ollama.generate(model=model_name, prompt=prompt)

    sources = sorted({h["metadata"]["filename"] for h in hits})
    return {
        "answer": response["response"].strip(),
        "sources": sources,
    }
