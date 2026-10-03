"""
Lightweight retrieval: finds past tickets that are textually similar to a
given ticket, to use as reference context when classifying a new one
(a small RAG-style step — retrieve, then feed into the LLM prompt).

Uses TF-IDF + cosine similarity rather than embeddings, since the mock
dataset is small and this keeps the project dependency-light. Swapping in
a real embedding model later (e.g. via the Anthropic/OpenAI embeddings API
or a local sentence-transformers model) would be a natural extension.
"""

from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_tickets(target: Dict, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Return up to `top_k` tickets from `candidates` most textually similar to
    `target`, excluding the target itself and any ticket still "open" (we
    only want past tickets that reached some resolved/triaged state to use
    as reference examples).
    """
    pool = [t for t in candidates if t["id"] != target["id"] and t["status"] != "open"]
    if not pool:
        return []

    texts = [f"{t['subject']} {t['body']}" for t in pool]
    target_text = f"{target['subject']} {target['body']}"

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts + [target_text])

    target_vector = matrix[-1]
    candidate_vectors = matrix[:-1]
    similarities = cosine_similarity(target_vector, candidate_vectors)[0]

    ranked = sorted(zip(pool, similarities), key=lambda pair: pair[1], reverse=True)
    return [ticket for ticket, score in ranked[:top_k] if score > 0]
