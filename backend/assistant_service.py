"""
This module contains the public answer functions used by the FastAPI routes:

- answer_question_simple:
    Direct model call (no retrieval).

- answer_question_with_persistent_guidelines_rag:
    Uses the persistent guideline library (stored/indexed in guidelines_service.py)
    to retrieve Top-K evidence chunks and answer with citations.
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple
from ollama import chat
from .guidelines_service import build_prompt, retrieve_guideline_evidence

# -----------------------------------------------------------------------------
# Configuration (models & environment variables)
# -----------------------------------------------------------------------------

# You can override these with environment variables
ASSISTANT_MODEL_NAME = os.getenv("ASSISTANT_MODEL_NAME", "MedAIBase/MedGemma1.0:4b")

# -----------------------------------------------------------------------------
# Model invocation (Ollama)
# -----------------------------------------------------------------------------
def call_model(prompt: str, model_name: str = ASSISTANT_MODEL_NAME) -> str:
    """Call the local Ollama chat model and return the assistant text."""
    resp = chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return (resp.message.content or "").strip()

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def answer_question_simple(
    question: str,
    patient_context: Optional[str] = None,
    model_name: str = ASSISTANT_MODEL_NAME,
) -> Dict[str, Any]:
    """Answer a question with no guideline retrieval (simple mode)."""
    prompt_parts = [
        "You are a clinical assistant. Answer in English.",
        "Be cautious. This tool supports clinicians but does not replace professional judgment.",
    ]
    if patient_context:
        prompt_parts.append("\nPATIENT CONTEXT (may be incomplete):\n" + patient_context)
    prompt_parts.append("\nQUESTION:\n" + question.strip())
    prompt_parts.append("\nRESPONSE:\n")

    prompt = "\n".join(prompt_parts).strip()
    answer = call_model(prompt, model_name=model_name)
    # No retriever evidence in the simple mode
    return {"answer": answer, "answer_text": answer, "sources": [], "confidence": "Low"}

def answer_question_with_persistent_guidelines_rag(
    question: str,
    patient_context: Optional[str] = None,
    top_k: int = 6,
) -> Dict[str, Any]:
    """Answer a question using persistent guideline RAG (vector retrieval + citations)."""
    retrieved = retrieve_guideline_evidence(question, top_k=top_k)
    prompt = build_prompt(question, retrieved, patient_context=patient_context)
    answer = call_model(prompt)

    sources: List[Dict[str, Any]] = []
    for ch, score in retrieved:
        title = ch.source_name
        if ch.page is not None:
            title += f" (p.{ch.page})"
        sources.append(
            {
                "kind": "guideline",
                "title": title,
                "source": ch.source_name,
                "page": ch.page,
                "score": round(float(score), 4),
                "snippet": (ch.text[:300] + "…") if len(ch.text) > 300 else ch.text,
                "meta": {"source_name": ch.source_name, "page": ch.page},
            }
        )

    if not retrieved:
        conf = "Low"
    else:
        best = max(float(s) for _c, s in retrieved)
        conf = "High" if best >= 0.60 else "Medium" if best >= 0.45 else "Low"

    return {"answer": answer, "answer_text": answer, "sources": sources, "confidence": conf}