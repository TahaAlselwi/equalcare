"""backend/assistant_service.py

Phase 1 (Guidelines RAG only)
----------------------------
This module implements a simple request-time RAG pipeline:

1) Extract text from guideline files (PDF/TXT/DOCX)
2) Chunk text into overlapping segments
3) Embed chunks with SentenceTransformers
4) Retrieve Top-K chunks for the question
5) Ask the local model via Ollama using only the retrieved context

Later phases can persist indexes and add patient-record RAG.

All text output is English-only.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ollama import chat

# Lazy-loaded embedding model
_EMBED_MODEL = None

# You can override these with environment variables
EMBED_MODEL_NAME = os.getenv("ASSISTANT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ASSISTANT_MODEL_NAME = os.getenv("ASSISTANT_MODEL_NAME", "MedAIBase/MedGemma1.0:4b")


@dataclass(frozen=True)
class GuidelineChunk:
    text: str
    source_name: str
    page: Optional[int] = None  # 1-indexed for PDF, None for non-PDF


def _clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u0000", " ")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r" \n", "\n", t)
    return t.strip()


def extract_text_from_file(filename: str, data: bytes) -> List[Tuple[str, Optional[int]]]:
    """Extract text from a guideline file.

    Returns a list of (text, page) tuples.
      - For TXT/DOCX: one item with page=None
      - For PDF: one item per page with page=1..N

    If a file yields no text, returns an empty list.
    """
    name = (filename or "").lower()

    if name.endswith(".txt"):
        text = _clean_text(data.decode("utf-8", errors="replace"))
        return [(text, None)] if text else []

    if name.endswith(".docx"):
        # Lazy import to keep startup fast
        from docx import Document

        doc = Document(io.BytesIO(data))
        parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        text = _clean_text("\n".join(parts))
        return [(text, None)] if text else []

    if name.endswith(".pdf"):
        # Lazy import
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        out: List[Tuple[str, Optional[int]]] = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            txt = _clean_text(txt)
            if txt:
                out.append((txt, i + 1))
        return out

    # Fallback: try decoding as text
    text = _clean_text(data.decode("utf-8", errors="replace"))
    return [(text, None)] if text else []


def chunk_text(
    text: str,
    source_name: str,
    page: Optional[int],
    chunk_chars: int = 1200,
    overlap: int = 200,
) -> List[GuidelineChunk]:
    """Split text into overlapping chunks.

    Uses a simple sliding window (character-based) and tries to cut at a newline
    or space near the end to reduce mid-sentence splits.
    """
    text = _clean_text(text)
    if not text:
        return []

    if chunk_chars < 200:
        chunk_chars = 200
    overlap = max(0, min(overlap, chunk_chars - 1))

    chunks: List[GuidelineChunk] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(n, start + chunk_chars)

        # Try to cut at a natural boundary
        if end < n:
            # Prefer paragraph boundary
            cut = text.rfind("\n", start, end)
            if cut == -1 or (end - cut) > 200:
                # Otherwise cut at whitespace
                cut = text.rfind(" ", start, end)
            if cut != -1 and cut > start + int(chunk_chars * 0.5):
                end = cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(GuidelineChunk(text=chunk, source_name=source_name, page=page))

        if end >= n:
            break

        start = max(0, end - overlap)

    return chunks


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED_MODEL


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model = _get_embed_model()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    return emb


def retrieve_top_k(
    question: str,
    chunks: List[GuidelineChunk],
    k: int = 6,
) -> List[Tuple[GuidelineChunk, float]]:
    if not chunks:
        return []

    q_emb = embed_texts([question])
    c_emb = embed_texts([c.text for c in chunks])

    if c_emb.size == 0:
        return []

    # Cosine similarity because we normalized embeddings
    sims = (c_emb @ q_emb[0]).astype(float)
    idx = np.argsort(-sims)[: min(k, len(chunks))]

    return [(chunks[int(i)], float(sims[int(i)])) for i in idx]


def build_prompt(
    question: str,
    retrieved: List[Tuple[GuidelineChunk, float]],
    patient_context: Optional[str] = None,
) -> str:
    context_blocks: List[str] = []

    for ch, _score in retrieved:
        label = f"{ch.source_name}"
        if ch.page is not None:
            label += f" (p.{ch.page})"

        txt = ch.text
        if len(txt) > 2200:
            txt = txt[:2200].rstrip() + "…"

        context_blocks.append(f"[SOURCE: {label}]\n{txt}")

    guidelines_context = "\n\n".join(context_blocks) if context_blocks else ""

    parts: List[str] = []
    parts.append("You are a clinical assistant. Answer in English.")
    parts.append(
        "Use ONLY the provided 'GUIDELINES CONTEXT' for guideline-based statements. "
        "If the context is insufficient, say so explicitly."
    )
    parts.append("Be cautious. This tool supports clinicians but does not replace professional judgment.")
    parts.append("When you use information from the context, cite it inline like: [SOURCE: filename (p.X)].")

    if patient_context:
        parts.append("\nPATIENT CONTEXT (may be incomplete):\n" + patient_context)

    if guidelines_context:
        parts.append("\nGUIDELINES CONTEXT:\n" + guidelines_context)
    else:
        parts.append("\nGUIDELINES CONTEXT:\n(No guideline context provided.)")

    parts.append("\nQUESTION:\n" + question.strip())
    parts.append(
        "\nRESPONSE REQUIREMENTS:\n"
        "- Provide a clear, practical answer.\n"
        "- Include red flags and when to escalate if relevant.\n"
        "- If you reference the guideline context, include citations.\n"
    )

    return "\n".join(parts).strip()


def call_model(prompt: str, model_name: str = ASSISTANT_MODEL_NAME) -> str:
    resp = chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return (resp.message.content or "").strip()


def answer_question_simple(
    question: str,
    patient_context: Optional[str] = None,
    model_name: str = ASSISTANT_MODEL_NAME,
) -> Dict[str, Any]:
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


def answer_question_with_guidelines_rag(
    question: str,
    guideline_files: List[Tuple[str, bytes]],
    patient_context: Optional[str] = None,
    top_k: int = 6,
    model_name: str = ASSISTANT_MODEL_NAME,
) -> Dict[str, Any]:
    # 1) Extract + chunk
    all_chunks: List[GuidelineChunk] = []
    for fname, data in guideline_files:
        pages = extract_text_from_file(fname, data)
        for page_text, page in pages:
            all_chunks.extend(chunk_text(page_text, source_name=fname, page=page))

    # Simple cap to prevent very large uploads from becoming too slow
    if len(all_chunks) > 1200:
        all_chunks = all_chunks[:1200]

    # 2) Retrieve
    retrieved = retrieve_top_k(question, all_chunks, k=top_k)

    # 3) Prompt + answer
    prompt = build_prompt(question, retrieved, patient_context=patient_context)
    answer = call_model(prompt, model_name=model_name)

    # 4) Return sources (Evidence UI expects structured sources)
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
                "score": round(score, 4),
                "snippet": (ch.text[:300] + "…") if len(ch.text) > 300 else ch.text,
                "meta": {"source_name": ch.source_name, "page": ch.page},
            }
        )

    # Simple confidence based on best similarity score
    if not retrieved:
        conf = "Low"
    else:
        best = max(float(s) for _c, s in retrieved)
        if best >= 0.60:
            conf = "High"
        elif best >= 0.45:
            conf = "Medium"
        else:
            conf = "Low"

    return {"answer": answer, "answer_text": answer, "sources": sources, "confidence": conf}
