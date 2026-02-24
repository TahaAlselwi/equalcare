"""backend/guidelines_service.py

Persistent Guidelines Library (Phase 1.5)
----------------------------------------

Goal
  - When a guideline file is uploaded once, we index it permanently.
  - The list of guideline titles persists across app restarts.
  - Uploading the same file again (same SHA256) skips re-indexing.
  - Removing a file also removes its chunks from the persistent vector index.

Design (no FAISS dependency)
  - Store raw files on disk: GUIDELINES_DIR / "files"
  - Store chunk text + metadata in SQLite tables:
      guidelines_documents, guidelines_chunks
  - Store a persistent vector index as a single NPZ file:
      GUIDELINES_DIR / "index.npz"
    containing:
      embeddings: float32 [N, D]
      chunk_ids: int64 [N]

At query time:
  - Embed the question (SentenceTransformers)
  - Dot-product with normalized embeddings
  - Take Top-K
  - Fetch chunk text + (filename,page) from SQLite

This is intentionally simple and offline-first. You can later upgrade to:
  - FAISS IndexIDMap for faster search
  - Hybrid retrieval (BM25 + vectors)
  - Per-country guideline sets and selection UI
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .paths import GUIDELINES_DIR
from .assistant_service import extract_text_from_file, chunk_text, embed_texts, call_model, build_prompt, GuidelineChunk
from .db import (
    get_guideline_by_sha256,
    create_guideline_document,
    update_guideline_chunk_count,
    insert_guideline_chunk,
    list_guidelines,
    get_guideline,
    delete_guideline_document,
    list_chunk_ids_for_doc,
    fetch_guideline_chunks_by_ids,
)


# ---------------- Storage paths ----------------

FILES_DIR = GUIDELINES_DIR / "files"
INDEX_PATH = GUIDELINES_DIR / "index.npz"

FILES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- In-memory index cache ----------------

_LOCK = threading.Lock()
_INDEX_CACHE: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (embeddings, chunk_ids)


def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _safe_filename(name: str) -> str:
    # keep it simple: remove path separators and odd chars
    name = (name or "file").replace("/", "_").replace("\\", "_")
    # avoid extremely long names on Windows
    return name[:160]


def _load_index_from_disk() -> Tuple[np.ndarray, np.ndarray]:
    if not INDEX_PATH.exists():
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    obj = np.load(INDEX_PATH, allow_pickle=False)
    emb = np.asarray(obj.get("embeddings"), dtype=np.float32)
    ids = np.asarray(obj.get("chunk_ids"), dtype=np.int64)
    if emb.ndim != 2 or ids.ndim != 1 or (emb.shape[0] != ids.shape[0]):
        # corrupted index; reset safely
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return emb, ids


def _save_index_to_disk(embeddings: np.ndarray, chunk_ids: np.ndarray) -> None:
    GUIDELINES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        INDEX_PATH,
        embeddings=embeddings.astype(np.float32),
        chunk_ids=chunk_ids.astype(np.int64),
    )


def _get_index_cached() -> Tuple[np.ndarray, np.ndarray]:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = _load_index_from_disk()
    return _INDEX_CACHE


def _set_index_cached(embeddings: np.ndarray, chunk_ids: np.ndarray) -> None:
    global _INDEX_CACHE
    _INDEX_CACHE = (embeddings, chunk_ids)


# ---------------- Public API ----------------

def list_guideline_documents() -> List[Dict[str, Any]]:
    return list_guidelines()


def add_guideline_file(filename: str, data: bytes) -> Dict[str, Any]:
    """Add a single guideline file. If already exists (SHA256 match), skip re-indexing."""
    if not filename:
        raise ValueError("filename is required")
    if not data:
        raise ValueError("empty file")

    sha = _sha256(data)
    existing = get_guideline_by_sha256(sha)
    if existing:
        # already indexed
        return {**existing, "already_exists": True}

    # Create DB row first to get doc_id
    safe_name = _safe_filename(filename)
    # temporary placeholder path; we'll update in the returned dict only
    tmp_storage = f"files/pending_{safe_name}"
    doc = create_guideline_document(
        original_filename=safe_name,
        sha256=sha,
        storage_path=tmp_storage,
        size_bytes=len(data),
    )
    doc_id = int(doc["id"])

    # Persist raw file
    storage_rel = f"files/{doc_id}_{safe_name}"
    storage_abs = GUIDELINES_DIR / storage_rel
    storage_abs.parent.mkdir(parents=True, exist_ok=True)
    storage_abs.write_bytes(data)

    # Update storage_path (simple direct UPDATE to keep patch minimal)
    # NOTE: We intentionally keep this in SQL here to avoid expanding db.py API too much.
    from .db import get_conn

    conn = get_conn()
    try:
        conn.execute(
            "UPDATE guidelines_documents SET storage_path = ? WHERE id = ?;",
            (storage_rel, doc_id),
        )
        conn.commit()
    finally:
        conn.close()

    # Extract + chunk
    chunks: List[Tuple[str, Optional[int]]] = extract_text_from_file(safe_name, data)
    all_chunks: List[GuidelineChunk] = []
    for page_text, page in chunks:
        all_chunks.extend(chunk_text(page_text, source_name=safe_name, page=page))

    # Cap to keep worst-case uploads bounded
    if len(all_chunks) > 2000:
        all_chunks = all_chunks[:2000]

    if not all_chunks:
        update_guideline_chunk_count(doc_id, 0)
        return {**get_guideline(doc_id), "already_exists": False, "warning": "No extractable text"}  # type: ignore

    # Embed texts
    vecs = embed_texts([c.text for c in all_chunks])
    if vecs.size == 0:
        update_guideline_chunk_count(doc_id, 0)
        return {**get_guideline(doc_id), "already_exists": False, "warning": "Embedding failed"}  # type: ignore

    # Insert chunks into DB to get stable chunk_ids
    chunk_ids: List[int] = []
    for ch in all_chunks:
        cid = insert_guideline_chunk(doc_id=doc_id, text=ch.text, page=ch.page)
        chunk_ids.append(int(cid))

    update_guideline_chunk_count(doc_id, len(chunk_ids))

    # Append to persistent index
    with _LOCK:
        emb, ids = _get_index_cached()
        if emb.size == 0:
            new_emb = vecs.astype(np.float32)
            new_ids = np.asarray(chunk_ids, dtype=np.int64)
        else:
            # Dimension guard
            if emb.shape[1] != vecs.shape[1]:
                # Different embedding model dimension → rebuild fresh index from scratch
                new_emb = vecs.astype(np.float32)
                new_ids = np.asarray(chunk_ids, dtype=np.int64)
            else:
                new_emb = np.vstack([emb, vecs.astype(np.float32)])
                new_ids = np.concatenate([ids, np.asarray(chunk_ids, dtype=np.int64)])

        _save_index_to_disk(new_emb, new_ids)
        _set_index_cached(new_emb, new_ids)

    out = get_guideline(doc_id)
    assert out is not None
    return {**out, "already_exists": False}


def remove_guideline_file(doc_id: int) -> None:
    doc = get_guideline(int(doc_id))
    if not doc:
        raise ValueError("Guideline not found")

    # Remove chunks from the vector index
    chunk_ids = list_chunk_ids_for_doc(int(doc_id))
    with _LOCK:
        emb, ids = _get_index_cached()
        if ids.size and chunk_ids:
            kill = set(int(x) for x in chunk_ids)
            keep_mask = np.array([int(x) not in kill for x in ids], dtype=bool)
            new_emb = emb[keep_mask]
            new_ids = ids[keep_mask]
            _save_index_to_disk(new_emb, new_ids)
            _set_index_cached(new_emb, new_ids)

    # Delete raw file on disk
    storage_rel = doc.get("storage_path")
    if storage_rel:
        rel = Path(str(storage_rel))
        # Back-compat: older rows might store "guidelines/files/..." (relative to DATA_DIR)
        if rel.parts and rel.parts[0] == "guidelines":
            p = GUIDELINES_DIR.parent / rel  # DATA_DIR / guidelines/...
        else:
            p = GUIDELINES_DIR / rel         # GUIDELINES_DIR / files/...
        try:
            if p.exists():
                p.unlink()
        except Exception:
            # non-fatal
            pass

    # Delete DB row (cascades chunks)
    delete_guideline_document(int(doc_id))


def retrieve_guideline_evidence(
    question: str,
    top_k: int = 6,
) -> List[Tuple[GuidelineChunk, float]]:
    """Vector search over the persistent guideline library."""
    q = (question or "").strip()
    if not q:
        return []

    with _LOCK:
        emb, ids = _get_index_cached()
        if emb.size == 0 or ids.size == 0:
            return []

        q_emb = embed_texts([q])
        if q_emb.size == 0:
            return []

        sims = (emb @ q_emb[0]).astype(float)
        k = min(int(top_k), int(sims.shape[0]))
        if k <= 0:
            return []

        # argsort top-k
        top_idx = np.argsort(-sims)[:k]
        top_chunk_ids = [int(ids[i]) for i in top_idx]
        top_scores = [float(sims[i]) for i in top_idx]

    # Fetch chunk rows from DB (join filename)
    rows = fetch_guideline_chunks_by_ids(top_chunk_ids)
    by_id = {int(r["chunk_id"]): r for r in rows}

    out: List[Tuple[GuidelineChunk, float]] = []
    for cid, score in zip(top_chunk_ids, top_scores):
        r = by_id.get(int(cid))
        if not r:
            continue
        ch = GuidelineChunk(text=r.get("text") or "", source_name=r.get("source_name") or "", page=r.get("page"))
        out.append((ch, float(score)))
    return out


def answer_question_with_persistent_guidelines_rag(
    question: str,
    patient_context: Optional[str] = None,
    top_k: int = 6,
) -> Dict[str, Any]:
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