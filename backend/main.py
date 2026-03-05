"""
EqualCare Backend (FastAPI)

This is the local API service used by the Electron + React desktop application.

High-level responsibilities:
- Patient CRUD (SQLite)
- Visit transcripts (ASR/diarization) + persistence
- Notes (SOAP draft + save)
- Orders extraction from transcript + manual order management
- Clinical Assistant Q&A (optional patient context + optional guidelines RAG)
- Imaging analysis + imaging history (store image on disk, metadata in SQLite)

"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

import os
import hashlib

from .transcript_service import diarize_and_transcribe, transcribe_wav_bytes
from .notes_service import generate_structured_soap, structured_soap_to_narrative
from .orders_service import extract_orders_from_transcript
from .image_service import analyze_image_bytes, DEFAULT_PROMPT
from .assistant_service import answer_question_simple, answer_question_with_persistent_guidelines_rag
from .guidelines_service import (
    list_guideline_documents,
    add_guideline_file,
    remove_guideline_file,
)
from .db import (
    init_db,
    list_patients,
    get_patient,
    create_patient,
    delete_patient,
    list_transcripts,
    create_transcript,
    delete_transcript_visit,
    list_notes,
    create_note,
    list_assistant_messages,
    create_assistant_message,
    delete_assistant_message,
    list_orders,
    create_order,
    get_order,
    update_order,
    update_order_status,
    delete_order,
    list_imaging_history,
    get_imaging_history,
    create_imaging_history,
    delete_imaging_history,
)

# -----------------------------------------------------------------------------
# App initialization
# -----------------------------------------------------------------------------

app = FastAPI()

# Allow React dev server to call the API (useful in development).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    """Create SQLite DB + tables on app startup."""
    init_db()


# -----------------------------------------------------------------------------
# Pydantic schemas (request payloads)
# -----------------------------------------------------------------------------

class PatientCreate(BaseModel):
    """Payload used to create a patient row."""
    full_name: str
    age: Optional[int] = None
    sex: Optional[str] = None  # "M"/"F"
    weight_kg: Optional[float] = None
    room: Optional[str] = None
    location: Optional[str] = None
    insurance: Optional[str] = None
    account_id: Optional[str] = None
    mrn: Optional[str] = None
    reason_for_visit: Optional[str] = None
    provider: Optional[str] = None


class OrderCreate(BaseModel):
    """Payload used to create an order row."""
    transcript_id: Optional[int] = None
    category: str
    title: str
    priority: Optional[str] = "routine"
    status: Optional[str] = "draft"
    source: Optional[str] = "manual"
    evidence: Optional[str] = None
    notes: Optional[str] = None
    details: Optional[dict] = None


class OrderPatch(BaseModel):
    """Partial update payload for an order row."""
    transcript_id: Optional[int] = None
    category: Optional[str] = None
    title: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    evidence: Optional[str] = None
    notes: Optional[str] = None
    details: Optional[dict] = None


class OrderStatusPatch(BaseModel):
    """Minimal payload to update only the status field."""
    status: str

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check used by the desktop app to verify the backend is up."""
    return {"ok": True}


# -----------------------------------------------------------------------------
# Dictation
# -----------------------------------------------------------------------------

@app.post("/dictation")
async def dictation(file: UploadFile = File(...)):
    """Simple dictation ASR."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio too large (max 50MB)")

    text = transcribe_wav_bytes(data)
    return {"text": text}

# -----------------------------------------------------------------------------
# Guidelines Library (Persistent)
# -----------------------------------------------------------------------------

@app.get("/guidelines")
def guidelines_list():
    """List indexed guideline/protocol documents."""
    return list_guideline_documents()

@app.post("/guidelines/upload")
async def guidelines_upload(files: list[UploadFile] = File(...)):
    """Upload guideline files and index them persistently.

    Dedupe rule:
      - same file content (SHA256) → skip re-indexing
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    total_bytes = 0
    out = []
    for f in files:
        if not f or not f.filename:
            continue
        data = await f.read()
        if not data:
            continue
        total_bytes += len(data)
        if total_bytes > 80 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Total uploaded files are too large (max 80MB)")
        try:
            out.append(add_guideline_file(f.filename, data))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to add guideline: {e}")

    if not out:
        raise HTTPException(status_code=400, detail="Files were empty or unreadable")
    return out

@app.delete("/guidelines/{doc_id}")
def guidelines_delete(doc_id: int):
    """Remove a guideline document and delete its chunks from the persistent index."""
    try:
        remove_guideline_file(int(doc_id))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}

# -----------------------------------------------------------------------------
# Clinical Assistant
# -----------------------------------------------------------------------------

@app.post("/assistant/ask")
async def assistant_ask(
    question: str = Form(...),
    patient_id: Optional[int] = Form(None),
    use_patient_context: bool = Form(True),
    use_guidelines: bool = Form(True),
    save_interaction: bool = Form(True),
    guideline_files: Optional[list[UploadFile]] = File(None),
):
    """Q&A endpoint for the Clinical Assistant page.

    Notes:
    - Patient context is *not* RAG: it attaches basic patient fields + last transcript + last note.
    - Guidelines uses the *persistent* guideline library. If guideline_files are provided here,
      they are indexed first (deduped) and then the question is answered using the persistent index.

    Returns:
      {
        "answer_text": str,
        "sources": list,
        "confidence": str,
        "mode": str,
        "message": Optional[dict]  # present only if saved
      }
    """


    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is required")

    # Optional patient context (NOT RAG): attach basic fields + last transcript + last note
    patient_context: Optional[str] = None
    patient_sources: list[dict] = []
    if use_patient_context and patient_id is not None:
        pinfo = get_patient(int(patient_id))
        if not pinfo:
            raise HTTPException(status_code=404, detail="Patient not found")

        trs = list_transcripts(int(patient_id))
        last_tr = trs[0] if trs else None

        notes = list_notes(int(patient_id))
        last_note = notes[0] if notes else None

        parts: list[str] = []
        parts.append("PATIENT")
        parts.append(
            f"Name: {pinfo.get('full_name','')} | Age: {pinfo.get('age','')} | Sex: {pinfo.get('sex','')}"
        )
        if pinfo.get("reason_for_visit"):
            parts.append(f"Reason for visit: {pinfo.get('reason_for_visit')}")
        if pinfo.get("provider"):
            parts.append(f"Provider: {pinfo.get('provider')}")

        if last_tr and (last_tr.get("text") or "").strip():
            parts.append("")
            parts.append("LAST TRANSCRIPT")
            parts.append((last_tr.get("text") or "").strip()[:8000])

        if last_note and (last_note.get("content") or "").strip():
            parts.append("")
            parts.append("LAST NOTE")
            parts.append((last_note.get("content") or "").strip()[:8000])

        patient_context = "\n".join(parts).strip()

        # Build lightweight patient sources for Evidence UI
        patient_sources.append(
            {
                "kind": "patient_record",
                "title": "Patient record",
                "source": f"patient:{int(patient_id)}",
                "snippet": (" ".join(parts[:3])[:300]).strip(),
                "score": None,
                "page": None,
                "meta": {"patient_id": int(patient_id)},
            }
        )

        if last_tr and (last_tr.get("text") or "").strip():
            patient_sources.append(
                {
                    "kind": "transcript",
                    "title": f"Transcript #{last_tr.get('id')} ({last_tr.get('created_at')})",
                    "source": f"transcript:{last_tr.get('id')}",
                    "snippet": ((last_tr.get("text") or "")[:300] + "…") if len((last_tr.get("text") or "")) > 300 else (last_tr.get("text") or ""),
                    "score": None,
                    "page": None,
                    "meta": {"transcript_id": last_tr.get("id"), "created_at": last_tr.get("created_at")},
                }
            )

        if last_note and (last_note.get("content") or "").strip():
            patient_sources.append(
                {
                    "kind": "note",
                    "title": f"Note #{last_note.get('id')} ({last_note.get('type')}) ({last_note.get('created_at')})",
                    "source": f"note:{last_note.get('id')}",
                    "snippet": ((last_note.get("content") or "")[:300] + "…") if len((last_note.get("content") or "")) > 300 else (last_note.get("content") or ""),
                    "score": None,
                    "page": None,
                    "meta": {"note_id": last_note.get("id"), "note_type": last_note.get("type"), "created_at": last_note.get("created_at")},
                }
            )

    # Decide mode (used by UI to show which sources were used).
    mode = "both" if (use_patient_context and use_guidelines) else "patient" if use_patient_context else "guidelines" if use_guidelines else "none"

    # Guidelines RAG (persistent guideline library)
    if use_guidelines:
        # Optional compatibility: if files are uploaded with the ask request,
        # we index them persistently first (dedupe by SHA256), then answer using the persistent index.
        if guideline_files:
            total_bytes = 0
            for f in guideline_files:
                if not f or not f.filename:
                    continue
                data = await f.read()
                if not data:
                    continue
                total_bytes += len(data)
                if total_bytes > 80 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Total uploaded files are too large (max 80MB)")
                try:
                    add_guideline_file(f.filename, data)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Failed to add guideline: {e}")

        # Must have at least one indexed guideline
        if len(list_guideline_documents()) == 0:
            raise HTTPException(status_code=400, detail="No indexed guidelines found. Upload guidelines first or disable Guidelines.")

        rag_resp = answer_question_with_persistent_guidelines_rag(
            question=q,
            patient_context=patient_context,
        )

        answer_text = (rag_resp.get("answer_text") or rag_resp.get("answer") or "").strip()
        guideline_sources = rag_resp.get("sources") or []
        confidence = (rag_resp.get("confidence") or "").strip() or None

        # Merge sources (patient evidence + guideline citations)
        merged_sources = [*patient_sources, *guideline_sources]

        # Fallback confidence
        if not confidence:
            confidence = "Medium" if guideline_sources else ("Medium" if patient_sources else "Low")

        saved = None
        if save_interaction and patient_id is not None:
            saved = create_assistant_message(
                patient_id=int(patient_id),
                question=q,
                answer_text=answer_text,
                use_patient_context=bool(use_patient_context),
                use_guidelines=bool(use_guidelines),
                confidence=confidence,
                sources=merged_sources,
            )

        return {
            "answer": answer_text,
            "answer_text": answer_text,
            "sources": merged_sources,
            "confidence": confidence,
            "mode": mode,
            "message": saved,
        }

    # No guidelines: simple model answer
    simple_resp = answer_question_simple(question=q, patient_context=patient_context)
    answer_text = (simple_resp.get("answer_text") or simple_resp.get("answer") or "").strip()
    confidence = (simple_resp.get("confidence") or "").strip() or None
    if not confidence:
        confidence = "Medium" if patient_sources else "Low"
    merged_sources = [*patient_sources]

    saved = None
    if save_interaction and patient_id is not None:
        saved = create_assistant_message(
            patient_id=int(patient_id),
            question=q,
            answer_text=answer_text,
            use_patient_context=bool(use_patient_context),
            use_guidelines=bool(use_guidelines),
            confidence=confidence,
            sources=merged_sources,
        )

    return {
        "answer": answer_text,
        "answer_text": answer_text,
        "sources": merged_sources,
        "confidence": confidence,
        "mode": mode,
        "message": saved,
    }

@app.get("/patients/{patient_id}/assistant/messages")
def patient_assistant_messages(patient_id: int, limit: int = 20):
    """List assistant messages for a patient (most recent first)."""
    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    limit = max(1, min(int(limit), 200))
    return list_assistant_messages(patient_id=patient_id, limit=limit)

@app.delete("/patients/{patient_id}/assistant/messages/{message_id}")
def patient_assistant_messages_delete(patient_id: int, message_id: int):
    """Delete a single assistant message row for a patient."""
    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    result = delete_assistant_message(patient_id=int(patient_id), message_id=int(message_id))
    if not result or not result.get("ok"):
        raise HTTPException(status_code=404, detail=(result or {}).get("detail", "Assistant message not found"))

    return result

# -----------------------------------------------------------------------------
# Patients
# -----------------------------------------------------------------------------

@app.get("/patients")
def patients_list():
    """List all patients."""
    return list_patients()

@app.get("/patients/{patient_id}")
def patients_get(patient_id: int):
    """Get a patient row by id."""
    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return p

@app.post("/patients")
def patients_create(payload: PatientCreate):
    """Create a patient row."""
    return create_patient(payload.model_dump())

@app.delete("/patients/{patient_id}")
def patients_delete(patient_id: int):
    """Delete a patient and all related data (transcripts, notes, assistant messages)."""
    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    delete_patient(patient_id)
    return {"ok": True, "deleted_id": patient_id}

# -----------------------------------------------------------------------------
# Transcripts (Patient-bound)
# -----------------------------------------------------------------------------

@app.get("/patients/{patient_id}/transcripts")
def patient_transcripts(patient_id: int):
    """List transcripts (visits) for a patient."""
    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return list_transcripts(patient_id)

@app.delete("/patients/{patient_id}/transcripts/{transcript_id}")
def patient_transcripts_delete(patient_id: int, transcript_id: int):
    """Delete a visit (transcript) and any derived records (SOAP notes, orders)."""
    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    result = delete_transcript_visit(int(patient_id), int(transcript_id))
    if not result or not result.get("ok"):
        raise HTTPException(status_code=404, detail=(result or {}).get("detail", "Transcript not found"))
    return result

@app.post("/patients/{patient_id}/transcripts/audio")
async def patient_transcript_from_audio(patient_id: int, file: UploadFile = File(...)):
    """Create a transcript from uploaded audio, then auto-generate SOAP + Orders."""
    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio too large (max 50MB)")

    text = diarize_and_transcribe(data)
    item = create_transcript(patient_id=patient_id, text=text, audio_filename=file.filename)

    # ✅ Auto-generate + save a SOAP note right after transcription.
    # This is synchronous so the Notes page can immediately show the saved summary.
    if (text or "").strip():
        # Notes
        try:
            structured = generate_structured_soap(text)
            narrative = structured_soap_to_narrative(structured)
            create_note(
                patient_id=patient_id,
                note_type="SOAP",
                content=narrative,
                source_transcript_id=int(item.get("id")) if item.get("id") is not None else None,
            )
        except Exception as e:
            # Keep transcript creation robust even if note generation fails.
            print(f"[WARN] Auto note generation failed: {e}")

        # Orders (based on transcript text)
        try:
            tid = int(item.get("id")) if item.get("id") is not None else None
            if tid is not None:
                candidates = extract_orders_from_transcript(text)
                for o in candidates:
                    create_order(
                        patient_id=int(patient_id),
                        transcript_id=int(tid),
                        category=str(o.get("category") or "").strip(),
                        title=str(o.get("title") or "").strip(),
                        priority=str(o.get("priority") or "routine").strip(),
                        status=str(o.get("status") or "draft").strip(),
                        source=str(o.get("source") or "auto_transcript").strip(),
                        # Auto Orders: do not store evidence snippets.
                        evidence=None,
                        details=(o.get("details") or {}),
                        notes=(o.get("notes") or None),
                    )
        except Exception as e:
            print(f"[WARN] Auto order generation failed: {e}")


    return item

# -----------------------------------------------------------------------------
# Orders
# -----------------------------------------------------------------------------

@app.get("/patients/{patient_id}/orders")
def patient_orders(patient_id: int, transcript_id: Optional[int] = None):
    """List orders for a patient."""
    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return list_orders(patient_id=patient_id, transcript_id=transcript_id)

@app.post("/patients/{patient_id}/orders")
def patient_orders_create(patient_id: int, payload: OrderCreate):
    """Create an order row (manual entry from UI)."""
    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    title = (payload.title or "").strip()
    cat = (payload.category or "").strip()
    if not title or not cat:
        raise HTTPException(status_code=400, detail="category and title are required")

    item = create_order(
        patient_id=int(patient_id),
        transcript_id=int(payload.transcript_id) if payload.transcript_id is not None else None,
        category=cat,
        title=title,
        priority=(payload.priority or "routine"),
        status=(payload.status or "draft"),
        source=(payload.source or "manual"),
        evidence=(payload.evidence or None),
        details=(payload.details or {}),
        notes=(payload.notes or None),
    )

    # If duplicate, return the current list item (best-effort).
    if item is None:
        raise HTTPException(status_code=409, detail="Duplicate order")
    return item

@app.patch("/orders/{order_id}")
def orders_patch(order_id: int, payload: OrderPatch):
    """Patch an order with provided fields."""
    current = get_order(int(order_id))
    if not current:
        raise HTTPException(status_code=404, detail="Order not found")

    updates = payload.model_dump(exclude_none=True)
    try:
        return update_order(int(order_id), updates)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.patch("/orders/{order_id}/status")
def orders_patch_status(order_id: int, payload: OrderStatusPatch):
    """Update only the status field."""
    current = get_order(int(order_id))
    if not current:
        raise HTTPException(status_code=404, detail="Order not found")
    try:
        return update_order_status(int(order_id), payload.status)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/orders/{order_id}")
def orders_delete(order_id: int):
    """Delete an order row."""
    current = get_order(int(order_id))
    if not current:
        raise HTTPException(status_code=404, detail="Order not found")
    delete_order(int(order_id))
    return {"ok": True}


# -----------------------------------------------------------------------------
# Notes (Patient-bound)
# -----------------------------------------------------------------------------

@app.get("/patients/{patient_id}/notes")
def patient_notes(patient_id: int):
    """List notes for a patient."""
    p = get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return list_notes(patient_id)

# -----------------------------------------------------------------------------
# Imaging 
# -----------------------------------------------------------------------------
@app.post("/imaging/analyze")
async def imaging_analyze(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 10MB)")

    try:
        text = analyze_image_bytes(data, prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    return {"result": text}

def _sha256_bytes(data: bytes) -> str:
    """Return SHA256 hex digest for raw bytes (used for dedupe)."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

@app.get("/imaging/history")
def imaging_history_list(limit: int = 50, patient_id: Optional[int] = None):
    """List imaging history items (image bytes served via /imaging/history/{id}/image)."""
    items = list_imaging_history(limit=limit, patient_id=patient_id)
    # The image bytes are served via /imaging/history/{id}/image
    return {"items": items}

@app.post("/imaging/history")
async def imaging_history_save(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    result_text: str = Form(""),
    patient_id: Optional[int] = Form(None),
    model_name: Optional[str] = Form(None),
):
    """Save an imaging analysis output to history.

    Stores the raw image on disk (deduped by sha256) and persists metadata in SQLite.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 10MB)")

    prompt_norm = (prompt or "").strip()
    result_norm = (result_text or "").strip()
    if not prompt_norm or not result_norm:
        raise HTTPException(status_code=400, detail="prompt and result_text are required")

    from .paths import IMAGES_DIR

    files_dir = IMAGES_DIR / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    sha = _sha256_bytes(data)

    # Keep extension if available
    ext = ""
    if file.filename and "." in file.filename:
        ext = "." + file.filename.split(".")[-1].lower().strip()
        if len(ext) > 10:
            ext = ""
    storage_name = f"{sha}{ext}" if ext else sha
    storage_path = files_dir / storage_name

    # Deduplicate by sha
    if not storage_path.exists():
        storage_path.write_bytes(data)

    rec = create_imaging_history(
        patient_id=patient_id,
        original_filename=file.filename,
        image_sha256=sha,
        image_storage_path=str(storage_path),
        prompt=prompt_norm,
        result_text=result_norm,
        model_name=model_name,
    )
    return {"item": rec}

@app.get("/imaging/history/{history_id}/image")
def imaging_history_image(history_id: int):
    """Serve the stored image bytes for a history row."""
    rec = get_imaging_history(history_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    path = rec.get("image_storage_path") or ""
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(path)

@app.delete("/imaging/history/{history_id}")
def imaging_history_delete(history_id: int):
    """Delete an imaging history row (and garbage-collect stored file if unreferenced)."""
    rec = get_imaging_history(history_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")

    ok = delete_imaging_history(history_id)

    # Attempt to delete the stored file if no other rows reference the same sha
    if ok:
        try:
            sha = rec.get("image_sha256")
            path = rec.get("image_storage_path")
            if sha and path and os.path.exists(path):
                from .db import get_conn

                conn = get_conn()
                try:
                    row = conn.execute(
                        "SELECT COUNT(*) AS c FROM imaging_history WHERE image_sha256 = ?;",
                        (sha,),
                    ).fetchone()
                    if row and int(row["c"]) == 0:
                        os.remove(path)
                finally:
                    conn.close()
        except Exception:
            pass

    return {"ok": bool(ok)}
