import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import DB_PATH

_DB_TIMEOUT_S = 15


def _wal_path() -> Path:
    return DB_PATH.with_name(DB_PATH.name + "-wal")


def _shm_path() -> Path:
    return DB_PATH.with_name(DB_PATH.name + "-shm")


def get_conn() -> sqlite3.Connection:
    # timeout + busy_timeout makes resets / writes more reliable on Windows
    conn = sqlite3.connect(str(DB_PATH), timeout=_DB_TIMEOUT_S, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    # WAL is good for app UX, but can leave -wal/-shm files (we handle this in reset)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                age INTEGER,
                sex TEXT,
                weight_kg REAL,
                room TEXT,
                location TEXT,
                insurance TEXT,
                account_id TEXT,
                mrn TEXT,
                reason_for_visit TEXT,
                provider TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                audio_filename TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                source_transcript_id INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
            );
            """
        )

        # Assistant Q/A messages (patient-bound)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                use_patient_context INTEGER NOT NULL DEFAULT 1,
                use_guidelines INTEGER NOT NULL DEFAULT 1,
                confidence TEXT,
                sources_json TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
            );
            """
        )

        # ---------------- Orders (Patient-bound, Visit-bound) ----------------
        # Each order can optionally be linked to a specific visit/encounter (transcript_id).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                transcript_id INTEGER,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                priority TEXT NOT NULL DEFAULT 'routine',
                status TEXT NOT NULL DEFAULT 'draft',
                source TEXT DEFAULT 'manual',
                evidence TEXT,
                details_json TEXT,
                notes TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id) ON DELETE SET NULL
            );
            """
        )

        # ---------------- Guidelines library (persistent RAG) ----------------
        # Stores uploaded guideline/protocol documents that are indexed once and reused.
        # We keep the raw file on disk and store chunk text + metadata in SQLite.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS guidelines_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                storage_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                chunk_count INTEGER DEFAULT 0
            );
            """
        )

        # Each chunk is assigned a stable INTEGER id (used by the vector index file).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS guidelines_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                page INTEGER,
                text TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (doc_id) REFERENCES guidelines_documents(id) ON DELETE CASCADE
            );
            """
        )


        # ---------------- Imaging History (persistent) ----------------
        # Stores image analyses so clinicians can reuse outputs without re-running the model.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS imaging_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                original_filename TEXT,
                image_sha256 TEXT NOT NULL,
                image_storage_path TEXT NOT NULL,
                prompt TEXT NOT NULL,
                result_text TEXT NOT NULL,
                model_name TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE SET NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------- Guidelines DB helpers ----------------

def get_guideline_by_sha256(sha256: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM guidelines_documents WHERE sha256 = ?;",
            (sha256,),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def list_guidelines() -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM guidelines_documents ORDER BY created_at DESC, id DESC;"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def create_guideline_document(
    original_filename: str,
    sha256: str,
    storage_path: str,
    size_bytes: int,
) -> Dict[str, Any]:
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO guidelines_documents (original_filename, sha256, storage_path, size_bytes)
            VALUES (?, ?, ?, ?);
            """,
            (original_filename, sha256, storage_path, int(size_bytes)),
        )
        conn.commit()
        did = conn.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"]
        row = conn.execute("SELECT * FROM guidelines_documents WHERE id = ?;", (int(did),)).fetchone()
        assert row is not None
        return _row_to_dict(row)
    finally:
        conn.close()


def update_guideline_chunk_count(doc_id: int, chunk_count: int) -> None:
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE guidelines_documents SET chunk_count = ? WHERE id = ?;",
            (int(chunk_count), int(doc_id)),
        )
        conn.commit()
    finally:
        conn.close()


def insert_guideline_chunk(doc_id: int, text: str, page: Optional[int]) -> int:
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO guidelines_chunks (doc_id, page, text) VALUES (?, ?, ?);",
            (int(doc_id), page if page is None else int(page), text),
        )
        conn.commit()
        cid = conn.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"]
        return int(cid)
    finally:
        conn.close()


def get_guideline(doc_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM guidelines_documents WHERE id = ?;", (int(doc_id),)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def delete_guideline_document(doc_id: int) -> None:
    conn = get_conn()
    try:
        conn.execute("DELETE FROM guidelines_documents WHERE id = ?;", (int(doc_id),))
        conn.commit()
    finally:
        conn.close()


def list_chunk_ids_for_doc(doc_id: int) -> List[int]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id FROM guidelines_chunks WHERE doc_id = ? ORDER BY id ASC;",
            (int(doc_id),),
        ).fetchall()
        return [int(r["id"]) for r in rows]
    finally:
        conn.close()


def fetch_guideline_chunks_by_ids(chunk_ids: List[int]) -> List[Dict[str, Any]]:
    if not chunk_ids:
        return []
    conn = get_conn()
    try:
        placeholders = ",".join(["?"] * len(chunk_ids))
        rows = conn.execute(
            f"""
            SELECT c.id as chunk_id, c.doc_id, c.page, c.text,
                   d.original_filename as source_name
            FROM guidelines_chunks c
            JOIN guidelines_documents d ON d.id = c.doc_id
            WHERE c.id IN ({placeholders});
            """,
            tuple(int(x) for x in chunk_ids),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _row_to_order(row: sqlite3.Row) -> Dict[str, Any]:
    d = _row_to_dict(row)
    try:
        d["details"] = json.loads(d.get("details_json") or "{}")
    except Exception:
        d["details"] = {}
    d.pop("details_json", None)
    return d


# ---------------- Patients ----------------

def list_patients() -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC, id DESC;").fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_patient(patient_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM patients WHERE id = ?;", (patient_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def create_patient(payload: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO patients
            (full_name, age, sex, weight_kg, room, location, insurance, account_id, mrn, reason_for_visit, provider)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                payload.get("full_name"),
                payload.get("age"),
                payload.get("sex"),
                payload.get("weight_kg"),
                payload.get("room"),
                payload.get("location"),
                payload.get("insurance"),
                payload.get("account_id"),
                payload.get("mrn"),
                payload.get("reason_for_visit"),
                payload.get("provider"),
            ),
        )
        conn.commit()
        pid = conn.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"]
        out = get_patient(int(pid))
        assert out is not None
        return out
    finally:
        conn.close()


def delete_patient(patient_id: int) -> None:
    """Delete a patient and all related rows (via ON DELETE CASCADE)."""
    conn = get_conn()
    try:
        conn.execute("DELETE FROM patients WHERE id = ?;", (int(patient_id),))
        conn.commit()
    finally:
        conn.close()


# ---------------- Orders ----------------

def list_orders(patient_id: int, transcript_id: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        if transcript_id is None:
            rows = conn.execute(
                """
                SELECT * FROM orders
                WHERE patient_id = ?
                ORDER BY created_at DESC, id DESC;
                """,
                (int(patient_id),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM orders
                WHERE patient_id = ? AND transcript_id = ?
                ORDER BY created_at DESC, id DESC;
                """,
                (int(patient_id), int(transcript_id)),
            ).fetchall()
        return [_row_to_order(r) for r in rows]
    finally:
        conn.close()


def get_order(order_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM orders WHERE id = ?;", (int(order_id),)).fetchone()
        return _row_to_order(row) if row else None
    finally:
        conn.close()


def _order_exists(patient_id: int, transcript_id: Optional[int], category: str, title: str) -> bool:
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT id FROM orders
            WHERE patient_id = ?
              AND (transcript_id IS ? OR transcript_id = ?)
              AND lower(category) = lower(?)
              AND lower(title) = lower(?)
            LIMIT 1;
            """,
            (int(patient_id), transcript_id, transcript_id, category, title),
        ).fetchone()
        return bool(row)
    finally:
        conn.close()


def create_order(
    patient_id: int,
    category: str,
    title: str,
    priority: str = "routine",
    status: str = "draft",
    transcript_id: Optional[int] = None,
    source: str = "manual",
    evidence: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Create an order.

    Returns None if a duplicate order already exists for the same (patient, transcript_id, category, title).
    """

    title_norm = (title or "").strip()
    category_norm = (category or "").strip()
    if not title_norm or not category_norm:
        raise ValueError("category and title are required")

    if _order_exists(int(patient_id), transcript_id, category_norm, title_norm):
        return None

    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO orders
              (patient_id, transcript_id, category, title, priority, status, source, evidence, details_json, notes)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                int(patient_id),
                int(transcript_id) if transcript_id is not None else None,
                category_norm,
                title_norm,
                (priority or "routine").strip(),
                (status or "draft").strip(),
                (source or "manual").strip(),
                (evidence or None),
                json.dumps(details or {}, ensure_ascii=False),
                (notes or None),
            ),
        )
        conn.commit()
        oid = conn.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"]
        row = conn.execute("SELECT * FROM orders WHERE id = ?;", (int(oid),)).fetchone()
        return _row_to_order(row)
    finally:
        conn.close()


def update_order(order_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "category",
        "title",
        "priority",
        "status",
        "notes",
        "details",
        "evidence",
        "transcript_id",
    }
    clean: Dict[str, Any] = {k: updates[k] for k in updates.keys() if k in allowed}

    if "details" in clean:
        clean["details_json"] = json.dumps(clean.pop("details") or {}, ensure_ascii=False)

    sets: list[str] = []
    vals: list[Any] = []
    for k, v in clean.items():
        sets.append(f"{k} = ?")
        vals.append(v)
    sets.append("updated_at = datetime('now')")
    vals.append(int(order_id))

    conn = get_conn()
    try:
        conn.execute(
            f"UPDATE orders SET {', '.join(sets)} WHERE id = ?;",
            tuple(vals),
        )
        conn.commit()
        out = get_order(int(order_id))
        if not out:
            raise ValueError("Order not found")
        return out
    finally:
        conn.close()


def update_order_status(order_id: int, status: str) -> Dict[str, Any]:
    return update_order(int(order_id), {"status": (status or "").strip()})


def delete_order(order_id: int) -> None:
    conn = get_conn()
    try:
        conn.execute("DELETE FROM orders WHERE id = ?;", (int(order_id),))
        conn.commit()
    finally:
        conn.close()


# ---------------- Transcripts ----------------

def list_transcripts(patient_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, patient_id, text, audio_filename, created_at
            FROM transcripts
            WHERE patient_id = ?
            ORDER BY created_at DESC, id DESC;
            """,
            (patient_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_transcript_for_patient(patient_id: int, transcript_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single transcript that belongs to a given patient."""
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT id, patient_id, text, audio_filename, created_at
            FROM transcripts
            WHERE patient_id = ? AND id = ?;
            """,
            (patient_id, transcript_id),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def create_transcript(patient_id: int, text: str, audio_filename: Optional[str] = None) -> Dict[str, Any]:
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO transcripts (patient_id, text, audio_filename)
            VALUES (?, ?, ?);
            """,
            (patient_id, text, audio_filename),
        )
        conn.commit()
        tid = conn.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"]
        row = conn.execute(
            "SELECT id, patient_id, text, audio_filename, created_at FROM transcripts WHERE id = ?;",
            (tid,),
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()




def delete_transcript_visit(patient_id: int, transcript_id: int) -> Dict[str, Any]:
    """Delete a transcript (encounter) and all visit-linked records.

    This removes:
      - Orders linked via orders.transcript_id
      - Notes linked via notes.source_transcript_id
      - The transcript row itself

    Returns counts for what was removed.
    """
    conn = get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE;")

        # Ensure transcript belongs to patient.
        row = conn.execute(
            "SELECT id FROM transcripts WHERE patient_id = ? AND id = ?;",
            (int(patient_id), int(transcript_id)),
        ).fetchone()
        if not row:
            conn.execute("ROLLBACK;")
            return {"ok": False, "detail": "Transcript not found"}

        # IMPORTANT: delete visit-linked rows *before* deleting transcript.
        # Orders has FK transcript_id with ON DELETE SET NULL, so deleting transcript first would orphan them.
        cur = conn.execute(
            "DELETE FROM orders WHERE patient_id = ? AND transcript_id = ?;",
            (int(patient_id), int(transcript_id)),
        )
        orders_deleted = int(cur.rowcount or 0)

        cur = conn.execute(
            "DELETE FROM notes WHERE patient_id = ? AND source_transcript_id = ?;",
            (int(patient_id), int(transcript_id)),
        )
        notes_deleted = int(cur.rowcount or 0)

        cur = conn.execute(
            "DELETE FROM transcripts WHERE patient_id = ? AND id = ?;",
            (int(patient_id), int(transcript_id)),
        )
        transcripts_deleted = int(cur.rowcount or 0)

        conn.commit()
        return {
            "ok": True,
            "transcript_id": int(transcript_id),
            "orders_deleted": orders_deleted,
            "notes_deleted": notes_deleted,
            "transcripts_deleted": transcripts_deleted,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ---------------- Notes ----------------

def list_notes(patient_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, patient_id, type, content, source_transcript_id, created_at
            FROM notes
            WHERE patient_id = ?
            ORDER BY created_at DESC, id DESC;
            """,
            (patient_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def create_note(
    patient_id: int,
    note_type: str,
    content: str,
    source_transcript_id: Optional[int] = None,
) -> Dict[str, Any]:
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO notes (patient_id, type, content, source_transcript_id)
            VALUES (?, ?, ?, ?);
            """,
            (patient_id, note_type, content, source_transcript_id),
        )
        conn.commit()
        nid = conn.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"]
        row = conn.execute(
            """
            SELECT id, patient_id, type, content, source_transcript_id, created_at
            FROM notes WHERE id = ?;
            """,
            (nid,),
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


# ---------------- Assistant Messages ----------------

def list_assistant_messages(patient_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """List most recent assistant Q/A messages for a patient."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, patient_id, question, answer_text,
                   use_patient_context, use_guidelines,
                   confidence, sources_json, created_at
            FROM assistant_messages
            WHERE patient_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?;
            """,
            (patient_id, int(limit)),
        ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            # Normalize booleans
            d["use_patient_context"] = bool(d.get("use_patient_context"))
            d["use_guidelines"] = bool(d.get("use_guidelines"))
            # Parse sources JSON
            try:
                d["sources"] = json.loads(d.get("sources_json") or "[]")
            except Exception:
                d["sources"] = []
            # Keep raw too (optional)
            returnable = {
                "id": d.get("id"),
                "patient_id": d.get("patient_id"),
                "question": d.get("question"),
                "answer_text": d.get("answer_text"),
                "use_patient_context": d.get("use_patient_context"),
                "use_guidelines": d.get("use_guidelines"),
                "confidence": d.get("confidence"),
                "sources": d.get("sources"),
                "created_at": d.get("created_at"),
            }
            out.append(returnable)
        return out
    finally:
        conn.close()


def create_assistant_message(
    patient_id: int,
    question: str,
    answer_text: str,
    use_patient_context: bool,
    use_guidelines: bool,
    confidence: Optional[str],
    sources: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Insert an assistant message row and return it."""
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO assistant_messages
              (patient_id, question, answer_text, use_patient_context, use_guidelines, confidence, sources_json)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                int(patient_id),
                (question or "").strip(),
                (answer_text or "").strip(),
                1 if use_patient_context else 0,
                1 if use_guidelines else 0,
                (confidence or "").strip() or None,
                sources_json,
            ),
        )
        conn.commit()
        mid = conn.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"]
        # Reuse list/read shape
        rows = conn.execute(
            """
            SELECT id, patient_id, question, answer_text,
                   use_patient_context, use_guidelines,
                   confidence, sources_json, created_at
            FROM assistant_messages
            WHERE id = ?;
            """,
            (int(mid),),
        ).fetchone()
        d = _row_to_dict(rows)
        d["use_patient_context"] = bool(d.get("use_patient_context"))
        d["use_guidelines"] = bool(d.get("use_guidelines"))
        try:
            d["sources"] = json.loads(d.get("sources_json") or "[]")
        except Exception:
            d["sources"] = []
        return {
            "id": d.get("id"),
            "patient_id": d.get("patient_id"),
            "question": d.get("question"),
            "answer_text": d.get("answer_text"),
            "use_patient_context": d.get("use_patient_context"),
            "use_guidelines": d.get("use_guidelines"),
            "confidence": d.get("confidence"),
            "sources": d.get("sources"),
            "created_at": d.get("created_at"),
        }
    finally:
        conn.close()


def delete_assistant_message(patient_id: int, message_id: int) -> Dict[str, Any]:
    """Delete a single assistant Q/A message for a patient."""
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT id
            FROM assistant_messages
            WHERE id = ? AND patient_id = ?;
            """,
            (int(message_id), int(patient_id)),
        ).fetchone()

        if not row:
            return {"ok": False, "detail": "Assistant message not found"}

        conn.execute(
            """
            DELETE FROM assistant_messages
            WHERE id = ? AND patient_id = ?;
            """,
            (int(message_id), int(patient_id)),
        )
        conn.commit()
        return {"ok": True, "deleted_id": int(message_id)}
    finally:
        conn.close()


# ---------------- Imaging History ----------------

def _row_to_imaging_history(row: sqlite3.Row) -> Dict[str, Any]:
    return _row_to_dict(row)


def create_imaging_history(
    patient_id: Optional[int],
    original_filename: Optional[str],
    image_sha256: str,
    image_storage_path: str,
    prompt: str,
    result_text: str,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO imaging_history
              (patient_id, original_filename, image_sha256, image_storage_path, prompt, result_text, model_name)
            VALUES
              (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                int(patient_id) if patient_id is not None else None,
                (original_filename or None),
                (image_sha256 or "").strip(),
                (image_storage_path or "").strip(),
                (prompt or "").strip(),
                (result_text or "").strip(),
                (model_name or None),
            ),
        )
        conn.commit()
        hid = conn.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"]
        row = conn.execute(
            """
            SELECT ih.*, p.full_name AS patient_name
            FROM imaging_history ih
            LEFT JOIN patients p ON p.id = ih.patient_id
            WHERE ih.id = ?;
            """,
            (int(hid),),
        ).fetchone()
        assert row is not None
        return _row_to_dict(row)
    finally:
        conn.close()


def list_imaging_history(
    limit: int = 50,
    patient_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    limit_n = int(limit) if int(limit) > 0 else 50
    limit_n = min(limit_n, 200)

    conn = get_conn()
    try:
        if patient_id is None:
            rows = conn.execute(
                """
                SELECT ih.*, p.full_name AS patient_name
                FROM imaging_history ih
                LEFT JOIN patients p ON p.id = ih.patient_id
                ORDER BY ih.created_at DESC, ih.id DESC
                LIMIT ?;
                """,
                (limit_n,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT ih.*, p.full_name AS patient_name
                FROM imaging_history ih
                LEFT JOIN patients p ON p.id = ih.patient_id
                WHERE ih.patient_id = ?
                ORDER BY ih.created_at DESC, ih.id DESC
                LIMIT ?;
                """,
                (int(patient_id), limit_n),
            ).fetchall()

        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_imaging_history(history_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT ih.*, p.full_name AS patient_name
            FROM imaging_history ih
            LEFT JOIN patients p ON p.id = ih.patient_id
            WHERE ih.id = ?;
            """,
            (int(history_id),),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def delete_imaging_history(history_id: int) -> bool:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM imaging_history WHERE id = ?;",
            (int(history_id),),
        ).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM imaging_history WHERE id = ?;", (int(history_id),))
        conn.commit()
        return True
    finally:
        conn.close()
