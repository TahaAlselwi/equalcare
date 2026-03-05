"""
Heuristic order extraction from visit transcript text.

This module extracts *explicit, actionable* medical orders from a transcript, such as:
- labs
- imaging
- procedures/therapies
- diet/activity instructions

Design notes
------------
- Conservative by default (aims to reduce false positives).
- Offline-first: pure Python + regex; intended to be easy to tweak.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import re
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
OrderCategory = str  # medication | lab | imaging | procedure | diet_activity | referral
OrderPriority = str  # routine | urgent


# -----------------------------------------------------------------------------
# Keyword lists (priority + action triggers)
# -----------------------------------------------------------------------------
_URGENT_WORDS = [
    "stat",
    "urgent",
    "immediately",
    "asap",
    "emergency",
    "now",
]
_ACTION_WORDS = [
    "order",
    "request",
    "prescribe",
    "start",
    "give",
    "administer",
    "schedule",
    "send for",
    "do",
    "run",
]

# -----------------------------------------------------------------------------
# Regex patterns (labs / imaging / procedures / meds)
# -----------------------------------------------------------------------------
_ACTION_RE = re.compile(r"(?:%s)" % "|".join(re.escape(w) for w in _ACTION_WORDS), re.IGNORECASE)

_LAB_RE = re.compile(
    r"\b(labs?|blood\s*test|urine|urinalysis|culture|cbc|cmp|bmp|lft|crp|esr|a1c|hba1c|troponin|d-?dimer|inr|pt\b|ptt|tsh|lipid|pregnancy\s*test|pcr)\b",
    re.IGNORECASE,
)

_IMAGING_RE = re.compile(
    r"\b(x\s*-?ray|xray|ct\b|mri\b|ultrasound|u/s\b|sonogram|radiograph|scan|echo|doppler)\b",
    re.IGNORECASE,
)

_PROCEDURE_RE = re.compile(
    r"\b(iv\s*fluids?|oxygen|o2\b|nebul(?:i[sz]er)?|suturing|suture|dressing|splint|cast|catheter|cannula|wound\s*care|admit|admission|discharge|procedure)\b",
    re.IGNORECASE,
)

_DIET_ACTIVITY_RE = re.compile(
    r"\b(npo|nil\s+per\s+os|diet|low\s*salt|low\s*sodium|fluid\s*restriction|bed\s*rest|activity|ambulate|mobilize|walk)\b",
    re.IGNORECASE,
)

_MED_DOSAGE_RE = re.compile(
    r"\b(\d+\s*(mg|mcg|g|ml)|mg\b|mcg\b|tablet|tab\b|capsule|cap\b|syrup|injection|iv\b|im\b|po\b|bid\b|tid\b|q\d+h|once\s+daily|daily)\b",
    re.IGNORECASE,
)
# -----------------------------------------------------------------------------
# Text normalization helpers
# -----------------------------------------------------------------------------
def _clean_text(t: str) -> str:
    if not t:
        return ""
    # Common tokenizer artifacts (some ASR pipelines)
    t = t.replace("</s>", " ")
    t = t.replace("\u0000", " ")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _strip_speaker_prefix(line: str) -> str:
    """Remove simple speaker prefixes like 'A:' or 'Speaker 1:' from transcript lines."""
    s = (line or "").strip()
    if not s:
        return ""

    # Examples: "A:", "B:", "SPEAKER_00:", "Speaker 1:", "Doctor:", "Patient:"
    s = re.sub(
        r"^(?:speaker[_\s-]*\d+|spk[_\s-]*\d+|doctor|patient|clinician|nurse)\s*:\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"^[A-Z]{1,3}\s*:\s*", "", s)  # A:, DR:, PT:
    return s.strip()

def _split_items(text: str) -> List[str]:
    """Split a free-form sentence/phrase into order-like items."""
    raw = (text or "").strip()
    if not raw:
        return []

    s = raw.replace("\r\n", "\n")
    s = re.sub(r"^[\s\t]*[-•\u2022]+\s+", "", s, flags=re.MULTILINE)

    # Prefer line-based splitting
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if len(lines) > 1:
        return lines

    # Otherwise, split on common separators (English transcripts)
    parts = re.split(r"\s*(?:;|,|\band\b|\bthen\b|\&)+\s*", s, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]

# -----------------------------------------------------------------------------
# Classification and heuristics
# -----------------------------------------------------------------------------
def _guess_priority(text: str) -> OrderPriority:
    t = (text or "").lower()
    return "urgent" if any(w in t for w in _URGENT_WORDS) else "routine"

def _classify_category(item: str) -> OrderCategory:
    t = (item or "").lower()

    # Imaging
    if _IMAGING_RE.search(t):
        return "imaging"

    # Labs
    if _LAB_RE.search(t):
        return "lab"

    # Diet / Activity
    if _DIET_ACTIVITY_RE.search(t):
        return "diet_activity"

    # Referrals / follow-up
    if re.search(r"\b(refer|referral|consult|follow\s*-?up|appointment|specialist)\b", t):
        return "referral"

    # Procedure / therapy
    if _PROCEDURE_RE.search(t):
        return "procedure"

    # Medication (only if it *looks* like a medication instruction)
    # NOTE: We avoid defaulting to medication for strict extraction.
    return "medication"

def _looks_like_strict_order_sentence(sentence: str) -> bool:
    """Return True if a sentence likely contains explicit, actionable orders.

    Strict signals:
      - Contains explicit orderable items (labs/imaging/procedure/diet patterns)
      - Or contains an action verb AND a medication dosage/route cue
    """
    s = (sentence or "").strip()
    if not s:
        return False

    # Strong category signals
    if _LAB_RE.search(s) or _IMAGING_RE.search(s) or _PROCEDURE_RE.search(s) or _DIET_ACTIVITY_RE.search(s):
        return True

    # Medication: action + dosage cue (keeps it strict)
    if _ACTION_RE.search(s) and _MED_DOSAGE_RE.search(s):
        return True

    # Medication: a "standalone" med line with dosage/route (common in dictations)
    # Example: "Paracetamol 500 mg PO BID"
    if _MED_DOSAGE_RE.search(s):
        # Require at least one word token (not purely numeric/units)
        if re.search(r"[A-Za-z]{3,}", s):
            return True

    # Medication: action + common generic/drug cues
    if _ACTION_RE.search(s) and re.search(
        r"\b(antibiotic|analgesic|pain\s*killer|paracetamol|acetaminophen|ibuprofen|amoxicillin)\b",
        s,
        re.IGNORECASE,
    ):
        return True

    return False

def _normalize_candidate_sentence(sentence: str) -> str:
    """Remove leading filler verbs to make nicer titles."""
    s = (sentence or "").strip()
    if not s:
        return ""

    # English filler
    s = re.sub(r"^(we\s+will|we'll|i\s+will|i'll|please|let's)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(order|request|prescribe|start|give|take|schedule|refer|consult|obtain|get)\s+", "", s, flags=re.IGNORECASE)

    # Trim punctuation
    s = s.strip(" .,:;\t\n\r!?")
    return s.strip()

# -----------------------------------------------------------------------------
# Sentence splitting
# -----------------------------------------------------------------------------
def _split_sentences(text: str) -> List[str]:
    """Split transcript into sentence-like chunks for keyword scanning."""
    if not text:
        return []

    # First split by newlines (transcripts are often line-based)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) > 1:
        return lines

    # Otherwise fall back to punctuation splitting
    parts = re.split(r"(?<=[\.!\?;])\s+", text)
    return [p.strip() for p in parts if p.strip()]

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def extract_orders_from_transcript(transcript_text: str) -> List[Dict[str, Any]]:
    """Extract order candidates from a visit transcript.

    Returns a list of dicts suitable for DB insertion.
    Each item includes:
      - category, title, priority, status, source, details

    This function is heuristic and conservative.
    """
    text = _clean_text(transcript_text)
    if not text:
        return []

    out: List[Dict[str, Any]] = []

    for raw in _split_sentences(text):
        sent = _strip_speaker_prefix(raw)
        if not sent:
            continue

        # Only consider explicit/strict order sentences.
        if not _looks_like_strict_order_sentence(sent):
            continue

        normalized = _normalize_candidate_sentence(sent)

        # Split into smaller items (e.g., "CBC and CMP")
        items = _split_items(normalized) or ([normalized] if normalized else [])

        for it in items:
            title = _normalize_candidate_sentence(it)
            if not title:
                continue

            cat = _classify_category(title)

            # Strict filtering:
            # - Skip referral by default for auto extraction (keeps "orders" as executable tasks)
            if cat == "referral":
                continue

            # - Medication must have a clearer cue (action or dosage) to avoid false positives.
            if cat == "medication":
                drug_cue = bool(
                    _MED_DOSAGE_RE.search(sent)
                    or re.search(
                        r"\b(antibiotic|analgesic|pain\s*killer|paracetamol|acetaminophen|ibuprofen|amoxicillin)\b",
                        sent,
                        re.IGNORECASE,
                    )
                )
                standalone_med = bool(_MED_DOSAGE_RE.search(sent) and re.search(r"[A-Za-z]{3,}", sent))
                if not ((bool(_ACTION_RE.search(sent)) and drug_cue) or standalone_med):
                    continue

            out.append(
                {
                    "category": cat,
                    "title": title,
                    "priority": _guess_priority(title),
                    "status": "draft",
                    "source": "auto_transcript",
                    "details": {},
                }
            )

    # Deduplicate by (category, title)
    seen = set()
    unique: List[Dict[str, Any]] = []
    for o in out:
        key = (str(o.get("category", "")).lower(), str(o.get("title", "")).strip().lower())
        if not key[0] or not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(o)

    return unique

