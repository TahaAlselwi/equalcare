import re
from typing import Any, Dict, List, Optional

OrderCategory = str  # medication | lab | imaging | procedure | diet_activity | referral
OrderPriority = str  # routine | urgent

# This module extracts order candidates from the VISIT TRANSCRIPT TEXT.
# It is heuristic-based (fast, offline-first) and designed to be edited later.


_URGENT_WORDS = [
    "stat",
    "urgent",
    "immediately",
    "asap",
    "emergency",
    "now",
    # Arabic
    "حالاً",
    "حالا",
    "فوراً",
    "فورا",
    "مستعجل",
    "طارئ",
    "طوارئ",
]


# Triggers: if a sentence contains any of these, we consider it a candidate.
# ---- Strict extraction philosophy ----
# We only want explicit, actionable medical orders ("things to execute"), e.g.
# labs, imaging, procedures, medications, diet/activity instructions.
#
# We intentionally avoid weak signals like "follow up" (which is ambiguous and
# often not an orderable task inside the clinic).

_ACTION_WORDS = [
    # English
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
    # Arabic
    "اطلب",
    "نطلب",
    "طلب",
    "وصف",
    "أكتب",
    "اكتب",
    "ابدأ",
    "ابدا",
    "اعط",
    "أعط",
    "نعطي",
    "اعطاء",
]

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

_AR_LAB_WORDS = ["تحاليل", "تحليل", "مزرعة", "صورة دم", "سكر", "وظائف كبد", "وظائف كلى", "بول"]
_AR_IMAGING_WORDS = ["أشعة", "اشعة", "تصوير", "سونار", "رنين", "طبقي", "ايكو", "إيكو"]
_AR_PROCEDURE_WORDS = ["محلول", "وريدي", "عضلي", "اوكسجين", "أوكسجين", "تبخير", "قسطرة", "تنويم", "خياطة", "ضماد", "غيار", "جبس", "تجبير"]
_AR_DIET_ACTIVITY_WORDS = ["صيام", "حمية", "قليل الملح", "راحة", "حركة", "مشي", "سوائل"]


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

    # Otherwise, split on common separators
    parts = re.split(r"\s*(?:;|,|\band\b|\&|\u060C|\bثم\b|\bو\b)\s*", s, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def _guess_priority(text: str) -> OrderPriority:
    t = (text or "").lower()
    return "urgent" if any(w.lower() in t for w in _URGENT_WORDS) else "routine"


def _classify_category(item: str) -> OrderCategory:
    t = (item or "").lower()

    # Imaging
    if _IMAGING_RE.search(t):
        return "imaging"
    if any(w in item for w in ["أشعة", "اشعة", "تصوير", "سونار", "رنين", "طبقي"]):
        return "imaging"

    # Labs
    if _LAB_RE.search(t):
        return "lab"
    if any(w in item for w in ["تحاليل", "تحليل", "مزرعة"]):
        return "lab"

    # Diet / Activity
    if _DIET_ACTIVITY_RE.search(t):
        return "diet_activity"
    if any(w in item for w in ["صيام", "حمية", "قليل الملح", "راحة", "حركة", "مشي"]):
        return "diet_activity"

    # Referrals / follow-up
    if re.search(r"\b(refer|referral|consult|follow\s*-?up|appointment|specialist)\b", t):
        return "referral"
    if any(w in item for w in ["إحالة", "احالة", "تحويل", "متابعة", "راجع", "استشارة", "اخصائي", "أخصائي"]):
        return "referral"

    # Procedure / therapy
    if _PROCEDURE_RE.search(t):
        return "procedure"
    if any(w in item for w in ["ضماد", "خياطة", "جلسة", "علاج طبيعي", "أوكسجين", "اوكسجين", "محلول"]):
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
    if any(w in s for w in _AR_LAB_WORDS + _AR_IMAGING_WORDS + _AR_PROCEDURE_WORDS + _AR_DIET_ACTIVITY_WORDS):
        return True

    # Medication: action + dosage cue (keeps it strict)
    if _ACTION_RE.search(s) and _MED_DOSAGE_RE.search(s):
        return True

    # Medication: a "standalone" med line with dosage/route (common in dictations)
    # Example: "Paracetamol 500 mg PO BID" or "باراسيتامول 500mg مرتين يومياً"
    if _MED_DOSAGE_RE.search(s):
        # Require at least one word token (not purely numeric/units)
        if re.search(r"[A-Za-z\u0600-\u06FF]{3,}", s):
            return True

    # Medication: action + common generic/drug cues
    if _ACTION_RE.search(s) and re.search(r"\b(antibiotic|analgesic|pain\s*killer|paracetamol|acetaminophen|ibuprofen|amoxicillin)\b", s, re.IGNORECASE):
        return True
    if _ACTION_RE.search(s) and any(w in s for w in ["مضاد", "مسكن", "باراسيتامول", "ايبوبروفين", "أموكس", "اموكس"]):
        return True

    return False


def _normalize_candidate_sentence(sentence: str) -> str:
    """Remove leading filler verbs to make nicer titles."""
    s = (sentence or "").strip()
    if not s:
        return ""

    # English filler
    s = re.sub(r"^(we\s+will|we'll|i\s+will|i'll|please|let's)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(order|request|prescribe|start|give|take|schedule|refer|consult)\s+", "", s, flags=re.IGNORECASE)

    # Arabic filler
    s = re.sub(r"^(سن|سوف)\s+", "", s)
    s = re.sub(r"^(اطلب|نطلب|طلب|وصف|اكتب|أكتب|ابدأ|ابدا|خذ|اعط|أعط|نعطي|حول|متابعة|راجع)\s+", "", s)

    # Trim punctuation
    s = s.strip(" .,:;\t\n\r\u061F!?" + "؟")
    return s.strip()


def _split_sentences(text: str) -> List[str]:
    """Split transcript into sentence-like chunks for keyword scanning."""
    if not text:
        return []

    # First split by newlines (transcripts are often line-based)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) > 1:
        return lines

    # Otherwise fall back to punctuation splitting
    parts = re.split(r"(?<=[\.!\?\u061F!؛;])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


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
        items = _split_items(normalized)
        if not items:
            items = [normalized] if normalized else []

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
                    or re.search(r"\b(antibiotic|analgesic|pain\s*killer|paracetamol|acetaminophen|ibuprofen|amoxicillin)\b", sent, re.IGNORECASE)
                    or any(w in sent for w in ["مضاد", "مسكن", "باراسيتامول", "ايبوبروفين", "أموكس", "اموكس"])
                )
                standalone_med = bool(_MED_DOSAGE_RE.search(sent) and re.search(r"[A-Za-z\u0600-\u06FF]{3,}", sent))
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


# Backwards-compatible helper for older code paths.
# If transcript_text is provided, we use it and ignore the structured Plan fields.
def extract_orders_from_structured_soap(
    structured: Dict[str, Dict[str, str]],
    transcript_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if transcript_text:
        return extract_orders_from_transcript(transcript_text)

    # Fallback to Plan-based extraction ONLY if transcript is missing.
    plan = (structured or {}).get("plan", {}) or {}
    plan_text = "\n".join(
        [
            plan.get("diagnostic_imaging", "") or "",
            plan.get("lab_reports", "") or "",
            plan.get("treatment", "") or "",
            plan.get("next_appointment", "") or "",
        ]
    ).strip()
    return extract_orders_from_transcript(plan_text)
