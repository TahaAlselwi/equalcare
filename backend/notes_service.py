"""
Notes / SOAP generation utilities.

What this module provides
-------------------------
Structured SOAP (JSON) extraction:
   - `generate_structured_soap(...)` returns a strict, UI-editable JSON structure.
   - `structured_soap_to_narrative(...)` converts that structure into a stable narrative note.

"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import re
from typing import Dict

from ollama import chat 

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
NOTE_MODEL_NAME = "MedAIBase/MedGemma1.0:4b"

# -----------------------------------------------------------------------------
# SOAP field definitions
# -----------------------------------------------------------------------------
SUBJECTIVE_FIELDS: Dict[str, str] = {
    "chief_complaints": "Chief Complaint(s)",
    "hpi": "HPI",
    "current_medication": "Current Medication",
    "medical_history": "Medical History",
    "allergies_intolerance": "Allergies/Intolerance",
    "surgical_history": "Surgical History",
    "family_history": "Family History",
    "social_history": "Social History",
    "ros": "ROS",
}

OBJECTIVE_FIELDS: Dict[str, str] = {
    "vitals": "Vitals",
    "past_results": "Past Results",
    "physical_examination": "Physical Examination",
}

ASSESSMENT_FIELDS: Dict[str, str] = {
    "assessment": "Assessment",
}

PLAN_FIELDS: Dict[str, str] = {
    "treatment": "Treatment",
    "diagnostic_imaging": "Diagnostic Imaging",
    "lab_reports": "Lab Reports",
    "next_appointment": "Next Appointment",
}

# -----------------------------------------------------------------------------
# Structured SOAP
# -----------------------------------------------------------------------------

def _empty_section(labels: Dict[str, str]) -> Dict[str, str]:
    """Create a default section dict filled with 'Not mentioned' for each field key."""
    return {k: "Not mentioned" for k in labels.keys()}

def empty_structured_soap() -> Dict[str, Dict[str, str]]:
    """Return an empty structured SOAP object with all fields set to 'Not mentioned'."""
    return {
        "subjective": _empty_section(SUBJECTIVE_FIELDS),
        "objective": _empty_section(OBJECTIVE_FIELDS),
        "assessment": _empty_section(ASSESSMENT_FIELDS),
        "plan": _empty_section(PLAN_FIELDS),
    }

def build_structured_soap_prompt(conversation_text: str) -> str:
    """Build a strict-JSON prompt for extracting a structured SOAP note from a transcript."""
    s_keys = list(SUBJECTIVE_FIELDS.keys())
    o_keys = list(OBJECTIVE_FIELDS.keys())
    a_keys = list(ASSESSMENT_FIELDS.keys())
    p_keys = list(PLAN_FIELDS.keys())

    # Provide the exact JSON skeleton expected.
    skeleton = {
        "subjective": {k: "" for k in s_keys},
        "objective": {k: "" for k in o_keys},
        "assessment": {k: "" for k in a_keys},
        "plan": {k: "" for k in p_keys},
    }

    return (
        "You are a clinical documentation assistant.\n\n"
        "Task: Extract a structured SOAP note from the conversation transcript.\n"
        "Rules:\n"
        "- Use ONLY information explicitly stated in the transcript.\n"
        "- If a field is not mentioned, write exactly: Not mentioned\n"
        "- Be concise and clinically phrased.\n"
        "- Return STRICT JSON only (no markdown, no commentary, no backticks).\n\n"
        "Return JSON with this exact shape and keys:\n"
        f"{json.dumps(skeleton, ensure_ascii=False)}\n\n"
        "Transcript:\n"
        f"{conversation_text}"
    ).strip()

# -----------------------------------------------------------------------------
# Response parsing and coercion
# -----------------------------------------------------------------------------
def _extract_json_blob(text: str) -> str:
    """Best-effort extraction of a JSON object from a model response."""
    if not text:
        return ""
    t = text.strip()
    # Remove common fences
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    # Try to slice the first top-level object
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return t[start : end + 1]
def _coerce_structured(obj: dict) -> Dict[str, Dict[str, str]]:
    out = empty_structured_soap()

    def take(section: str, labels: Dict[str, str]):
        src = obj.get(section) if isinstance(obj.get(section), dict) else {}
        for k in labels.keys():
            v = src.get(k, out[section][k])
            if v is None:
                v = out[section][k]
            out[section][k] = str(v).strip() if str(v).strip() else out[section][k]

    take("subjective", SUBJECTIVE_FIELDS)
    take("objective", OBJECTIVE_FIELDS)
    take("assessment", ASSESSMENT_FIELDS)
    take("plan", PLAN_FIELDS)

    return out

def generate_structured_soap(conversation_text: str, model_name: str = NOTE_MODEL_NAME) -> Dict[str, Dict[str, str]]:
    """Generate structured SOAP JSON for editable fields."""
    prompt = build_structured_soap_prompt(conversation_text)
    response = chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    raw = (response.message.content or "").strip()
    blob = _extract_json_blob(raw)

    if blob:
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return _coerce_structured(obj)
        except Exception:
            pass

    # Fallback: return empty (safe) structure rather than a wrong parse.
    return empty_structured_soap()

# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------
def structured_soap_to_narrative(structured: Dict[str, Dict[str, str]]) -> str:
    """Convert structured SOAP JSON into a stable narrative format.

    This is used for saving notes automatically after transcription, while keeping
    the content easy to render back into the UI fields.
    """

    def norm(v: str) -> str:
        vv = (v or "").strip()
        if not vv:
            return ""
        if vv.lower() == "not mentioned":
            return ""
        return vv

    lines: list[str] = []

    def push_section(title: str, labels: Dict[str, str], sec: Dict[str, str]):
        lines.append(f"{title}:")
        for k, label in labels.items():
            val = norm(sec.get(k, ""))
            if not val:
                continue
            lines.append(f"{label}:")
            lines.append(val)
        lines.append("")

    push_section("Subjective", SUBJECTIVE_FIELDS, structured.get("subjective", {}) or {})
    push_section("Objective", OBJECTIVE_FIELDS, structured.get("objective", {}) or {})
    push_section("Assessment", ASSESSMENT_FIELDS, structured.get("assessment", {}) or {})
    push_section("Plan", PLAN_FIELDS, structured.get("plan", {}) or {})

    return "\n".join(lines).strip()

