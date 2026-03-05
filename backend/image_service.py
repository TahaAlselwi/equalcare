"""
Image analysis utilities (vision model).

Purpose
-------
This module provides a small helper layer around `ollama.chat(...)` for sending an
image (as base64) with a user prompt, and returning the model's text response.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import base64
import ollama  

# -----------------------------------------------------------------------------
# Model configuration (defaults)
# -----------------------------------------------------------------------------
MODEL_NAME = "thiagomoraes/medgemma-1.5-4b-it:Q4_K_S"
DEFAULT_PROMPT = "Describe this X-ray. What do you see?"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def image_bytes_to_base64(img_bytes: bytes) -> str:
    """Convert raw image bytes to a base64-encoded UTF-8 string."""
    return base64.b64encode(img_bytes).decode("utf-8")

def extract_content(resp) -> str:
    """Extract a text response from the object returned by `ollama.chat`."""
    try:
        msg = getattr(resp, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if content:
                return content
    except Exception:
        pass

    if isinstance(resp, dict):
        if isinstance(resp.get("message"), dict):
            return (resp["message"].get("content") or "")
        return (resp.get("content") or "")

    return str(resp)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def analyze_image_bytes(
    img_bytes: bytes,
    prompt: str = DEFAULT_PROMPT,
    model_name: str = MODEL_NAME,
) -> str:
    """
    Analyze an image by sending it to an Ollama vision-capable model.

    Parameters
    ----------
    img_bytes:
        Raw image bytes (e.g., from an uploaded PNG/JPG).
    prompt:
        User instruction given to the vision model.
    model_name:
        Ollama model tag to use (must support images).

    Returns
    -------
    str:
        The model's text response.
    """
    if not img_bytes:
        raise ValueError("img_bytes is empty")

    image_base64 = image_bytes_to_base64(img_bytes)

    resp = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64],
            }
        ],
    )

    return extract_content(resp).strip()
