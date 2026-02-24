# backend/image_service.py
import base64
import ollama  # pip install ollama

MODEL_NAME = "thiagomoraes/medgemma-1.5-4b-it:Q4_K_S"
DEFAULT_PROMPT = "Describe this chest X-ray. What do you see?"

def image_bytes_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")

def extract_content(resp) -> str:
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

def analyze_image_bytes(
    img_bytes: bytes,
    prompt: str = DEFAULT_PROMPT,
    model_name: str = MODEL_NAME,
) -> str:
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
