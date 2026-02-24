# backend/run_backend.py
"""
import os
import uvicorn

def main():
    host = os.environ.get("EQUALCARE_HOST", "127.0.0.1")
    port = int(os.environ.get("EQUALCARE_BACKEND_PORT", "8000"))

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        log_level="info",
        workers=1,
    )

if __name__ == "__main__":
    main()
"""