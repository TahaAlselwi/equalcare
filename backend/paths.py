# backend/paths.py
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent

# Dev default: backend/data
# Product later: we'll set EQUALCARE_DATA_DIR to a writable user folder from Electron.
DATA_DIR = Path(os.environ.get("EQUALCARE_DATA_DIR", str(BACKEND_DIR / "data"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)


DB_PATH = Path(os.environ.get("EQUALCARE_DB_PATH", str(DATA_DIR / "equalcare.db"))).resolve()

GUIDELINES_DIR = Path(
    os.environ.get("EQUALCARE_GUIDELINES_DIR", str(DATA_DIR / "guidelines"))
).resolve()
GUIDELINES_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_DIR = Path(
    os.environ.get("EQUALCARE_IMAGES_DIR", str(DATA_DIR / "images"))
).resolve()
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
