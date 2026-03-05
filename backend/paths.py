"""
Single place for backend filesystem paths ( backend/paths.py)

Purpose
-------
This file centralizes all persistent storage locations used by the backend, so
they are easy to find and update.

Environment variables
---------------------
- EQUALCARE_DATA_DIR: Base data folder (defaults to backend/data)
- EQUALCARE_DB_PATH: SQLite DB file path (defaults to <DATA_DIR>/equalcare.db)
- EQUALCARE_GUIDELINES_DIR: Guidelines folder (defaults to <DATA_DIR>/guidelines)
- EQUALCARE_IMAGES_DIR: Images folder (defaults to <DATA_DIR>/images)
"""
import os
from pathlib import Path

# Current backend directory (backend/)
BACKEND_DIR = Path(__file__).resolve().parent

# Dev default: backend/data
# Product later: we'll set EQUALCARE_DATA_DIR to a writable user folder from Electron.
DATA_DIR = Path(os.environ.get("EQUALCARE_DATA_DIR", str(BACKEND_DIR / "data"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# SQLite database path (defaults inside DATA_DIR, override via EQUALCARE_DB_PATH)
DB_PATH = Path(os.environ.get("EQUALCARE_DB_PATH", str(DATA_DIR / "equalcare.db"))).resolve()

# Folder to store / index guideline documents (override via EQUALCARE_GUIDELINES_DIR)
GUIDELINES_DIR = Path(
    os.environ.get("EQUALCARE_GUIDELINES_DIR", str(DATA_DIR / "guidelines"))
).resolve()
GUIDELINES_DIR.mkdir(parents=True, exist_ok=True)

# Folder to store imaging inputs/outputs (override via EQUALCARE_IMAGES_DIR)
IMAGES_DIR = Path(
    os.environ.get("EQUALCARE_IMAGES_DIR", str(DATA_DIR / "images"))
).resolve()
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
