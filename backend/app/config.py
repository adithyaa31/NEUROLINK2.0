import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", ROOT / "neurolink.db"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "backend" / "models" / "weights"))
FRONTEND_ORIGINS = [origin.strip() for origin in os.getenv(
    "FRONTEND_ORIGINS", os.getenv("FRONTEND_ORIGIN", "http://localhost:5173,http://127.0.0.1:5173")
).split(",") if origin.strip()]
