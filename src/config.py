from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TESTING_DIR = DATA_DIR / "Testing"
TESTING_LABELS = TESTING_DIR / "testing_labels.csv"
TESTING_WORDS = TESTING_DIR / "testing_words"
TRAINING_LABELS = DATA_DIR / "Training" / "training_labels.csv"

DB_PATH = ROOT / "lancedb_handwriting_demo"
TABLE_NAME = "doctor_handwriting"
OUTPUT_DIR = ROOT / "outputs"

SOURCE_DATASET = "doctor-handwriting-recognition-dataset"
DEFAULT_TEST_LIMIT: int | None = None
DEFAULT_PREVIEW_SAMPLES = 8
VECTOR_DIM = 384

OPENAI_VISION_MODEL = "gpt-5.4-mini"
OPENAI_EXTRACTION_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LATE_INTERACTION_MODEL = "colbert-ir/colbertv2.0"


def load_local_env() -> None:
    load_dotenv(ROOT / ".env")


def ensure_output_dir(*parts: str) -> Path:
    path = OUTPUT_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
