from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / MODEL_NAME


def download_model() -> Path:
    """Download all-MiniLM-L6-v2 once and save it inside the repository."""
    if LOCAL_MODEL_DIR.exists() and (LOCAL_MODEL_DIR / "modules.json").exists():
        print(f"Local Sentence Transformer already exists at {LOCAL_MODEL_DIR}")
        return LOCAL_MODEL_DIR

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    model.save_pretrained(str(LOCAL_MODEL_DIR))
    print(f"Saved {MODEL_NAME} to {LOCAL_MODEL_DIR}")
    return LOCAL_MODEL_DIR


if __name__ == "__main__":
    download_model()
