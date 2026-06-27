from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

try:
    from .embedding_model import EMBEDDING_MODEL_NAME, ONNX_MODEL_DIR
    from .utils import ensure_directories
except ImportError:  # pragma: no cover - supports `python src/export_onnx_model.py`
    from embedding_model import EMBEDDING_MODEL_NAME, ONNX_MODEL_DIR
    from utils import ensure_directories


def export_onnx_model() -> Path:
    """Export the local Sentence Transformer model for ONNX Runtime CPU inference."""
    from sentence_transformers import SentenceTransformer

    ensure_directories()
    ONNX_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, backend="onnx", local_files_only=True)
    except Exception:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, backend="onnx")
    model.save_pretrained(str(ONNX_MODEL_DIR))
    print(f"Saved ONNX Sentence Transformer to {ONNX_MODEL_DIR}")
    return ONNX_MODEL_DIR


if __name__ == "__main__":
    export_onnx_model()


