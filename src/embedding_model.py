from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from .utils import MODELS_DIR
except ImportError:  # pragma: no cover - supports `python src/*.py`
    from utils import MODELS_DIR


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
ONNX_MODEL_DIR = MODELS_DIR / "sentence_transformer_onnx"
EMBEDDING_BATCH_SIZE = 64


def _has_onnx_model(model_dir: Path = ONNX_MODEL_DIR) -> bool:
    return model_dir.exists() and any(model_dir.rglob("*.onnx"))


@lru_cache(maxsize=2)
def _load_sentence_transformer(prefer_onnx: bool = True):
    from sentence_transformers import SentenceTransformer

    if prefer_onnx and _has_onnx_model():
        return SentenceTransformer(str(ONNX_MODEL_DIR), backend="onnx", local_files_only=True)
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_embedding_backend(prefer_onnx: bool = True) -> str:
    if prefer_onnx and _has_onnx_model():
        return "onnx"
    return "torch"


def encode_texts(
    texts: list[object] | tuple[object, ...],
    normalize: bool = True,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    prefer_onnx: bool = True,
) -> np.ndarray:
    """Convert text into Sentence Transformer embeddings as float32 arrays."""
    clean_texts = ["" if text is None else str(text) for text in texts]
    model = _load_sentence_transformer(prefer_onnx=prefer_onnx)
    embeddings = model.encode(
        clean_texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=len(clean_texts) > batch_size,
    )
    return np.asarray(embeddings, dtype=np.float32)


