from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_directories() -> None:
    """Create runtime output directories if they do not exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save a Python object with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, output_path)


def load_pickle(path: str | Path) -> Any:
    """Load a Python object saved with joblib."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing required artifact: {model_path}. Run `python src/train_model.py` first."
        )
    return joblib.load(model_path)


def format_list(items: list[str]) -> str:
    """Return a readable comma-separated list for UI output."""
    return ", ".join(items) if items else "None found"
