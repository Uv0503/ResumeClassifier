from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

try:
    from nltk.corpus import stopwords
except Exception:  # pragma: no cover - nltk data may not be installed yet
    stopwords = None

try:
    from .utils import DATA_DIR
except ImportError:  # pragma: no cover - supports script path imports
    from utils import DATA_DIR


RESUME_PROFILE_COLUMNS = [
    "skills",
    "career_objective",
    "responsibilities",
    "positions",
    "degree_names",
    "major_field_of_studies",
    "educational_institution_name",
]

LEAKAGE_COLUMNS = {
    "job_position_name",
    "matched_score",
    "skills_required",
    "educational_requirements",
    "experience_requirement",
    "job_responsibilities",
}

COLUMN_RENAMES = {
    "educationaL_requirements": "educational_requirements",
    "experiencere_requirement": "experience_requirement",
    "responsibilities.1": "job_responsibilities",
}

ENCODING_REPLACEMENTS = {
    "\ufeff": "",
    "Ã¯Â»Â¿": "",
    "Ã¢â‚¬Â¢": " ",
    "Ã¢Å¡Â«": " ",
    "Ã‚Â·": " ",
    "â€¢": " ",
    "â€“": "-",
    "â€”": "-",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "Â": " ",
}

TECHNICAL_TERMS_TO_KEEP = {
    "api",
    "apis",
    "aws",
    "azure",
    "c",
    "c++",
    "c#",
    "cloud",
    "data",
    "docker",
    "fastapi",
    "gcp",
    "git",
    "java",
    "javascript",
    "kubernetes",
    "linux",
    "machine",
    "mongodb",
    "node.js",
    "nlp",
    "postgresql",
    "python",
    "react.js",
    "sql",
}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataset columns, including hidden BOM characters and known typos."""
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return df.rename(columns=COLUMN_RENAMES)


def _fix_encoding_artifacts(text: str) -> str:
    for source, target in ENCODING_REPLACEMENTS.items():
        text = text.replace(source, target)
    return text


def _get_stop_words() -> set[str]:
    if stopwords is None:
        return set()
    try:
        words = set(stopwords.words("english"))
    except LookupError:
        return set()
    return words.difference(TECHNICAL_TERMS_TO_KEEP)


def clean_resume_text(text: object, remove_stopwords: bool = False) -> str:
    """Clean resume or job-description text while preserving useful tech tokens."""
    if text is None or pd.isna(text):
        return ""

    text = _fix_encoding_artifacts(str(text)).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"(\+?\d[\d\s().-]{7,}\d)", " ", text)

    replacements = {
        "c++": " c++ ",
        "c#": " c# ",
        "node.js": " node.js ",
        "react.js": " react.js ",
        "rest api": " rest api ",
        "rest apis": " rest apis ",
        "scikit-learn": " scikit learn ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    text = re.sub(r"(?<![a-z])[-_/](?![a-z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        stop_words = _get_stop_words()
        text = " ".join(word for word in text.split() if word not in stop_words)

    return text


def build_profile_text(df: pd.DataFrame) -> pd.Series:
    """Combine resume-side columns into one model input without target leakage."""
    available_columns = [col for col in RESUME_PROFILE_COLUMNS if col in df.columns]
    if not available_columns:
        raise ValueError(
            "No resume profile columns found. Expected at least one of: "
            f"{RESUME_PROFILE_COLUMNS}"
        )

    profile_parts = df[available_columns].fillna("").astype(str)
    return profile_parts.apply(
        lambda row: " ".join(value.strip() for value in row if value.strip()),
        axis=1,
    )


def load_job_role_data(path: str | Path = DATA_DIR / "resume_data.csv") -> pd.DataFrame:
    """Load resume_data.csv and return leakage-free training rows."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    df = clean_column_names(df)
    if "job_position_name" not in df.columns:
        raise ValueError(
            "Dataset is missing target column 'job_position_name'. "
            f"Available columns: {sorted(df.columns)}"
        )

    df["profile_text"] = build_profile_text(df)
    df["profile_text"] = df["profile_text"].apply(clean_resume_text)
    df["job_position_name"] = df["job_position_name"].fillna("").astype(str).str.strip()

    df = df[(df["profile_text"] != "") & (df["job_position_name"] != "")]
    df = df.drop_duplicates(subset=["profile_text", "job_position_name"])
    return df.reset_index(drop=True)


def load_resume_data(path: str | Path = DATA_DIR / "Resume.csv") -> pd.DataFrame:
    """Legacy loader retained for old notebooks or comparisons."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    required_columns = {"Resume_str", "Category"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    df = df[["Resume_str", "Category"]].rename(columns={"Resume_str": "Resume"})
    df = df.dropna(subset=["Resume", "Category"]).drop_duplicates()
    df["Resume"] = df["Resume"].astype(str)
    df["Category"] = df["Category"].astype(str)
    return df.reset_index(drop=True)
