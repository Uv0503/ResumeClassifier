from __future__ import annotations

import json
from functools import lru_cache

from scipy.sparse import csr_matrix

try:
    from .utils import DATA_DIR
except ImportError:  # pragma: no cover - supports script path imports
    from utils import DATA_DIR


TECH_SKILL_LABEL = "TECH_SKILL"
SKILL_PATTERNS_PATH = DATA_DIR / "skills_patterns.json"
CONTEXT_WINDOW = 8
CONTEXT_TERMS = {
    "api",
    "apis",
    "built",
    "develop",
    "developed",
    "developing",
    "deployment",
    "deployed",
    "engineer",
    "engineering",
    "experience",
    "framework",
    "frameworks",
    "hiring",
    "knowledge",
    "need",
    "needs",
    "platform",
    "platforms",
    "proficient",
    "required",
    "requires",
    "skills",
    "stack",
    "technology",
    "technologies",
    "tool",
    "tools",
    "using",
    "with",
}

SKILL_FEATURES: list[str] = [
    "Java",
    "Python",
    "SQL",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Spring Boot",
    "REST APIs",
    "Docker",
    "Kubernetes",
    "AWS",
    "Kafka",
    "Redis",
    "React.js",
    "Node.js",
    "Machine Learning",
    "NLP",
    "Scikit-learn",
    "Pandas",
    "NumPy",
]


def _load_skill_patterns() -> list[dict[str, str]]:
    if not SKILL_PATTERNS_PATH.exists():
        raise FileNotFoundError(f"Missing skill patterns file: {SKILL_PATTERNS_PATH}")

    patterns = json.loads(SKILL_PATTERNS_PATH.read_text(encoding="utf-8"))
    if not isinstance(patterns, list):
        raise ValueError("Skill patterns file must contain a JSON list of EntityRuler patterns.")
    return patterns


@lru_cache(maxsize=1)
def _load_skill_nlp():
    try:
        import spacy
    except Exception as exc:
        raise ImportError(
            "spaCy is required for NER skill extraction. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True, "phrase_matcher_attr": "LOWER"})
    ruler.add_patterns(_load_skill_patterns())
    return nlp


def _has_skill_context(ent) -> bool:
    doc_tokens = [token.text.lower() for token in ent.doc if not token.is_space]
    if len(doc_tokens) <= 12:
        return True

    sentence = ent.sent if ent.sent is not None else ent.doc
    tokens = [token.text.lower() for token in sentence if not token.is_space]
    if any(token in CONTEXT_TERMS for token in tokens):
        return True

    start = max(ent.start - CONTEXT_WINDOW, 0)
    end = min(ent.end + CONTEXT_WINDOW, len(ent.doc))
    nearby = [token.text.lower() for token in ent.doc[start:end] if not token.is_space]
    return any(token in CONTEXT_TERMS for token in nearby)


def extract_skills(text: object) -> list[str]:
    """Extract technical skills from free-form text with spaCy EntityRuler NER."""
    text = "" if text is None else str(text)
    if not text.strip():
        return []

    doc = _load_skill_nlp()(text)
    found_skills = {
        ent.ent_id_ or ent.text
        for ent in doc.ents
        if ent.label_ == TECH_SKILL_LABEL and _has_skill_context(ent)
    }
    return sorted(found_skills)


def find_missing_skills(resume_text: object, job_description: object) -> dict[str, list[str]]:
    """Compare resume skills against job-description skills."""
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)
    missing_skills = sorted(set(jd_skills).difference(resume_skills))
    matched_skills = sorted(set(jd_skills).intersection(resume_skills))

    return {
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }


def build_skill_feature_matrix(texts: list[object], skill_features: list[str] | None = None):
    """Create a binary sparse matrix for selected technical skills."""
    selected_features = skill_features or SKILL_FEATURES
    rows: list[list[int]] = []

    for text in texts:
        found_skills = set(extract_skills(text))
        rows.append([1 if skill in found_skills else 0 for skill in selected_features])

    return csr_matrix(rows, dtype="int8")
