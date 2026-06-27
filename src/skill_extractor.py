from __future__ import annotations

from functools import lru_cache

from scipy.sparse import csr_matrix


TECH_SKILL_LABEL = "TECH_SKILL"
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

SKILL_PATTERNS: dict[str, list[str]] = {
    "Python": ["Python"],
    "Java": ["Java"],
    "JavaScript": ["JavaScript", "JS"],
    "TypeScript": ["TypeScript", "TS"],
    "C++": ["C++", "CPP"],
    "C#": ["C#", "C Sharp", "CSharp"],
    "SQL": ["SQL"],
    "MySQL": ["MySQL"],
    "PostgreSQL": ["PostgreSQL", "Postgres"],
    "MongoDB": ["MongoDB", "Mongo DB"],
    "Machine Learning": ["Machine Learning", "ML"],
    "Deep Learning": ["Deep Learning", "DL"],
    "NLP": ["NLP", "Natural Language Processing"],
    "TensorFlow": ["TensorFlow", "Tensor Flow"],
    "PyTorch": ["PyTorch", "Torch"],
    "Scikit-learn": ["Scikit-learn", "Scikit learn", "sklearn"],
    "Pandas": ["Pandas"],
    "NumPy": ["NumPy", "Numpy"],
    "Docker": ["Docker"],
    "Kubernetes": ["Kubernetes", "K8s"],
    "AWS": ["AWS", "Amazon Web Services"],
    "Azure": ["Azure"],
    "GCP": ["GCP", "Google Cloud"],
    "Git": ["Git", "GitHub"],
    "GitHub Actions": ["GitHub Actions"],
    "CI/CD": ["CI/CD", "CI CD", "CICD"],
    "FastAPI": ["FastAPI", "Fast API"],
    "Flask": ["Flask"],
    "Django": ["Django"],
    "Streamlit": ["Streamlit"],
    "Spring Boot": ["Spring Boot"],
    "React.js": ["React.js", "ReactJS", "React JS", "React"],
    "Node.js": ["Node.js", "NodeJS", "Node JS"],
    "Express.js": ["Express.js", "ExpressJS", "Express JS"],
    "Kafka": ["Kafka"],
    "Redis": ["Redis"],
    "Linux": ["Linux"],
    "REST APIs": ["REST API", "REST APIs", "RESTful API"],
    "GraphQL": ["GraphQL"],
    "FAISS": ["FAISS"],
    "Sentence Transformers": ["Sentence Transformers", "Sentence-Transformers"],
    "ONNX Runtime": ["ONNX Runtime", "ONNX"],
    "XGBoost": ["XGBoost", "XGBClassifier"],
    "LightGBM": ["LightGBM"],
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
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
    patterns = []
    for canonical_skill, phrases in SKILL_PATTERNS.items():
        for phrase in phrases:
            patterns.append(
                {
                    "label": TECH_SKILL_LABEL,
                    "pattern": phrase,
                    "id": canonical_skill,
                }
            )
    ruler.add_patterns(patterns)
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
    """Extract technical skills from free-form text with a spaCy NER-style pipeline."""
    text = "" if text is None else str(text)
    if not text.strip():
        return []

    nlp = _load_skill_nlp()
    doc = nlp(text)
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
