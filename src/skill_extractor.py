from __future__ import annotations

import re

from scipy.sparse import csr_matrix


SKILL_ALIASES: dict[str, list[str]] = {
    "Python": ["python"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "c sharp", "csharp"],
    "SQL": ["sql"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb", "mongo db"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "dl"],
    "NLP": ["nlp", "natural language processing"],
    "TensorFlow": ["tensorflow", "tensor flow"],
    "PyTorch": ["pytorch", "torch"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy", "num py"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Git": ["git", "github"],
    "GitHub Actions": ["github actions"],
    "CI/CD": ["ci/cd", "ci cd", "cicd"],
    "FastAPI": ["fastapi", "fast api"],
    "Flask": ["flask"],
    "Streamlit": ["streamlit"],
    "Spring Boot": ["spring boot"],
    "React.js": ["react.js", "reactjs", "react js", "react"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Express.js": ["express.js", "expressjs", "express js"],
    "Kafka": ["kafka"],
    "Redis": ["redis"],
    "Linux": ["linux"],
    "REST APIs": ["rest api", "rest apis", "restful api", "rest"],
    "FAISS": ["faiss"],
    "Sentence Transformers": ["sentence transformers", "sentence-transformers"],
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


def _normalize(text: object) -> str:
    text = "" if text is None else str(text).lower()
    replacements = {
        "node.js": "nodejs",
        "react.js": "reactjs",
        "express.js": "expressjs",
        "c++": " cpp ",
        "c#": " csharp ",
        "ci/cd": " cicd ",
        "scikit-learn": "scikit learn",
        "sentence-transformers": "sentence transformers",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    text = re.sub(r"[.;,:)\]}]", " ", text)
    text = re.sub(r"[(\[{]", " ", text)
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} "


def extract_skills(text: object) -> list[str]:
    """Extract known technical skills from free-form text."""
    normalized_text = _normalize(text)
    found_skills: list[str] = []

    for skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize(alias).strip()
            pattern = rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])"
            if re.search(pattern, normalized_text):
                found_skills.append(skill)
                break

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
