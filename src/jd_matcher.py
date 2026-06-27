from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_resume_text
from skill_extractor import find_missing_skills


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _embedding_similarity(resume_text: str, job_description: str) -> float:
    model = _load_sentence_transformer()
    embeddings = model.encode([resume_text, job_description], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def _tfidf_similarity(resume_text: str, job_description: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    matrix = vectorizer.fit_transform([resume_text, job_description])
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def calculate_jd_match_score(resume_text: object, job_description: object) -> dict[str, object]:
    """Return semantic, skill-overlap, and final JD fit scores."""
    cleaned_resume = clean_resume_text(resume_text)
    cleaned_jd = clean_resume_text(job_description)

    if not cleaned_resume:
        raise ValueError("Resume text is empty. Please enter or upload a resume.")
    if not cleaned_jd:
        raise ValueError("Job description is empty. Please enter a job description.")

    method = f"sentence-transformers ({EMBEDDING_MODEL_NAME})"
    fallback_reason = None
    try:
        similarity = _embedding_similarity(cleaned_resume, cleaned_jd)
    except Exception as exc:
        similarity = _tfidf_similarity(cleaned_resume, cleaned_jd)
        method = "tf-idf fallback"
        fallback_reason = str(exc)

    semantic_similarity_score = round(max(0.0, min(similarity, 1.0)) * 100, 2)
    skills = find_missing_skills(resume_text, job_description)

    total_required_skills = len(skills["jd_skills"])
    matched_required_skills = len(skills["matched_skills"])
    if total_required_skills:
        skill_match_percentage = round((matched_required_skills / total_required_skills) * 100, 2)
    else:
        skill_match_percentage = 0.0

    final_jd_score = round((0.7 * semantic_similarity_score) + (0.3 * skill_match_percentage), 2)

    return {
        "match_score": final_jd_score,
        "final_jd_score": final_jd_score,
        "semantic_similarity_score": semantic_similarity_score,
        "skill_match_percentage": skill_match_percentage,
        "method": method,
        "fallback_reason": fallback_reason,
        "matched_keywords": skills["matched_skills"],
        "missing_keywords": skills["missing_skills"],
        "resume_skills": skills["resume_skills"],
        "jd_skills": skills["jd_skills"],
    }

