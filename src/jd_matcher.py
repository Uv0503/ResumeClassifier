from __future__ import annotations

import numpy as np

try:
    from .embedding_model import EMBEDDING_MODEL_NAME, encode_texts, get_embedding_backend
    from .preprocessing import clean_resume_text
    from .skill_extractor import find_missing_skills
except ImportError:  # pragma: no cover - supports Streamlit/script path imports
    from embedding_model import EMBEDDING_MODEL_NAME, encode_texts, get_embedding_backend
    from preprocessing import clean_resume_text
    from skill_extractor import find_missing_skills


def _load_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS is required for JD matching. Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return faiss


def build_jd_faiss_index(job_descriptions: list[str]):
    """Encode job descriptions with local ONNX Sentence Transformers and add them to FAISS."""
    faiss = _load_faiss()
    jd_embeddings = encode_texts(job_descriptions, normalize=True, prefer_onnx=True)
    jd_embeddings = np.ascontiguousarray(jd_embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(jd_embeddings.shape[1])
    index.add(jd_embeddings)
    return index, jd_embeddings


def search_job_descriptions(resume_text: str, job_descriptions: list[str], top_k: int = 1):
    """Return FAISS semantic similarity scores for the best matching job descriptions."""
    index, _ = build_jd_faiss_index(job_descriptions)
    resume_embedding = encode_texts([resume_text], normalize=True, prefer_onnx=True)
    resume_embedding = np.ascontiguousarray(resume_embedding, dtype=np.float32)
    scores, indices = index.search(resume_embedding, min(top_k, len(job_descriptions)))
    return scores[0], indices[0]


def _faiss_similarity(resume_text: str, job_description: str) -> float:
    scores, _ = search_job_descriptions(resume_text, [job_description], top_k=1)
    return float(scores[0])


def calculate_jd_match_score(resume_text: object, job_description: object) -> dict[str, object]:
    """Return semantic, skill-overlap, and final JD fit scores."""
    cleaned_resume = clean_resume_text(resume_text)
    cleaned_jd = clean_resume_text(job_description)

    if not cleaned_resume:
        raise ValueError("Resume text is empty. Please enter or upload a resume.")
    if not cleaned_jd:
        raise ValueError("Job description is empty. Please enter a job description.")

    similarity = _faiss_similarity(cleaned_resume, cleaned_jd)
    backend = get_embedding_backend(prefer_onnx=True)
    method = f"faiss + {'onnx ' if backend == 'onnx' else 'local '}sentence-transformers ({EMBEDDING_MODEL_NAME})"

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
        "fallback_reason": None,
        "matched_keywords": skills["matched_skills"],
        "missing_keywords": skills["missing_skills"],
        "resume_skills": skills["resume_skills"],
        "jd_skills": skills["jd_skills"],
    }
