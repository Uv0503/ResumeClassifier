from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from jd_matcher import calculate_jd_match_score
from predict import predict_job_role
from skill_extractor import extract_skills
from utils import format_list


def _extract_pdf_text(uploaded_file) -> str:
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as exc:
        raise ValueError(f"Could not read PDF file: {exc}") from exc


def _read_uploaded_resume(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()
    if file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if file_name.endswith(".pdf"):
        return _extract_pdf_text(uploaded_file)
    raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")


def _recommendation(score: float, missing_skills: list[str]) -> str:
    if score >= 75 and not missing_skills:
        return "Strong match. This resume aligns well with the job description."
    if score >= 60:
        return "Good match. The candidate is close, but a few skills could be strengthened."
    if score >= 40:
        return "Moderate match. Review experience carefully and address the missing skills."
    return "Low match. The resume needs stronger alignment with this job description."


def _confidence_message(confidence: float | None) -> dict[str, str]:
    if confidence is None:
        return {
            "level": "Confidence not available",
            "explanation": "This model does not provide probability-like confidence scores.",
        }
    if confidence >= 70:
        return {
            "level": "High confidence prediction",
            "explanation": "The model strongly supports this job role.",
        }
    if confidence >= 40:
        return {
            "level": "Medium confidence prediction",
            "explanation": "The model moderately supports this role. Review top alternatives too.",
        }
    return {
        "level": "Low confidence prediction",
        "explanation": "The resume may overlap multiple job roles. Review the top predictions carefully.",
    }


st.set_page_config(page_title="AI Resume Screening", layout="wide")
st.title("AI Resume Screening and Job Fit Prediction System")

uploaded_resume = st.file_uploader("Upload resume (.txt or .pdf)", type=["txt", "pdf"])
uploaded_text = ""
if uploaded_resume is not None:
    try:
        uploaded_text = _read_uploaded_resume(uploaded_resume)
    except ValueError as exc:
        st.error(str(exc))

resume_text = st.text_area("Resume text", value=uploaded_text, height=280)
job_description = st.text_area("Job description", height=220)

if st.button("Analyze Resume", type="primary"):
    if not resume_text.strip():
        st.error("Please enter or upload resume text.")
    elif not job_description.strip():
        st.error("Please enter a job description.")
    else:
        try:
            prediction = predict_job_role(resume_text)
            match_result = calculate_jd_match_score(resume_text, job_description)
            resume_skills = extract_skills(resume_text)

            st.subheader("Predicted Job Role")
            st.success(prediction["predicted_job_role"])

            col1, col2, col3 = st.columns(3)
            confidence = prediction.get("confidence")
            col1.metric("Prediction Confidence", f"{confidence:.2f}%" if confidence is not None else "N/A")
            col2.metric("Semantic Similarity", f"{match_result['semantic_similarity_score']:.2f}%")
            col3.metric("Skill Match", f"{match_result['skill_match_percentage']:.2f}%")
            st.metric("Final JD Match Score", f"{match_result['final_jd_score']:.2f}%")

            confidence_message = _confidence_message(confidence)
            st.info(f"{confidence_message['level']}: {confidence_message['explanation']}")

            top_predictions = prediction.get("top_predictions", [])
            if top_predictions:
                st.subheader("Top 3 Predicted Job Roles")
                st.table(
                    [
                        {
                            "Job Role": item["job_role"],
                            "Confidence": f"{item['confidence']:.2f}%" if item.get("confidence") is not None else "N/A",
                        }
                        for item in top_predictions
                    ]
                )

            st.subheader("Screening Insights")
            st.write(f"Matching method: {match_result['method']}")
            if match_result.get("fallback_reason"):
                st.warning(f"Semantic matcher failed, using TF-IDF fallback. Match score may be less accurate. Details: {match_result['fallback_reason']}")
            st.write(f"Skills found in resume: {format_list(resume_skills)}")
            st.write(f"Skills required by JD: {format_list(match_result['jd_skills'])}")
            st.write(f"Matched skills: {format_list(match_result['matched_keywords'])}")
            st.write(f"Missing skills: {format_list(match_result['missing_keywords'])}")
            st.info(_recommendation(match_result["final_jd_score"], match_result["missing_keywords"]))
        except FileNotFoundError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

