from __future__ import annotations

from pathlib import Path

import numpy as np

from preprocessing import clean_resume_text
from utils import MODELS_DIR, load_pickle


MODEL_PATH = MODELS_DIR / "job_role_classifier.pkl"
VECTORIZER_PATH = MODELS_DIR / "job_role_vectorizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "job_role_label_encoder.pkl"


def load_artifacts(
    model_path: str | Path = MODEL_PATH,
    vectorizer_path: str | Path = VECTORIZER_PATH,
    label_encoder_path: str | Path = LABEL_ENCODER_PATH,
):
    """Load the saved job-role classifier, TF-IDF vectorizer, and label encoder."""
    model = load_pickle(model_path)
    vectorizer = load_pickle(vectorizer_path)
    label_encoder = load_pickle(label_encoder_path)
    return model, vectorizer, label_encoder


def get_confidence_level(confidence: float | None) -> dict[str, str]:
    """Explain how strongly the model supports the predicted job role."""
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


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    total = exp_scores.sum()
    if total == 0:
        return np.zeros_like(scores, dtype=float)
    return exp_scores / total


def _top_predictions_from_values(
    values: np.ndarray,
    label_encoder,
    top_n: int = 3,
) -> list[dict[str, object]]:
    top_indices = np.argsort(values)[::-1][:top_n]
    return [
        {
            "job_role": str(label_encoder.inverse_transform([index])[0]),
            "confidence": round(float(values[index]) * 100, 2),
        }
        for index in top_indices
    ]


def _prediction_details(model, label_encoder, vectorized_resume) -> tuple[str, float | None, list[dict[str, object]]]:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(vectorized_resume)[0], dtype=float)
        top_predictions = _top_predictions_from_values(probabilities, label_encoder)
        predicted_role = top_predictions[0]["job_role"]
        confidence = top_predictions[0]["confidence"]
        return predicted_role, confidence, top_predictions

    if hasattr(model, "decision_function"):
        decision_scores = np.asarray(model.decision_function(vectorized_resume))
        if decision_scores.ndim == 1 and len(label_encoder.classes_) == 2:
            positive_score = float(decision_scores.ravel()[0])
            decision_scores = np.array([[-positive_score, positive_score]])

        # These softmax values are display scores for SVM margins, not calibrated probabilities.
        display_scores = _softmax(decision_scores[0])
        top_predictions = _top_predictions_from_values(display_scores, label_encoder)
        predicted_role = str(label_encoder.inverse_transform([model.predict(vectorized_resume)[0]])[0])
        confidence = next(
            (item["confidence"] for item in top_predictions if item["job_role"] == predicted_role),
            top_predictions[0]["confidence"] if top_predictions else None,
        )
        return predicted_role, confidence, top_predictions

    prediction = model.predict(vectorized_resume)[0]
    predicted_role = str(label_encoder.inverse_transform([prediction])[0])
    return predicted_role, None, [{"job_role": predicted_role, "confidence": None}]


def predict_job_role(resume_text: object) -> dict[str, object]:
    """Predict the most likely job role using the trained model only."""
    cleaned_text = clean_resume_text(resume_text)
    if not cleaned_text:
        raise ValueError("Resume text is empty. Please enter or upload a resume.")

    model, vectorizer, label_encoder = load_artifacts()
    vectorized_resume = vectorizer.transform([cleaned_text])
    predicted_role, confidence, top_predictions = _prediction_details(
        model,
        label_encoder,
        vectorized_resume,
    )

    return {
        "predicted_job_role": predicted_role,
        "confidence": confidence,
        "top_predictions": top_predictions,
    }
