from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from preprocessing import load_job_role_data
from utils import DATA_DIR, MODELS_DIR, REPORTS_DIR, ensure_directories, save_pickle


RANDOM_STATE = 42
TEST_SIZE = 0.2
DATASET_FILE = DATA_DIR / "resume_data.csv"
FINAL_MODEL_NAME = "Logistic Regression"


def _build_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=1,
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
    }


def _build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )


def _prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    class_counts = df["job_position_name"].value_counts()
    removed_roles = class_counts[class_counts < 2]
    notes: list[str] = []

    if not removed_roles.empty:
        df = df[df["job_position_name"].isin(class_counts[class_counts >= 2].index)].copy()
        notes.append(
            "Removed job roles with fewer than 2 samples so stratified train/test split is valid."
        )

    if df["job_position_name"].nunique() < 2:
        raise ValueError("Need at least two job roles with enough samples to train a classifier.")

    return df.reset_index(drop=True), {
        "removed_low_sample_roles": removed_roles.to_dict(),
        "notes": notes,
    }


def _evaluate_model(model_name: str, model, X_test, y_test, labels: list[str]) -> dict[str, object]:
    predictions = model.predict(X_test)
    precision, recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(y_test, predictions)
    label_ids = list(range(len(labels)))
    report_text = classification_report(
        y_test,
        predictions,
        labels=label_ids,
        target_names=labels,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        predictions,
        labels=label_ids,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n{model_name} results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall: {recall:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print("Classification report:")
    print(report_text)

    return {
        "model": model_name,
        "accuracy": round(float(accuracy), 4),
        "weighted_precision": round(float(precision), 4),
        "weighted_recall": round(float(recall), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "predictions": predictions,
    }


def _public_result(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"predictions", "classification_report_text"}
    }


def train_and_evaluate() -> dict[str, object]:
    ensure_directories()

    df = load_job_role_data(DATASET_FILE)
    print(f"Loaded {len(df)} clean resume records from {DATASET_FILE}")
    print(f"Job roles before low-sample filtering: {df['job_position_name'].nunique()}")

    original_counts = df["job_position_name"].value_counts()
    df, split_notes = _prepare_training_frame(df)
    role_counts = df["job_position_name"].value_counts()

    print(f"Records used for training/evaluation: {len(df)}")
    print(f"Job roles used: {df['job_position_name'].nunique()}")
    print("\nTop 10 job roles by count:")
    print(role_counts.head(10).to_string())
    if split_notes["removed_low_sample_roles"]:
        print("\nRemoved low-sample job roles:")
        print(pd.Series(split_notes["removed_low_sample_roles"]).to_string())

    (REPORTS_DIR / "job_role_distribution.csv").write_text(
        role_counts.to_csv(header=["count"]),
        encoding="utf-8",
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["job_position_name"])
    labels = label_encoder.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        df["profile_text"],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    vectorizer = _build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF feature matrix: {X_train_tfidf.shape[1]} features")

    results: list[dict[str, object]] = []
    trained_models: dict[str, object] = {}

    for model_name, model in _build_models().items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train_tfidf, y_train)
        trained_models[model_name] = model
        results.append(_evaluate_model(model_name, model, X_test_tfidf, y_test, labels))

    results_df = pd.DataFrame(
        [{k: v for k, v in _public_result(result).items() if k != "classification_report"} for result in results]
    )
    tie_break_priority = {
        "Logistic Regression": 0,
        "Linear SVM": 1,
        "Multinomial Naive Bayes": 2,
        "Random Forest": 3,
    }
    results_df["tie_break_priority"] = results_df["model"].map(tie_break_priority).fillna(99)
    results_df = results_df.sort_values(
        ["weighted_f1", "accuracy", "tie_break_priority"],
        ascending=[False, False, True],
    )
    results_df.to_csv(REPORTS_DIR / "job_role_model_comparison.csv", index=False)

    best_model_name = FINAL_MODEL_NAME
    best_model = trained_models[best_model_name]
    best_result = next(result for result in results if result["model"] == best_model_name)

    save_pickle(best_model, MODELS_DIR / "job_role_classifier.pkl")
    save_pickle(vectorizer, MODELS_DIR / "job_role_vectorizer.pkl")
    save_pickle(label_encoder, MODELS_DIR / "job_role_label_encoder.pkl")

    (REPORTS_DIR / "job_role_classification_report.txt").write_text(
        str(best_result["classification_report_text"]),
        encoding="utf-8",
    )
    (REPORTS_DIR / "job_role_classification_report.json").write_text(
        json.dumps(best_result["classification_report"], indent=2),
        encoding="utf-8",
    )

    summary = {
        "best_model": best_model_name,
        "best_accuracy": float(best_result["accuracy"]),
        "best_weighted_f1": float(best_result["weighted_f1"]),
        "records_used": int(len(df)),
        "total_job_roles": int(df["job_position_name"].nunique()),
        "model_results": [_public_result(result) for result in results],
        "top_10_job_roles_by_count": role_counts.head(10).astype(int).to_dict(),
        "dataset_file": str(DATASET_FILE),
        "split_notes": split_notes,
        "original_total_job_roles": int(original_counts.shape[0]),
    }
    (MODELS_DIR / "job_role_model_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (REPORTS_DIR / "job_role_training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nModel comparison sorted by weighted F1-score:")
    print(results_df.to_string(index=False))
    print(f"\nFinal saved model: {best_model_name}")
    print(f"Final model weighted F1-score: {best_result['weighted_f1']:.4f}")
    print(f"Final model accuracy: {best_result['accuracy']:.4f}")
    print(f"Saved job-role model artifacts and report to {MODELS_DIR}")
    return summary


if __name__ == "__main__":
    train_and_evaluate()




