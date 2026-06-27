from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

try:
    from .embedding_model import EMBEDDING_MODEL_NAME, encode_texts, get_embedding_backend
    from .preprocessing import load_job_role_data
    from .utils import DATA_DIR, MODELS_DIR, REPORTS_DIR, ensure_directories, save_pickle
except ImportError:  # pragma: no cover - supports `python src/train_model.py`
    from embedding_model import EMBEDDING_MODEL_NAME, encode_texts, get_embedding_backend
    from preprocessing import load_job_role_data
    from utils import DATA_DIR, MODELS_DIR, REPORTS_DIR, ensure_directories, save_pickle


RANDOM_STATE = 42
TEST_SIZE = 0.2
DATASET_FILE = DATA_DIR / "resume_data.csv"
FINAL_MODEL_NAME = "XGBoost"


def _build_model(num_classes: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
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

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["profile_text"],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    embedding_backend = get_embedding_backend()
    print(f"\nEmbedding resumes with {EMBEDDING_MODEL_NAME} ({embedding_backend} backend)...")
    X_train = encode_texts(X_train_text.tolist(), normalize=True)
    X_test = encode_texts(X_test_text.tolist(), normalize=True)
    print(f"Embedding matrix: {X_train.shape[0]} rows x {X_train.shape[1]} dimensions")

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = _build_model(num_classes=len(labels))

    print(f"\nTraining {FINAL_MODEL_NAME} with balanced sample weights...")
    model.fit(X_train, y_train, sample_weight=sample_weights)
    best_result = _evaluate_model(FINAL_MODEL_NAME, model, X_test, y_test, labels)

    results_df = pd.DataFrame(
        [{k: v for k, v in _public_result(best_result).items() if k != "classification_report"}]
    )
    results_df.to_csv(REPORTS_DIR / "job_role_model_comparison.csv", index=False)

    save_pickle(model, MODELS_DIR / "job_role_classifier.pkl")
    save_pickle(label_encoder, MODELS_DIR / "job_role_label_encoder.pkl")

    embedding_config = {
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "embedding_backend": embedding_backend,
        "normalize_embeddings": True,
        "embedding_dimensions": int(X_train.shape[1]),
        "classifier": FINAL_MODEL_NAME,
        "class_imbalance_strategy": "compute_sample_weight(class_weight='balanced')",
    }
    (MODELS_DIR / "job_role_embedding_config.json").write_text(
        json.dumps(embedding_config, indent=2),
        encoding="utf-8",
    )

    (REPORTS_DIR / "job_role_classification_report.txt").write_text(
        str(best_result["classification_report_text"]),
        encoding="utf-8",
    )
    (REPORTS_DIR / "job_role_classification_report.json").write_text(
        json.dumps(best_result["classification_report"], indent=2),
        encoding="utf-8",
    )

    summary = {
        "best_model": FINAL_MODEL_NAME,
        "best_accuracy": float(best_result["accuracy"]),
        "best_weighted_f1": float(best_result["weighted_f1"]),
        "records_used": int(len(df)),
        "total_job_roles": int(df["job_position_name"].nunique()),
        "model_results": [_public_result(best_result)],
        "top_10_job_roles_by_count": role_counts.head(10).astype(int).to_dict(),
        "dataset_file": str(DATASET_FILE),
        "split_notes": split_notes,
        "original_total_job_roles": int(original_counts.shape[0]),
        "embedding_config": embedding_config,
    }
    (MODELS_DIR / "job_role_model_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (REPORTS_DIR / "job_role_training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nFinal saved model: {FINAL_MODEL_NAME}")
    print(f"Final model weighted F1-score: {best_result['weighted_f1']:.4f}")
    print(f"Final model accuracy: {best_result['accuracy']:.4f}")
    print(f"Saved job-role model artifacts and report to {MODELS_DIR}")
    return summary


if __name__ == "__main__":
    train_and_evaluate()


