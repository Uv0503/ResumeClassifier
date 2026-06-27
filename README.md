# AI Resume Screening and Job Fit Prediction System v2

This project predicts the best-fit job role from resume text and compares a resume with a job description. The upgraded pipeline uses a repository-local Sentence Transformer, XGBoost, spaCy EntityRuler skill extraction, ONNX Runtime, and FAISS instead of TF-IDF and hardcoded keyword matching.

## What Changed

- `download_model.py` saves `all-MiniLM-L6-v2` locally to `models/all-MiniLM-L6-v2`.
- Resume text is converted into dense vectors with the local Sentence Transformer model.
- Job-role classification is trained with `XGBClassifier` on those dense embeddings.
- Rare job roles are handled with `compute_sample_weight(class_weight="balanced")` during training.
- Skill extraction uses spaCy `EntityRuler` patterns loaded from `data/skills_patterns.json`.
- JD semantic matching embeds text with the ONNX-preferred local model and searches job-description vectors with FAISS `IndexFlatIP`.

## Dataset

Training data is expected at:

```text
data/resume_data.csv
```

The target column is:

```text
job_position_name
```

The model input is a leakage-free `profile_text` field built from resume-side columns when present:

```text
skills
career_objective
responsibilities
positions
degree_names
major_field_of_studies
educational_institution_name
```

## Project Structure

```text
resume-screening-ml-v2/
|-- app/
|   |-- streamlit_app.py
|-- data/
|   |-- resume_data.csv
|   |-- skills_patterns.json
|-- models/
|   |-- all-MiniLM-L6-v2/
|   |-- sentence_transformer_onnx/
|   |-- job_role_classifier.pkl
|   |-- job_role_label_encoder.pkl
|   |-- job_role_embedding_config.json
|   |-- job_role_model_report.json
|-- reports/
|-- src/
|   |-- embedding_model.py
|   |-- export_onnx_model.py
|   |-- jd_matcher.py
|   |-- predict.py
|   |-- preprocessing.py
|   |-- skill_extractor.py
|   |-- train_model.py
|   |-- utils.py
|-- download_model.py
|-- requirements.txt
```

## Install And Prepare Models

```bash
pip install -r requirements.txt
python download_model.py
python src/export_onnx_model.py
```

`download_model.py` skips work if `models/all-MiniLM-L6-v2` already exists. The ONNX export is used by JD matching for faster CPU inference.

## Training

```bash
python src/train_model.py
```

Training now does this:

1. Loads and cleans `data/resume_data.csv`.
2. Builds leakage-free resume `profile_text`.
3. Removes classes with fewer than 2 samples so stratified splitting is valid.
4. Encodes resume text with `models/all-MiniLM-L6-v2`.
5. Computes balanced sample weights for imbalanced job-role classes.
6. Trains XGBoost with `multi:softprob`.
7. Saves the classifier, label encoder, embedding config, and evaluation reports.

`models/job_role_vectorizer.pkl` has been removed and is no longer saved or loaded.

## Running The App

```bash
streamlit run app/streamlit_app.py
```

The app shows the predicted job role, confidence, top 3 roles, semantic JD similarity, skill match percentage, matched skills, missing skills, and final recommendation.

## JD Matching

JD matching embeds the resume and job description with the local ONNX Sentence Transformer, normalizes the vectors, creates a FAISS `IndexFlatIP` index for job descriptions, and searches that index for the highest semantic match.

The final score is:

```text
final_jd_score = 0.7 * semantic_similarity_score + 0.3 * skill_match_percentage
```

Skill overlap is based on spaCy EntityRuler entities loaded from `data/skills_patterns.json`.

## Smoke Tests

```bash
python -c "from src.predict import predict_job_role; print(predict_job_role('Python React Docker AWS machine learning engineer'))"
```

```bash
python -c "from src.jd_matcher import calculate_jd_match_score; print(calculate_jd_match_score('Python Docker AWS ML', 'Need Python, AWS, Docker'))"
```

```bash
python -c "from src.skill_extractor import extract_skills; print(extract_skills('Built APIs with Python, FastAPI, Docker, Kubernetes and AWS'))"
```

## Notes

- The trained local XGBoost model currently reports accuracy `0.8591` and weighted F1 `0.8627` on the validation split.
- Model quality depends on the coverage and cleanliness of `data/resume_data.csv`.
- `xgboost`, `spacy`, `faiss-cpu`, and `onnxruntime` are required runtime dependencies.
