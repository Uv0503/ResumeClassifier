# AI Resume Screening and Job Fit Prediction System v2

This project predicts the best-fit job role from resume text and compares a resume with a job description. The ML pipeline now uses semantic embeddings, XGBoost, spaCy skill extraction, and FAISS similarity search instead of TF-IDF and basic keyword matching.

## What Changed

- Resume text is converted into dense vectors with `sentence-transformers/all-MiniLM-L6-v2`.
- Job-role classification is trained with `XGBClassifier` on those embedding vectors.
- Rare job roles are handled with `compute_sample_weight(class_weight="balanced")`, so underrepresented roles matter during training.
- Skill extraction uses a spaCy `EntityRuler` NER-style pipeline with technical skill patterns and sentence context filters.
- Resume-to-JD semantic matching uses FAISS `IndexFlatIP` over normalized embeddings for fast cosine-style similarity.
- Optional ONNX Runtime support can speed up local CPU embedding inference.

## Dataset

The training dataset is expected at:

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

Job-side or target-related fields are not used as model input.

## Project Structure

```text
resume-screening-ml-v2/
|-- app/
|   |-- streamlit_app.py
|-- data/
|   |-- resume_data.csv
|-- models/
|   |-- job_role_classifier.pkl
|   |-- job_role_label_encoder.pkl
|   |-- job_role_embedding_config.json
|   |-- job_role_model_report.json
|   |-- sentence_transformer_onnx/        # optional after export
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
|-- requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

If spaCy was already installed and fails with a NumPy binary compatibility error, reinstall from `requirements.txt` so `spacy>=3.8.2` and the installed NumPy version agree.

## Optional ONNX Export

Run this once after installing dependencies if you want faster CPU embedding inference:

```bash
python src/export_onnx_model.py
```

This writes the optimized Sentence Transformer files to:

```text
models/sentence_transformer_onnx/
```

The shared embedding helper automatically prefers that ONNX model when it exists. If it is missing, the project falls back to the normal Torch Sentence Transformer backend.

## Training

```bash
python src/train_model.py
```

Training now does this:

1. Loads and cleans `data/resume_data.csv`.
2. Builds leakage-free resume `profile_text`.
3. Removes classes with fewer than 2 samples so stratified splitting is valid.
4. Encodes resume text with `all-MiniLM-L6-v2` embeddings.
5. Computes balanced sample weights for imbalanced job-role classes.
6. Trains XGBoost with `multi:softprob` for top-role probabilities.
7. Saves the classifier, label encoder, embedding config, and evaluation reports.

Expected output artifacts:

```text
models/job_role_classifier.pkl
models/job_role_label_encoder.pkl
models/job_role_embedding_config.json
models/job_role_model_report.json
reports/job_role_classification_report.json
reports/job_role_classification_report.txt
reports/job_role_model_comparison.csv
```

`models/job_role_vectorizer.pkl` is obsolete and is no longer used.

## Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app shows:

- Predicted job role
- Prediction confidence
- Top 3 predicted job roles
- Semantic similarity score
- Skill match percentage
- Final JD match score
- Matching method
- Skills found in resume
- Skills required by JD
- Matched skills
- Missing skills
- Final screening recommendation

## JD Matching

Job-role prediction and JD matching are separate steps.

The model predicts a role from resume embeddings. JD matching compares the resume and job description by embedding both texts, normalizing the vectors, and searching with FAISS `IndexFlatIP`.

The final score is still:

```text
final_jd_score = 0.7 * semantic_similarity_score + 0.3 * skill_match_percentage
```

Skill overlap is based on spaCy EntityRuler entities labeled as technical skills.

## Smoke Tests

After installing dependencies and training the model:

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

- The first run may download the Sentence Transformer model if it is not already cached.
- FAISS and XGBoost are required for the upgraded runtime and training pipeline.
- ONNX is optional but recommended for CPU speed.
- Model quality still depends on the coverage and cleanliness of `resume_data.csv`.
