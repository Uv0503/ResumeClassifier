# AI Resume Screening and Job Fit Prediction System v2

This v2 project upgrades the resume screening workflow into a job-role prediction and resume-to-job fit system. The model predicts practical roles such as `Software Engineer`, `Full Stack Developer`, `Machine Learning Engineer`, `DevOps Engineer`, and related job titles, then compares the resume against a job description with semantic matching and skill overlap.

## Problem Statement

Older resume screening datasets often group resumes into coarse labels that are not specific enough for technical hiring. This version predicts a concrete job role from resume profile text, then evaluates fit against a job description using semantic similarity and matched/missing skills.

## Why Job-Role Prediction Is Better

Coarse resume labels answer: "What general area does this resume look like?"

Job-role prediction answers: "What job role is this candidate most aligned with?"

That makes the output more useful for screening because recruiters and interviewers usually compare candidates against concrete roles, not generic resume groups.

## Dataset

The training dataset is expected at:

```text
data/resume_data.csv
```

The target column is:

```text
job_position_name
```

The training input is a leakage-free `profile_text` field built from resume-side columns when present:

```text
skills
career_objective
responsibilities
positions
degree_names
major_field_of_studies
educational_institution_name
```

The pipeline cleans hidden BOM characters and known typo columns, including:

```text
educationaL_requirements -> educational_requirements
experiencere_requirement -> experience_requirement
responsibilities.1 -> job_responsibilities
```

The following job-side or target-related fields are not used as model input:

```text
job_position_name
matched_score
skills_required
educational_requirements
experience_requirement
job_responsibilities
```

## Features

- Predicts a specific job role from resume text.
- Uses the trained ML model directly for role prediction; no hardcoded role boosting is used.
- Shows prediction confidence when the classifier supports it.
- Shows the top 3 predicted job roles.
- Compares resume and JD with `sentence-transformers/all-MiniLM-L6-v2`.
- Falls back to TF-IDF cosine similarity if the semantic matcher fails.
- Extracts resume skills, JD skills, matched skills, and missing skills.
- Combines semantic similarity and skill overlap into a final JD score.
- Keeps the pipeline explainable and interview-friendly.

## Project Structure

```text
resume-screening-ml-v2/
|-- app/
|   |-- streamlit_app.py
|-- data/
|   |-- resume_data.csv
|-- legacy_v1/
|   |-- data/
|   |-- models/
|   |-- reports/
|-- models/
|   |-- job_role_classifier.pkl
|   |-- job_role_vectorizer.pkl
|   |-- job_role_label_encoder.pkl
|   |-- job_role_model_report.json
|-- notebooks/
|   |-- 01_job_role_prediction_training.ipynb
|-- reports/
|-- src/
|   |-- preprocessing.py
|   |-- train_model.py
|   |-- predict.py
|   |-- jd_matcher.py
|   |-- skill_extractor.py
|   |-- utils.py
|-- README.md
|-- requirements.txt
|-- .gitignore
```

## Model Training Process

1. Load `data/resume_data.csv`.
2. Clean column names and fix known typo columns.
3. Build `profile_text` from resume-side fields only.
4. Clean text while preserving technical tokens such as `C++`, `C#`, `Node.js`, `React.js`, `Python`, `Java`, `SQL`, `AWS`, `Docker`, `Kubernetes`, `Machine Learning`, `NLP`, `FastAPI`, `MongoDB`, and `PostgreSQL`.
5. Drop rows with empty profile text or empty target labels.
6. Remove classes with fewer than 2 samples so stratified splitting is valid.
7. Encode `job_position_name` labels.
8. Convert profile text to TF-IDF features.
9. Compare these classifiers:

```python
LogisticRegression(max_iter=2000, class_weight="balanced")
LinearSVC(class_weight="balanced")
RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
MultinomialNB()
```

10. Compare models using weighted F1-score.
11. Save Logistic Regression as the final model, along with the vectorizer, label encoder, and report.

## Evaluation Metrics

Each model is evaluated with:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Classification report

Models are compared with weighted F1-score because it is more reliable than accuracy when job-role classes are imbalanced. Logistic Regression is saved as the final classifier so the app can use probability-based confidence scores.

Note: The job-role model achieved very high validation performance on the structured `resume_data.csv` dataset. This may be because the dataset contains highly role-specific resume fields. Real-world resumes can be noisier, so the app also shows top predictions, confidence, JD match score, matched skills, and missing skills instead of relying only on one label.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python src/train_model.py
```

Expected output artifacts:

```text
models/job_role_classifier.pkl
models/job_role_vectorizer.pkl
models/job_role_label_encoder.pkl
models/job_role_model_report.json
```

## Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app shows:

- Predicted Job Role
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

Job-role prediction and JD matching are separate parts of the system.

The model predicts a job role from resume profile text using TF-IDF and ML classifiers. JD matching compares the resume against a specific job description using sentence-transformer embeddings and skill-overlap scoring.

The main JD score is calculated as:

```text
final_jd_score = 0.7 * semantic_similarity_score + 0.3 * skill_match_percentage
```

Where:

```text
skill_match_percentage = matched_required_skills / total_required_skills * 100
```

If the semantic matcher fails, the app falls back to TF-IDF similarity and shows a warning that the match score may be less accurate.

## Frontend Test Cases

### Test Case 1: SDE / Full Stack Resume

Resume:

```text
Software engineering student with experience in C++, Python, Go, JavaScript, React.js, Node.js, Express.js, MongoDB, PostgreSQL, FastAPI, Docker, Git, REST APIs, machine learning, FAISS, Sentence Transformers, and full-stack application development. Built trading systems, video platforms, RAG document Q&A assistants, and secure banking systems.
```

JD:

```text
We are hiring a Software Development Engineer Intern with strong data structures and algorithms, full-stack development, backend APIs, React.js, Node.js, FastAPI, Python, C++, MongoDB, PostgreSQL, Docker, Git, and machine learning knowledge.
```

Expected:

- Predicted Job Role: Full Stack Developer, Senior Software Engineer, AI Engineer, ML Engineer, or a similar technical role
- Top 3 predicted roles shown
- JD match score: medium/high
- Matched and missing skills displayed clearly

### Test Case 2: ML Engineer Resume

Resume:

```text
Machine learning engineer with experience in Python, Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, NLP, embeddings, model training, feature engineering, classification, regression, and model evaluation.
```

JD:

```text
Looking for a Machine Learning Engineer with Python, Scikit-learn, TensorFlow, PyTorch, NLP, embeddings, feature engineering, and model deployment experience.
```

Expected:

- Predicted Job Role: Machine Learning Engineer / AI Engineer / Data Science Engineer
- JD match score: high

### Test Case 3: DevOps Resume

Resume:

```text
DevOps engineer with experience in Docker, Kubernetes, AWS, CI/CD pipelines, Linux, monitoring, GitHub Actions, cloud deployments, infrastructure automation, and backend deployment workflows.
```

JD:

```text
Hiring DevOps Engineer with Docker, Kubernetes, AWS, CI/CD, Linux, cloud deployment, monitoring, and automation experience.
```

Expected:

- Predicted Job Role: DevOps Engineer or related infrastructure role
- JD match score: high

## Limitations

- The model quality depends on the coverage and consistency of `resume_data.csv`.
- Rare job roles with fewer than 2 samples are removed before stratified splitting.
- TF-IDF models are explainable but do not understand context as deeply as transformer classifiers.
- Skill extraction uses a curated skill dictionary, so unseen tools may need to be added.
- Sentence-transformer matching needs model download/cache access; TF-IDF is used if loading fails.

## Future Improvements

- Expand the skill dictionary from real job descriptions.
- Add role-family grouping for similar labels.
- Add confidence calibration for margin-based classifiers.
- Add more training examples for rare roles.
- Add model monitoring for prediction drift.

## Resume Bullets

- Upgraded a resume screening system to specific job-role prediction using TF-IDF and classical ML classifiers.
- Built leakage-free resume profile features from skills, objectives, responsibilities, positions, and education fields.
- Compared Logistic Regression, Linear SVM, Random Forest, and Naive Bayes using weighted F1-score.
- Integrated JD matching with sentence-transformer semantic similarity, TF-IDF fallback, and skill-overlap scoring.
- Developed a Streamlit interface showing predicted job role, top alternatives, JD match score, matched skills, missing skills, and screening recommendation.


