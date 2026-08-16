# 🫀 Heart Disease Risk Prediction Using Data Mining
> **Demonstration & Educational Reference Mini-Project**

---

> ⚠️ **DEMONSTRATION / SAMPLE PROJECT WARNING**
>
> This repository is an educational reference project created to demonstrate the expected structure, methodology, documentation, Git workflow, and final presentation of a Data Mining mini-project.
>
> For this demonstration, a **SINGLE GitHub account** is used to create and publish the repository.
>
> The student codes **AISD05**, **SIAD19**, and **RSD20** represent **SIMULATED CONTRIBUTORS** used to demonstrate how a three-student project can be organized through commit labels, branches, Pull Requests, and merges.
>
> This does **NOT** represent the GitHub contribution requirement for an actual student project.
>
> In a real student project, every student must use their own GitHub account and make genuine individual contributions.
>
> Therefore, the Git history in this repository demonstrates the intended workflow but must **NOT** be interpreted as evidence that three separate GitHub accounts contributed to this repository.

---

> ⚠️ **MEDICAL DISCLAIMER**
>
> This application and associated code are developed exclusively for academic and educational Data Mining purposes. Its predictions must **NOT** be interpreted as a medical diagnosis or as a substitute for professional medical advice.

---

## 📌 Project Overview & Problem Statement
Cardiovascular diseases (CVDs) remain the leading cause of mortality worldwide. Early detection of individuals at high risk allows targeted preventative interventions and lifestyle modifications.

**Core Research Question:**
*"Can routine patient demographics, physiological markers, blood chemistry parameters, and lifestyle factors accurately predict whether an individual is at high risk of heart disease?"*

This repository demonstrates the end-to-end Data Mining pipeline:
1. Automated data fetch via `kagglehub`.
2. Data quality audit ($N = 9,000$ patient records, 25 clinical features).
3. Leakage-free preprocessing via scikit-learn `ColumnTransformer`.
4. Exploratory Data Analysis & Statistical Hypothesis Testing ($\alpha = 0.05$).
5. Supervised Classification benchmark across **6 algorithms**: Zero-R Baseline, KNN, Decision Tree, Random Forest, Naive Bayes, and Support Vector Machine (SVM).
6. Multi-page interactive Streamlit web application.
7. Publication-ready scientific report in LaTeX compiled to PDF.

---

## 📊 Dataset & Kaggle Source Documentation
- **Kaggle Dataset Identifier:** `uditjain13/heart-disease-risk-2026`
- **Acquisition Method:** Downloaded dynamically using `kagglehub`:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("uditjain13/heart-disease-risk-2026")
print("Path to dataset files:", path)
```

### Dataset Audit Summary
- **Total Records:** 9,000 patients
- **Total Attributes:** 27 columns (1 identifier `patient_id`, 25 predictor attributes, 1 target `has_heart_disease`)
- **Missing Values:** 0
- **Duplicates:** 0
- **Target Distribution:** 6,273 Negative (`0`, 69.7%) vs 2,727 Positive (`1`, 30.3%)

> **Dataset Size Note (Section 5 Requirement):**
> The Kaggle dataset contains 9,000 observations. No synthetic or duplicate rows were artificially created to reach 10,000 records. The actual sample size is documented across all analysis files and report artifacts.

---

## 🏆 Model Performance Benchmark

All 6 algorithms were trained on 7,200 training records using **5-Fold Stratified Cross-Validation** with `GridSearchCV` hyperparameter tuning and evaluated on a hold-out test set ($N_{test} = 1,800$):

| Model | Accuracy | Precision | Recall | F1 (Weighted) | ROC-AUC | 5-Fold CV Score (F1) |
|---|---:|---:|---:|---:|---:|---:|
| Zero-R Baseline | 0.6972 | 0.0000 | 0.0000 | 0.5728 | 0.5000 | 0.5725 ± 0.0004 |
| K-Nearest Neighbors | 0.8317 | 0.8201 | 0.5688 | 0.8217 | 0.8678 | 0.8069 ± 0.0037 |
| Decision Tree | 0.8644 | 0.8265 | 0.6991 | 0.8610 | 0.9097 | 0.8561 ± 0.0085 |
| Naive Bayes | 0.8611 | 0.7496 | 0.8128 | 0.8626 | 0.9271 | 0.8513 ± 0.0120 |
| Random Forest | 0.8822 | 0.8612 | 0.7284 | 0.8792 | 0.9361 | 0.8757 ± 0.0051 |
| **Support Vector Machine (Best)** | **0.8917** | **0.8601** | **0.7670** | **0.8898** | **0.9422** | **0.8954 ± 0.0047** |

---

## 🌐 Interactive Streamlit Application

The interactive web application contains 7 dedicated pages:
1. **Overview:** Problem statement, objectives, dataset metadata.
2. **Dataset Explorer:** Raw data table preview, missing value audit, column types.
3. **EDA Dashboard:** Distributions, correlation heatmap, target class balance.
4. **Statistical Analysis:** Mann-Whitney U & Chi-Square test statistics and p-values.
5. **Model Comparison:** Performance benchmark table, ROC curves, confusion matrices, Random Forest feature importances.
6. **Heart Disease Risk Prediction:** Dynamic form for inputting patient biomarkers and inferring heart disease risk score with probability badge.
7. **About & Contributions:** Team responsibilities, software stack, demonstration warnings.

### How to Run Streamlit Application
```bash
streamlit run src/app.py
```

---

## 👥 Demonstration Git Workflow vs Real Student Requirement

| Aspect | This Demonstration Repository | Real Student Project Requirement |
|---|---|---|
| GitHub accounts | 1 account | 1 account per student |
| Student identities | Simulated (`AISD05`, `SIAD19`, `RSD20`) | Authentic individual student identities |
| Commit prefixes | `feat(AISD05): ...`, `feat(SIAD19): ...` | Genuine individual commit histories |
| Git Branches | `feature/AISD05-eda`, `feature/SIAD19-modeling`, `feature/RSD20-streamlit` | Authentic student feature branches |
| Pull Requests | Workflow demonstration PRs | Genuine team code reviews & merges |
| Purpose | Educational reference / Teaching | Official academic assessment |

### Simulated Team Responsibilities
- **`AISD05`**: Data acquisition script, quality audit, EDA dashboards, statistical hypothesis testing.
- **`SIAD19`**: Preprocessing pipeline, model development (Zero-R, KNN, Decision Tree, Random Forest, Naive Bayes, SVM), hyperparameter tuning, cross-validation.
- **`RSD20`**: Streamlit web application, predictor UI, unit tests, LaTeX report writing, PDF compilation.

---

## 📁 Repository Architecture

```
dataminingproject/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── heart_disease_risk_2026.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       └── model_comparison.csv
│
├── scripts/
│   ├── download_data.py
│   ├── train_pipeline.py
│   └── create_notebooks.py
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_preprocessing_encoding.ipynb
│   └── 03_model_training_evaluation.ipynb
│
├── src/
│   ├── app.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py
│
├── models/
│   ├── best_model.joblib
│   ├── preprocessing_pipeline.joblib
│   └── model_metadata.json
│
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   └── test_models.py
│
└── docs/
    ├── Rapport_MiniProject.tex
    ├── Rapport_MiniProject.pdf
    └── figures/
        ├── target_distribution.png
        ├── correlation_matrix.png
        ├── feature_distributions.png
        ├── roc_curves.png
        ├── confusion_matrices.png
        └── feature_importance.png
```

---

## 🔄 Reproducibility Instructions

To reproduce the entire project end-to-end:

1. **Clone repository & install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Kaggle dataset:**
   ```bash
   python scripts/download_data.py
   ```

3. **Run end-to-end Data Mining & Training Pipeline:**
   ```bash
   PYTHONPATH=. python scripts/train_pipeline.py
   ```

4. **Run Unit Tests:**
   ```bash
   PYTHONPATH=. pytest tests/
   ```

5. **Launch Streamlit Web App:**
   ```bash
   streamlit run src/app.py
   ```

6. **Compile Scientific LaTeX Report:**
   ```bash
   cd docs && pdflatex Rapport_MiniProject.tex
   ```

---

## 📜 References
- Udit Jain, *Heart Disease Risk Dataset 2026*, Kaggle, 2026.
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR, 12:2825-2830, 2011.
