# 🫀 Heart Disease Risk Prediction Using Data Mining
> **Advanced Data Mining Research & Reference Mini-Project**

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

---

> ⚠️ **MEDICAL DISCLAIMER**
>
> This application and associated code are developed exclusively for academic and educational Data Mining purposes. Its predictions must **NOT** be interpreted as a medical diagnosis or as a substitute for professional medical advice.

---

## 📌 Project Overview & Problem Statement
Cardiovascular diseases (CVDs) remain the leading cause of mortality worldwide. Early detection of individuals at high risk allows targeted preventative interventions and lifestyle modifications.

**Core Research Question:**
*"Can routine patient demographics, physiological markers, blood chemistry parameters, and lifestyle factors accurately predict whether an individual is at high risk of heart disease?"*

This repository demonstrates an expanded, research-style Data Mining pipeline:
1. Automated data fetch via `kagglehub` (`uditjain13/heart-disease-risk-2026`).
2. Data quality audit ($N = 9,000$ patient records, 25 clinical features).
3. Leakage-free preprocessing via scikit-learn `ColumnTransformer`.
4. Exploratory Data Analysis & 6-Hypothesis Statistical Testing Suite ($\alpha = 0.05$) with Benjamini-Hochberg FDR correction and practical effect sizes.
5. Multi-Perspective Feature Importance analysis (Mutual Info, Gini Importance, Permutation Importance, Correlation Magnitude).
6. Supervised Classification benchmark across **6 algorithms**: Zero-R Baseline, KNN, Decision Tree, Random Forest, Naive Bayes, and Support Vector Machine (SVM).
7. 9-Page interactive Streamlit web dashboard.
8. Publication-ready 10-page scientific report in LaTeX compiled to PDF.

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

---

## 🧪 Statistical Hypothesis Testing Suite

| ID | Test Applied | Statistic | Raw $p$-val | FDR $p$-val | Effect Size | Decision |
|---|---|---:|---:|---:|---|---|
| H1 | Chi-Square Independence | 942.50 | < 0.0001 | < 0.0001 | Cramér's V = 0.3236 | Reject H0 |
| H2 | Mann-Whitney U Test | 2,850,200.0 | < 0.0001 | < 0.0001 | Rank-Biserial r = 0.4210 | Reject H0 |
| H3 | Mann-Whitney U Test | 2,410,500.0 | < 0.0001 | < 0.0001 | Cohen's d = 1.3420 | Reject H0 |
| H4 | Spearman Rank Corr | 0.3150 | < 0.0001 | < 0.0001 | Spearman $\rho$ = 0.3150 | Reject H0 |
| H5 | Kruskal-Wallis Test | 124.80 | < 0.0001 | < 0.0001 | Eta-squared = 0.0137 | Reject H0 |
| H6 | Chi-Square / Fisher Exact | 485.20 | < 0.0001 | < 0.0001 | Odds Ratio = 3.4250 | Reject H0 |

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

## 💡 Key Empirical Discoveries

1. **Prevalence Profile:** 30.30% positive class prevalence ($N=9,000$), requiring Weighted F1 and ROC-AUC metrics over raw accuracy.
2. **Chronotropic Response Deficit:** Max Heart Rate Achieved showed a large effect size (Cohen's $d = 1.34$, $p < 0.0001$).
3. **Myocardial Ischemia Signal:** ST depression exhibited strong non-linear diagnostic value (Rank-Biserial $r = 0.42$).
4. **Classifier Superiority:** RBF Support Vector Machine achieved optimal hyperplane margin separation ($89.17\%$ Accuracy, $0.9422$ ROC-AUC).
5. **Synergistic Risk Triad:** Age, LDL Cholesterol, and HbA1c interact synergistically to compound individual cardiovascular risk.

---

## 🌐 Interactive Streamlit Web Application

The interactive web application contains 9 dedicated pages:
1. **Home & Key Metrics:** Executive KPI dashboard, problem statement, project summary.
2. **Dataset Explorer:** Data dictionary, raw preview, missing value audit.
3. **EDA Dashboard:** Univariate, bivariate panels, correlation matrix, PCA scatter plot.
4. **Statistical Analysis:** Hypothesis test suite, $p$-values (raw vs FDR corrected), effect sizes, decision badges.
5. **Feature Importance Comparison:** Interactive side-by-side comparison of Gini vs Permutation vs Mutual Information.
6. **Model Benchmark & Trade-offs:** Comparison table, ROC curves, confusion matrices.
7. **Heart Disease Risk Predictor:** Dynamic patient risk predictor interface with probability badge & medical disclaimer.
8. **Key Discoveries & Findings:** Question $\rightarrow$ Analysis $\rightarrow$ Result $\rightarrow$ Interpretation $\rightarrow$ DM Implication breakdown.
9. **About & Contributions:** Team responsibilities, software stack, demonstration warnings.

### How to Run Streamlit Application
```bash
python3 -m streamlit run src/app.py
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
│   ├── generate_report_assets.py
│   ├── create_notebooks.py
│   └── setup_git_history.py
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_statistical_analysis.ipynb
│   ├── 03_preprocessing.ipynb
│   └── 04_model_training_evaluation.ipynb
│
├── src/
│   ├── app.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── statistics.py
│   ├── feature_analysis.py
│   ├── findings.py
│   ├── visualization.py
│   └── utils.py
│
├── results/
│   ├── statistical_tests.json
│   ├── feature_importances.json
│   ├── key_findings.json
│   └── feature_importances.csv
│
├── models/
│   ├── best_model.joblib
│   ├── preprocessing_pipeline.joblib
│   └── model_metadata.json
│
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_statistics.py
│   └── test_models.py
│
└── docs/
    ├── Rapport_MiniProject.tex
    ├── Rapport_MiniProject.pdf (10 pages)
    └── figures/
        ├── target_distribution.png
        ├── correlation_matrix.png
        ├── feature_distributions.png
        ├── bivariate_panel.png
        ├── pca_bivariate_scatter.png
        ├── feature_importance_comp.png
        ├── roc_curves.png
        └── confusion_matrices.png
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

4. **Generate Report & Dashboard Assets:**
   ```bash
   PYTHONPATH=. python scripts/generate_report_assets.py
   ```

5. **Run Unit Tests:**
   ```bash
   PYTHONPATH=. pytest tests/
   ```

6. **Launch Streamlit Web App:**
   ```bash
   python3 -m streamlit run src/app.py
   ```

7. **Compile Scientific LaTeX Report:**
   ```bash
   cd docs && pdflatex Rapport_MiniProject.tex
   ```

---

## 📜 References
- Udit Jain, *Heart Disease Risk Dataset 2026*, Kaggle, 2026.
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR, 12:2825-2830, 2011.
