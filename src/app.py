"""
Streamlit Web Application for Heart Disease Risk Prediction.
Interactive Data Mining Demonstration App containing 7 dedicated pages:
1. Overview
2. Dataset Explorer
3. EDA Dashboard
4. Statistical Analysis
5. Model Comparison
6. Heart Disease Risk Prediction
7. About & Team Contributions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy import stats
import json

# Setup page configuration
st.set_page_config(
    page_title="Heart Disease Risk Prediction — Data Mining Demo",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark/Modern sleek palette)
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-left: 5px solid #2563EB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #FEE2E2;
        border-left: 5px solid #DC2626;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        color: #991B1B;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper Path Loaders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease_risk_2026.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

@st.cache_data
def load_data():
    if DATA_RAW_PATH.exists():
        return pd.read_csv(DATA_RAW_PATH)
    return None

@st.cache_resource
def load_model_artifacts():
    best_model_path = MODELS_DIR / "best_model.joblib"
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"
    comp_path = DATA_PROCESSED_DIR / "model_comparison.csv"
    
    model = joblib.load(best_model_path) if best_model_path.exists() else None
    pipeline = joblib.load(pipeline_path) if pipeline_path.exists() else None
    
    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    comparison_df = pd.read_csv(comp_path) if comp_path.exists() else None
    return model, pipeline, metadata, comparison_df

# Navigation Sidebar
st.sidebar.title("🫀 Heart Disease DM")
st.sidebar.markdown("**Educational Data Mining Demonstration**")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Overview",
        "2. Dataset Explorer",
        "3. EDA Dashboard",
        "4. Statistical Analysis",
        "5. Model Comparison",
        "6. Heart Disease Risk Prediction",
        "7. About & Contributions"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Simulated Team:**
- `AISD05`: Data & EDA
- `SIAD19`: ML Models
- `RSD20`: Streamlit & App
""")

df = load_data()
best_model, preprocessor_pipeline, model_metadata, comparison_df = load_model_artifacts()

# ==========================================
# PAGE 1 — OVERVIEW
# ==========================================
if page == "1. Overview":
    st.markdown('<div class="main-header">🫀 Heart Disease Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data Mining & Supervised Machine Learning Reference Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
    > ⚠️ **DEMONSTRATION / SAMPLE PROJECT WARNING**  
    > This repository and application are educational reference projects created to demonstrate the expected structure, methodology, documentation, Git workflow, and presentation of a Data Mining mini-project.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📌 Project Problem Statement")
        st.write("""
        Cardiovascular diseases remain a leading global cause of mortality. Early identification of individuals at high risk of heart disease based on clinical metrics, lifestyle factors, and physiological markers enables timely medical intervention and lifestyle modification.
        
        **Core Research Question:**
        *“Can patient demographics, clinical biomarkers, and lifestyle characteristics accurately predict whether an individual is at high risk of developing heart disease?”*
        """)
        
        st.subheader("🎯 Objectives")
        st.markdown("""
        - **Data Quality & Audit:** Inspect raw dataset schema, audit missing values, duplicates, and distributions.
        - **Exploratory Data Analysis (EDA):** Visualize key risk drivers (ST depression, maximum heart rate, LDL/HDL, HbA1c).
        - **Statistical Rigor:** Formulate formal null/alternative hypotheses ($H_0$, $H_1$) and test associations via Chi-Square and Mann-Whitney U tests.
        - **Supervised Classification Benchmark:** Train and tune **6 algorithms**: Zero-R, KNN, Decision Tree, Random Forest, Naive Bayes, and SVM using 5-fold Stratified Cross-Validation.
        - **Interactive Deployment:** Deliver a production-grade Streamlit web interface for risk prediction.
        """)
        
    with col2:
        st.subheader("📊 Dataset At A Glance")
        if df is not None:
            st.metric("Total Patient Records", f"{len(df):,}")
            st.metric("Total Attributes", f"{df.shape[1]}")
            st.metric("Predictor Features", f"{df.shape[1] - 2}")
            target_prev = (df["has_heart_disease"].mean()) * 100
            st.metric("Heart Disease Prevalence", f"{target_prev:.1f}%")
        else:
            st.warning("Raw dataset file not found in data/raw/")

# ==========================================
# PAGE 2 — DATASET EXPLORER
# ==========================================
elif page == "2. Dataset Explorer":
    st.title("📂 Dataset Explorer & Quality Audit")
    
    if df is not None:
        st.markdown("""
        <div class="warning-box">
        <strong>Dataset Size Requirement Audit (Section 5 Note):</strong><br>
        The official Kaggle dataset <code>uditjain13/heart-disease-risk-2026</code> contains <strong>9,000 records</strong> and <strong>27 features</strong>.
        No synthetic rows or fake records were fabricated to force 10,000 rows.
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Preview Raw Dataset")
        rows_to_show = st.slider("Number of records to display", 5, 100, 10)
        st.dataframe(df.head(rows_to_show), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Types & Missing Value Audit")
            null_audit = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Missing Values": df.isnull().sum(),
                "Unique Values": df.nunique()
            })
            st.dataframe(null_audit, use_container_width=True)
            
        with col2:
            st.subheader("Summary Statistics (Numerical)")
            st.dataframe(df.describe().T[["mean", "std", "min", "50%", "max"]], use_container_width=True)
    else:
        st.error("Dataset not loaded.")

# ==========================================
# PAGE 3 — EDA DASHBOARD
# ==========================================
elif page == "3. EDA Dashboard":
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    fig_target = FIGURES_DIR / "target_distribution.png"
    fig_corr = FIGURES_DIR / "correlation_matrix.png"
    fig_dist = FIGURES_DIR / "feature_distributions.png"
    
    if fig_target.exists():
        st.subheader("Target Variable Audit (Class Balance)")
        st.image(str(fig_target), use_container_width=True)
        st.caption("Figure 1: Target variable distribution showing ~30.3% positive class prevalence.")
        
    if fig_corr.exists():
        st.subheader("Top Clinical Feature Correlations")
        st.image(str(fig_corr), use_container_width=True)
        st.caption("Figure 2: Pearson correlation matrix highlighting top clinical risk drivers.")
        
    if fig_dist.exists():
        st.subheader("Clinical Biomarker Distributions Stratified by Target Class")
        st.image(str(fig_dist), use_container_width=True)
        st.caption("Figure 3: Density plots comparing key clinical markers between diseased and healthy patients.")

# ==========================================
# PAGE 4 — STATISTICAL ANALYSIS
# ==========================================
elif page == "4. Statistical Analysis":
    st.title("🧪 Statistical Hypothesis Testing")
    st.markdown("""
    Rigorous statistical tests were conducted to test significant associations between clinical features and heart disease risk ($\alpha = 0.05$).
    """)
    
    if df is not None:
        st.subheader("1. Continuous Clinical Attributes (Mann-Whitney U Test)")
        num_cols = ["age", "st_depression", "max_heart_rate_achieved", "ldl", "hdl", "hba1c", "resting_bp_systolic", "exercise_minutes_per_week"]
        
        mw_results = []
        for col in num_cols:
            group0 = df[df["has_heart_disease"] == 0][col]
            group1 = df[df["has_heart_disease"] == 1][col]
            stat_val, p_val = stats.mannwhitneyu(group0, group1, alternative="two-sided")
            mw_results.append({
                "Clinical Feature": col,
                "Mean (No Disease)": round(group0.mean(), 2),
                "Mean (Heart Disease)": round(group1.mean(), 2),
                "U Statistic": round(stat_val, 1),
                "p-value": "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}",
                "Significant?": "Yes (p < 0.05)" if p_val < 0.05 else "No"
            })
        st.table(pd.DataFrame(mw_results))
        
        st.subheader("2. Categorical & Lifestyle Attributes (Chi-Square Test of Independence)")
        cat_cols = ["sex", "chest_pain_type", "exercise_induced_angina", "family_history", "smoker_status", "wearable_owner"]
        
        chi2_results = []
        for col in cat_cols:
            contingency = pd.crosstab(df[col], df["has_heart_disease"])
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            chi2_results.append({
                "Categorical Feature": col,
                "Chi-Square Stat ($\chi^2$)": round(chi2, 2),
                "Degrees of Freedom": dof,
                "p-value": "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}",
                "Significant?": "Yes (p < 0.05)" if p_val < 0.05 else "No"
            })
        st.table(pd.DataFrame(chi2_results))

# ==========================================
# PAGE 5 — MODEL COMPARISON
# ==========================================
elif page == "5. Model Comparison":
    st.title("🤖 Supervised Classification Model Benchmark")
    
    if comparison_df is not None:
        st.subheader("Performance Comparison Table (Test Set)")
        st.dataframe(comparison_df, use_container_width=True)
        
        if model_metadata:
            st.success(f"**Best Performing Model Identified:** {model_metadata.get('best_model_name')} "
                       f"(ROC-AUC: {model_metadata.get('test_metrics', {}).get('ROC-AUC')})")
            
        fig_roc = FIGURES_DIR / "roc_curves.png"
        fig_cm = FIGURES_DIR / "confusion_matrices.png"
        fig_fi = FIGURES_DIR / "feature_importance.png"
        
        col1, col2 = st.columns(2)
        with col1:
            if fig_roc.exists():
                st.subheader("ROC Curves Comparison")
                st.image(str(fig_roc), use_container_width=True)
        with col2:
            if fig_fi.exists():
                st.subheader("Random Forest Feature Importances")
                st.image(str(fig_fi), use_container_width=True)
                
        if fig_cm.exists():
            st.subheader("Confusion Matrix Grid across 6 Models")
            st.image(str(fig_cm), use_container_width=True)
    else:
        st.info("Run `python scripts/train_pipeline.py` to generate model benchmark artifacts.")

# ==========================================
# PAGE 6 — HEART DISEASE RISK PREDICTION
# ==========================================
elif page == "6. Heart Disease Risk Prediction":
    st.title("🫀 Patient Heart Disease Risk Predictor")
    
    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ <strong>MEDICAL DISCLAIMER:</strong><br>
    This application is developed exclusively for educational and Data Mining demonstration purposes.
    Its predictions must NOT be interpreted as a medical diagnosis or as a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)
    
    if best_model is not None and preprocessor_pipeline is not None:
        st.subheader("Input Patient Characteristics & Clinical Biomarkers")
        
        with st.form("patient_prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### 👤 Demographics & Vitals")
                age = st.number_input("Age (years)", 18, 100, 55)
                sex = st.selectbox("Sex", ["Male", "Female"])
                sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", 80, 220, 130)
                dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 50, 140, 82)
                resting_hr = st.number_input("Resting Heart Rate (bpm)", 40, 130, 75)
                max_hr = st.number_input("Max Heart Rate Achieved (bpm)", 60, 220, 150)
                
            with col2:
                st.markdown("##### 🧪 Lab & Clinical Biomarkers")
                chol_total = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 210)
                hdl = st.number_input("HDL Cholesterol (mg/dL)", 15, 120, 48)
                ldl = st.number_input("LDL Cholesterol (mg/dL)", 30, 250, 125)
                triglycerides = st.number_input("Triglycerides (mg/dL)", 30, 500, 160)
                fbs = st.number_input("Fasting Blood Sugar (mg/dL)", 60, 250, 110)
                hba1c = st.number_input("HbA1c (%)", 4.0, 14.0, 5.8)
                bmi = st.number_input("Body Mass Index (BMI)", 14.0, 50.0, 26.5)
                st_dep = st.number_input("ST Depression (mm)", 0.0, 7.0, 1.2)

            with col3:
                st.markdown("##### 🏃 Lifestyle & Symptoms")
                cp_type = st.selectbox("Chest Pain Type", ["Asymptomatic", "Non-Anginal Pain", "Atypical Angina", "Typical Angina"])
                ex_angina = st.checkbox("Exercise Induced Angina", value=False)
                family_hist = st.checkbox("Family History of Heart Disease", value=False)
                smoker = st.selectbox("Smoker Status", ["Never", "Former", "Current"])
                alcohol = st.number_input("Alcohol Units / Week", 0.0, 60.0, 4.0)
                exercise_min = st.number_input("Exercise Min / Week", 0, 600, 150)
                sleep_hrs = st.number_input("Sleep Hours / Day", 3.0, 12.0, 7.0)
                stress = st.number_input("Stress Score (0-100)", 0.0, 100.0, 45.0)
                wearable = st.checkbox("Wearable Device Owner", value=True)
                daily_steps = st.number_input("Daily Steps", 500, 25000, 6500)
                diet_score = st.number_input("Diet Quality Score (0-100)", 0.0, 100.0, 60.0)

            submit_btn = st.form_submit_button("🔍 Predict Heart Disease Risk")
            
        if submit_btn:
            # Construct patient DataFrame
            patient_dict = {
                "age": [age],
                "sex": [sex],
                "resting_bp_systolic": [sys_bp],
                "resting_bp_diastolic": [dia_bp],
                "cholesterol_total": [chol_total],
                "hdl": [hdl],
                "ldl": [ldl],
                "triglycerides": [triglycerides],
                "fasting_blood_sugar": [fbs],
                "hba1c": [hba1c],
                "bmi": [bmi],
                "resting_heart_rate": [resting_hr],
                "max_heart_rate_achieved": [max_hr],
                "chest_pain_type": [cp_type],
                "exercise_induced_angina": [ex_angina],
                "st_depression": [st_dep],
                "family_history": [family_hist],
                "smoker_status": [smoker],
                "alcohol_units_per_week": [alcohol],
                "exercise_minutes_per_week": [exercise_min],
                "sleep_hours": [sleep_hrs],
                "stress_score": [stress],
                "wearable_owner": [wearable],
                "daily_steps": [daily_steps],
                "diet_quality_score": [diet_score]
            }
            patient_df = pd.DataFrame(patient_dict)
            
            # Apply feature engineering
            patient_df["cholesterol_hdl_ratio"] = patient_df["cholesterol_total"] / np.maximum(patient_df["hdl"], 1.0)
            patient_df["pulse_pressure"] = patient_df["resting_bp_systolic"] - patient_df["resting_bp_diastolic"]
            patient_df["mean_arterial_pressure"] = patient_df["resting_bp_diastolic"] + (patient_df["pulse_pressure"] / 3.0)
            
            # Transform and Predict
            patient_proc = preprocessor_pipeline.transform(patient_df)
            pred_class = int(best_model.predict(patient_proc)[0])
            
            if hasattr(best_model, "predict_proba"):
                pred_prob = float(best_model.predict_proba(patient_proc)[0][1])
            else:
                pred_prob = 1.0 if pred_class == 1 else 0.0
                
            st.markdown("---")
            st.subheader("🎯 Risk Prediction Result")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if pred_class == 1:
                    st.error(f"### ⚠️ High Heart Disease Risk Predicted\nEstimated Risk Probability: **{pred_prob:.1%}**")
                else:
                    st.success(f"### ✅ Low Heart Disease Risk Predicted\nEstimated Risk Probability: **{pred_prob:.1%}**")
                    
            with res_col2:
                st.metric("Model Selected", model_metadata.get("best_model_name", "Best Model"))
                st.metric("Model ROC-AUC Score", f"{model_metadata.get('test_metrics', {}).get('ROC-AUC', 0.0):.3f}")
    else:
        st.error("Model artifacts not loaded. Please execute `python scripts/train_pipeline.py` first.")

# ==========================================
# PAGE 7 — ABOUT & CONTRIBUTIONS
# ==========================================
elif page == "7. About & Contributions":
    st.title("ℹ️ About & Student Contribution Matrix")
    
    st.markdown("""
    > ⚠️ **DEMONSTRATION GIT WORKFLOW NOTICE**  
    > In this educational sample repository, a **SINGLE GitHub account** was used to publish the project.
    > The student codes `AISD05`, `SIAD19`, and `RSD20` represent **simulated contributors** used to demonstrate how a multi-student team structures Git branches and commit messages.
    > In an actual student project, **every student must use their own GitHub account**.
    """)
    
    st.subheader("Simulated Student Contribution Matrix")
    contrib_data = [
        {"Student Code": "AISD05", "Responsibility Area": "Data Acquisition, Quality Audit, EDA, Statistical Testing", "Git Branch": "feature/AISD05-eda"},
        {"Student Code": "SIAD19", "Responsibility Area": "Preprocessing Pipeline, ML Models (6 Algorithms), CV & GridSearch", "Git Branch": "feature/SIAD19-modeling"},
        {"Student Code": "RSD20", "Responsibility Area": "Streamlit Web App, Predictor UI, Unit Tests, LaTeX Report & PDF", "Git Branch": "feature/RSD20-streamlit"}
    ]
    st.table(pd.DataFrame(contrib_data))
    
    st.subheader("Software Stack & Dependencies")
    st.markdown("""
    - **Language:** Python 3.13 / 3.10
    - **Machine Learning:** `scikit-learn`, `scipy`, `numpy`, `pandas`
    - **Data Fetching:** `kagglehub`
    - **Visualization:** `matplotlib`, `seaborn`
    - **Web Framework:** `streamlit`
    - **Report Generation:** LaTeX (`pdflatex`)
    """)
