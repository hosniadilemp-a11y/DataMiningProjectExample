"""
Streamlit Web Application for Heart Disease Risk Prediction.
Interactive Data Mining Academic Dashboard containing 9 dedicated pages:
1. Home & Executive KPI Summary
2. Dataset Explorer & Quality Audit
3. EDA Dashboard (Univariate, Bivariate, Multivariate, PCA)
4. Statistical Hypothesis Suite (FDR Corrected & Effect Sizes)
5. Multi-Perspective Feature Importance Comparison
6. Model Benchmark & Trade-offs
7. Interactive Heart Disease Risk Predictor
8. Key Discoveries & Findings
9. About & Student Contribution Matrix
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

# Setup page configuration
st.set_page_config(
    page_title="Heart Disease Risk Prediction — Advanced Data Mining",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Academic / Sleek Dashboard Palette)
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        padding: 1.2rem;
        border-radius: 0.6rem;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    .finding-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1.2rem;
    }
    .badge-reject {
        background-color: #DC2626;
        color: white;
        padding: 0.25rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-fdr {
        background-color: #059669;
        color: white;
        padding: 0.25rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: bold;
        font-size: 0.85rem;
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

# Helper Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease_risk_2026.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

@st.cache_data
def load_raw():
    if DATA_RAW_PATH.exists():
        return pd.read_csv(DATA_RAW_PATH)
    return None

@st.cache_data
def load_json_artifact(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@st.cache_resource
def load_ml_artifacts():
    best_model_path = MODELS_DIR / "best_model.joblib"
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"
    comp_path = DATA_PROCESSED_DIR / "model_comparison.csv"
    
    model = joblib.load(best_model_path) if best_model_path.exists() else None
    pipeline = joblib.load(pipeline_path) if pipeline_path.exists() else None
    metadata = load_json_artifact(meta_path) if meta_path.exists() else {}
    comparison_df = pd.read_csv(comp_path) if comp_path.exists() else None
    
    return model, pipeline, metadata, comparison_df

df = load_raw()
best_model, preprocessor_pipeline, model_metadata, comparison_df = load_ml_artifacts()
stats_data = load_json_artifact(RESULTS_DIR / "statistical_tests.json")
findings_data = load_json_artifact(RESULTS_DIR / "key_findings.json")
fi_data = load_json_artifact(RESULTS_DIR / "feature_importances.json")

# Navigation Sidebar
st.sidebar.title("🫀 Heart Disease DM")
st.sidebar.markdown("**Advanced Academic Dashboard**")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Home & Key Metrics",
        "2. Dataset Explorer",
        "3. EDA Dashboard",
        "4. Statistical Analysis (Hypotheses)",
        "5. Feature Importance Comparison",
        "6. Model Benchmark & Trade-offs",
        "7. Heart Disease Risk Predictor",
        "8. Key Discoveries & Findings",
        "9. About & Student Matrix"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Simulated Team:**
- `AISD05`: Data, Audit, EDA & Hypothesis Tests
- `SIAD19`: Preprocessing & ML Modeling
- `RSD20`: Streamlit Dashboard, Report & PDF
""")

# ==========================================
# PAGE 1 — HOME & KEY METRICS
# ==========================================
if page == "1. Home & Key Metrics":
    st.markdown('<div class="main-title">🫀 Heart Disease Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Advanced Supervised Data Mining & Statistical Analysis Study</div>', unsafe_allow_html=True)
    
    st.markdown("""
    > ⚠️ **DEMONSTRATION / SAMPLE PROJECT NOTICE**  
    > Educational reference project created to demonstrate scientific rigor, statistical testing, machine learning benchmarking, and Streamlit dashboard design. Uses **ONE GitHub account** with simulated student contributors (`AISD05`, `SIAD19`, `RSD20`).
    """)
    
    st.subheader("📌 Executive KPI Dashboard")
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    
    with kpi1:
        st.markdown('<div class="kpi-card"><div class="kpi-title">PATIENTS</div><div class="kpi-value">9,000</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown('<div class="kpi-card"><div class="kpi-title">FEATURES</div><div class="kpi-value">25</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown('<div class="kpi-card"><div class="kpi-title">PREVALENCE</div><div class="kpi-value">30.3%</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown('<div class="kpi-card"><div class="kpi-title">BEST MODEL</div><div class="kpi-value">SVM</div></div>', unsafe_allow_html=True)
    with kpi5:
        st.markdown('<div class="kpi-card"><div class="kpi-title">ROC-AUC</div><div class="kpi-value">0.942</div></div>', unsafe_allow_html=True)
    with kpi6:
        st.markdown('<div class="kpi-card"><div class="kpi-title">TEST RECALL</div><div class="kpi-value">76.7%</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📖 Project Summary & Scientific Motivation")
        st.write("""
        Cardiovascular diseases (CVDs) remain the leading global cause of mortality. This project investigates whether non-invasive routine patient characteristics, physiological stress tests, blood chemistry biomarkers, and self-reported lifestyle habits can accurately predict individual heart disease risk.
        
        **Core Investigation Framework:**
        $$\\text{Question} \\longrightarrow \\text{EDA} \\longrightarrow \\text{Hypothesis Test} \\longrightarrow \\text{Effect Size} \\longrightarrow \\text{ML Benchmark} \\longrightarrow \\text{Deployment}$$
        """)
        
    with col2:
        st.subheader("🎯 Primary Scientific Contributions")
        st.markdown("""
        - **Data Audit ($N=9,000$):** Zero missing values, zero duplicates, moderate class imbalance (69.7% vs 30.3%).
        - **6 Statistical Hypotheses:** Non-parametric tests with Benjamini-Hochberg FDR correction & effect sizes.
        - **Multi-Perspective Feature Importance:** Comparison of Mutual Info, Gini Importance, and Permutation Importance.
        - **6 Classifier Benchmark:** Zero-R, KNN, Decision Tree, Random Forest, Naive Bayes, and RBF SVM.
        """)

# ==========================================
# PAGE 2 — DATASET EXPLORER
# ==========================================
elif page == "2. Dataset Explorer":
    st.title("📂 Dataset Explorer & Quality Audit")
    
    if df is not None:
        st.markdown("""
        <div class="finding-box">
        <strong>Dataset Audit Note (Section 5 Requirement):</strong><br>
        Audited dataset <code>uditjain13/heart-disease-risk-2026</code> contains <strong>9,000 patient records</strong> and <strong>27 columns</strong> (1 identifier <code>patient_id</code>, 25 feature attributes, 1 binary target <code>has_heart_disease</code>).
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Preview Raw Data", "Data Dictionary & Types", "Summary Statistics"])
        
        with tab1:
            rows_to_show = st.slider("Select row count to display", 5, 100, 10)
            st.dataframe(df.head(rows_to_show), width="stretch")
            
        with tab2:
            dtype_df = pd.DataFrame({
                "Column Name": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Missing Values": df.isnull().sum(),
                "Unique Values": df.nunique(),
                "Sample Value": [df[c].iloc[0] for c in df.columns]
            })
            st.dataframe(dtype_df, width="stretch")
            
        with tab3:
            st.dataframe(df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]], width="stretch")

# ==========================================
# PAGE 3 — EDA DASHBOARD
# ==========================================
elif page == "3. EDA Dashboard":
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Univariate & Target", "Bivariate Panels", "Correlation Matrix", "PCA Dimensionality Reduction"])
    
    with tab1:
        fig1 = FIGURES_DIR / "target_distribution.png"
        fig3 = FIGURES_DIR / "feature_distributions.png"
        if fig1.exists():
            st.image(str(fig1), width="stretch")
        if fig3.exists():
            st.image(str(fig3), width="stretch")
            
    with tab2:
        fig_biv = FIGURES_DIR / "bivariate_panel.png"
        if fig_biv.exists():
            st.image(str(fig_biv), width="stretch")
            
    with tab3:
        fig2 = FIGURES_DIR / "correlation_matrix.png"
        if fig2.exists():
            st.image(str(fig2), width="stretch")
            
    with tab4:
        fig_pca = FIGURES_DIR / "pca_bivariate_scatter.png"
        if fig_pca.exists():
            st.image(str(fig_pca), width="stretch")
            st.caption("Figure: 2D Principal Component Analysis (PCA) projection of continuous clinical biomarker space.")

# ==========================================
# PAGE 4 — STATISTICAL ANALYSIS
# ==========================================
elif page == "4. Statistical Analysis (Hypotheses)":
    st.title("🧪 Statistical Hypothesis Testing Suite")
    st.markdown("""
    Rigorous statistical tests conducted with significance threshold $\\alpha = 0.05$.
    Includes Benjamini-Hochberg False Discovery Rate (FDR) multiple testing corrections and practical effect sizes.
    """)
    
    if stats_data:
        for h in stats_data:
            with st.expander(f"📌 {h['id']}: {h['question']}", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**H0:** {h['h0']}")
                    st.write(f"**H1:** {h['h1']}")
                    st.write(f"**Test Selected:** {h['test']} (Reason: {h['reason']})")
                    st.write(f"**Plain Language Interpretation:** {h['interpretation']}")
                    st.write(f"**Data Mining Implication:** {h['dm_implication']}")
                with col2:
                    st.metric("Test Statistic", f"{h['statistic']}")
                    p_val_display = "< 0.0001" if h['p_value_raw'] < 0.0001 else f"{h['p_value_raw']:.4f}"
                    st.metric("p-value (Raw)", p_val_display)
                    st.metric(f"Effect Size ({h['effect_size_type']})", f"{h['effect_size_val']}")
                    st.markdown(f'<span class="badge-reject">{h["decision_raw"]}</span> <span class="badge-fdr">FDR Sig: {h["significant_fdr"]}</span>', unsafe_allow_html=True)
                    
        st.markdown("---")
        st.subheader("Summary Table of Statistical Hypotheses")
        stat_summary_df = pd.DataFrame([
            {
                "ID": h["id"],
                "Test": h["test"],
                "Statistic": h["statistic"],
                "Raw p-value": "< 0.0001" if h["p_value_raw"] < 0.0001 else f"{h['p_value_raw']:.4f}",
                "FDR p-value": "< 0.0001" if h["p_value_fdr"] < 0.0001 else f"{h['p_value_fdr']:.4f}",
                "Effect Size": f"{h['effect_size_type']} = {h['effect_size_val']}",
                "Decision": h["decision_raw"]
            } for h in stats_data
        ])
        st.table(stat_summary_df)

# ==========================================
# PAGE 5 — FEATURE IMPORTANCE COMPARISON
# ==========================================
elif page == "5. Feature Importance Comparison":
    st.title("⚖️ Multi-Perspective Feature Importance")
    st.markdown("""
    Comparing 4 distinct perspectives of feature importance:
    1. **Mutual Information:** Information-theoretic dependency.
    2. **Tree Gini Importance:** Random Forest impurity reduction.
    3. **Permutation Importance:** Out-of-fold performance drop on test set.
    4. **Correlation Magnitude:** Bivariate linear correlation magnitude.
    """)
    
    fig_fic = FIGURES_DIR / "feature_importance_comp.png"
    if fig_fic.exists():
        st.image(str(fig_fic), width="stretch")
        
    if fi_data:
        st.subheader("Normalized Feature Importance Rankings Table")
        st.dataframe(pd.DataFrame(fi_data), width="stretch")

# ==========================================
# PAGE 6 — MODEL BENCHMARK
# ==========================================
elif page == "6. Model Benchmark & Trade-offs":
    st.title("🤖 Supervised Classification Model Benchmark")
    
    if comparison_df is not None:
        st.subheader("Supervised Classifier Comparison (Test Set, N = 1,800)")
        st.dataframe(comparison_df, width="stretch")
        
        if model_metadata:
            st.success(f"**Best Model:** {model_metadata.get('best_model_name')} "
                       f"(Accuracy: {model_metadata.get('test_metrics', {}).get('Accuracy')}, "
                       f"ROC-AUC: {model_metadata.get('test_metrics', {}).get('ROC-AUC')})")
            
        fig_roc = FIGURES_DIR / "roc_curves.png"
        fig_cm = FIGURES_DIR / "confusion_matrices.png"
        
        col1, col2 = st.columns(2)
        with col1:
            if fig_roc.exists():
                st.subheader("ROC Curves Comparison")
                st.image(str(fig_roc), width="stretch")
        with col2:
            if fig_cm.exists():
                st.subheader("Confusion Matrix Grid")
                st.image(str(fig_cm), width="stretch")

# ==========================================
# PAGE 7 — HEART DISEASE RISK PREDICTOR
# ==========================================
elif page == "7. Heart Disease Risk Predictor":
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
            patient_dict = {
                "age": [age], "sex": [sex], "resting_bp_systolic": [sys_bp], "resting_bp_diastolic": [dia_bp],
                "cholesterol_total": [chol_total], "hdl": [hdl], "ldl": [ldl], "triglycerides": [triglycerides],
                "fasting_blood_sugar": [fbs], "hba1c": [hba1c], "bmi": [bmi], "resting_heart_rate": [resting_hr],
                "max_heart_rate_achieved": [max_hr], "chest_pain_type": [cp_type], "exercise_induced_angina": [ex_angina],
                "st_depression": [st_dep], "family_history": [family_hist], "smoker_status": [smoker],
                "alcohol_units_per_week": [alcohol], "exercise_minutes_per_week": [exercise_min],
                "sleep_hours": [sleep_hrs], "stress_score": [stress], "wearable_owner": [wearable],
                "daily_steps": [daily_steps], "diet_quality_score": [diet_score]
            }
            patient_df = pd.DataFrame(patient_dict)
            patient_df["cholesterol_hdl_ratio"] = patient_df["cholesterol_total"] / np.maximum(patient_df["hdl"], 1.0)
            patient_df["pulse_pressure"] = patient_df["resting_bp_systolic"] - patient_df["resting_bp_diastolic"]
            patient_df["mean_arterial_pressure"] = patient_df["resting_bp_diastolic"] + (patient_df["pulse_pressure"] / 3.0)
            
            patient_proc = preprocessor_pipeline.transform(patient_df)
            pred_class = int(best_model.predict(patient_proc)[0])
            
            if hasattr(best_model, "predict_proba"):
                pred_prob = float(best_model.predict_proba(patient_proc)[0][1])
            else:
                pred_prob = 1.0 if pred_class == 1 else 0.0
                
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if pred_class == 1:
                    st.error(f"### ⚠️ High Heart Disease Risk Predicted\nEstimated Risk Probability: **{pred_prob:.1%}**")
                else:
                    st.success(f"### ✅ Low Heart Disease Risk Predicted\nEstimated Risk Probability: **{pred_prob:.1%}**")
            with res_col2:
                st.metric("Model Selected", model_metadata.get("best_model_name", "Best Model"))
                st.metric("Model ROC-AUC", f"{model_metadata.get('test_metrics', {}).get('ROC-AUC', 0.0):.3f}")

# ==========================================
# PAGE 8 — KEY DISCOVERIES & FINDINGS
# ==========================================
elif page == "8. Key Discoveries & Findings":
    st.title("💡 Key Empirical Findings & Discoveries")
    st.markdown("Structured analytical discoveries synthesized from statistical tests, EDA, and Machine Learning models:")
    
    if findings_data:
        for f in findings_data:
            with st.container():
                st.markdown(f"### 🔍 {f['title']}")
                st.markdown(f"""
                - **Research Question:** *{f['question']}*
                - **Analytical Method:** {f['analysis']}
                - **Empirical Result:** **{f['result']}** ({f['effect_size']})
                - **Clinical & Statistical Interpretation:** {f['interpretation']}
                - **Data Mining Implication:** `{f['dm_implication']}`
                """)
                st.markdown("---")

# ==========================================
# PAGE 9 — ABOUT & STUDENT MATRIX
# ==========================================
elif page == "9. About & Student Matrix":
    st.title("ℹ️ About & Student Contribution Matrix")
    
    st.markdown("""
    > ⚠️ **DEMONSTRATION GIT WORKFLOW NOTICE**  
    > Published using **ONE GitHub account** for reference/demonstration purposes.
    > The student codes `AISD05`, `SIAD19`, and `RSD20` represent **simulated contributors**.
    > In actual student mini-projects, **each student must use their own GitHub account**.
    """)
    
    st.subheader("Simulated Student Contribution Matrix")
    contrib_data = [
        {"Student Code": "AISD05", "Responsibility Area": "Dataset Acquisition, Quality Audit, EDA, Hypothesis Testing Suite", "Git Branch": "feature/AISD05-eda"},
        {"Student Code": "SIAD19", "Responsibility Area": "Preprocessing Pipeline, ML Models (6 Algorithms), CV & GridSearch", "Git Branch": "feature/SIAD19-modeling"},
        {"Student Code": "RSD20", "Responsibility Area": "Streamlit Web App, Predictor UI, Unit Tests, LaTeX Report & PDF", "Git Branch": "feature/RSD20-streamlit"}
    ]
    st.table(pd.DataFrame(contrib_data))
    
    st.subheader("Software Stack & Dependencies")
    st.markdown("""
    - **Language:** Python 3.13 / 3.10
    - **Machine Learning:** `scikit-learn`, `scipy`, `numpy`, `pandas`, `pyarrow`
    - **Data Acquisition:** `kagglehub`
    - **Visualization:** `matplotlib`, `seaborn`
    - **Web Framework:** `streamlit`
    - **Report Generation:** LaTeX (`pdflatex`)
    """)
