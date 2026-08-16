"""
Streamlit Web Application for Heart Disease Risk Prediction.
Interactive Data Mining Academic Dashboard containing 9 dedicated pages:
1. Home & Executive Summary (Academic Objectives & Formal Metrics Framework)
2. Dataset Explorer & Quality Audit
3. EDA Dashboard (Univariate, Bivariate Panels, Correlations, PCA)
4. Statistical Analysis Suite (FDR Corrected & Effect Sizes)
5. Multi-Perspective Feature Importance Comparison
6. Model Benchmark & Hyperparameters Transparency
7. Categorized & Importance-Color-Coded Heart Disease Risk Predictor
8. Key Discoveries & Findings
9. About, Student Matrix & Online Deployment Guide
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Setup page configuration
st.set_page_config(
    page_title="Heart Disease Risk Prediction — Academic Data Mining",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Academic / Professional Palette)
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
        border: 1px solid #CBD5E1;
        border-left: 5px solid #2563EB;
        padding: 1.2rem;
        border-radius: 0.6rem;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    .badge-critical {
        background-color: #EF4444;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-high {
        background-color: #F97316;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-moderate {
        background-color: #EAB308;
        color: black;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-lifestyle {
        background-color: #10B981;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .finding-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1.2rem;
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
st.sidebar.markdown("**Academic Data Mining Dashboard**")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Home & Executive Summary",
        "2. Dataset Explorer",
        "3. EDA Dashboard",
        "4. Statistical Analysis Suite",
        "5. Feature Importance Comparison",
        "6. Model Benchmark & Hyperparameters",
        "7. Heart Disease Risk Predictor",
        "8. Key Discoveries & Findings",
        "9. About & Online Deployment"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Simulated Contributors:**
- `AISD05`: Data, Audit, EDA & Hypotheses
- `SIAD19`: Preprocessing & ML Benchmark
- `RSD20`: Streamlit UI, Report & PDF
""")

# ==========================================
# PAGE 1 — HOME & EXECUTIVE SUMMARY
# ==========================================
if page == "1. Home & Executive Summary":
    st.markdown('<div class="main-title">🫀 Heart Disease Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Advanced Data Mining Research & Machine Learning Benchmark</div>', unsafe_allow_html=True)
    
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
        st.markdown('<div class="kpi-card"><div class="kpi-title">BEST MODEL</div><div class="kpi-value">SVM (RBF)</div></div>', unsafe_allow_html=True)
    with kpi5:
        st.markdown('<div class="kpi-card"><div class="kpi-title">ROC-AUC</div><div class="kpi-value">0.942</div></div>', unsafe_allow_html=True)
    with kpi6:
        st.markdown('<div class="kpi-card"><div class="kpi-title">TEST RECALL</div><div class="kpi-value">76.7%</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Data Mining Objectives")
        st.markdown("""
        - **Data Quality & Schema Audit:** Audit 9,000 patient records ($0$ missing values, $0$ duplicates) for target imbalance ($30.3\\%$ positive class prevalence).
        - **Leakage-Free Preprocessing:** Standardize continuous features ($\mu=0, \sigma=1$) and One-Hot encode categories via `ColumnTransformer` inside training pipelines.
        - **Statistical Hypothesis Suite:** Conduct 6 non-parametric tests ($\alpha=0.05$) incorporating Benjamini-Hochberg FDR correction and effect sizes (Cramér's V, Cohen's d, Odds Ratio).
        - **Multi-Classifier Benchmark:** Optimize and evaluate 6 algorithms (Zero-R, KNN, Decision Tree, Random Forest, Naive Bayes, SVM) via 5-Fold Stratified Cross-Validation.
        """)
        
    with col2:
        st.subheader("📐 Formal Evaluation Metrics Framework")
        st.markdown("""
        To evaluate model performance under moderate class imbalance ($2.3:1$), we utilize formal metrics:
        
        - **Accuracy:** $\\text{Acc} = \\frac{TP + TN}{TP + TN + FP + FN} = 89.17\\%$
        - **Precision:** $\\text{Prec} = \\frac{TP}{TP + FP} = 86.01\\%$ (Positive Predictive Value)
        - **Recall / Sensitivity:** $\\text{Rec} = \\frac{TP}{TP + FN} = 76.70\\%$ (True Positive Rate)
        - **Specificity:** $\\text{Spec} = \\frac{TN}{TN + FP} = 94.58\\%$ (True Negative Rate)
        - **Weighted F1-Score:** $F_1 = 2 \\cdot \\frac{\\text{Prec} \\cdot \\text{Rec}}{\\text{Prec} + \\text{Rec}} = 0.8898$
        - **ROC-AUC:** $\\text{AUC} = \\int_0^1 \\text{TPR}(f) d(\\text{FPR}) = 0.9422$
        - **5-Fold Stratified Cross-Validation Score:** $0.8954 \\pm 0.0047$
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
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Interactive Feature Inspector",
        "Target Distribution & Class Audit",
        "Bivariate Boxplots & Violins",
        "Correlation Matrix",
        "PCA 2D Projection"
    ])
    
    with tab1:
        st.subheader("Interactive Numerical Distribution Inspector")
        num_cols = ["age", "st_depression", "max_heart_rate_achieved", "ldl", "hdl", "hba1c", "resting_bp_systolic", "bmi", "stress_score"]
        selected_feat = st.selectbox("Select Clinical Feature to Inspect:", num_cols, index=0)
        
        if df is not None:
            fig_ins, ax_ins = plt.subplots(figsize=(9, 4.5))
            sns.histplot(data=df, x=selected_feat, hue="has_heart_disease", kde=True, ax=ax_ins,
                         palette=["#2b5c8f", "#d95f02"], element="step", stat="density", common_norm=False)
            ax_ins.set_title(f"Density Distribution of {selected_feat.upper()} Stratified by Heart Disease", fontsize=12, fontweight="bold")
            st.pyplot(fig_ins)
            
    with tab2:
        fig1 = FIGURES_DIR / "target_distribution.png"
        fig3 = FIGURES_DIR / "feature_distributions.png"
        if fig1.exists():
            st.image(str(fig1), width="stretch")
        if fig3.exists():
            st.image(str(fig3), width="stretch")
            
    with tab3:
        fig_biv = FIGURES_DIR / "bivariate_panel.png"
        if fig_biv.exists():
            st.image(str(fig_biv), width="stretch")
            
    with tab4:
        fig2 = FIGURES_DIR / "correlation_matrix.png"
        if fig2.exists():
            st.image(str(fig2), width="stretch")
            
    with tab5:
        fig_pca = FIGURES_DIR / "pca_bivariate_scatter.png"
        if fig_pca.exists():
            st.image(str(fig_pca), width="stretch")
            st.caption("Figure: 2D Principal Component Analysis (PCA) projection of continuous clinical biomarker space.")

# ==========================================
# PAGE 4 — STATISTICAL ANALYSIS
# ==========================================
elif page == "4. Statistical Analysis Suite":
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
                    st.write(f"**Data Mining Implication:** `{h['dm_implication']}`")
                with col2:
                    st.metric("Test Statistic", f"{h['statistic']}")
                    p_val_display = "< 0.0001" if h['p_value_raw'] < 0.0001 else f"{h['p_value_raw']:.4f}"
                    st.metric("p-value (Raw)", p_val_display)
                    st.metric(f"Effect Size ({h['effect_size_type']})", f"{h['effect_size_val']}")
                    st.markdown(f'<span style="background:#DC2626;color:white;padding:3px 8px;border-radius:4px;font-weight:bold;">{h["decision_raw"]}</span> '
                                f'<span style="background:#059669;color:white;padding:3px 8px;border-radius:4px;font-weight:bold;">FDR Sig: {h["significant_fdr"]}</span>', unsafe_allow_html=True)
                    
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
# PAGE 6 — MODEL BENCHMARK & HYPERPARAMETERS
# ==========================================
elif page == "6. Model Benchmark & Hyperparameters":
    st.title("🤖 Supervised Classification Model Benchmark & Hyperparameters")
    
    if comparison_df is not None:
        st.subheader("1. Supervised Classifier Comparison (Test Set, N = 1,800)")
        st.dataframe(comparison_df, width="stretch")
        
        if model_metadata:
            st.success(f"**Best Performing Classifier:** {model_metadata.get('best_model_name')} "
                       f"(Accuracy: {model_metadata.get('test_metrics', {}).get('Accuracy')}, "
                       f"Weighted F1: {model_metadata.get('test_metrics', {}).get('F1 (Weighted)')}, "
                       f"ROC-AUC: {model_metadata.get('test_metrics', {}).get('ROC-AUC')})")
            
        st.subheader("2. Optimal Hyperparameters Selected via 5-Fold Stratified GridSearchCV")
        hyperparams_df = pd.DataFrame([
            {"Model": "Zero-R Baseline", "Optimal Hyperparameters": "strategy='most_frequent'", "CV Score": "0.5725"},
            {"Model": "K-Nearest Neighbors", "Optimal Hyperparameters": "metric='euclidean', n_neighbors=11, weights='uniform'", "CV Score": "0.8069"},
            {"Model": "Decision Tree", "Optimal Hyperparameters": "criterion='entropy', max_depth=7, min_samples_leaf=4, min_samples_split=10", "CV Score": "0.8561"},
            {"Model": "Random Forest", "Optimal Hyperparameters": "n_estimators=100, max_depth=10, min_samples_split=2, max_features='sqrt'", "CV Score": "0.8757"},
            {"Model": "Naive Bayes", "Optimal Hyperparameters": "var_smoothing=1e-09", "CV Score": "0.8513"},
            {"Model": "Support Vector Machine (Best)", "Optimal Hyperparameters": "C=1.0, kernel='rbf', gamma='scale', probability=True", "CV Score": "0.8954"}
        ])
        st.table(hyperparams_df)
        
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
# PAGE 7 — CATEGORIZED RISK PREDICTOR
# ==========================================
elif page == "7. Heart Disease Risk Predictor":
    st.title("🫀 Dynamic Patient Heart Disease Risk Predictor")
    
    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ <strong>EDUCATIONAL MEDICAL DISCLAIMER:</strong><br>
    This application is developed exclusively for academic Data Mining demonstration purposes.
    Its predictions must NOT be interpreted as a medical diagnosis or as a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)
    
    if best_model is not None and preprocessor_pipeline is not None:
        st.subheader("Input Patient Characteristics & Clinical Biomarkers")
        st.markdown("Features are grouped into 4 clinical categories and color-coded by importance gradient (🔴 **High Impact** $\\rightarrow$ 🟧 **Moderate-High** $\\rightarrow$ 🟨 **Moderate** $\\rightarrow$ 🟩 **Behavioral/Lifestyle**):")
        
        with st.form("patient_prediction_form"):
            # Category 1: High Impact Cardiac Risk Markers
            st.markdown("#### 🔴 Category 1: Primary Cardiac Risk Markers <span class='badge-critical'>CRITICAL IMPACT</span>", unsafe_allow_html=True)
            c1_1, c1_2, c1_3, c1_4 = st.columns(4)
            with c1_1:
                max_hr = st.number_input("Max Heart Rate (bpm)", 60, 220, 150, help="🔴 Highest negative correlation (-0.58)")
            with c1_2:
                st_dep = st.number_input("ST Depression (mm)", 0.0, 7.0, 1.2, help="🔴 Highest positive correlation (+0.36)")
            with c1_3:
                cp_type = st.selectbox("Chest Pain Type", ["Asymptomatic", "Non-Anginal Pain", "Atypical Angina", "Typical Angina"])
            with c1_4:
                ex_angina = st.checkbox("Exercise Induced Angina", value=False)
                
            st.markdown("---")
            # Category 2: Metabolic & Blood Biomarkers
            st.markdown("#### 🟧 Category 2: Metabolic & Blood Chemistry Biomarkers <span class='badge-high'>HIGH IMPACT</span>", unsafe_allow_html=True)
            c2_1, c2_2, c2_3, c2_4, c2_5, c2_6 = st.columns(6)
            with c2_1:
                ldl = st.number_input("LDL (mg/dL)", 30, 250, 125)
            with c2_2:
                hdl = st.number_input("HDL (mg/dL)", 15, 120, 48)
            with c2_3:
                chol_total = st.number_input("Total Chol (mg/dL)", 100, 400, 210)
            with c2_4:
                hba1c = st.number_input("HbA1c (%)", 4.0, 14.0, 5.8)
            with c2_5:
                fbs = st.number_input("Fasting BS (mg/dL)", 60, 250, 110)
            with c2_6:
                triglycerides = st.number_input("Triglycerides", 30, 500, 160)

            st.markdown("---")
            # Category 3: Hemodynamic & Demographic Vitals
            st.markdown("#### 🟨 Category 3: Hemodynamic Vitals & Demographics <span class='badge-moderate'>MODERATE IMPACT</span>", unsafe_allow_html=True)
            c3_1, c3_2, c3_3, c3_4, c3_5, c3_6 = st.columns(6)
            with c3_1:
                age = st.number_input("Age (years)", 18, 100, 55)
            with c3_2:
                sex = st.selectbox("Sex", ["Male", "Female"])
            with c3_3:
                sys_bp = st.number_input("Systolic BP (mmHg)", 80, 220, 130)
            with c3_4:
                dia_bp = st.number_input("Diastolic BP (mmHg)", 50, 140, 82)
            with c3_5:
                resting_hr = st.number_input("Resting HR (bpm)", 40, 130, 75)
            with c3_6:
                bmi = st.number_input("BMI (kg/m²)", 14.0, 50.0, 26.5)

            st.markdown("---")
            # Category 4: Lifestyle & Behavioral Factors
            st.markdown("#### 🟩 Category 4: Lifestyle & Behavioral Parameters <span class='badge-lifestyle'>BEHAVIORAL/LIFESTYLE</span>", unsafe_allow_html=True)
            c4_1, c4_2, c4_3, c4_4, c4_5 = st.columns(5)
            with c4_1:
                smoker = st.selectbox("Smoker Status", ["Never", "Former", "Current"])
                exercise_min = st.number_input("Exercise Min/Wk", 0, 600, 150)
            with c4_2:
                daily_steps = st.number_input("Daily Steps", 500, 25000, 6500)
                diet_score = st.number_input("Diet Score (0-100)", 0.0, 100.0, 60.0)
            with c4_3:
                stress = st.number_input("Stress Score (0-100)", 0.0, 100.0, 45.0)
                alcohol = st.number_input("Alcohol Units/Wk", 0.0, 60.0, 4.0)
            with c4_4:
                sleep_hrs = st.number_input("Sleep Hrs/Day", 3.0, 12.0, 7.0)
                wearable = st.checkbox("Wearable Owner", value=True)
            with c4_5:
                family_hist = st.checkbox("Family History", value=False)

            st.markdown("---")
            submit_btn = st.form_submit_button("🔍 Compute Heart Disease Risk Score")
            
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
            st.subheader("🎯 Clinical Prediction & Risk Gauge Output")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                if pred_class == 1:
                    st.error(f"### ⚠️ HIGH RISK PREDICTED\nEstimated Probability: **{pred_prob:.1%}**")
                else:
                    st.success(f"### ✅ LOW RISK PREDICTED\nEstimated Probability: **{pred_prob:.1%}**")
                    
            with res_col2:
                st.metric("Model Engine", model_metadata.get("best_model_name", "Support Vector Machine"))
                st.metric("Model Test ROC-AUC", f"{model_metadata.get('test_metrics', {}).get('ROC-AUC', 0.9422):.4f}")
                
            with res_col3:
                st.metric("Model Test Accuracy", f"{model_metadata.get('test_metrics', {}).get('Accuracy', 0.8917):.2%}")
                st.metric("Model Test Recall", f"{model_metadata.get('test_metrics', {}).get('Recall', 0.7670):.2%}")

# ==========================================
# PAGE 8 — KEY DISCOVERIES & FINDINGS
# ==========================================
elif page == "8. Key Discoveries & Findings":
    st.title("💡 Key Empirical Discoveries & Findings")
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
# PAGE 9 — ABOUT & ONLINE DEPLOYMENT
# ==========================================
elif page == "9. About & Online Deployment":
    st.title("ℹ️ About, Student Matrix & Online Publishing Guide")
    
    st.markdown("""
    > ⚠️ **DEMONSTRATION GIT WORKFLOW NOTICE**  
    > Published using **ONE GitHub account** for reference/demonstration purposes.
    > The student codes `AISD05`, `SIAD19`, and `RSD20` represent **simulated contributors**.
    """)
    
    st.subheader("1. Simulated Student Contribution Matrix")
    contrib_data = [
        {"Student Code": "AISD05", "Responsibility Area": "Dataset Acquisition, Quality Audit, EDA, Hypothesis Testing Suite", "Git Branch": "feature/AISD05-eda"},
        {"Student Code": "SIAD19", "Responsibility Area": "Preprocessing Pipeline, ML Models (6 Algorithms), CV & GridSearch", "Git Branch": "feature/SIAD19-modeling"},
        {"Student Code": "RSD20", "Responsibility Area": "Streamlit Web App, Predictor UI, Unit Tests, LaTeX Report & PDF", "Git Branch": "feature/RSD20-streamlit"}
    ]
    st.table(pd.DataFrame(contrib_data))
    
    st.subheader("🌐 2. How to Publish This Interactive App Online Free (Streamlit Cloud)")
    st.markdown("""
    You can publish this interactive website online for free using **Streamlit Community Cloud**:
    
    1. Sign in to **[share.streamlit.io](https://share.streamlit.io)** using your GitHub account (`hosniadilemp-a11y`).
    2. Click **"New App"** and enter your repository details:
       - **Repository:** `hosniadilemp-a11y/DataMiningProjectExample`
       - **Branch:** `main`
       - **Main file path:** `src/app.py`
    3. Click **"Deploy!"**
    
    Within 1 minute, your interactive Data Mining application will be live online with a public URL (e.g. `https://dataminingprojectexample.streamlit.app`).
    """)
    
    st.subheader("3. Software Stack & Dependencies")
    st.markdown("""
    - **Language:** Python 3.13 / 3.10
    - **Machine Learning:** `scikit-learn`, `scipy`, `numpy`, `pandas`, `pyarrow`
    - **Data Acquisition:** `kagglehub`
    - **Visualization:** `matplotlib`, `seaborn`
    - **Web Framework:** `streamlit`
    - **Report Generation:** LaTeX (`pdflatex`)
    """)
