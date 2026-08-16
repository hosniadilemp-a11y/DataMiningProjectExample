"""
Streamlit Web Application for Heart Disease Risk Prediction.
Interactive Data Mining Academic Dashboard containing 9 dedicated pages:
1. Home & Executive Summary (Academic Objectives & Formal Metrics Framework)
2. Dataset Explorer & Quality Audit (Interactive Filtering & Breakdown)
3. EDA Dashboard (Univariate, Bivariate, Categorical Panels, Correlations, PCA)
4. Statistical Analysis Suite (FDR Corrected & Effect Sizes)
5. Multi-Perspective Feature Importance Comparison
6. Model Benchmark & Hyperparameters Transparency
7. Categorized & Importance-Color-Coded Heart Disease Risk Predictor (Top Output & Interactive Sliders)
8. Key Discoveries & Findings
9. About & Student Contribution Matrix
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
    .param-badge {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: 600;
        color: #334155;
        display: inline-block;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
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
        "9. About & Student Matrix"
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
        - **Data Quality & Schema Audit:** Audit 9,000 patient records (0 missing values, 0 duplicates) for target imbalance (30.3% positive class prevalence).
        - **Leakage-Free Preprocessing:** Standardize continuous features (mean=0, variance=1) and One-Hot encode categories via `ColumnTransformer` inside training pipelines.
        - **Statistical Hypothesis Suite:** Conduct 6 non-parametric tests (significance threshold α = 0.05) incorporating Benjamini-Hochberg FDR correction and effect sizes (Cramér's V, Cohen's d, Odds Ratio).
        - **Multi-Classifier Benchmark:** Optimize and evaluate 6 algorithms (Zero-R, KNN, Decision Tree, Random Forest, Naive Bayes, SVM) via 5-Fold Stratified Cross-Validation.
        """)
        
    with col2:
        st.subheader("📐 Evaluation Metrics Framework")
        st.markdown("""
        To evaluate model performance under moderate class imbalance (2.3:1 ratio), we utilize formal evaluation criteria:
        
        - **Overall Accuracy:** Measures overall correctness across healthy and diseased patients (89.17%).
        - **Precision (Positive Predictive Value):** Proportion of true disease cases among all positive predictions (86.01%).
        - **Recall / Sensitivity (True Positive Rate):** Proportion of actual heart disease cases correctly caught (76.70%).
        - **Specificity (True Negative Rate):** Proportion of healthy individuals correctly identified (94.58%).
        - **Weighted F1-Score:** Harmonic mean of precision and recall accounting for class proportions (0.8898).
        - **ROC-AUC (Area Under ROC Curve):** Overall discrimination capacity across all probability thresholds (0.9422).
        - **5-Fold Stratified Cross-Validation Score:** Cross-validated training benchmark (0.8954 ± 0.0047).
        """)

# ==========================================
# PAGE 2 — DATASET EXPLORER
# ==========================================
elif page == "2. Dataset Explorer":
    st.title("📂 Dataset Explorer & Quality Audit")
    
    if df is not None:
        st.markdown("""
        <div class="finding-box">
        <strong>Dataset Quality & Size Audit:</strong><br>
        Audited dataset <code>uditjain13/heart-disease-risk-2026</code> contains <strong>9,000 patient records</strong> and <strong>27 columns</strong> (1 identifier <code>patient_id</code>, 25 feature attributes, 1 binary target <code>has_heart_disease</code>).
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Interactive Data Filter",
            "📊 Categorical Attribute Inspector",
            "📋 Data Dictionary & Types",
            "📈 Summary Statistics"
        ])
        
        with tab1:
            st.subheader("Interactive Patient Data Filter")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                target_filter = st.multiselect("Filter by Target Class", [0, 1], default=[0, 1], format_func=lambda x: "Heart Disease (1)" if x == 1 else "No Disease (0)")
            with f_col2:
                age_range = st.slider("Age Range Filter", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
            with f_col3:
                search_query = st.text_input("Search Patient Records", "")
                
            filtered_df = df[
                (df["has_heart_disease"].isin(target_filter)) &
                (df["age"] >= age_range[0]) &
                (df["age"] <= age_range[1])
            ]
            
            if search_query:
                filtered_df = filtered_df[filtered_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)]
                
            st.write(f"Showing **{len(filtered_df):,}** matching patient records out of **{len(df):,}** total:")
            st.dataframe(filtered_df.head(100), width="stretch")
            
        with tab2:
            st.subheader("Categorical Value Counts & Proportions")
            cat_col = st.selectbox("Select Categorical Attribute to Analyze:", ["chest_pain_type", "smoker_status", "exercise_induced_angina", "family_history", "sex"])
            
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                st.dataframe(df[cat_col].value_counts().reset_index().rename(columns={"index": cat_col, "count": "Patient Count"}), width="stretch")
            with c_col2:
                fig_cat, ax_cat = plt.subplots(figsize=(6, 4))
                df[cat_col].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=sns.color_palette("Set2"), ax=ax_cat)
                ax_cat.set_ylabel("")
                ax_cat.set_title(f"Percentage Distribution: {cat_col}", fontsize=11, fontweight="bold")
                st.pyplot(fig_cat)
                
        with tab3:
            dtype_df = pd.DataFrame({
                "Column Name": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Missing Values": df.isnull().sum(),
                "Unique Values": df.nunique(),
                "Sample Value": [df[c].iloc[0] for c in df.columns]
            })
            st.dataframe(dtype_df, width="stretch")
            
        with tab4:
            st.dataframe(df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]], width="stretch")

# ==========================================
# PAGE 3 — EDA DASHBOARD
# ==========================================
elif page == "3. EDA Dashboard":
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Density Inspector",
        "Categorical Proportions",
        "Target Class Audit",
        "Bivariate Boxplots",
        "Correlation Heatmap",
        "PCA 2D Projection"
    ])
    
    with tab1:
        st.subheader("Interactive Feature Density Plot")
        num_cols = ["age", "st_depression", "max_heart_rate_achieved", "ldl", "hdl", "hba1c", "resting_bp_systolic", "bmi", "stress_score"]
        selected_feat = st.selectbox("Select Continuous Clinical Biomarker:", num_cols, index=0)
        
        if df is not None:
            fig_ins, ax_ins = plt.subplots(figsize=(9, 4.5))
            sns.histplot(data=df, x=selected_feat, hue="has_heart_disease", kde=True, ax=ax_ins,
                         palette=["#2b5c8f", "#d95f02"], element="step", stat="density", common_norm=False)
            ax_ins.set_title(f"Density Distribution of {selected_feat.upper()} Stratified by Target Class", fontsize=12, fontweight="bold")
            st.pyplot(fig_ins)
            
    with tab2:
        fig_cat_brk = FIGURES_DIR / "categorical_breakdowns.png"
        if fig_cat_brk.exists():
            st.image(str(fig_cat_brk), width="stretch")
            
    with tab3:
        fig1 = FIGURES_DIR / "target_distribution.png"
        fig3 = FIGURES_DIR / "feature_distributions.png"
        if fig1.exists():
            st.image(str(fig1), width="stretch")
        if fig3.exists():
            st.image(str(fig3), width="stretch")
            
    with tab4:
        fig_biv = FIGURES_DIR / "bivariate_panel.png"
        if fig_biv.exists():
            st.image(str(fig_biv), width="stretch")
            
    with tab5:
        fig2 = FIGURES_DIR / "correlation_matrix.png"
        if fig2.exists():
            st.image(str(fig2), width="stretch")
            
    with tab6:
        fig_pca = FIGURES_DIR / "pca_bivariate_scatter.png"
        if fig_pca.exists():
            st.image(str(fig_pca), width="stretch")

# ==========================================
# PAGE 4 — STATISTICAL ANALYSIS
# ==========================================
elif page == "4. Statistical Analysis Suite":
    st.title("🧪 Statistical Hypothesis Testing Suite")
    st.markdown("""
    Rigorous statistical hypothesis tests conducted with significance threshold **α = 0.05**.
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
        st.subheader("Summary Table of Statistical Hypotheses (α = 0.05)")
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
            
        st.subheader("2. Optimal Model Hyperparameters Selected via 5-Fold Stratified Cross-Validation")
        
        models_params = [
            {
                "Model": "Zero-R Baseline",
                "Parameters": [("Strategy", "Most Frequent Majority Class")],
                "CV Score": "0.5725"
            },
            {
                "Model": "K-Nearest Neighbors",
                "Parameters": [("Distance Metric", "Euclidean"), ("Neighbors (k)", "11"), ("Weights", "Uniform")],
                "CV Score": "0.8069"
            },
            {
                "Model": "Decision Tree",
                "Parameters": [("Split Criterion", "Entropy / Information Gain"), ("Max Depth", "7"), ("Min Samples Leaf", "4"), ("Min Samples Split", "10")],
                "CV Score": "0.8561"
            },
            {
                "Model": "Random Forest",
                "Parameters": [("Estimators", "100 Trees"), ("Max Depth", "10 Trees"), ("Max Features", "Square Root (sqrt)"), ("Min Samples Split", "2")],
                "CV Score": "0.8757"
            },
            {
                "Model": "Naive Bayes",
                "Parameters": [("Model Variant", "Gaussian Naive Bayes"), ("Variance Smoothing", "1e-09")],
                "CV Score": "0.8513"
            },
            {
                "Model": "Support Vector Machine (Best Classifier)",
                "Parameters": [("Kernel Function", "Radial Basis Function (RBF)"), ("Regularization (C)", "1.0"), ("Gamma Scaling", "Scale"), ("Probability Estimation", "Enabled (Platt Scaling)")],
                "CV Score": "0.8954"
            }
        ]
        
        for mp in models_params:
            with st.expander(f"⚙️ {mp['Model']} — 5-Fold CV Score: {mp['CV Score']}", expanded=(mp['Model'].startswith("Support Vector"))):
                st.write("**Optimal Hyperparameter Configuration:**")
                badges_html = "".join([f'<span class="param-badge"><strong>{k}:</strong> {v}</span>' for k, v in mp["Parameters"]])
                st.markdown(badges_html, unsafe_allow_html=True)
                
        st.markdown("---")
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
# PAGE 7 — CATEGORIZED DYNAMIC RISK PREDICTOR WITH SLIDERS
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
        
        # -------------------------------------------------------------
        # TOP CONTAINER FOR REAL-TIME DYNAMIC PREDICTION OUTPUT
        # -------------------------------------------------------------
        top_prediction_card = st.container()
        st.markdown("---")
        
        # -------------------------------------------------------------
        # INTERACTIVE SLIDER CONTROLS (REAL-TIME INSTANT DYNAMIC UPDATE)
        # -------------------------------------------------------------
        st.subheader("🎛️ Interactive Patient Clinical Parameter Sliders")
        st.markdown("Drag any slider to observe real-time risk probability changes at the top of the page. Features are categorized by importance gradient (🔴 **Critical** $\\rightarrow$ 🟧 **High** $\\rightarrow$ 🟨 **Moderate** $\\rightarrow$ 🟩 **Lifestyle**):")
        
        # Category 1: High Impact Cardiac Risk Markers
        st.markdown("#### 🔴 Category 1: Primary Cardiac Risk Markers <span class='badge-critical'>CRITICAL IMPACT</span>", unsafe_allow_html=True)
        c1_1, c1_2, c1_3, c1_4 = st.columns(4)
        with c1_1:
            max_hr = st.slider("Max Heart Rate (bpm)", 60, 220, 150, key="p7_max_hr", help="🔴 Highest negative correlation (-0.58)")
        with c1_2:
            st_dep = st.slider("ST Depression (mm)", 0.0, 7.0, 1.2, step=0.1, key="p7_st_dep", help="🔴 Highest positive correlation (+0.36)")
        with c1_3:
            cp_type = st.selectbox("Chest Pain Type", ["Asymptomatic", "Non-Anginal Pain", "Atypical Angina", "Typical Angina"], key="p7_cp_type")
        with c1_4:
            ex_angina = st.checkbox("Exercise Induced Angina", value=False, key="p7_ex_angina")
            
        st.markdown("---")
        # Category 2: Metabolic & Blood Biomarkers
        st.markdown("#### 🟧 Category 2: Metabolic & Blood Chemistry Biomarkers <span class='badge-high'>HIGH IMPACT</span>", unsafe_allow_html=True)
        c2_1, c2_2, c2_3, c2_4, c2_5, c2_6 = st.columns(6)
        with c2_1:
            ldl = st.slider("LDL (mg/dL)", 30, 250, 125, key="p7_ldl")
        with c2_2:
            hdl = st.slider("HDL (mg/dL)", 15, 120, 48, key="p7_hdl")
        with c2_3:
            chol_total = st.slider("Total Chol (mg/dL)", 100, 400, 210, key="p7_chol_total")
        with c2_4:
            hba1c = st.slider("HbA1c (%)", 4.0, 14.0, 5.8, step=0.1, key="p7_hba1c")
        with c2_5:
            fbs = st.slider("Fasting BS (mg/dL)", 60, 250, 110, key="p7_fbs")
        with c2_6:
            triglycerides = st.slider("Triglycerides", 30, 500, 160, key="p7_triglycerides")

        st.markdown("---")
        # Category 3: Hemodynamic & Demographic Vitals
        st.markdown("#### 🟨 Category 3: Hemodynamic Vitals & Demographics <span class='badge-moderate'>MODERATE IMPACT</span>", unsafe_allow_html=True)
        c3_1, c3_2, c3_3, c3_4, c3_5, c3_6 = st.columns(6)
        with c3_1:
            age = st.slider("Age (years)", 18, 100, 55, key="p7_age")
        with c3_2:
            sex = st.selectbox("Sex", ["Male", "Female"], key="p7_sex")
        with c3_3:
            sys_bp = st.slider("Systolic BP (mmHg)", 80, 220, 130, key="p7_sys_bp")
        with c3_4:
            dia_bp = st.slider("Diastolic BP (mmHg)", 50, 140, 82, key="p7_dia_bp")
        with c3_5:
            resting_hr = st.slider("Resting HR (bpm)", 40, 130, 75, key="p7_resting_hr")
        with c3_6:
            bmi = st.slider("BMI (kg/m²)", 14.0, 50.0, 26.5, step=0.1, key="p7_bmi")

        st.markdown("---")
        # Category 4: Lifestyle & Behavioral Factors
        st.markdown("#### 🟩 Category 4: Lifestyle & Behavioral Parameters <span class='badge-lifestyle'>BEHAVIORAL/LIFESTYLE</span>", unsafe_allow_html=True)
        c4_1, c4_2, c4_3, c4_4, c4_5 = st.columns(5)
        with c4_1:
            smoker = st.selectbox("Smoker Status", ["Never", "Former", "Current"], key="p7_smoker")
            exercise_min = st.slider("Exercise Min/Wk", 0, 600, 150, key="p7_exercise_min")
        with c4_2:
            daily_steps = st.slider("Daily Steps", 500, 25000, 6500, step=250, key="p7_daily_steps")
            diet_score = st.slider("Diet Score (0-100)", 0, 100, 60, key="p7_diet_score")
        with c4_3:
            stress = st.slider("Stress Score (0-100)", 0, 100, 45, key="p7_stress")
            alcohol = st.slider("Alcohol Units/Wk", 0.0, 60.0, 4.0, step=0.5, key="p7_alcohol")
        with c4_4:
            sleep_hrs = st.slider("Sleep Hrs/Day", 3.0, 12.0, 7.0, step=0.5, key="p7_sleep_hrs")
            wearable = st.checkbox("Wearable Owner", value=True, key="p7_wearable")
        with c4_5:
            family_hist = st.checkbox("Family History", value=False, key="p7_family_hist")

        # -------------------------------------------------------------
        # COMPUTE DYNAMIC REAL-TIME PREDICTION & POPULATE TOP CONTAINER
        # -------------------------------------------------------------
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
            
        with top_prediction_card:
            st.subheader("🎯 Real-Time Clinical Risk Prediction & Gauge Output")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                if pred_class == 1:
                    st.error(f"### ⚠️ HIGH RISK PREDICTED\nEstimated Risk Probability: **{pred_prob:.1%}**")
                else:
                    st.success(f"### ✅ LOW RISK PREDICTED\nEstimated Risk Probability: **{pred_prob:.1%}**")
                    
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
# PAGE 9 — ABOUT & STUDENT MATRIX
# ==========================================
elif page == "9. About & Student Matrix":
    st.title("ℹ️ About & Student Contribution Matrix")
    
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
    
    st.subheader("2. Software Stack & Dependencies")
    st.markdown("""
    - **Language:** Python 3.13 / 3.10
    - **Machine Learning:** `scikit-learn`, `scipy`, `numpy`, `pandas`, `pyarrow`
    - **Data Acquisition:** `kagglehub`
    - **Visualization:** `matplotlib`, `seaborn`
    - **Web Framework:** `streamlit`
    - **Report Generation:** LaTeX (`pdflatex`)
    """)
