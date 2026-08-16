"""
Visualization Module for Generating Publication-Quality Figures.
Exports high-resolution figures to docs/figures/ for LaTeX paper and Streamlit dashboards.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, confusion_matrix
from src.utils import FIGURES_DIR, NUMERICAL_COLS, TARGET_COL, load_raw_data

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#eeeeee",
    "grid.linestyle": "--",
    "figure.autolayout": True
})

PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

def plot_target_distribution(df: pd.DataFrame = None, save: bool = True) -> plt.Figure:
    """Plot target variable distribution."""
    if df is None:
        df = load_raw_data()
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["No Disease (0)", "Heart Disease (1)"]
    colors = ["#2b5c8f", "#d95f02"]
    
    # Bar Chart
    bars = ax1.bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Patient Count", fontsize=11, fontweight="bold")
    ax1.set_title("Target Class Distribution (Absolute Count)", fontsize=12, fontweight="bold", pad=10)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height:,} ({height/len(df):.1%})",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    # Donut Chart
    ax2.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90,
            colors=colors, wedgeprops=dict(width=0.4, edgecolor="w", linewidth=2))
    ax2.set_title("Target Class Ratio (Percentage)", fontsize=12, fontweight="bold", pad=10)
    
    plt.suptitle("Heart Disease Risk Target Variable Audit (N = 9,000)", fontsize=14, fontweight="bold", y=1.02)
    
    if save:
        fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=300, bbox_inches="tight")
    return fig

def plot_categorical_breakdowns(df: pd.DataFrame = None, save: bool = True) -> plt.Figure:
    """Plot stacked proportions of categorical risk factors."""
    if df is None:
        df = load_raw_data()
        
    cats = ["chest_pain_type", "smoker_status", "exercise_induced_angina", "family_history"]
    titles = ["Chest Pain Type", "Smoker Status", "Exercise Angina", "Family History"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (cat, title) in enumerate(zip(cats, titles)):
        ax = axes[i]
        ct = pd.crosstab(df[cat], df[TARGET_COL], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, color=["#2b5c8f", "#d95f02"], ax=ax, width=0.5, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=9)
        ax.set_xlabel("")
        ax.legend(["No Disease", "Heart Disease"], loc="upper right")
        ax.tick_params(axis='x', rotation=15)
        
    plt.suptitle("Categorical Predictor Proportions Stratified by Target Class", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save:
        fig.savefig(FIGURES_DIR / "categorical_breakdowns.png", dpi=300, bbox_inches="tight")
    return fig

def plot_correlation_matrix(df: pd.DataFrame = None, save: bool = True) -> plt.Figure:
    """Plot top numerical feature correlation matrix."""
    if df is None:
        df = load_raw_data()
        
    num_df = df[NUMERICAL_COLS + [TARGET_COL]]
    corr = num_df.corr()
    
    # Sort features by absolute correlation with target
    top_cols = corr[TARGET_COL].abs().sort_values(ascending=False).index[:14]
    sub_corr = corr.loc[top_cols, top_cols]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sub_corr, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                cbar_kws={"label": "Pearson Correlation Coefficient"},
                linewidths=0.5, linecolor="#ffffff")
    ax.set_title("Top Clinical Feature Correlations with Heart Disease Risk", fontsize=13, fontweight="bold", pad=12)
    
    if save:
        fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=300, bbox_inches="tight")
    return fig

def plot_feature_distributions(df: pd.DataFrame = None, save: bool = True) -> plt.Figure:
    """Plot histograms/KDEs of key clinical risk predictors."""
    if df is None:
        df = load_raw_data()
        
    key_features = ["age", "st_depression", "max_heart_rate_achieved", "ldl", "hba1c", "resting_bp_systolic"]
    titles = ["Age (years)", "ST Depression (mm)", "Max Heart Rate (bpm)", "LDL Cholesterol (mg/dL)", "HbA1c (%)", "Systolic BP (mmHg)"]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, (feat, title) in enumerate(zip(key_features, titles)):
        ax = axes[i]
        sns.histplot(data=df, x=feat, hue=TARGET_COL, kde=True, ax=ax,
                     palette=["#2b5c8f", "#d95f02"], element="step", stat="density", common_norm=False)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=9)
        
    plt.suptitle("Clinical Variable Distributions Stratified by Target Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    
    if save:
        fig.savefig(FIGURES_DIR / "feature_distributions.png", dpi=300, bbox_inches="tight")
    return fig

def plot_bivariate_panel(df: pd.DataFrame = None, save: bool = True) -> plt.Figure:
    """Plot 2x2 panel boxplots of top clinical predictors."""
    if df is None:
        df = load_raw_data()
        
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    
    predictors = ["st_depression", "max_heart_rate_achieved", "ldl", "age"]
    labels = ["ST Depression (mm)", "Max Heart Rate (bpm)", "LDL Cholesterol (mg/dL)", "Age (years)"]
    
    for i, (pred, label) in enumerate(zip(predictors, labels)):
        ax = axes[i]
        sns.boxplot(data=df, x=TARGET_COL, y=pred, hue=TARGET_COL, palette=["#2b5c8f", "#d95f02"], ax=ax, legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Disease", "Heart Disease"])
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        
    plt.suptitle("Bivariate Boxplots of Key Risk Factors", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save:
        fig.savefig(FIGURES_DIR / "bivariate_panel.png", dpi=300, bbox_inches="tight")
    return fig

def plot_pca_scatter(save: bool = True) -> plt.Figure:
    """Plot 2D PCA projection of clinical patient feature space."""
    from src.utils import DATA_PROCESSED
    X_train_path = DATA_PROCESSED / "X_train.csv"
    y_train_path = DATA_PROCESSED / "y_train.csv"
    
    if not X_train_path.exists():
        return None
        
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train)
    var_exp = pca.explained_variance_ratio_
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=matplotlib.colors.ListedColormap(["#2b5c8f", "#d95f02"]),
                         alpha=0.6, s=15, edgecolors="none")
    
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1%} var)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1%} var)", fontsize=11, fontweight="bold")
    ax.set_title(f"PCA 2D Projection of Patient Feature Space (Cumulative Var: {sum(var_exp):.1%})", fontsize=12, fontweight="bold", pad=10)
    
    # Custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8) for c in ["#2b5c8f", "#d95f02"]]
    ax.legend(handles, ["No Disease (0)", "Heart Disease (1)"], loc="best", frameon=True)
    
    if save:
        fig.savefig(FIGURES_DIR / "pca_bivariate_scatter.png", dpi=300, bbox_inches="tight")
    return fig

def plot_feature_importance_comparison(save: bool = True) -> plt.Figure:
    """Plot side-by-side comparison of Gini vs Permutation vs Mutual Information."""
    from src.utils import PROJECT_ROOT
    fi_path = PROJECT_ROOT / "results" / "feature_importances.csv"
    
    if not fi_path.exists():
        return None
        
    fi_df = pd.read_csv(fi_path).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(fi_df))
    width = 0.25
    
    ax.barh(x - width, fi_df["MI Normalized"], width, label="Mutual Information", color="#2b5c8f")
    ax.barh(x, fi_df["Gini Normalized"], width, label="Random Forest Gini", color="#7570b3")
    ax.barh(x + width, fi_df["Permutation Normalized"], width, label="Permutation Importance", color="#d95f02")
    
    ax.set_yticks(x)
    ax.set_yticklabels(fi_df["Feature"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Relative Importance", fontsize=11, fontweight="bold")
    ax.set_title("Multi-Perspective Feature Importance Comparison (Top 10 Attributes)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    
    if save:
        fig.savefig(FIGURES_DIR / "feature_importance_comp.png", dpi=300, bbox_inches="tight")
    return fig

def plot_roc_curves(best_estimators: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, save: bool = True) -> plt.Figure:
    """Plot ROC curves for all 6 models."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    for i, (name, model) in enumerate(best_estimators.items()):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = model.predict(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, lw=2, color=PALETTE[i % len(PALETTE)],
                label=f"{name} (AUC = {roc_auc:.3f})")
        
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Benchmark Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    
    if save:
        fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=300, bbox_inches="tight")
    return fig

def plot_confusion_matrices(best_estimators: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, save: bool = True) -> plt.Figure:
    """Plot 2x3 grid of confusion matrices for all 6 models."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, (name, model) in enumerate(best_estimators.items()):
        ax = axes[i]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"])
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)
        
    plt.suptitle("Confusion Matrix Grid across 6 Classification Algorithms", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    
    if save:
        fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    return fig
