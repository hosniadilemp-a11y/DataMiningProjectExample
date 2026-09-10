"""
Interactive HTML Report Generator for the Heart Attack Classification Benchmark.
Renders an executive, responsive dashboard with rich aesthetics, interactive charts,
side-by-side balanced vs unbalanced comparison, and pedagogical course insights.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from .preprocess import DatasetProfile
from .evaluate import ModelEvaluationResult


class ReportGenerator:
    """Generates a standalone, rich HTML report."""

    def __init__(self, profile: DatasetProfile, benchmark_results: Dict[str, Dict[str, ModelEvaluationResult]]):
        self.profile = profile
        self.results = benchmark_results

    def _serialize_results_to_json(self) -> str:
        """Serializes results structure into JSON for client-side Chart.js rendering."""
        serializable = {"unbalanced": {}, "balanced": {}}

        for scenario in ["unbalanced", "balanced"]:
            for name, res in self.results[scenario].items():
                serializable[scenario][name] = {
                    "model_name": res.model_name,
                    "tier": res.tier,
                    "type": res.model_type,
                    "description": res.description,
                    "course_note": res.course_note,
                    "fit_time": res.fit_time_seconds,
                    "accuracy": {"mean": round(res.accuracy.mean, 4), "std": round(res.accuracy.std, 4), "str": res.accuracy.formatted},
                    "precision": {"mean": round(res.precision.mean, 4), "std": round(res.precision.std, 4), "str": res.precision.formatted},
                    "recall": {"mean": round(res.recall.mean, 4), "std": round(res.recall.std, 4), "str": res.recall.formatted},
                    "f1_score": {"mean": round(res.f1_score.mean, 4), "std": round(res.f1_score.std, 4), "str": res.f1_score.formatted},
                    "precision_macro": {"mean": round(res.precision_macro.mean, 4), "std": round(res.precision_macro.std, 4), "str": res.precision_macro.formatted},
                    "recall_macro": {"mean": round(res.recall_macro.mean, 4), "std": round(res.recall_macro.std, 4), "str": res.recall_macro.formatted},
                    "f1_macro": {"mean": round(res.f1_macro.mean, 4), "std": round(res.f1_macro.std, 4), "str": res.f1_macro.formatted},
                    "roc_auc": {"mean": round(res.roc_auc.mean, 4), "std": round(res.roc_auc.std, 4), "str": res.roc_auc.formatted},
                    "ap": {"mean": round(res.average_precision.mean, 4), "std": round(res.average_precision.std, 4), "str": res.average_precision.formatted},
                    "confusion_matrix": res.confusion_matrix_total,
                    "roc_curve": res.roc_curve_data,
                    "pr_curve": res.pr_curve_data,
                    "feature_importances": res.feature_importances,
                }
        return json.dumps(serializable)

    def _serialize_profile_to_json(self) -> str:
        """Serializes dataset profile metadata."""
        p_dict = {
            "name": self.profile.name,
            "total_samples": self.profile.total_samples,
            "num_features": self.profile.num_features,
            "numeric_count": len(self.profile.numeric_features),
            "categorical_count": len(self.profile.categorical_features),
            "target_column": self.profile.target_column,
            "distribution": self.profile.target_distribution,
            "percentages": self.profile.target_percentages,
            "is_imbalanced": self.profile.is_imbalanced,
            "imbalance_ratio": self.profile.imbalance_ratio,
            "missing_count": self.profile.missing_count,
            "duplicate_count": self.profile.duplicate_count,
        }
        return json.dumps(p_dict)

    def generate_html(self, output_filepath: Path) -> Path:
        """Generates the full HTML file and writes to disk."""
        data_json = self._serialize_results_to_json()
        profile_json = self._serialize_profile_to_json()

        # Find best models for unbalanced and balanced
        unbal_res = self.results["unbalanced"]
        bal_res = self.results["balanced"]

        best_unbal_f1 = max(unbal_res.values(), key=lambda r: r.f1_score.mean)
        best_bal_recall = max(bal_res.values(), key=lambda r: r.recall.mean)
        zero_r = unbal_res.get("Zero-R (Majoritaire)")

        html_content = f"""<!DOCTYPE html>
<html lang="fr" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Heart Attack Classification Benchmark & Data Mining Hierarchy</title>
  <meta name="description" content="Pipeline de préparation de données, équilibrage SMOTE, et benchmark supervisé multi-modèles (Aléatoire -> Zero-R -> One-R -> Modèles Avancés) avec validation croisée 5-Fold.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

  <style>
    :root {{
      --bg-base: #0b0f19;
      --bg-surface: #111827;
      --bg-surface-elevated: #1e293b;
      --bg-card: rgba(255, 255, 255, 0.03);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(239, 68, 68, 0.3);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --primary: #ef4444;
      --primary-glow: rgba(239, 68, 68, 0.25);
      --secondary: #06b6d4;
      --accent: #8b5cf6;
      --success: #10b981;
      --warning: #f59e0b;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --shadow-card: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
      --font-display: 'Outfit', sans-serif;
      --font-body: 'Inter', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}

    [data-theme="light"] {{
      --bg-base: #f8fafc;
      --bg-surface: #ffffff;
      --bg-surface-elevated: #f1f5f9;
      --bg-card: rgba(0, 0, 0, 0.02);
      --border-subtle: rgba(0, 0, 0, 0.08);
      --border-accent: rgba(239, 68, 68, 0.4);
      --text-main: #0f172a;
      --text-muted: #475569;
      --text-dim: #94a3b8;
      --primary-glow: rgba(239, 68, 68, 0.15);
      --shadow-card: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: var(--font-body);
      background-color: var(--bg-base);
      color: var(--text-main);
      line-height: 1.6;
      padding-bottom: 60px;
      transition: background-color 0.3s ease, color 0.3s ease;
    }}

    /* Container */
    .container {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 0 24px;
    }}

    /* Header */
    header {{
      background: linear-gradient(180deg, rgba(239, 68, 68, 0.08) 0%, transparent 100%);
      border-bottom: 1px solid var(--border-subtle);
      padding: 36px 0 24px 0;
      margin-bottom: 32px;
    }}

    .header-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 16px;
    }}

    .badge-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 600;
      background: rgba(239, 68, 68, 0.12);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.25);
    }}

    .badge-pill svg {{
      animation: heartbeat 1.4s ease-in-out infinite;
    }}

    @keyframes heartbeat {{
      0%, 100% {{ transform: scale(1); }}
      15% {{ transform: scale(1.25); }}
      30% {{ transform: scale(1); }}
      45% {{ transform: scale(1.18); }}
    }}

    h1 {{
      font-family: var(--font-display);
      font-size: 2.3rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 40%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }}

    [data-theme="light"] h1 {{
      background: linear-gradient(135deg, #0f172a 40%, #334155 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .subtitle {{
      color: var(--text-muted);
      font-size: 1.05rem;
      max-width: 900px;
    }}

    .theme-toggle-btn {{
      background: var(--bg-surface-elevated);
      color: var(--text-main);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 8px 14px;
      font-size: 0.88rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
    }}

    .theme-toggle-btn:hover {{
      border-color: var(--primary);
    }}

    /* Hierarchy Banner */
    .hierarchy-strip {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 18px 24px;
      margin-bottom: 30px;
      box-shadow: var(--shadow-card);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 16px;
    }}

    .hierarchy-title {{
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-dim);
      font-weight: 700;
    }}

    .hierarchy-flow {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      font-size: 0.95rem;
      font-weight: 600;
    }}

    .flow-node {{
      padding: 6px 14px;
      border-radius: var(--radius-sm);
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .flow-node.baseline {{
      border-color: rgba(148, 163, 184, 0.4);
      color: #94a3b8;
    }}

    .flow-node.zero-r {{
      border-color: rgba(245, 158, 11, 0.4);
      background: rgba(245, 158, 11, 0.08);
      color: #fbbf24;
    }}

    .flow-node.one-r {{
      border-color: rgba(6, 182, 212, 0.4);
      background: rgba(6, 182, 212, 0.08);
      color: #38bdf8;
    }}

    .flow-node.advanced {{
      border-color: rgba(139, 92, 246, 0.4);
      background: rgba(139, 92, 246, 0.08);
      color: #c084fc;
    }}

    .flow-arrow {{
      color: var(--text-dim);
      font-weight: bold;
    }}

    /* KPI Cards */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 18px;
      margin-bottom: 32px;
    }}

    .kpi-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 20px;
      position: relative;
      overflow: hidden;
      box-shadow: var(--shadow-card);
      transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .kpi-card:hover {{
      transform: translateY(-3px);
      border-color: rgba(255, 255, 255, 0.15);
    }}

    .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--primary), var(--secondary));
    }}

    .kpi-label {{
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
      font-weight: 600;
      margin-bottom: 6px;
    }}

    .kpi-value {{
      font-family: var(--font-display);
      font-size: 1.85rem;
      font-weight: 800;
      color: var(--text-main);
      display: flex;
      align-items: baseline;
      gap: 6px;
    }}

    .kpi-subtext {{
      font-size: 0.82rem;
      color: var(--text-muted);
      margin-top: 6px;
    }}

    /* Tabs Navigation */
    .nav-tabs {{
      display: flex;
      gap: 10px;
      border-bottom: 1px solid var(--border-subtle);
      margin-bottom: 28px;
      overflow-x: auto;
      padding-bottom: 4px;
    }}

    .tab-btn {{
      padding: 10px 18px;
      background: transparent;
      color: var(--text-muted);
      border: none;
      border-bottom: 2px solid transparent;
      font-weight: 600;
      font-size: 0.92rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
      transition: all 0.2s ease;
    }}

    .tab-btn:hover {{
      color: var(--text-main);
    }}

    .tab-btn.active {{
      color: var(--primary);
      border-bottom-color: var(--primary);
      background: rgba(239, 68, 68, 0.05);
      border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    }}

    .tab-panel {{
      display: none;
      animation: fadeIn 0.3s ease forwards;
    }}

    .tab-panel.active {{
      display: block;
    }}

    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Control Switchers */
    .switcher-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 14px;
      background: var(--bg-surface-elevated);
      padding: 12px 18px;
      border-radius: var(--radius-md);
      margin-bottom: 22px;
      border: 1px solid var(--border-subtle);
    }}

    .scenario-toggle {{
      display: inline-flex;
      background: var(--bg-surface);
      border-radius: var(--radius-sm);
      padding: 4px;
      border: 1px solid var(--border-subtle);
    }}

    .scenario-btn {{
      border: none;
      background: transparent;
      color: var(--text-muted);
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 0.86rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }}

    .scenario-btn.active {{
      background: var(--primary);
      color: #ffffff;
      box-shadow: 0 2px 8px var(--primary-glow);
    }}

    /* Card styling */
    .card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-card);
    }}

    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
      flex-wrap: wrap;
      gap: 12px;
    }}

    .card-title {{
      font-family: var(--font-display);
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .card-desc {{
      color: var(--text-muted);
      font-size: 0.88rem;
      margin-top: 4px;
    }}

    /* Grid layouts */
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 24px;
    }}

    /* Table */
    .table-container {{
      overflow-x: auto;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-subtle);
    }}

    table.benchmark-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
      text-align: left;
    }}

    table.benchmark-table th {{
      background: var(--bg-surface-elevated);
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 0.76rem;
      padding: 14px 16px;
      border-bottom: 1px solid var(--border-subtle);
      white-space: nowrap;
      cursor: pointer;
      user-select: none;
    }}

    table.benchmark-table th:hover {{
      color: var(--text-main);
    }}

    table.benchmark-table td {{
      padding: 13px 16px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-main);
      white-space: nowrap;
    }}

    table.benchmark-table tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
    }}

    .score-cell {{
      font-family: var(--font-mono);
      font-size: 0.85rem;
    }}

    .score-std {{
      color: var(--text-dim);
      font-size: 0.78rem;
    }}

    .best-score {{
      color: var(--success);
      font-weight: 700;
      background: rgba(16, 185, 129, 0.08);
      padding: 3px 6px;
      border-radius: 4px;
      border: 1px solid rgba(16, 185, 129, 0.2);
    }}

    .baseline-score {{
      color: var(--warning);
      font-style: italic;
    }}

    .tier-badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}

    .tier-baseline {{ background: rgba(148, 163, 184, 0.15); color: #cbd5e1; }}
    .tier-simple {{ background: rgba(6, 182, 212, 0.15); color: #38bdf8; }}
    .tier-linear {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; }}
    .tier-ensemble {{ background: rgba(139, 92, 246, 0.15); color: #c084fc; }}
    .tier-stacking {{ background: rgba(236, 72, 153, 0.15); color: #f472b6; }}

    /* Educational Alert Boxes */
    .edu-alert {{
      border-left: 4px solid var(--primary);
      background: rgba(239, 68, 68, 0.06);
      padding: 16px 20px;
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
      margin-bottom: 22px;
      font-size: 0.9rem;
    }}

    .edu-alert.warning {{
      border-left-color: var(--warning);
      background: rgba(245, 158, 11, 0.06);
    }}

    .edu-alert.info {{
      border-left-color: var(--secondary);
      background: rgba(6, 182, 212, 0.06);
    }}

    .edu-alert.success {{
      border-left-color: var(--success);
      background: rgba(16, 185, 129, 0.06);
    }}

    .edu-alert-title {{
      font-weight: 700;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    /* Confusion matrix styling */
    .cm-wrapper {{
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
    }}

    .cm-card {{
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 16px;
      flex: 1;
      min-width: 280px;
    }}

    .cm-grid {{
      display: grid;
      grid-template-columns: auto 1fr 1fr;
      gap: 6px;
      margin-top: 12px;
      font-size: 0.84rem;
      text-align: center;
    }}

    .cm-header {{
      font-weight: 600;
      color: var(--text-dim);
      padding: 6px;
    }}

    .cm-box {{
      padding: 14px 10px;
      border-radius: 6px;
      font-family: var(--font-mono);
      font-size: 1rem;
      font-weight: 700;
    }}

    .cm-tn {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .cm-tp {{ background: rgba(6, 182, 212, 0.15); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); }}
    .cm-fp {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
    .cm-fn {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}

    /* Canvas container */
    .chart-container {{
      position: relative;
      height: 380px;
      width: 100%;
    }}

    /* Feature Importance Bars */
    .fi-item {{
      margin-bottom: 12px;
    }}

    .fi-label-row {{
      display: flex;
      justify-content: space-between;
      font-size: 0.84rem;
      font-weight: 500;
      margin-bottom: 4px;
    }}

    .fi-bar-bg {{
      width: 100%;
      height: 8px;
      background: var(--bg-surface-elevated);
      border-radius: 4px;
      overflow: hidden;
    }}

    .fi-bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--secondary), var(--primary));
      border-radius: 4px;
      transition: width 0.6s ease;
    }}

    /* Footer */
    footer {{
      margin-top: 50px;
      border-top: 1px solid var(--border-subtle);
      padding-top: 24px;
      text-align: center;
      color: var(--text-dim);
      font-size: 0.85rem;
    }}
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="container">
      <div class="header-top">
        <div class="badge-pill">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
          </svg>
          Cardiology & Data Mining Benchmark
        </div>
        <button id="themeToggle" class="theme-toggle-btn" onclick="toggleTheme()">
          🌓 Mode Clair / Sombre
        </button>
      </div>

      <h1>Prédiction des Crises Cardiaques & Modélisation Supervisée</h1>
      <p class="subtitle">
        Pipeline complet de prétraitement, étude du déséquilibre (Brut vs SMOTE), progression hiérarchique du cours
        (Aléatoire ➔ Zero-R ➔ One-R ➔ Modèles Avancés ➔ Méta-Ensemble Stacking) validée par <strong>5-Fold Stratified Cross-Validation</strong>.
      </p>
    </div>
  </header>

  <div class="container">

    <!-- Hierarchy Visual Flow -->
    <div class="hierarchy-strip">
      <div class="hierarchy-title">Progression Pédagogique du Cours :</div>
      <div class="hierarchy-flow">
        <div class="flow-node baseline">🎲 Aléatoire (Uniforme)</div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node zero-r">🎯 Zero-R (Classe Majoritaire)</div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node one-r">⚡ One-R (1 Règle / Decision Stump)</div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node advanced">🚀 Modèles Avancés (DT, RF, Bayi, SVM, KNN, Boosting, Stacking)</div>
      </div>
    </div>

    <!-- KPI Summary Grid -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Échantillons & Déséquilibre</div>
        <div class="kpi-value">{self.profile.total_samples:,} <span style="font-size: 1rem; color: var(--text-dim); font-weight: normal;">patients</span></div>
        <div class="kpi-subtext">
          {self.profile.target_percentages.get(0, 0)}% Sains / {self.profile.target_percentages.get(1, 0)}% Malades (Ratio: {self.profile.imbalance_ratio}:1)
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Seuil Plancher Zero-R (Accuracy)</div>
        <div class="kpi-value" style="color: #fbbf24;">
          {(zero_r.accuracy.mean * 100 if zero_r else 0.0):.2f}%
        </div>
        <div class="kpi-subtext">
          Obtenu en prédisant toujours 0 (Rappel de la maladie = 0.0%)
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Meilleur Modèle (F1 Données Brutes)</div>
        <div class="kpi-value" style="color: #38bdf8;">
          {best_unbal_f1.f1_score.mean:.3f}
        </div>
        <div class="kpi-subtext">
          {best_unbal_f1.model_name} (ROC-AUC: {best_unbal_f1.roc_auc.mean:.3f})
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Meilleur Rappel (Équilibré SMOTE)</div>
        <div class="kpi-value" style="color: #34d399;">
          {(best_bal_recall.recall.mean * 100):.1f}%
        </div>
        <div class="kpi-subtext">
          {best_bal_recall.model_name} (Détection maximale des crises)
        </div>
      </div>
    </div>

    <!-- Global Scenario Switcher -->
    <div class="switcher-bar">
      <div style="font-weight: 600; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
        <span>Affichage actif du scénario :</span>
      </div>
      <div class="scenario-toggle">
        <button id="btnUnbalanced" class="scenario-btn active" onclick="setScenario('unbalanced')">Données Brutes (Non Équilibrées)</button>
        <button id="btnBalanced" class="scenario-btn" onclick="setScenario('balanced')">Données Équilibrées (SMOTE Fold-Wise)</button>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="nav-tabs">
      <button class="tab-btn active" onclick="switchTab('tab-comparison')">📊 Comparatif des Modèles & Graphiques</button>
      <button class="tab-btn" onclick="switchTab('tab-table')">📋 Tableau Complet des Métriques (Mean ± Std)</button>
      <button class="tab-btn" onclick="switchTab('tab-curves')">📈 Courbes ROC & Précision-Rappel (PR / AP)</button>
      <button class="tab-btn" onclick="switchTab('tab-cm')">🔲 Matrices de Confusion & Diagnostic Médical</button>
      <button class="tab-btn" onclick="switchTab('tab-features')">🔍 Importance des Variables Biomédicales</button>
      <button class="tab-btn" onclick="switchTab('tab-course')">🎓 Enseignements & Concepts du Cours</button>
    </div>

    <!-- TAB 1: Comparison & Charts -->
    <div id="tab-comparison" class="tab-panel active">
      <div class="edu-alert info">
        <div class="edu-alert-title">
          <span>ℹ️</span> Analyse Comparative de la Hiérarchie des Modèles
        </div>
        Le graphique ci-dessous illustre l'évolution des performances depuis le tirage aléatoire et Zero-R jusqu'aux modèles avancés et au stacking.
        Chaque barre représente la moyenne calculée sur 5 plis de validation croisée stratifiée, avec l'écart-type affiché dans l'infobulle.
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <div>
              <div class="card-title">Exactitude (Accuracy) & F1-Score</div>
              <div class="card-desc">Comparaison des modèles par rapport au seuil plancher Zero-R</div>
            </div>
          </div>
          <div class="chart-container">
            <canvas id="chartAccF1"></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div>
              <div class="card-title">Rappel (Recall) vs Précision (Precision)</div>
              <div class="card-desc">Capacité à détecter tous les patients à risque sans fausse alerte excessive</div>
            </div>
          </div>
          <div class="chart-container">
            <canvas id="chartPrecRec"></canvas>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">ROC-AUC & Average Precision (AP)</div>
            <div class="card-desc">Robustesse du classement des probabilités et performance sous déséquilibre</div>
          </div>
        </div>
        <div class="chart-container">
          <canvas id="chartRocAp"></canvas>
        </div>
      </div>
    </div>

    <!-- TAB 2: Full Table -->
    <div id="tab-table" class="tab-panel">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Tableau Récapitulatif 5-Fold Cross-Validation</div>
            <div class="card-desc">Toutes les métriques sont exprimées sous la forme <code>Moyenne ± Écart-Type (std)</code> sur les 5 plis.</div>
          </div>
        </div>

        <div class="table-container">
          <table class="benchmark-table" id="resultsTable">
            <thead>
              <tr>
                <th onclick="sortTable(0)">Algorithme</th>
                <th onclick="sortTable(1)">Palier / Type</th>
                <th onclick="sortTable(2)">Accuracy</th>
                <th onclick="sortTable(3)">Précision</th>
                <th onclick="sortTable(4)">Rappel (Sensibilité)</th>
                <th onclick="sortTable(5)">F1-Score</th>
                <th onclick="sortTable(6)">F1 Macro</th>
                <th onclick="sortTable(7)">ROC-AUC</th>
                <th onclick="sortTable(8)">Avg Precision (AP)</th>
                <th onclick="sortTable(9)">Temps (s)</th>
              </tr>
            </thead>
            <tbody id="tableBody">
              <!-- Dynamically populated via JS -->
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 3: ROC & PR Curves -->
    <div id="tab-curves" class="tab-panel">
      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <div>
              <div class="card-title">Courbes ROC (Receiver Operating Characteristic)</div>
              <div class="card-desc">Taux de Vrais Positifs (TPR) vs Taux de Faux Positifs (FPR)</div>
            </div>
          </div>
          <div class="chart-container">
            <canvas id="chartRocCurve"></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div>
              <div class="card-title">Courbes Précision-Rappel (PR Curves)</div>
              <div class="card-desc">Particulièrement informative pour les classes positives minoritaires</div>
            </div>
          </div>
          <div class="chart-container">
            <canvas id="chartPrCurve"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: Confusion Matrices -->
    <div id="tab-cm" class="tab-panel">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Matrices de Confusion Cumulées (Somme des 5 Plis de Test)</div>
            <div class="card-desc">Analyse critique des Faux Négatifs (patients cardiaques non diagnostiqués) vs Faux Positifs</div>
          </div>
        </div>

        <div class="edu-alert warning">
          <div class="edu-alert-title">⚠️ Le Dilemme Médical : Faux Négatif vs Faux Positif</div>
          En cardiologie, un <strong>Faux Négatif</strong> (patient cardiaque renvoyé chez lui) peut avoir des conséquences vitales.
          Observer comment l'application de <strong>SMOTE</strong> déplace la frontière de décision pour réduire massivement les Faux Négatifs au prix d'une légère hausse de Faux Positifs.
        </div>

        <div class="cm-wrapper" id="cmContainer">
          <!-- Dynamically populated via JS -->
        </div>
      </div>
    </div>

    <!-- TAB 5: Feature Importances -->
    <div id="tab-features" class="tab-panel">
      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <div>
              <div class="card-title">Facteurs de Risque Dominants (Random Forest)</div>
              <div class="card-desc">Importance relative (Gini Impurity Decrease) des caractéristiques</div>
            </div>
          </div>
          <div id="rfFeatureList"></div>
        </div>

        <div class="card">
          <div class="card-header">
            <div>
              <div class="card-title">Facteurs de Risque Dominants (XGBoost)</div>
              <div class="card-desc">Poids des gradients dans l'ensemble séquentiel</div>
            </div>
          </div>
          <div id="xgbFeatureList"></div>
        </div>
      </div>
    </div>

    <!-- TAB 6: Course Concepts -->
    <div id="tab-course" class="tab-panel">
      <div class="card">
        <div class="card-title" style="margin-bottom: 16px;">🎓 Alignement avec le Cours de Data Mining & Classification</div>

        <div class="edu-alert warning">
          <div class="edu-alert-title">1. Le « Piège de l'Accuracy » et la Règle d'Or de Zero-R</div>
          <p>
            Le classifieur <strong>Zero-R</strong> prédit toujours la classe majoritaire sans regarder les variables explicatives $X$.
            Sur notre jeu de données déséquilibré ({self.profile.imbalance_ratio:.1f}:1), Zero-R obtient environ <strong>{(zero_r.accuracy.mean * 100 if zero_r else 0):.2f}% d'Accuracy</strong>, mais avec un <strong>Rappel de 0%</strong> sur les patients malades !
          </p>
          <p style="margin-top: 6px;">
            <em>Règle d'or du cours :</em> Tout modèle qui affiche 70% d'exactitude n'a rien appris de plus qu'une constante triviale. L'exactitude de Zero-R est le <strong>seuil plancher universel</strong>.
          </p>
        </div>

        <div class="edu-alert info">
          <div class="edu-alert-title">2. One-R (Decision Stump) : La puissance d'une seule règle</div>
          <p>
            One-R (Holte, 1993) sélectionne la meilleure variable explicative unique. On constate qu'avec une seule division univariée,
            One-R dépasse déjà significativement le F1-score de Zero-R, fournissant une baseline interprétable essentielle pour juger du gain apporté par des arbres plus profonds ou des forêts.
          </p>
        </div>

        <div class="edu-alert success">
          <div class="edu-alert-title">3. Impact du Rééquilibrage SMOTE (Prévention de Fuite de Données)</div>
          <p>
            Appliqué <em>strictement à l'intérieur des plis d'entraînement</em>, le sur-échantillonnage synthétique (SMOTE) rééquilibre la sensibilité du gradient et des séparateurs.
            Le Rappel sur la classe malade augmente spectaculairement (notamment pour KNN et SVM), optimisant le score F1-Macro.
          </p>
        </div>

        <div class="edu-alert">
          <div class="edu-alert-title">4. Supériorité du Méta-Apprentissage (Stacking) & des Ensembles</div>
          <p>
            Le modèle de <strong>Stacking</strong> combine les prédictions d'algorithmes aux biais inductifs distincts (arbres décorrélés pour RF, frontière à vaste marge pour SVM, votes de voisinage pour KNN, et descente de gradient de second ordre pour XGBoost) au moyen d'un méta-classifieur de régression logistique.
            Cette synergie produit l'aire sous la courbe (ROC-AUC) la plus stable et régulière sur l'ensemble des 5 plis.
          </p>
        </div>
      </div>
    </div>

  </div>

  <footer>
    <div class="container">
      Généré automatiquement par le pipeline de Classification Data Mining & Machine Learning • 5-Fold Stratified Cross-Validation • Scikit-Learn, Imbalanced-Learn & XGBoost
    </div>
  </footer>

  <script>
    // Embedded Data from Python
    const RAW_DATA = {data_json};
    const PROFILE = {profile_json};

    let currentScenario = 'unbalanced';
    let chartInstances = {{}};

    function setScenario(scen) {{
      currentScenario = scen;
      document.getElementById('btnUnbalanced').classList.toggle('active', scen === 'unbalanced');
      document.getElementById('btnBalanced').classList.toggle('active', scen === 'balanced');
      updateAllViews();
    }}

    function switchTab(tabId) {{
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      event.currentTarget.classList.add('active');
      document.getElementById(tabId).classList.add('active');
    }}

    function toggleTheme() {{
      const html = document.documentElement;
      const current = html.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
      Chart.defaults.color = next === 'dark' ? '#94a3b8' : '#475569';
      Chart.defaults.borderColor = next === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)';
      updateCharts();
    }}

    function getTierClass(tier) {{
      if (tier.includes('Baseline')) return 'tier-baseline';
      if (tier.includes('Simple')) return 'tier-simple';
      if (tier.includes('Linéaire')) return 'tier-linear';
      if (tier.includes('Ensemble')) return 'tier-ensemble';
      if (tier.includes('Stacking')) return 'tier-stacking';
      return 'tier-simple';
    }}

    function renderTable() {{
      const tbody = document.getElementById('tableBody');
      tbody.innerHTML = '';
      const data = RAW_DATA[currentScenario];
      const modelNames = Object.keys(data);

      // Find best values for highlighting
      let maxAcc = -1, maxPrec = -1, maxRec = -1, maxF1 = -1, maxF1Macro = -1, maxRoc = -1, maxAp = -1;
      modelNames.forEach(k => {{
        const m = data[k];
        if (m.accuracy.mean > maxAcc) maxAcc = m.accuracy.mean;
        if (m.precision.mean > maxPrec) maxPrec = m.precision.mean;
        if (m.recall.mean > maxRec) maxRec = m.recall.mean;
        if (m.f1_score.mean > maxF1) maxF1 = m.f1_score.mean;
        if (m.f1_macro.mean > maxF1Macro) maxF1Macro = m.f1_macro.mean;
        if (m.roc_auc.mean > maxRoc) maxRoc = m.roc_auc.mean;
        if (m.ap.mean > maxAp) maxAp = m.ap.mean;
      }});

      modelNames.forEach(name => {{
        const m = data[name];
        const tr = document.createElement('tr');

        const isZeroR = name.includes('Zero-R');

        tr.innerHTML = `
          <td style="font-weight: 600;">${{name}}</td>
          <td><span class="tier-badge ${{getTierClass(m.tier)}}">${{m.type}}</span></td>
          <td class="score-cell ${{m.accuracy.mean === maxAcc ? 'best-score' : (isZeroR ? 'baseline-score' : '')}}">
            ${{m.accuracy.mean.toFixed(4)}} <span class="score-std">±${{m.accuracy.std.toFixed(4)}}</span>
          </td>
          <td class="score-cell ${{m.precision.mean === maxPrec ? 'best-score' : ''}}">
            ${{m.precision.mean.toFixed(4)}} <span class="score-std">±${{m.precision.std.toFixed(4)}}</span>
          </td>
          <td class="score-cell ${{m.recall.mean === maxRec ? 'best-score' : ''}}">
            ${{m.recall.mean.toFixed(4)}} <span class="score-std">±${{m.recall.std.toFixed(4)}}</span>
          </td>
          <td class="score-cell ${{m.f1_score.mean === maxF1 ? 'best-score' : ''}}">
            ${{m.f1_score.mean.toFixed(4)}} <span class="score-std">±${{m.f1_score.std.toFixed(4)}}</span>
          </td>
          <td class="score-cell ${{m.f1_macro.mean === maxF1Macro ? 'best-score' : ''}}">
            ${{m.f1_macro.mean.toFixed(4)}} <span class="score-std">±${{m.f1_macro.std.toFixed(4)}}</span>
          </td>
          <td class="score-cell ${{m.roc_auc.mean === maxRoc ? 'best-score' : ''}}">
            ${{m.roc_auc.mean.toFixed(4)}} <span class="score-std">±${{m.roc_auc.std.toFixed(4)}}</span>
          </td>
          <td class="score-cell ${{m.ap.mean === maxAp ? 'best-score' : ''}}">
            ${{m.ap.mean.toFixed(4)}} <span class="score-std">±${{m.ap.std.toFixed(4)}}</span>
          </td>
          <td class="score-cell" style="color: var(--text-dim);">${{m.fit_time.toFixed(2)}}s</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function renderConfusionMatrices() {{
      const container = document.getElementById('cmContainer');
      container.innerHTML = '';
      const data = RAW_DATA[currentScenario];

      // Showcase key models: Zero-R, One-R, Decision Tree, Random Forest, XGBoost, Stacking
      const featured = ['Zero-R (Majoritaire)', 'One-R (1 Règle)', 'Arbre de Décision (DT)', 'Random Forest (RF / RM)', 'XGBoost', 'Stacking Classifier'];

      featured.forEach(name => {{
        if (!data[name]) return;
        const m = data[name];
        const cm = m.confusion_matrix; // [[TN, FP], [FN, TP]]
        const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
        const total = tn + fp + fn + tp;

        const card = document.createElement('div');
        card.className = 'cm-card';
        card.innerHTML = `
          <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">${{name}}</div>
          <div style="font-size: 0.8rem; color: var(--text-dim); margin-bottom: 10px;">${{m.type}}</div>
          <div class="cm-grid">
            <div></div>
            <div class="cm-header">Prédit: Sain (0)</div>
            <div class="cm-header">Prédit: Malade (1)</div>

            <div class="cm-header" style="text-align: right;">Réel: 0</div>
            <div class="cm-box cm-tn" title="Vrais Négatifs">${{tn}} <span style="font-size: 0.72rem; display:block;">TN (${{(tn/total*100).toFixed(1)}}%)</span></div>
            <div class="cm-box cm-fp" title="Faux Positifs">${{fp}} <span style="font-size: 0.72rem; display:block;">FP (${{(fp/total*100).toFixed(1)}}%)</span></div>

            <div class="cm-header" style="text-align: right;">Réel: 1</div>
            <div class="cm-box cm-fn" title="Faux Négatifs (DANGER)">${{fn}} <span style="font-size: 0.72rem; display:block;">FN (${{(fn/total*100).toFixed(1)}}%)</span></div>
            <div class="cm-box cm-tp" title="Vrais Positifs">${{tp}} <span style="font-size: 0.72rem; display:block;">TP (${{(tp/total*100).toFixed(1)}}%)</span></div>
          </div>
          <div style="margin-top: 10px; font-size: 0.82rem; display: flex; justify-content: space-between; color: var(--text-muted);">
            <span>Rappel: <strong>${{(m.recall.mean*100).toFixed(1)}}%</strong></span>
            <span>Précision: <strong>${{(m.precision.mean*100).toFixed(1)}}%</strong></span>
          </div>
        `;
        container.appendChild(card);
      }});
    }}

    function renderFeatureImportances() {{
      const data = RAW_DATA[currentScenario];
      const rfData = data['Random Forest (RF / RM)']?.feature_importances;
      const xgbData = data['XGBoost']?.feature_importances;

      function renderList(targetId, featObj) {{
        const target = document.getElementById(targetId);
        if (!featObj) {{
          target.innerHTML = '<div style="color: var(--text-dim);">Non disponible</div>';
          return;
        }}
        const entries = Object.entries(featObj).sort((a, b) => b[1] - a[1]).slice(0, 10);
        const maxVal = entries[0][1] || 1;
        target.innerHTML = entries.map(([name, val]) => `
          <div class="fi-item">
            <div class="fi-label-row">
              <span>${{name}}</span>
              <span style="font-family: var(--font-mono);">${{(val * 100).toFixed(2)}}%</span>
            </div>
            <div class="fi-bar-bg">
              <div class="fi-bar-fill" style="width: ${{Math.max(5, (val / maxVal * 100))}}%;"></div>
            </div>
          </div>
        `).join('');
      }}

      renderList('rfFeatureList', rfData);
      renderList('xgbFeatureList', xgbData);
    }}

    function updateCharts() {{
      const data = RAW_DATA[currentScenario];
      const labels = Object.keys(data);

      const accMeans = labels.map(k => data[k].accuracy.mean);
      const f1Means = labels.map(k => data[k].f1_score.mean);
      const precMeans = labels.map(k => data[k].precision.mean);
      const recMeans = labels.map(k => data[k].recall.mean);
      const rocAucMeans = labels.map(k => data[k].roc_auc.mean);
      const apMeans = labels.map(k => data[k].ap.mean);

      // Chart 1: Accuracy & F1
      if (chartInstances.accF1) chartInstances.accF1.destroy();
      chartInstances.accF1 = new Chart(document.getElementById('chartAccF1'), {{
        type: 'bar',
        data: {{
          labels: labels.map(l => l.split(' (')[0]),
          datasets: [
            {{
              label: 'Exactitude (Accuracy)',
              data: accMeans,
              backgroundColor: 'rgba(6, 182, 212, 0.7)',
              borderRadius: 6,
            }},
            {{
              label: 'F1-Score',
              data: f1Means,
              backgroundColor: 'rgba(239, 68, 68, 0.75)',
              borderRadius: 6,
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          scales: {{
            y: {{ min: 0, max: 1.0, grid: {{ color: 'rgba(255,255,255,0.06)' }} }},
            x: {{ grid: {{ display: false }} }}
          }}
        }}
      }});

      // Chart 2: Precision vs Recall
      if (chartInstances.precRec) chartInstances.precRec.destroy();
      chartInstances.precRec = new Chart(document.getElementById('chartPrecRec'), {{
        type: 'bar',
        data: {{
          labels: labels.map(l => l.split(' (')[0]),
          datasets: [
            {{
              label: 'Précision',
              data: precMeans,
              backgroundColor: 'rgba(139, 92, 246, 0.7)',
              borderRadius: 6,
            }},
            {{
              label: 'Rappel (Sensibilité)',
              data: recMeans,
              backgroundColor: 'rgba(16, 185, 129, 0.75)',
              borderRadius: 6,
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          scales: {{
            y: {{ min: 0, max: 1.0, grid: {{ color: 'rgba(255,255,255,0.06)' }} }},
            x: {{ grid: {{ display: false }} }}
          }}
        }}
      }});

      // Chart 3: ROC-AUC vs AP
      if (chartInstances.rocAp) chartInstances.rocAp.destroy();
      chartInstances.rocAp = new Chart(document.getElementById('chartRocAp'), {{
        type: 'bar',
        data: {{
          labels: labels.map(l => l.split(' (')[0]),
          datasets: [
            {{
              label: 'ROC-AUC',
              data: rocAucMeans,
              backgroundColor: 'rgba(245, 158, 11, 0.75)',
              borderRadius: 6,
            }},
            {{
              label: 'Average Precision (AP)',
              data: apMeans,
              backgroundColor: 'rgba(236, 72, 153, 0.7)',
              borderRadius: 6,
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          scales: {{
            y: {{ min: 0, max: 1.0, grid: {{ color: 'rgba(255,255,255,0.06)' }} }},
            x: {{ grid: {{ display: false }} }}
          }}
        }}
      }});

      // Chart 4: ROC Curves
      if (chartInstances.rocCurves) chartInstances.rocCurves.destroy();
      const colors = ['#ef4444', '#06b6d4', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899', '#3b82f6'];
      const topModels = ['Zero-R (Majoritaire)', 'One-R (1 Règle)', 'Arbre de Décision (DT)', 'Random Forest (RF / RM)', 'SVM (Support Vector Machine)', 'XGBoost', 'Stacking Classifier'];

      const rocDatasets = topModels.filter(m => data[m]).map((m, idx) => {{
        const r = data[m].roc_curve;
        const pts = r.fpr.map((x, i) => ({{ x: x, y: r.tpr[i] }}));
        return {{
          label: `${{m}} (AUC = ${{r.auc}})`,
          data: pts,
          borderColor: colors[idx % colors.length],
          borderWidth: 2,
          fill: false,
          showLine: true,
          pointRadius: 0,
        }};
      }});

      // Add diagonal reference line
      rocDatasets.push({{
        label: 'Aléatoire pur (AUC = 0.50)',
        data: [{{x: 0, y: 0}}, {{x: 1, y: 1}}],
        borderColor: 'rgba(255,255,255,0.3)',
        borderDash: [5, 5],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
      }});

      chartInstances.rocCurves = new Chart(document.getElementById('chartRocCurve'), {{
        type: 'scatter',
        data: {{ datasets: rocDatasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          scales: {{
            x: {{ type: 'linear', min: 0, max: 1, title: {{ display: true, text: 'Taux de Faux Positifs (FPR)' }} }},
            y: {{ type: 'linear', min: 0, max: 1, title: {{ display: true, text: 'Taux de Vrais Positifs (TPR / Rappel)' }} }},
          }}
        }}
      }});

      // Chart 5: PR Curves
      if (chartInstances.prCurves) chartInstances.prCurves.destroy();
      const prDatasets = topModels.filter(m => data[m]).map((m, idx) => {{
        const p = data[m].pr_curve;
        const pts = p.recall.map((x, i) => ({{ x: x, y: p.precision[i] }}));
        return {{
          label: `${{m}} (AP = ${{p.ap}})`,
          data: pts,
          borderColor: colors[idx % colors.length],
          borderWidth: 2,
          fill: false,
          showLine: true,
          pointRadius: 0,
        }};
      }});

      chartInstances.prCurves = new Chart(document.getElementById('chartPrCurve'), {{
        type: 'scatter',
        data: {{ datasets: prDatasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          scales: {{
            x: {{ type: 'linear', min: 0, max: 1, title: {{ display: true, text: 'Rappel (Recall)' }} }},
            y: {{ type: 'linear', min: 0, max: 1, title: {{ display: true, text: 'Précision (Precision)' }} }},
          }}
        }}
      }});
    }}

    function updateAllViews() {{
      renderTable();
      renderConfusionMatrices();
      renderFeatureImportances();
      updateCharts();
    }}

    // Sorting utility for results table
    let sortAsc = false;
    function sortTable(colIndex) {{
      const table = document.getElementById('resultsTable');
      const rows = Array.from(table.rows).slice(1);
      sortAsc = !sortAsc;

      rows.sort((r1, r2) => {{
        const t1 = r1.cells[colIndex].innerText.split('±')[0].trim();
        const t2 = r2.cells[colIndex].innerText.split('±')[0].trim();
        const n1 = parseFloat(t1);
        const n2 = parseFloat(t2);

        if (!isNaN(n1) && !isNaN(n2)) {{
          return sortAsc ? n1 - n2 : n2 - n1;
        }}
        return sortAsc ? t1.localeCompare(t2) : t2.localeCompare(t1);
      }});

      const tbody = document.getElementById('tableBody');
      rows.forEach(r => tbody.appendChild(r));
    }}

    // Initial load
    window.addEventListener('DOMContentLoaded', () => {{
      Chart.defaults.color = '#94a3b8';
      Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';
      updateAllViews();
    }});
  </script>
</body>
</html>
"""
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_filepath
