#!/usr/bin/env python3
"""
CLI Runner for the Course Classification Progression Benchmark (Aléatoire -> Zero-R -> One-R -> Modèles Avancés).
Executes 5-fold cross-validation with leak-free SMOTE and generates interactive HTML and CSV reports.
"""

import sys
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.heart_attack_pipeline.run_pipeline import main

if __name__ == "__main__":
    main()
