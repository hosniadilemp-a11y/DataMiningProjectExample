#!/usr/bin/env python3
"""
Git Commit Workflow Builder for Heart Disease Risk Prediction Reference Project.
Creates main, develop, and feature branches with simulated student commit messages.
"""

import subprocess
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_git(args, allow_fail=False):
    res = subprocess.run(["git"] + args, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if res.returncode != 0 and not allow_fail:
        print(f"Git error: {res.stderr}")
    return res

def setup_git_history():
    print("=== Initializing Git Repository & Simulated Workflow ===")
    
    # Init git if not present
    if not (PROJECT_ROOT / ".git").exists():
        run_git(["init", "-b", "main"])
        run_git(["config", "user.name", "hosniadilemp-a11y"])
        run_git(["config", "user.email", "hosni.adil.emp@gmail.com"])
    
    # 1. Base commit on main
    run_git(["add", ".gitignore", "LICENSE", "requirements.txt", "README.md"])
    run_git(["commit", "-m", "chore: initialize project structure and repository configuration"])
    
    # 2. Branch develop
    run_git(["checkout", "-b", "develop"])
    
    # 3. Branch feature/AISD05-eda
    run_git(["checkout", "-b", "feature/AISD05-eda"])
    run_git(["add", "scripts/download_data.py", "data/raw/.gitkeep", "src/utils.py"])
    run_git(["commit", "-m", "feat(AISD05): add Kaggle dataset acquisition script via KaggleHub"])
    
    run_git(["add", "src/visualization.py", "notebooks/01_eda_and_cleaning.ipynb"])
    run_git(["commit", "-m", "feat(AISD05): implement exploratory data analysis and statistical testing"])
    
    # Merge AISD05 into develop
    run_git(["checkout", "develop"])
    run_git(["merge", "--no-ff", "feature/AISD05-eda", "-m", "merge: pull request #1 from feature/AISD05-eda into develop"])
    
    # 4. Branch feature/SIAD19-modeling
    run_git(["checkout", "-b", "feature/SIAD19-modeling"])
    run_git(["add", "src/preprocessing.py", "notebooks/02_preprocessing_encoding.ipynb", "data/processed/.gitkeep"])
    run_git(["commit", "-m", "feat(SIAD19): implement data preprocessing pipeline and column transformers"])
    
    run_git(["add", "src/models.py", "src/evaluation.py", "scripts/train_pipeline.py", "models/.gitkeep", "notebooks/03_model_training_evaluation.ipynb"])
    run_git(["commit", "-m", "feat(SIAD19): implement 6 classification models, 5-fold CV, and hyperparameter tuning"])
    
    # Merge SIAD19 into develop
    run_git(["checkout", "develop"])
    run_git(["merge", "--no-ff", "feature/SIAD19-modeling", "-m", "merge: pull request #2 from feature/SIAD19-modeling into develop"])
    
    # 5. Branch feature/RSD20-streamlit
    run_git(["checkout", "-b", "feature/RSD20-streamlit"])
    run_git(["add", "src/app.py", "tests/"])
    run_git(["commit", "-m", "feat(RSD20): initialize Streamlit web application and unit test suite"])
    
    run_git(["add", "docs/", "scripts/create_notebooks.py"])
    run_git(["commit", "-m", "docs(RSD20): add scientific LaTeX report and compile final PDF"])
    
    # Merge RSD20 into develop
    run_git(["checkout", "develop"])
    run_git(["merge", "--no-ff", "feature/RSD20-streamlit", "-m", "merge: pull request #3 from feature/RSD20-streamlit into develop"])
    
    # Add any remaining files
    run_git(["add", "."])
    run_git(["commit", "-m", "release: finalize heart disease risk prediction reference project"], allow_fail=True)
    
    # Merge develop into main
    run_git(["checkout", "main"])
    run_git(["merge", "--no-ff", "develop", "-m", "release(v1.0.0): merge final validated project into main"])
    
    print("\nGit Branch Structure:")
    res_branch = run_git(["branch", "-a"])
    print(res_branch.stdout)
    
    print("\nGit Log History:")
    res_log = run_git(["log", "--oneline", "-n", "15"])
    print(res_log.stdout)

if __name__ == "__main__":
    setup_git_history()
