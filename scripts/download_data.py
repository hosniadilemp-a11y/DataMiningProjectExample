#!/usr/bin/env python3
"""
Data Acquisition Script for Heart Disease Risk Prediction Project.
Downloads dataset from Kaggle via KaggleHub API and validates raw contents.

Usage:
    python scripts/download_data.py
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import kagglehub

def download_and_setup():
    print("=== Downloading Kaggle Dataset: uditjain13/heart-disease-risk-2026 ===")
    
    # KaggleHub official download code
    path = kagglehub.dataset_download("uditjain13/heart-disease-risk-2026")
    print("Path to dataset files:", path)
    
    dataset_dir = Path(path)
    csv_files = list(dataset_dir.rglob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in KaggleHub downloaded directory: {path}")
    
    target_csv = csv_files[0]
    print(f"Identified dataset file: {target_csv}")
    
    # Destination in project root
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = raw_dir / "heart_disease_risk_2026.csv"
    shutil.copy2(target_csv, dest_path)
    print(f"Successfully copied raw dataset to: {dest_path}")
    
    # Validate raw data
    df = pd.read_csv(dest_path)
    print("\n=== RAW DATASET AUDIT & VALIDATION ===")
    print(f"Shape: {df.shape[0]} records, {df.shape[1]} columns")
    print(f"Missing Values Total: {df.isnull().sum().sum()}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")
    print("Columns:", list(df.columns))
    
    if "has_heart_disease" in df.columns:
        counts = df["has_heart_disease"].value_counts().to_dict()
        print(f"Target 'has_heart_disease' distribution: {counts}")
        print(f"Positive Class Prevalence: {counts.get(1, 0)/len(df):.2%}")
    
    print("\nDataset download and setup complete!")

if __name__ == "__main__":
    download_and_setup()
