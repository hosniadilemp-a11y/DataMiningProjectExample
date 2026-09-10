"""
Heart Attack Classification Benchmark Package.
Implements the pedagogical Data Mining hierarchy:
Aléatoire ➔ Zero-R ➔ One-R ➔ Modèles Avancés ➔ Stacking
"""

from .preprocess import HeartDataPreprocessor, DatasetProfile
from .models import get_model_catalog
from .evaluate import Evaluator, ModelEvaluationResult
from .generate_report import ReportGenerator

__all__ = [
    "HeartDataPreprocessor",
    "DatasetProfile",
    "get_model_catalog",
    "Evaluator",
    "ModelEvaluationResult",
    "ReportGenerator",
]
