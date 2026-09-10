"""
Model definitions and catalog following the pedagogical classification hierarchy:
Aléatoire ➔ Zero-R (Majoritaire) ➔ One-R (1 Règle) ➔ Modèles Avancés (DT, RF, Bayi, SVM, KNN, AdaBoost, XGBoost, Stacking)
"""

from typing import Dict, List, Tuple
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    StackingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_model_catalog(random_state: int = 42) -> Dict[str, Dict]:
    """
    Returns the complete hierarchy of models with metadata, hyperparameters,
    and pedagogical explanations matching the Data Mining course.
    """
    # XGBoost setup with fallback
    if XGBOOST_AVAILABLE:
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        xgb_model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            random_state=random_state,
        )

    # Base estimators for stacking (heterogeneous bias models: bagging, probabilistic, instance-based, boosting)
    stacking_base = [
        ("rf", RandomForestClassifier(n_estimators=60, max_depth=6, random_state=random_state, n_jobs=-1)),
        ("nb", GaussianNB()),
        ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ("xgb", xgb_model),
    ]

    stacking_model = StackingClassifier(
        estimators=stacking_base,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=random_state),
        cv=3,
        n_jobs=-1,
    )

    catalog = {
        "Aléatoire (Uniform)": {
            "estimator": DummyClassifier(strategy="uniform", random_state=random_state),
            "tier": "1. Baseline Aléatoire",
            "type": "Dummy / Random",
            "description": "Tirage équiprobable pur (1/K) sans aucun a priori.",
            "course_note": "Score plancher théorique d'un classifieur sans information (50% pour binaire).",
        },
        "Zero-R (Majoritaire)": {
            "estimator": DummyClassifier(strategy="most_frequent"),
            "tier": "2. Baseline Zero-R",
            "type": "Zero-Rule (Majority)",
            "description": "Prédit systématiquement la classe majoritaire de l'entraînement.",
            "course_note": "Seuil plancher universel d'Accuracy. Tout modèle avec une accuracy < Zero-R est pire qu'inutile.",
        },
        "One-R (1 Règle)": {
            "estimator": DecisionTreeClassifier(max_depth=1, random_state=random_state),
            "tier": "3. Baseline One-R",
            "type": "One-Rule (Decision Stump)",
            "description": "Classifieur basé sur un seul attribut explicatif (arbre de décision à profondeur 1).",
            "course_note": "Règle si-alors univariée interprétable. Atteint souvent 80-90% de la performance de modèles complexes.",
        },
        "Arbre de Décision (DT)": {
            "estimator": DecisionTreeClassifier(
                max_depth=6,
                min_samples_split=15,
                min_samples_leaf=5,
                random_state=random_state,
            ),
            "tier": "4. Modèle Simple",
            "type": "Decision Tree",
            "description": "Arbre de classification hiérarchique avec division par impureté de Gini.",
            "course_note": "Modèle boîte blanche hautement interprétable, sensible au surapprentissage si non élagué.",
        },
        "Naïve Bayes (Bayi)": {
            "estimator": GaussianNB(),
            "tier": "4. Modèle Simple",
            "type": "Probabilistic / Bayesian",
            "description": "Théorème de Bayes avec hypothèse forte d'indépendance conditionnelle des attributs.",
            "course_note": "Rapide, robuste au bruit, performant même avec de petits échantillons.",
        },
        "K-Plus Proches Voisins (KNN)": {
            "estimator": KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                metric="euclidean",
            ),
            "tier": "4. Modèle Simple",
            "type": "Instance-based (Lazy Learning)",
            "description": "Classification basée sur la proximité géométrique dans l'espace normalisé.",
            "course_note": "Non-paramétrique, très sensible à la standardisation des données et au déséquilibre.",
        },
        "SVM (Support Vector Machine)": {
            "estimator": SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                random_state=random_state,
            ),
            "tier": "5. Modèle Linéaire & Noyau",
            "type": "Kernel / Margin Maximizer",
            "description": "Séparateur à vaste marge dans un espace de Hilbert via noyau RBF.",
            "course_note": "Excellent pouvoir de généralisation dans les espaces multidimensionnels.",
        },
        "Random Forest (RF / RM)": {
            "estimator": RandomForestClassifier(
                n_estimators=120,
                max_depth=8,
                min_samples_split=6,
                random_state=random_state,
                n_jobs=-1,
            ),
            "tier": "6. Ensemble Learning (Bagging)",
            "type": "Ensemble (Bagging)",
            "description": "Forêt aléatoire d'arbres décorrélés avec bootstrap et sous-échantillonnage d'attributs.",
            "course_note": "Réduction drastique de la variance sans augmenter le biais. Fournit l'importance des variables.",
        },
        "AdaBoost": {
            "estimator": AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=random_state,
            ),
            "tier": "6. Ensemble Learning (Boosting)",
            "type": "Ensemble (Sequential Boosting)",
            "description": "Boosting adaptatif augmentant le poids des individus mal classés au fil des itérations.",
            "course_note": "Focalisation sur les cas difficiles (ex: faux négatifs cardiaques).",
        },
        "XGBoost": {
            "estimator": xgb_model,
            "tier": "6. Ensemble Learning (Boosting)",
            "type": "Extreme Gradient Boosting",
            "description": "Gradient Boosting optimisé de second ordre avec régularisation L1/L2 intégrée.",
            "course_note": "L'un des algorithmes les plus puissants sur données tabulaires en Data Mining.",
        },
        "Stacking Classifier": {
            "estimator": stacking_model,
            "tier": "7. Méta-Ensemble (Stacking)",
            "type": "Meta-Learning",
            "description": "Combinaison hétérogène (RF + Naïve Bayes + KNN + XGBoost) entraînée par un méta-classifieur LogisticRegression.",
            "course_note": "Exploite les forces complémentaires de paradigmes de classification hétérogènes.",
        },
    }

    return catalog
