"""
Data preprocessing and preparation module for Heart Attack / Heart Disease datasets.
Handles data loading, inspection, cleaning, encoding, scaling, and balancing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


@dataclass
class DatasetProfile:
    """Stores statistical and structural metadata about the dataset."""
    name: str
    total_samples: int
    num_features: int
    numeric_features: List[str]
    categorical_features: List[str]
    target_column: str
    target_distribution: Dict[Union[int, str], int]
    target_percentages: Dict[Union[int, str], float]
    is_imbalanced: bool
    imbalance_ratio: float  # majority / minority
    minority_class: int
    majority_class: int
    missing_count: int
    duplicate_count: int
    feature_names_out: List[str] = field(default_factory=list)


class HeartDataPreprocessor:
    """
    Robust preprocessor for Heart Attack & Heart Disease datasets.
    Supports automatic column inference, leak-free fold preprocessing,
    and balancing via SMOTE.
    """

    TARGET_CANDIDATES = [
        "has_heart_disease",
        "target",
        "HeartDisease",
        "output",
        "condition",
        "heart_attack",
        "num",
    ]

    ID_CANDIDATES = ["patient_id", "id", "Id", "ID", "patientid", "Unnamed: 0"]

    def __init__(self, target_column: Optional[str] = None, imbalance_threshold: float = 1.3):
        self.target_column = target_column
        self.imbalance_threshold = imbalance_threshold
        self.profile: Optional[DatasetProfile] = None
        self.preprocessor_pipeline: Optional[ColumnTransformer] = None
        self.feature_names_out: List[str] = []

    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Loads dataset from CSV file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        df = pd.read_csv(path)
        return df

    def find_target_column(self, df: pd.DataFrame) -> str:
        """Identifies target column automatically if not explicitly given."""
        if self.target_column and self.target_column in df.columns:
            return self.target_column

        for candidate in self.TARGET_CANDIDATES:
            if candidate in df.columns:
                self.target_column = candidate
                return candidate

        # Fallback: check last column if binary
        last_col = df.columns[-1]
        if df[last_col].nunique() == 2:
            self.target_column = last_col
            return last_col

        raise ValueError(
            f"Could not automatically detect target column. Available columns: {df.columns.tolist()}"
        )

    def prepare_raw_data(
        self, df: pd.DataFrame, sample_size: Optional[int] = None, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series, DatasetProfile]:
        """
        Cleans identifiers, inspects data, separates X and y, applies optional stratified
        subsampling (preserving the exact class ratio), and creates a dataset profile.
        """
        data = df.copy()

        # Remove byte order marks in column names if present
        data.columns = [c.strip().replace("\ufeff", "") for c in data.columns]

        target_col = self.find_target_column(data)

        # Drop ID columns
        cols_to_drop = [c for c in self.ID_CANDIDATES if c in data.columns]
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)

        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Convert boolean target to int if needed
        if y.dtype == bool or y.dtype == object:
            if set(y.unique()) <= {True, False}:
                y = y.astype(int)
            elif set(y.unique()) <= {"1", "0", 1, 0}:
                y = y.astype(int)
            else:
                # Map binary categorical target
                classes = sorted(y.unique())
                mapping = {classes[0]: 0, classes[1]: 1}
                y = y.map(mapping).astype(int)

        # Stratified subsample if requested (maintaining original class proportions)
        if sample_size is not None and 0 < sample_size < len(y):
            from sklearn.model_selection import train_test_split
            X, _, y, _ = train_test_split(
                X, y, train_size=sample_size, stratify=y, random_state=random_state
            )
            # Reset indices
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

        # Duplicate & missing count before processing
        dup_count = int(X.duplicated().sum())
        missing_count = int(X.isnull().sum().sum())

        # Class counts
        val_counts = y.value_counts().to_dict()
        total_samples = len(y)
        val_pcts = {k: round(v / total_samples * 100, 2) for k, v in val_counts.items()}

        classes_sorted = sorted(val_counts.keys(), key=lambda k: val_counts[k])
        minority_cls = classes_sorted[0]
        majority_cls = classes_sorted[-1]
        imbalance_ratio = val_counts[majority_cls] / max(1, val_counts[minority_cls])
        is_imbalanced = imbalance_ratio > self.imbalance_threshold

        # Identify numeric vs categorical columns
        numeric_cols = []
        categorical_cols = []

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
                # If numeric has very low cardinality (<= 4) and integer, check if categorical
                if X[col].nunique() <= 4 and (X[col] % 1 == 0).all():
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        self.profile = DatasetProfile(
            name="Heart Attack / Disease Risk Dataset",
            total_samples=total_samples,
            num_features=X.shape[1],
            numeric_features=numeric_cols,
            categorical_features=categorical_cols,
            target_column=target_col,
            target_distribution=val_counts,
            target_percentages=val_pcts,
            is_imbalanced=is_imbalanced,
            imbalance_ratio=round(imbalance_ratio, 2),
            minority_class=minority_cls,
            majority_class=majority_cls,
            missing_count=missing_count,
            duplicate_count=dup_count,
        )

        return X, y, self.profile

    def build_transformer(
        self, numeric_cols: List[str], categorical_cols: List[str]
    ) -> ColumnTransformer:
        """Constructs an sklearn ColumnTransformer for numeric and categorical pipelines."""
        transformers = []

        if numeric_cols:
            num_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", num_pipe, numeric_cols))

        if categorical_cols:
            cat_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False, drop=None
                        ),
                    ),
                ]
            )
            transformers.append(("cat", cat_pipe, categorical_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        return preprocessor

    def fit_transform(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """Fits the transformer on X and returns transformed matrix with feature names."""
        if not self.profile:
            raise RuntimeError("Must call prepare_raw_data before fit_transform.")

        self.preprocessor_pipeline = self.build_transformer(
            self.profile.numeric_features, self.profile.categorical_features
        )
        X_trans = self.preprocessor_pipeline.fit_transform(X)

        # Extract output feature names
        feature_names = []
        if self.profile.numeric_features:
            feature_names.extend(self.profile.numeric_features)

        if self.profile.categorical_features:
            ohe = self.preprocessor_pipeline.named_transformers_["cat"].named_steps["onehot"]
            ohe_names = list(ohe.get_feature_names_out(self.profile.categorical_features))
            feature_names.extend(ohe_names)

        self.feature_names_out = feature_names
        self.profile.feature_names_out = feature_names
        return X_trans, feature_names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms validation or test X using the fitted transformer."""
        if not self.preprocessor_pipeline:
            raise RuntimeError("Transformer is not fitted yet.")
        return self.preprocessor_pipeline.transform(X)

    @staticmethod
    def balance_training_fold(
        X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Applies balancing to the training fold only (preventing data leakage).
        Uses SMOTE if available; falls back to RandomOverSampler.
        """
        if IMBLEARN_AVAILABLE:
            try:
                sampler = SMOTE(random_state=random_state)
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                return X_res, y_res, "SMOTE (Synthetic Minority Over-sampling)"
            except Exception:
                sampler = RandomOverSampler(random_state=random_state)
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                return X_res, y_res, "RandomOverSampler"
        else:
            # Simple native random oversampling fallback if imblearn missing
            unique, counts = np.unique(y_train, return_counts=True)
            max_count = np.max(counts)
            indices = []
            for cls in unique:
                cls_idx = np.where(y_train == cls)[0]
                resampled_idx = np.random.choice(cls_idx, size=max_count, replace=True)
                indices.extend(resampled_idx)
            np.random.shuffle(indices)
            return X_train[indices], y_train[indices], "Native Random Oversampling"
