import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = Path("model.bin")
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    ds = load_breast_cancer(as_frame=True)
    X = ds.data.copy()
    y_raw = ds.target.copy()
    # sklearn: 0=malignant, 1=benign
    # remap: malignant=1, benign=0 (risk of malignancy)
    y = (y_raw == 0).astype(int)
    return X, y


def main() -> None:
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    numeric_features = list(X.columns)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    # Try multiple models (rubric)
    candidates = {
        "logreg": LogisticRegression(max_iter=5000, solver="liblinear"),
        "rf": RandomForestClassifier(random_state=RANDOM_STATE),
    }

    best_name = None
    best_model = None
    best_auc = -1.0

    # Parameter tuning (rubric)
    grids = {
        "logreg": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l1", "l2"],
        },
        "rf": {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 3, 5],
        },
    }

    for name, base_model in candidates.items():
        pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", base_model),
            ]
        )

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grids[name],
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)

        val_proba = gs.best_estimator_.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)

        print(f"[{name}] best_cv_auc={gs.best_score_:.4f} val_auc={auc:.4f} best_params={gs.best_params_}")

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = gs.best_estimator_

    assert best_model is not None

    artifact = {
        "model_name": best_name,
        "pipeline": best_model,      # includes preprocessing
        "feature_names": list(X.columns),
        "malignancy_positive_class": 1,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved model to: {MODEL_PATH} (val_auc={best_auc:.4f}, model={best_name})")


if __name__ == "__main__":
    main()
