from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ("short", "flat", "long")


@dataclass(frozen=True)
class ValidationResult:
    fold: int
    log_loss: float
    report: str


def make_model(random_state: int) -> VotingClassifier:
    linear = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    boosting = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=300,
                    l2_regularization=0.2,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return VotingClassifier(
        estimators=[("linear", linear), ("forest", forest), ("boosting", boosting)],
        voting="soft",
        weights=[1, 2, 2],
    )


def validate_model(
    x: pd.DataFrame,
    y: pd.Series,
    splits: int,
    embargo_bars: int,
    random_state: int,
) -> list[ValidationResult]:
    splitter = TimeSeriesSplit(n_splits=splits)
    results: list[ValidationResult] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x), start=1):
        if embargo_bars:
            train_idx = train_idx[train_idx < test_idx[0] - embargo_bars]
        model = make_model(random_state)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        probabilities = model.predict_proba(x.iloc[test_idx])
        predictions = model.predict(x.iloc[test_idx])
        results.append(
            ValidationResult(
                fold=fold,
                log_loss=float(log_loss(y.iloc[test_idx], probabilities, labels=list(model.classes_))),
                report=classification_report(y.iloc[test_idx], predictions, labels=list(LABELS), zero_division=0),
            )
        )
    return results


def fit_model(x: pd.DataFrame, y: pd.Series, random_state: int) -> VotingClassifier:
    model = make_model(random_state)
    model.fit(x, y)
    return model


def save_artifact(model: BaseEstimator, feature_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, path)


def load_artifact(path: Path) -> tuple[BaseEstimator, list[str]]:
    artifact = joblib.load(path)
    return artifact["model"], artifact["feature_names"]


def signal_from_probabilities(
    classes: np.ndarray,
    probabilities: np.ndarray,
    min_confidence: float,
) -> dict[str, float | str]:
    best_idx = int(np.argmax(probabilities))
    label = str(classes[best_idx])
    confidence = float(probabilities[best_idx])
    if confidence < min_confidence:
        label = "flat"
    return {
        "signal": label,
        "confidence": confidence,
        **{f"prob_{klass}": float(prob) for klass, prob in zip(classes, probabilities, strict=True)},
    }
