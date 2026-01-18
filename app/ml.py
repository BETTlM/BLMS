from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURES = [
    "annual_income",
    "employment_years",
    "credit_score",
    "existing_debt",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    # derived:
    "dti",  # existing_debt / annual_income
    "lti",  # loan_amount / annual_income
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def build_features_matrix(rows: list[dict]) -> np.ndarray:
    # rows: list of dicts with base columns in the DB
    annual_income = np.array([float(r["annual_income"]) for r in rows], dtype=float)
    employment_years = np.array([float(r["employment_years"]) for r in rows], dtype=float)
    credit_score = np.array([float(r["credit_score"]) for r in rows], dtype=float)
    existing_debt = np.array([float(r["existing_debt"]) for r in rows], dtype=float)
    loan_amount = np.array([float(r["loan_amount"]) for r in rows], dtype=float)
    loan_term_months = np.array([float(r["loan_term_months"]) for r in rows], dtype=float)
    interest_rate = np.array([float(r["interest_rate"]) for r in rows], dtype=float)

    denom = np.maximum(annual_income, 1.0)
    dti = existing_debt / denom
    lti = loan_amount / denom

    X = np.column_stack(
        [
            annual_income,
            employment_years,
            credit_score,
            existing_debt,
            loan_amount,
            loan_term_months,
            interest_rate,
            dti,
            lti,
        ]
    )
    return X


@dataclass
class LogisticModel:
    feature_names: list[str]
    mean_: list[float]
    std_: list[float]
    weights: list[float]
    bias: float
    threshold: float = 0.5

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        mu = np.array(self.mean_, dtype=float)
        sd = np.array(self.std_, dtype=float)
        sd = np.where(sd == 0, 1.0, sd)
        Xn = (X - mu) / sd
        w = np.array(self.weights, dtype=float)
        z = Xn @ w + float(self.bias)
        return _sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return (p >= self.threshold).astype(int)

    def to_json(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "mean_": self.mean_,
            "std_": self.std_,
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold,
        }

    @staticmethod
    def from_json(d: dict) -> "LogisticModel":
        return LogisticModel(
            feature_names=list(d["feature_names"]),
            mean_=list(d["mean_"]),
            std_=list(d["std_"]),
            weights=list(d["weights"]),
            bias=float(d["bias"]),
            threshold=float(d.get("threshold", 0.5)),
        )


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.2,
    steps: int = 1200,
    l2: float = 0.08,
    threshold: float = 0.5,
    seed: int = 7,
) -> LogisticModel:
    rng = np.random.default_rng(seed)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xn = (X - mu) / sd

    n, d = Xn.shape
    w = rng.normal(0, 0.01, size=d)
    b = 0.0

    y = y.astype(float)
    for _ in range(steps):
        z = Xn @ w + b
        p = _sigmoid(z)
        # gradients
        grad_w = (Xn.T @ (p - y)) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b

    return LogisticModel(
        feature_names=list(FEATURES),
        mean_=[float(x) for x in mu],
        std_=[float(x) for x in sd],
        weights=[float(x) for x in w],
        bias=float(b),
        threshold=float(threshold),
    )


def save_model(model: LogisticModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json(), indent=2), encoding="utf-8")


def load_model(path: Path) -> LogisticModel:
    d = json.loads(path.read_text(encoding="utf-8"))
    return LogisticModel.from_json(d)


def explain_instance(model: LogisticModel, X_row: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
    """
    Simple explanation: contribution = standardized_feature * weight.
    Returns top_k by absolute contribution.
    """
    mu = np.array(model.mean_, dtype=float)
    sd = np.array(model.std_, dtype=float)
    sd = np.where(sd == 0, 1.0, sd)
    xz = (X_row.reshape(1, -1) - mu) / sd
    w = np.array(model.weights, dtype=float).reshape(1, -1)
    contrib = (xz * w).ravel()
    pairs = list(zip(model.feature_names, [float(c) for c in contrib]))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return pairs[:top_k]

