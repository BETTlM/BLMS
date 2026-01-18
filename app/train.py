from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running as: python3 app/train.py
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from app import db
from app.ml import build_features_matrix, save_model, train_logistic_regression


def train(model_path: Path | None = None) -> Path:
    paths = db.get_paths()
    conn = db.connect(paths.db_path)
    db.init_db(conn)

    rows = db.fetch_df(
        conn,
        """
        SELECT
          annual_income, employment_years, credit_score, existing_debt,
          loan_amount, loan_term_months, interest_rate,
          actual_default
        FROM loans
        WHERE actual_default IS NOT NULL
        """,
    )
    if len(rows) < 50:
        raise RuntimeError("Not enough labeled loans to train. Run: python3 app/seed.py")

    y = np.array([int(r["actual_default"]) for r in rows], dtype=int)
    X = build_features_matrix(rows)

    model = train_logistic_regression(X, y)

    # quick metrics (for demo)
    p = model.predict_proba(X)
    pred = (p >= model.threshold).astype(int)
    acc = float((pred == y).mean())
    base = float(y.mean())

    out = model_path or (paths.artifacts_dir / "model.json")
    save_model(model, out)
    print(f"Trained model saved to: {out}")
    print(f"Train accuracy (on synthetic data): {acc:.4f}")
    print(f"Default rate baseline: {base:.4f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()
    out = Path(args.out) if args.out else None
    train(out)


if __name__ == "__main__":
    main()

