from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running as: python3 app/predict_and_store.py
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from app import db
from app.ml import build_features_matrix, explain_instance, load_model


def _risk_band(p: float) -> str:
    if p < 0.33:
        return "LOW"
    if p < 0.66:
        return "MEDIUM"
    return "HIGH"


def predict_and_store(
    *,
    model_path: Path | None = None,
    customer_id: int | None = None,
) -> None:
    paths = db.get_paths()
    conn = db.connect(paths.db_path)
    db.init_db(conn)

    model_file = model_path or (paths.artifacts_dir / "model.json")
    if not model_file.exists():
        raise RuntimeError("Model not found. Run: python3 app/train.py")
    model = load_model(model_file)

    print("=== New Loan Application (will be stored + predicted) ===")

    if customer_id is None:
        full_name = input("Customer full name: ").strip() or "New Customer"
        age = int(input("Age (18-100): ").strip() or "30")
        gender = (input("Gender [M/F/O]: ").strip().upper() or "O")
        city = input("City: ").strip() or "Kochi"
        phone = input("Phone: ").strip() or "+91-0000000000"
        cur = conn.execute(
            "INSERT INTO customers(full_name, age, gender, city, phone) VALUES(?,?,?,?,?)",
            (full_name, age, gender, city, phone),
        )
        customer_id = int(cur.lastrowid)
        conn.commit()
    else:
        exists = conn.execute("SELECT 1 FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
        if not exists:
            raise RuntimeError(f"customer_id {customer_id} not found.")

    # pick an officer (first one) if exists
    officer_row = conn.execute("SELECT officer_id FROM loan_officers ORDER BY officer_id LIMIT 1").fetchone()
    if not officer_row:
        raise RuntimeError("No loan officers found. Run: python3 app/seed.py (it creates officers).")
    officer_id = int(officer_row["officer_id"])

    annual_income = float(input("Annual income (INR/year): ").strip() or "650000")
    employment_years = int(input("Employment years: ").strip() or "5")
    credit_score = int(input("Credit score (300-900): ").strip() or "700")
    existing_debt = float(input("Existing debt (INR): ").strip() or "100000")
    loan_amount = float(input("Loan amount (INR): ").strip() or "450000")
    loan_term_months = int(input("Loan term months (e.g. 12/24/36/60/120/240/360): ").strip() or "60")
    interest_rate = float(input("Interest rate (%): ").strip() or "12")
    officer_decision_approve = int(input("Officer decision approve (1=yes,0=no): ").strip() or "1")

    cur = conn.execute(
        """
        INSERT INTO loans(
          customer_id, officer_id,
          annual_income, employment_years, credit_score, existing_debt,
          loan_amount, loan_term_months, interest_rate,
          officer_decision_approve, actual_default
        ) VALUES(?,?,?,?,?,?,?,?,?,?,NULL)
        """,
        (
            customer_id,
            officer_id,
            annual_income,
            employment_years,
            credit_score,
            existing_debt,
            loan_amount,
            loan_term_months,
            interest_rate,
            officer_decision_approve,
        ),
    )
    loan_id = int(cur.lastrowid)
    conn.commit()

    row = db.fetch_df(
        conn,
        """
        SELECT annual_income, employment_years, credit_score, existing_debt,
               loan_amount, loan_term_months, interest_rate
        FROM loans WHERE loan_id = ?
        """,
        (loan_id,),
    )[0]

    X = build_features_matrix([row])
    prob = float(model.predict_proba(X)[0])
    pred = int(prob >= model.threshold)

    conn.execute(
        "UPDATE loans SET ai_default_prob = ?, ai_default_pred = ? WHERE loan_id = ?",
        (prob, pred, loan_id),
    )
    conn.commit()

    band = _risk_band(prob)
    expl = explain_instance(model, X[0], top_k=5)

    print()
    print(f"Stored loan_id: {loan_id}")
    print(f"AI default probability: {prob:.4f}  |  AI prediction (1=default): {pred}  |  Risk band: {band}")
    print("Top contributing features (standardized * weight):")
    for name, val in expl:
        print(f"  - {name}: {val:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="")
    p.add_argument("--customer-id", type=int, default=0)
    args = p.parse_args()
    predict_and_store(
        model_path=Path(args.model) if args.model else None,
        customer_id=args.customer_id if args.customer_id != 0 else None,
    )


if __name__ == "__main__":
    main()

