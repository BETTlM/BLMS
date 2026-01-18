from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as: python3 app/seed.py
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import db
from app.synth import generate_customers, generate_loans_for_customer, generate_officers


def seed(n_customers: int, loans_per_customer: int, seed_value: int) -> None:
    paths = db.get_paths()
    conn = db.connect(paths.db_path)
    db.init_db(conn)

    rng = __import__("numpy").random.default_rng(seed_value)

    # Officers
    officers = generate_officers(rng)
    for o in officers:
        conn.execute(
            "INSERT INTO loan_officers(full_name, branch_name, experience_years) VALUES(?,?,?)",
            (o.full_name, o.branch_name, o.experience_years),
        )
    conn.commit()

    officer_ids = [r["officer_id"] for r in conn.execute("SELECT officer_id FROM loan_officers").fetchall()]
    if not officer_ids:
        raise RuntimeError("No loan officers created.")

    # Customers + loans
    customers = generate_customers(rng, n_customers)
    for c in customers:
        cur = conn.execute(
            "INSERT INTO customers(full_name, age, gender, city, phone) VALUES(?,?,?,?,?)",
            (c.full_name, c.age, c.gender, c.city, c.phone),
        )
        customer_id = int(cur.lastrowid)

        loans = generate_loans_for_customer(rng, loans_per_customer, historical=True)
        for l in loans:
            officer_id = int(rng.choice(officer_ids))
            conn.execute(
                """
                INSERT INTO loans(
                  customer_id, officer_id,
                  annual_income, employment_years, credit_score, existing_debt,
                  loan_amount, loan_term_months, interest_rate,
                  officer_decision_approve, actual_default
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    customer_id,
                    officer_id,
                    l.annual_income,
                    l.employment_years,
                    l.credit_score,
                    l.existing_debt,
                    l.loan_amount,
                    l.loan_term_months,
                    l.interest_rate,
                    l.officer_decision_approve,
                    l.actual_default,
                ),
            )
    conn.commit()

    # Quick summary
    total_customers = conn.execute("SELECT COUNT(*) AS n FROM customers").fetchone()["n"]
    total_loans = conn.execute("SELECT COUNT(*) AS n FROM loans").fetchone()["n"]
    labeled = conn.execute("SELECT COUNT(*) AS n FROM loans WHERE actual_default IS NOT NULL").fetchone()["n"]
    default_rate = conn.execute(
        "SELECT ROUND(AVG(actual_default), 4) AS r FROM loans WHERE actual_default IS NOT NULL"
    ).fetchone()["r"]

    print(f"Seeded DB at: {paths.db_path}")
    print(f"Customers: {total_customers}, Loans: {total_loans}, Labeled loans: {labeled}, Default rate: {default_rate}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--customers", type=int, default=400)
    p.add_argument("--loans-per-customer", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    seed(args.customers, args.loans_per_customer, args.seed)


if __name__ == "__main__":
    main()

