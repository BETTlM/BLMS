from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Gender = Literal["M", "F", "O"]


@dataclass(frozen=True)
class CustomerRow:
    full_name: str
    age: int
    gender: Gender
    city: str
    phone: str


@dataclass(frozen=True)
class OfficerRow:
    full_name: str
    branch_name: str
    experience_years: int


@dataclass(frozen=True)
class LoanRow:
    annual_income: float
    employment_years: int
    credit_score: int
    existing_debt: float
    loan_amount: float
    loan_term_months: int
    interest_rate: float
    officer_decision_approve: int  # 0/1
    actual_default: int | None  # 0/1 for historical; None for new app


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def _choose(rng: np.random.Generator, items: list[str], n: int) -> list[str]:
    idx = rng.integers(0, len(items), size=n)
    return [items[i] for i in idx]


def generate_customers(rng: np.random.Generator, n: int) -> list[CustomerRow]:
    first = ["Aarav", "Aditi", "Arjun", "Ananya", "Dev", "Diya", "Ishan", "Kavya", "Nikhil", "Priya", "Rahul", "Sneha"]
    last = ["Sharma", "Nair", "Iyer", "Menon", "Patel", "Reddy", "Khan", "Das", "Gupta", "Singh", "Joshi", "Kumar"]
    cities = ["Kochi", "Trivandrum", "Kozhikode", "Thrissur", "Bengaluru", "Chennai", "Hyderabad", "Mumbai"]
    genders: list[Gender] = ["M", "F", "O"]

    rows: list[CustomerRow] = []
    for _ in range(n):
        name = f"{rng.choice(first)} {rng.choice(last)}"
        age = int(rng.integers(21, 66))
        gender = genders[int(rng.integers(0, 3))]
        city = str(rng.choice(cities))
        phone = f"+91-{int(rng.integers(6000000000, 9999999999))}"
        rows.append(CustomerRow(full_name=name, age=age, gender=gender, city=city, phone=phone))
    return rows


def generate_officers(rng: np.random.Generator) -> list[OfficerRow]:
    names = ["S. Mathew", "R. Nair", "A. Menon", "P. Varma", "K. Iqbal"]
    branches = ["Kochi Main", "Trivandrum", "Kozhikode", "Thrissur", "Bengaluru"]
    rows: list[OfficerRow] = []
    for i in range(len(names)):
        rows.append(
            OfficerRow(
                full_name=names[i],
                branch_name=branches[i],
                experience_years=int(rng.integers(2, 25)),
            )
        )
    return rows


def generate_loans_for_customer(
    rng: np.random.Generator,
    n: int,
    historical: bool = True,
) -> list[LoanRow]:
    # Base features
    annual_income = rng.lognormal(mean=np.log(650000), sigma=0.55, size=n)  # INR/year
    annual_income = np.clip(annual_income, 150000, 5000000)

    employment_years = rng.integers(0, 31, size=n)

    credit_score = rng.normal(loc=690, scale=70, size=n)
    credit_score = np.clip(credit_score, 300, 900).round().astype(int)

    existing_debt = rng.lognormal(mean=np.log(120000), sigma=0.8, size=n)
    existing_debt = np.clip(existing_debt, 0, 2500000)

    loan_amount = rng.lognormal(mean=np.log(450000), sigma=0.7, size=n)
    loan_amount = np.clip(loan_amount, 30000, 5000000)

    term_choices = np.array([12, 24, 36, 48, 60, 72, 120, 180, 240, 360])
    loan_term_months = rng.choice(term_choices, size=n, replace=True)

    # Interest rate correlates with credit_score and term (roughly)
    base_rate = 22 - (credit_score - 300) * (12 / 600)  # lower score -> higher rate
    term_bump = (loan_term_months / 360) * 6
    noise = rng.normal(0, 1.25, size=n)
    interest_rate = np.clip(base_rate + term_bump + noise, 6.0, 36.0)

    # Officer decision rule-of-thumb (human label for DB demo)
    dti = existing_debt / np.maximum(annual_income, 1.0)
    lti = loan_amount / np.maximum(annual_income, 1.0)
    approve_score = (
        0.008 * (credit_score - 600)
        - 2.2 * dti
        - 0.9 * lti
        + 0.05 * employment_years
        - 0.03 * (interest_rate - 12)
    )
    officer_decision_approve = (approve_score > 0.15).astype(int)

    # Actual default probability (synthetic ground truth)
    # Higher when: low credit score, high DTI/LTI, high interest rate, low employment.
    z = (
        -1.2
        - 0.010 * (credit_score - 700)
        + 3.5 * dti
        + 1.5 * lti
        + 0.06 * (interest_rate - 12)
        - 0.05 * employment_years
        + rng.normal(0, 0.35, size=n)
    )
    p_default = _sigmoid(z)

    if historical:
        actual_default = (rng.random(size=n) < p_default).astype(int)
        actual_default_list: list[int | None] = [int(x) for x in actual_default]
    else:
        actual_default_list = [None for _ in range(n)]

    rows: list[LoanRow] = []
    for i in range(n):
        rows.append(
            LoanRow(
                annual_income=float(annual_income[i]),
                employment_years=int(employment_years[i]),
                credit_score=int(credit_score[i]),
                existing_debt=float(existing_debt[i]),
                loan_amount=float(loan_amount[i]),
                loan_term_months=int(loan_term_months[i]),
                interest_rate=float(interest_rate[i]),
                officer_decision_approve=int(officer_decision_approve[i]),
                actual_default=actual_default_list[i],
            )
        )
    return rows

