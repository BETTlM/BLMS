-- Bank Loan Management System (SQLite)
-- Exactly 3 entity sets:
--   1) Customer
--   2) LoanOfficer
--   3) Loan

PRAGMA foreign_keys = ON;

-- CUSTOMER (entity set 1)
CREATE TABLE IF NOT EXISTS customers (
  customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name   TEXT NOT NULL,
  age         INTEGER NOT NULL CHECK (age BETWEEN 18 AND 100),
  gender      TEXT NOT NULL CHECK (gender IN ('M','F','O')),
  city        TEXT NOT NULL,
  phone       TEXT,
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- LOAN_OFFICER (entity set 2)
CREATE TABLE IF NOT EXISTS loan_officers (
  officer_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name        TEXT NOT NULL,
  branch_name      TEXT NOT NULL,
  experience_years INTEGER NOT NULL CHECK (experience_years BETWEEN 0 AND 60),
  created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- LOAN (entity set 3)
CREATE TABLE IF NOT EXISTS loans (
  loan_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  customer_id  INTEGER NOT NULL,
  officer_id   INTEGER NOT NULL,

  -- Loan/application features
  annual_income      REAL NOT NULL CHECK (annual_income > 0),
  employment_years   INTEGER NOT NULL CHECK (employment_years BETWEEN 0 AND 60),
  credit_score       INTEGER NOT NULL CHECK (credit_score BETWEEN 300 AND 900),
  existing_debt      REAL NOT NULL CHECK (existing_debt >= 0),
  loan_amount        REAL NOT NULL CHECK (loan_amount > 0),
  loan_term_months   INTEGER NOT NULL CHECK (loan_term_months IN (6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240, 360)),
  interest_rate      REAL NOT NULL CHECK (interest_rate BETWEEN 0 AND 50),

  -- Human decision (for DBMS integration demos)
  officer_decision_approve INTEGER NOT NULL CHECK (officer_decision_approve IN (0,1)),

  -- Ground truth label for training (historical loans)
  -- NULL for new applications where outcome is unknown
  actual_default INTEGER CHECK (actual_default IN (0,1)),

  -- AI decision (must be updated by ML integration code)
  ai_default_prob REAL CHECK (ai_default_prob BETWEEN 0 AND 1),
  ai_default_pred INTEGER CHECK (ai_default_pred IN (0,1)),

  created_at TEXT NOT NULL DEFAULT (datetime('now')),

  FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE RESTRICT,
  FOREIGN KEY (officer_id)  REFERENCES loan_officers(officer_id) ON DELETE RESTRICT
);

-- Helpful indexes (performance + viva discussion)
CREATE INDEX IF NOT EXISTS idx_loans_customer_id ON loans(customer_id);
CREATE INDEX IF NOT EXISTS idx_loans_officer_id ON loans(officer_id);
CREATE INDEX IF NOT EXISTS idx_loans_actual_default ON loans(actual_default);

