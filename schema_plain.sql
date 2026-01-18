-- Plain SQL Schema (SQLite-compatible, no PRAGMA)
-- Entity sets (exactly 3):
--   1) customers
--   2) loan_officers
--   3) loans
--
-- Notes:
-- - This file is intentionally kept "plain": only CREATE TABLE + CREATE INDEX.
-- - Uses SQLite-friendly defaults like CURRENT_TIMESTAMP (no datetime('now')).
-- - If you want a MySQL/PostgreSQL version, the main change is the auto-id syntax.

-- 1) CUSTOMER
CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name   TEXT NOT NULL,
  age         INTEGER NOT NULL CHECK (age BETWEEN 18 AND 100),
  gender      TEXT NOT NULL CHECK (gender IN ('M','F','O')),
  city        TEXT NOT NULL,
  phone       TEXT,
  created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2) LOAN_OFFICER
CREATE TABLE loan_officers (
  officer_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name        TEXT NOT NULL,
  branch_name      TEXT NOT NULL,
  experience_years INTEGER NOT NULL CHECK (experience_years BETWEEN 0 AND 60),
  created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 3) LOAN
CREATE TABLE loans (
  loan_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  customer_id  INTEGER NOT NULL,
  officer_id   INTEGER NOT NULL,

  annual_income     REAL NOT NULL CHECK (annual_income > 0),
  employment_years  INTEGER NOT NULL CHECK (employment_years BETWEEN 0 AND 60),
  credit_score      INTEGER NOT NULL CHECK (credit_score BETWEEN 300 AND 900),
  existing_debt     REAL NOT NULL CHECK (existing_debt >= 0),
  loan_amount       REAL NOT NULL CHECK (loan_amount > 0),
  loan_term_months  INTEGER NOT NULL CHECK (loan_term_months IN (6,12,18,24,36,48,60,72,84,96,120,180,240,360)),
  interest_rate     REAL NOT NULL CHECK (interest_rate BETWEEN 0 AND 50),

  -- Human decision
  officer_decision_approve INTEGER NOT NULL CHECK (officer_decision_approve IN (0,1)),

  -- Ground truth label (historical loans); NULL for new applications
  actual_default INTEGER CHECK (actual_default IN (0,1)),

  -- AI decision (updated by ML integration)
  ai_default_prob REAL CHECK (ai_default_prob BETWEEN 0 AND 1),
  ai_default_pred INTEGER CHECK (ai_default_pred IN (0,1)),

  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

  FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE RESTRICT,
  FOREIGN KEY (officer_id) REFERENCES loan_officers(officer_id) ON DELETE RESTRICT
);

-- Indexes
CREATE INDEX idx_loans_customer_id ON loans(customer_id);
CREATE INDEX idx_loans_officer_id ON loans(officer_id);
CREATE INDEX idx_loans_actual_default ON loans(actual_default);

