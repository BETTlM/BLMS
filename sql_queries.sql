-- Viva-ready SQL queries (SQLite)
PRAGMA foreign_keys = ON;

-- 1) Show all loans with customer + officer (JOIN)
SELECT
  l.loan_id,
  c.full_name AS customer_name,
  o.full_name AS officer_name,
  l.loan_amount,
  l.credit_score,
  l.actual_default,
  l.ai_default_prob,
  l.ai_default_pred,
  l.created_at
FROM loans l
JOIN customers c ON c.customer_id = l.customer_id
JOIN loan_officers o ON o.officer_id = l.officer_id
ORDER BY l.loan_id DESC;

-- 2) Count loans per officer (GROUP BY)
SELECT
  o.officer_id,
  o.full_name,
  o.branch_name,
  COUNT(*) AS total_loans_handled
FROM loan_officers o
JOIN loans l ON l.officer_id = o.officer_id
GROUP BY o.officer_id, o.full_name, o.branch_name
ORDER BY total_loans_handled DESC;

-- 3) Default rate per branch (GROUP BY + aggregation)
SELECT
  o.branch_name,
  COUNT(*) AS total_loans,
  SUM(CASE WHEN l.actual_default = 1 THEN 1 ELSE 0 END) AS defaults,
  ROUND(1.0 * SUM(CASE WHEN l.actual_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS default_rate
FROM loans l
JOIN loan_officers o ON o.officer_id = l.officer_id
WHERE l.actual_default IS NOT NULL
GROUP BY o.branch_name
ORDER BY default_rate DESC;

-- 4) Customers with high debt-to-income ratio (computed column)
SELECT
  c.customer_id,
  c.full_name,
  l.loan_id,
  l.existing_debt,
  l.annual_income,
  ROUND(l.existing_debt / l.annual_income, 4) AS dti
FROM loans l
JOIN customers c ON c.customer_id = l.customer_id
WHERE (l.existing_debt / l.annual_income) >= 0.4
ORDER BY dti DESC;

-- 5) Correlated subquery: latest loan per customer
SELECT
  c.customer_id,
  c.full_name,
  l.loan_id,
  l.created_at,
  l.loan_amount
FROM customers c
JOIN loans l ON l.customer_id = c.customer_id
WHERE l.created_at = (
  SELECT MAX(l2.created_at) FROM loans l2 WHERE l2.customer_id = c.customer_id
)
ORDER BY c.customer_id;

-- 6) View: readable loan dashboard
CREATE VIEW IF NOT EXISTS v_loan_dashboard AS
SELECT
  l.loan_id,
  c.full_name AS customer_name,
  o.full_name AS officer_name,
  o.branch_name,
  l.loan_amount,
  l.loan_term_months,
  l.interest_rate,
  l.credit_score,
  l.officer_decision_approve,
  l.actual_default,
  l.ai_default_prob,
  l.ai_default_pred,
  l.created_at
FROM loans l
JOIN customers c ON c.customer_id = l.customer_id
JOIN loan_officers o ON o.officer_id = l.officer_id;

SELECT * FROM v_loan_dashboard ORDER BY loan_id DESC LIMIT 20;

-- 7) Find potential mismatches between human approval and AI default prediction
SELECT
  loan_id,
  customer_name,
  officer_name,
  loan_amount,
  officer_decision_approve,
  ai_default_prob,
  ai_default_pred
FROM v_loan_dashboard
WHERE ai_default_pred IS NOT NULL
  AND officer_decision_approve = 1
  AND ai_default_pred = 1
ORDER BY ai_default_prob DESC;

