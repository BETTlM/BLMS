## Bank Loan Management System with Default Risk Analysis (DBMS + ML)

Batch: S4 AID  
Course: 23AID214 – DBMS Mini Project List (Integration with AI/ML)

This mini-project demonstrates **end-to-end integration** of:
- a **3-entity-set relational database** (SQLite)
- a **binary classification model** (default risk)
- an **integration pipeline** that **inserts new loan data → predicts → stores AI decision** in the database.

### Entity sets (exactly 3)
- **Customer**
- **LoanOfficer**
- **Loan**

The **ML target** is: **Will this loan default?** (binary: 1 = default, 0 = no default).

### ER diagram
See `er_diagram.md`.

### Project structure
- `schema.sql`: SQLite schema (tables, FKs, indexes)
- `schema_plain.sql`: **plain/standard SQL** version of the schema (for writing in record / lab manual)
- `sql_queries.sql`: viva-ready SQL queries (joins, group by, subquery, view, etc.)
- `app/seed.py`: generate + insert **synthetic data** into SQLite
- `app/train.py`: train the default-risk model and save weights to `artifacts/model.json`
- `app/predict_and_store.py`: interactive script that:
  - inserts customer + loan application
  - loads the model
  - predicts default probability
  - updates the `Loan` row with `ai_default_prob` and `ai_default_pred`
- `app/ui.py`: **modern GUI** (Streamlit) with dashboard + new application form + analytics + export

### How to run (recommended order)
From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python3 app/seed.py
python3 app/train.py
python3 app/predict_and_store.py
```

This creates:
- `artifacts/bank_loan.db`
- `artifacts/model.json`

### Run the modern GUI (recommended for final demo)
```bash
streamlit run app/ui.py
```

### What to show in evaluation
- **Mid-term**:
  - ER diagram (`er_diagram.md`)
  - schema (`schema.sql`) with exactly 3 entity sets
  - ML problem definition: **default risk prediction**
- **Final**:
  - run the GUI: `streamlit run app/ui.py`
  - Seed synthetic data (sidebar) → Train model (sidebar)
  - Create a **New Loan Application** (page)
  - Show prediction stored in DB:
    - `loans.ai_default_prob`
    - `loans.ai_default_pred`

### Notes / assumptions (reasonable + realistic)
- The synthetic dataset simulates typical banking features:
  - `annual_income`, `employment_years`, `credit_score`, `existing_debt`,
    `loan_amount`, `loan_term_months`, `interest_rate`
- Historical loans (synthetic) include `actual_default` for training.
- New applications store `actual_default = NULL` (unknown at application time).

