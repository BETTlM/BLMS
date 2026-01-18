from __future__ import annotations

from pathlib import Path
import sys

# Allow running as: streamlit run app/ui.py
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from app import db
from app.ml import build_features_matrix, explain_instance, load_model


st.set_page_config(
    page_title="Bank Loan Management + Default Risk AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-aware modern styling (works in light + dark mode)
st.markdown(
    """
<style>
  /* Layout: fluid but not cramped */
  .block-container {
    padding-top: 1.1rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
  }

  /* Theme-aware tokens */
  html[data-theme="light"] {
    --app-card-bg: rgba(255, 255, 255, 0.85);
    --app-card-border: rgba(15, 23, 42, 0.10);
    --app-muted: rgba(15, 23, 42, 0.62);
    --app-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
  }
  html[data-theme="dark"] {
    --app-card-bg: rgba(255, 255, 255, 0.06);
    --app-card-border: rgba(255, 255, 255, 0.12);
    --app-muted: rgba(255, 255, 255, 0.70);
    --app-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
  }

  /* Cards: metrics + forms + expanders */
  div[data-testid="stMetric"],
  div[data-testid="stForm"],
  details[data-testid="stExpander"] > summary,
  details[data-testid="stExpander"] > div {
    background: var(--app-card-bg) !important;
    border: 1px solid var(--app-card-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--app-shadow);
  }

  /* Metric spacing */
  div[data-testid="stMetric"] { padding: 0.75rem 0.9rem; }

  /* Expander summary alignment */
  details[data-testid="stExpander"] > summary { padding: 0.65rem 0.85rem; }
  details[data-testid="stExpander"] > div { padding: 0.6rem 0.85rem 0.85rem; }

  /* DataFrames: softer edges */
  .stDataFrame { border-radius: 14px; overflow: hidden; }

  /* Sidebar polish */
  section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
  section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p {
    color: var(--app-muted);
  }

  /* Buttons: consistent height */
  button[kind="primary"], button[kind="secondary"] { border-radius: 12px; }

  /* Mobile tweaks */
  @media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
  }
</style>
""",
    unsafe_allow_html=True,
)


def _risk_band(p: float) -> str:
    if p < 0.33:
        return "LOW"
    if p < 0.66:
        return "MEDIUM"
    return "HIGH"


def _paths():
    return db.get_paths()


def _conn():
    paths = _paths()
    conn = db.connect(paths.db_path)
    db.init_db(conn)
    return conn


def _ensure_model() -> Path | None:
    paths = _paths()
    model_path = paths.artifacts_dir / "model.json"
    return model_path if model_path.exists() else None


def _kpi(conn):
    c = conn.execute("SELECT COUNT(*) AS n FROM customers").fetchone()["n"]
    o = conn.execute("SELECT COUNT(*) AS n FROM loan_officers").fetchone()["n"]
    l = conn.execute("SELECT COUNT(*) AS n FROM loans").fetchone()["n"]
    labeled = conn.execute("SELECT COUNT(*) AS n FROM loans WHERE actual_default IS NOT NULL").fetchone()["n"]
    dr = conn.execute(
        "SELECT ROUND(AVG(actual_default), 4) AS r FROM loans WHERE actual_default IS NOT NULL"
    ).fetchone()["r"]
    return int(c), int(o), int(l), int(labeled), (float(dr) if dr is not None else None)


def _df(conn, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.DataFrame(db.fetch_df(conn, sql, params))


def _sidebar_admin(conn):
    st.sidebar.subheader("Admin")
    with st.sidebar.expander("Database", expanded=True):
        col1, col2 = st.columns(2)
        if col1.button("Initialize DB", use_container_width=True):
            db.init_db(conn)
            st.success("DB initialized.")
            st.rerun()
        if col2.button("Reset DB", use_container_width=True):
            db.reset_db(conn)
            st.warning("DB reset (tables dropped and re-created).")
            st.rerun()

    with st.sidebar.expander("Synthetic data", expanded=False):
        customers = st.number_input("Customers", min_value=50, max_value=5000, value=400, step=50)
        loans_per_customer = st.number_input("Loans/customer", min_value=1, max_value=10, value=2, step=1)
        seed_value = st.number_input("Seed", min_value=1, max_value=9999, value=7, step=1)
        if st.button("Seed synthetic data", use_container_width=True):
            from app.seed import seed

            seed(int(customers), int(loans_per_customer), int(seed_value))
            st.success("Synthetic data generated and inserted.")
            st.rerun()

    with st.sidebar.expander("Model", expanded=False):
        if st.button("Train / Retrain model", use_container_width=True):
            from app.train import train

            out = train()
            st.success(f"Model trained: {out.name}")
            st.rerun()

        model_path = _ensure_model()
        if model_path:
            st.caption(f"Model: `{model_path.name}`")
        else:
            st.caption("Model: not trained yet.")


def page_dashboard(conn):
    st.title("🏦 Bank Loan Management + Default Risk AI")
    st.caption("3-entity-set DB (Customer, LoanOfficer, Loan) + ML integration (predict + store AI decision).")

    c, o, l, labeled, dr = _kpi(conn)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers", f"{c}")
    k2.metric("Loan officers", f"{o}")
    k3.metric("Loans", f"{l}")
    k4.metric("Labeled loans (trainable)", f"{labeled}")
    k5.metric("Historical default rate", f"{dr:.2%}" if dr is not None else "—")

    st.divider()

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Recent loans (with AI fields)")
        df = _df(
            conn,
            """
            SELECT
              l.loan_id,
              c.full_name AS customer_name,
              o.branch_name,
              l.loan_amount,
              l.credit_score,
              l.officer_decision_approve,
              l.actual_default,
              l.ai_default_prob,
              l.ai_default_pred,
              l.created_at
            FROM loans l
            JOIN customers c ON c.customer_id = l.customer_id
            JOIN loan_officers o ON o.officer_id = l.officer_id
            ORDER BY l.loan_id DESC
            LIMIT 25
            """,
        )
        st.dataframe(df, use_container_width=True, height=420)

    with right:
        st.subheader("Analytics")
        model_path = _ensure_model()
        if model_path:
            model = load_model(model_path)
            w = pd.DataFrame(
                {"feature": model.feature_names, "weight": model.weights}
            ).sort_values("weight", ascending=False)
            st.caption("Model weights (higher → more default risk after standardization)")
            st.bar_chart(w.set_index("feature")[["weight"]])
        else:
            st.info("Train the model to see AI insights (sidebar → Model).")

        by_branch = _df(
            conn,
            """
            SELECT
              o.branch_name,
              COUNT(*) AS total_loans,
              AVG(CASE WHEN l.actual_default IS NULL THEN NULL ELSE l.actual_default END) AS default_rate
            FROM loans l
            JOIN loan_officers o ON o.officer_id = l.officer_id
            WHERE l.actual_default IS NOT NULL
            GROUP BY o.branch_name
            ORDER BY default_rate DESC
            """,
        )
        if not by_branch.empty:
            chart = by_branch.set_index("branch_name")[["default_rate"]]
            st.bar_chart(chart)
        else:
            st.info("No labeled loans yet. Seed data first.")

        st.subheader("Officer approval vs AI risk")
        mismatch = _df(
            conn,
            """
            SELECT
              l.loan_id,
              c.full_name AS customer_name,
              l.loan_amount,
              l.officer_decision_approve,
              l.ai_default_prob,
              l.ai_default_pred
            FROM loans l
            JOIN customers c ON c.customer_id = l.customer_id
            WHERE l.ai_default_pred IS NOT NULL
              AND l.officer_decision_approve = 1
              AND l.ai_default_pred = 1
            ORDER BY l.ai_default_prob DESC
            LIMIT 10
            """,
        )
        st.dataframe(mismatch, use_container_width=True, height=240)


def page_new_application(conn):
    st.title("New Loan Application")
    st.caption("Enter applicant + loan details → AI predicts default risk → result is stored in `loans` table.")

    model_path = _ensure_model()
    if not model_path:
        st.warning("Train the model first (sidebar → Model → Train).")
        return
    model = load_model(model_path)

    officers = _df(conn, "SELECT officer_id, full_name, branch_name FROM loan_officers ORDER BY officer_id")
    if officers.empty:
        st.warning("No loan officers in DB. Seed synthetic data first.")
        return

    tab1, tab2 = st.tabs(["Create new customer", "Use existing customer"])

    customer_id = None
    with tab1:
        with st.form("new_customer_form", border=True):
            c1, c2, c3 = st.columns(3)
            full_name = c1.text_input("Full name", value="")
            age = c2.number_input("Age", min_value=18, max_value=100, value=30)
            gender = c3.selectbox("Gender", options=["M", "F", "O"], index=2)
            c4, c5 = st.columns(2)
            city = c4.text_input("City", value="Kochi")
            phone = c5.text_input("Phone", value="+91-")
            create_customer = st.form_submit_button("Create customer", use_container_width=True)
        if create_customer:
            cur = conn.execute(
                "INSERT INTO customers(full_name, age, gender, city, phone) VALUES(?,?,?,?,?)",
                (full_name or "New Customer", int(age), gender, city, phone),
            )
            conn.commit()
            customer_id = int(cur.lastrowid)
            st.success(f"Customer created: customer_id={customer_id}")

    with tab2:
        customers = _df(conn, "SELECT customer_id, full_name, city FROM customers ORDER BY customer_id DESC LIMIT 500")
        if customers.empty:
            st.info("No customers yet. Create one in the first tab.")
        else:
            customers["label"] = customers["customer_id"].astype(str) + " — " + customers["full_name"] + " (" + customers["city"] + ")"
            picked = st.selectbox("Select customer", customers["label"].tolist())
            if picked:
                customer_id = int(picked.split("—")[0].strip())

    st.divider()

    if customer_id is None:
        st.info("Select or create a customer to continue.")
        return

    st.subheader("Loan details")
    with st.form("loan_form", border=True):
        col1, col2, col3 = st.columns(3)
        officer_label = (
            officers["officer_id"].astype(str)
            + " — "
            + officers["full_name"]
            + " ("
            + officers["branch_name"]
            + ")"
        ).tolist()
        officer_pick = col1.selectbox("Assign loan officer", officer_label)
        annual_income = col2.number_input("Annual income (INR/year)", min_value=50000.0, max_value=10000000.0, value=650000.0, step=10000.0)
        employment_years = col3.number_input("Employment years", min_value=0, max_value=60, value=5, step=1)

        col4, col5, col6 = st.columns(3)
        credit_score = col4.number_input("Credit score", min_value=300, max_value=900, value=700, step=1)
        existing_debt = col5.number_input("Existing debt (INR)", min_value=0.0, max_value=10000000.0, value=100000.0, step=5000.0)
        loan_amount = col6.number_input("Loan amount (INR)", min_value=10000.0, max_value=10000000.0, value=450000.0, step=10000.0)

        col7, col8, col9 = st.columns(3)
        loan_term_months = col7.selectbox("Loan term (months)", options=[12, 24, 36, 48, 60, 72, 120, 180, 240, 360], index=4)
        interest_rate = col8.number_input("Interest rate (%)", min_value=1.0, max_value=50.0, value=12.0, step=0.25)
        officer_decision_approve = col9.selectbox("Officer decision", options=[1, 0], format_func=lambda x: "Approve" if x == 1 else "Reject")

        submit = st.form_submit_button("Predict default risk and store", use_container_width=True)

    if not submit:
        return

    officer_id = int(officer_pick.split("—")[0].strip())
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
            int(customer_id),
            officer_id,
            float(annual_income),
            int(employment_years),
            int(credit_score),
            float(existing_debt),
            float(loan_amount),
            int(loan_term_months),
            float(interest_rate),
            int(officer_decision_approve),
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
    band = _risk_band(prob)

    conn.execute(
        "UPDATE loans SET ai_default_prob = ?, ai_default_pred = ? WHERE loan_id = ?",
        (prob, pred, loan_id),
    )
    conn.commit()

    st.success(f"Stored loan_id={loan_id} with AI decision.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Default probability", f"{prob:.2%}")
    c2.metric("AI prediction (1=default)", f"{pred}")
    c3.metric("Risk band", band)

    st.subheader("Reason codes (simple explanation)")
    expl = explain_instance(model, X[0], top_k=6)
    expl_df = pd.DataFrame(expl, columns=["feature", "contribution"]).assign(
        contribution=lambda d: d["contribution"].round(4)
    )
    st.dataframe(expl_df, use_container_width=True)


def page_browse(conn):
    st.title("Browse database")
    tab1, tab2, tab3 = st.tabs(["Loans", "Customers", "Loan officers"])

    with tab1:
        df = _df(
            conn,
            """
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
            JOIN loan_officers o ON o.officer_id = l.officer_id
            ORDER BY l.loan_id DESC
            LIMIT 1000
            """,
        )
        st.dataframe(df, use_container_width=True, height=520)

    with tab2:
        df = _df(conn, "SELECT * FROM customers ORDER BY customer_id DESC LIMIT 1000")
        st.dataframe(df, use_container_width=True, height=520)

    with tab3:
        df = _df(conn, "SELECT * FROM loan_officers ORDER BY officer_id")
        st.dataframe(df, use_container_width=True, height=520)


def page_sql(conn):
    st.title("SQL Queries (Viva)")
    st.caption("Run prepared queries or write your own SELECT queries (read-only).")

    queries = {
        "Recent loans with joins": """
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
ORDER BY l.loan_id DESC
LIMIT 30;
""".strip(),
        "Default rate by branch": """
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
""".strip(),
        "Officer approvals vs AI high risk": """
SELECT
  l.loan_id,
  c.full_name AS customer_name,
  l.loan_amount,
  l.officer_decision_approve,
  l.ai_default_prob,
  l.ai_default_pred
FROM loans l
JOIN customers c ON c.customer_id = l.customer_id
WHERE l.ai_default_pred IS NOT NULL
  AND l.officer_decision_approve = 1
  AND l.ai_default_pred = 1
ORDER BY l.ai_default_prob DESC
LIMIT 20;
""".strip(),
    }

    choice = st.selectbox("Prepared query", list(queries.keys()))
    sql = st.text_area("SQL", value=queries[choice], height=220)
    run = st.button("Run", use_container_width=True)
    if run:
        if not sql.strip().lower().startswith("select"):
            st.error("Only SELECT queries are allowed here.")
            return
        try:
            df = _df(conn, sql)
            st.dataframe(df, use_container_width=True, height=520)
        except Exception as e:
            st.error(str(e))


def page_export(conn):
    st.title("Export")
    st.caption("Download CSV exports for demo/report submission.")

    loans = _df(
        conn,
        """
        SELECT
          l.loan_id, l.customer_id, l.officer_id,
          l.annual_income, l.employment_years, l.credit_score, l.existing_debt,
          l.loan_amount, l.loan_term_months, l.interest_rate,
          l.officer_decision_approve, l.actual_default, l.ai_default_prob, l.ai_default_pred,
          l.created_at
        FROM loans l
        ORDER BY l.loan_id
        """,
    )
    customers = _df(conn, "SELECT * FROM customers ORDER BY customer_id")
    officers = _df(conn, "SELECT * FROM loan_officers ORDER BY officer_id")

    st.download_button(
        "Download loans.csv",
        data=loans.to_csv(index=False).encode("utf-8"),
        file_name="loans.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Download customers.csv",
        data=customers.to_csv(index=False).encode("utf-8"),
        file_name="customers.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Download loan_officers.csv",
        data=officers.to_csv(index=False).encode("utf-8"),
        file_name="loan_officers.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main():
    conn = _conn()
    _sidebar_admin(conn)

    st.sidebar.divider()
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Dashboard", "New application", "Browse DB", "SQL (viva)", "Export"],
        label_visibility="collapsed",
    )

    if page == "Dashboard":
        page_dashboard(conn)
    elif page == "New application":
        page_new_application(conn)
    elif page == "Browse DB":
        page_browse(conn)
    elif page == "SQL (viva)":
        page_sql(conn)
    elif page == "Export":
        page_export(conn)


if __name__ == "__main__":
    main()

