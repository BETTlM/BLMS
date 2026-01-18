## ER Diagram (3 Entity Sets)

### Entity sets
- **Customer**
- **LoanOfficer**
- **Loan**

### Relationships
- **Customer (1) — (M) Loan**
  - A customer can have many loans/applications.
  - Each loan belongs to exactly one customer.
- **LoanOfficer (1) — (M) Loan**
  - A loan officer can handle many loans/applications.
  - Each loan is assigned to exactly one loan officer.

### Mermaid ER diagram (use in presentation)

```mermaid
erDiagram
  CUSTOMER ||--o{ LOAN : applies_for
  LOAN_OFFICER ||--o{ LOAN : processes

  CUSTOMER {
    int customer_id PK
    string full_name
    int age
    string gender
    string city
    string phone
    string created_at
  }

  LOAN_OFFICER {
    int officer_id PK
    string full_name
    string branch_name
    int experience_years
    string created_at
  }

  LOAN {
    int loan_id PK
    int customer_id FK
    int officer_id FK
    float annual_income
    int employment_years
    int credit_score
    float existing_debt
    float loan_amount
    int loan_term_months
    float interest_rate
    int officer_decision_approve
    int actual_default
    float ai_default_prob
    int ai_default_pred
    string created_at
  }
```

### ML problem definition (Binary Classification)
- **Input (X)**: loan + customer financial features (income, credit score, debt, loan amount, etc.)
- **Output (y)**: `actual_default`  
  - 1 → loan defaulted  
  - 0 → loan did not default
- **Goal**: predict default risk for new loan applications, then store:
  - `ai_default_prob` (probability of default)
  - `ai_default_pred` (0/1 using a threshold)

