## Bank Loan Management System with Default Risk Analysis (Simple Report)

### What is this project?
This is a small banking application that stores loan details in a database and uses AI to **predict the risk of loan default**.

In simple words:
- The **database** stores customers, officers, and loans.
- The **AI model** looks at loan details (income, credit score, debt, etc.).
- The AI gives a **default risk** and we store that result back into the database.

### Why are we doing this?
Banks process many applications. If the data is stored properly and risk is predicted, it becomes easier to:
- keep records neatly
- analyze loan performance
- support decisions with data

### Database design (exactly 3 entity sets)
We use exactly **three tables** (entity sets). This matches the project requirement.

#### 1) `customers` (Customer table)
Stores basic customer details:
- `customer_id` (Primary Key)
- `full_name`, `age`, `gender`, `city`, `phone`
- `created_at` (when the record was created)

#### 2) `loan_officers` (Loan Officer table)
Stores officer details:
- `officer_id` (Primary Key)
- `full_name`, `branch_name`, `experience_years`
- `created_at`

#### 3) `loans` (Loan table)
Stores each loan/application:
- `loan_id` (Primary Key)
- `customer_id` (Foreign Key → `customers.customer_id`)
- `officer_id` (Foreign Key → `loan_officers.officer_id`)

It also stores:
- **Loan details**: income, credit score, existing debt, loan amount, term, interest rate
- **Officer decision**: `officer_decision_approve` (approve/reject)
- **Actual result (history)**: `actual_default` (1=default, 0=not default, NULL if unknown)
- **AI result (prediction)**:
  - `ai_default_prob` (probability between 0 and 1)
  - `ai_default_pred` (1=default, 0=not default)

### Relationships (easy explanation)
- One customer can apply for **many loans**.
- One loan officer can handle **many loans**.
- Every loan belongs to **one** customer and **one** officer.

### AI / ML part (binary classification)
**Question the model answers:**
“Will this loan default?” → Yes/No

- **Type**: Binary Classification
- **Output/Target**: `actual_default`
- **Inputs**: values like annual income, credit score, debt, loan amount, etc.

### How integration works (what happens in the final demo)
1. We create sample (synthetic) data in the database.
2. We train the AI model using historical loans (`actual_default` available).
3. When a new loan application is entered in the GUI:
   - the loan is inserted into the `loans` table
   - the AI predicts default risk
   - the app updates the same row with:
     - `ai_default_prob`
     - `ai_default_pred`

### Extra features (to make it presentable)
- Dashboard showing total customers, officers, loans, and default rate
- Branch-wise default rate analytics
- List of loans where officer approved but AI says “high risk”
- Simple “reason codes” (top factors affecting prediction)
- Export the tables as CSV for report submission

