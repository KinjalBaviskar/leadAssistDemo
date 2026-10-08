# LeadAssist – Prototype

A small working prototype of the AI lead-management assistant: one FastAPI file, one HTML page, Gemini as the LLM (`gemini-3.5-flash` by default, falling back to `gemini-3.8-flash` if the configured model is retired), and SQLite for storage.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then put your key in GEMINI_API_KEY
.venv/bin/uvicorn app:app --reload
```

Open http://localhost:8000

## Logins

The demo accounts below are for development only. They are created automatically on first start.

| Role | Email | Password |
|---|---|---|
| Company | company@leadassist.local | company123 |
| Client | client@leadassist.local | client123 |

New clients can also create an account with **Client sign up** on the login screen.

### Client portal
- Upload one or more requirement documents (**PDF, DOCX, TXT or MD**), with optional notes.
- The AI reads the documents and shows the estimate (cost, effort, duration and team), the feasibility scores, the extracted features, the missing information and the questions to answer.
- The client sees the status of each submission (Pending review, Approved or Rejected) and can comment.
- Clients only see their own submissions. They never see internal data such as profit margin, opportunity score or the AI's recommendation rationale.
- Once a lead is approved, the client can see the proposal.

### Company portal
- Sees all leads from every client, with status filters and the AI recommendation and score for each lead.
- Can approve or reject a lead with a comment, and add further comments. Clients see these comments.
- Can upload documents on behalf of a client.
- Can analyse scope changes (cost, hours and days impact, then approve or reject the change).
- Can generate a proposal.

Gemini never decides the numbers. Prices, scores and estimates always come from the backend formulas, which are a simple rule-based baseline, not a trained ML model.

## Sample documents (`samples/`)

Log in as the client and upload these PDFs.

| File | Scenario | Expected AI recommendation (company view) |
|---|---|---|
| `1_clinic_booking_strong.pdf` | Simple scope, healthy budget and timeline | PRIORITIZE |
| `2_food_delivery_low_budget.pdf` | Large scope on a ₹3 lakh budget | DEPRIORITIZE |
| `3_ai_credit_risk_tight_deadline.pdf` | Complex ML platform due in 6 weeks | REVIEW (the deadline is not feasible) |

The exact numbers depend on Gemini's extraction, so they can vary a little between runs.
