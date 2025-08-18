# AgriXchange: Multi Agent AI-Powered Farmer–Buyer Negotiation Platform

This repo implements the final approach:
- CSV **data backbone** (farmers, buyers, mandi prices)
- **Hybrid Training** → ML model trained on 5000+ synthetic + real farmer-buyer pairs  
- **Farmer & Buyer Negotiation Agents** → Multi-round bargaining simulation  
- **Explainability** for every suggested match
- **FastAPI** endpoints for demos and integrations

## Quickstart

Run Commands:
pip install fastapi uvicorn pandas joblib 
pip install python-multipart

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Seed demo CSVs
python seeds/generate_synthetic.py

# 2) Train ML on 5000+ synthetic samples
python app/ml/train.py

# 3) Run API
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive API.

## Key Endpoints
- `POST /upload` - upload farmer.csv, buyer.csv, pricing.csv / run normally as we already have these in our files
- `GET /health` – service status
- `POST /seed` – reset + seed data from CSVs (idempotent for demo)
- `POST /train` – (re)train the ML model
- `GET /negotiate/{farmer_id}` – Multi-agent negotiation between one farmer & buyers
- `GET /negotiate_all` – Negotiation results for all farmers

## Repository Structure
```bash
agriX/
│── app/
│   ├── main.py              # FastAPI API (final endpoints)
│   ├── data.py              # Data loading & preprocessing
│   ├── models.py            # Pydantic models
│   ├── agent/
│   │   ├── negotiation.py   # Farmer & Buyer agents, negotiation logic
│   ├── ml/
│   │   └── train.py         # ML training pipeline (RandomForest)
│   └── utils/
│       └── explain.py       # Explainability for matches
│
│── data/                    # farmers.csv, buyers.csv, prices.csv
│── seeds/
│   └── generate_synthetic.py # Script to seed demo data
│── requirements.txt
│── README.md

```
