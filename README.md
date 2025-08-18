# AgriXchange – Fair AI Matching Agent (Hackathon Build)

This repo implements the final approach:
- CSV **data backbone** (farmers, buyers, mandi prices)
- **Synthetic+Real hybrid** training to bootstrap an ML model
- **AI Matching Agent** that enforces mandi-price fairness and ranks buyers
- **Explainability** for every suggested match
- **FastAPI** endpoints for demos and integrations

## Quickstart

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
- `GET /health` – service status
- `POST /seed` – reset + seed data from CSVs (idempotent for demo)
- `POST /train` – (re)train the ML model
- `GET /ai/match/{farmer_id}` – top-k buyer suggestions for a single farmer
- `GET /ai/match_all` – top suggestions for all farmers

## Files
- `data/*.csv` – core CSVs
- `app/data.py` – data loading, preprocessing
- `app/agent/matcher.py` – fairness-aware agent
- `app/ml/train.py` – synthetic training + model save/load
- `app/main.py` – FastAPI API
