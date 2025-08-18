from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Optional
import pandas as pd
from .data import FARMERS_CSV, BUYERS_CSV, PRICES_CSV, load_farmers, load_buyers, load_prices, MODEL_PATH
from .agent.matcher import match_for_farmer
from .agent.negotiation import FarmerAgent, BuyerAgent, negotiate
from .models import FarmerMatchResponse, BuyerSuggestion, TrainResponse, SeedResponse
from joblib import dump
from pathlib import Path
import subprocess, sys
import shutil

app = FastAPI(title="AgriXchange – Fair AI Matching Agent", version="0.1.0")


@app.post("/upload", response_model=SeedResponse)
async def upload_csv(
    farmers_file: UploadFile = File(None),
    buyers_file: UploadFile = File(None),
    prices_file: UploadFile = File(None),
):
    try:
        if farmers_file:
            with open(FARMERS_CSV, "wb") as f:
                shutil.copyfileobj(farmers_file.file, f)
            farmers_file.file.close()

        if buyers_file:
            with open(BUYERS_CSV, "wb") as f:
                shutil.copyfileobj(buyers_file.file, f)
            buyers_file.file.close()

        if prices_file:
            with open(PRICES_CSV, "wb") as f:
                shutil.copyfileobj(prices_file.file, f)
            prices_file.file.close()

        farmers = load_farmers()
        buyers = load_buyers()
        prices = load_prices()

        return SeedResponse(
            status="uploaded",
            farmers=len(farmers),
            buyers=len(buyers),
            prices=len(prices),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "model_present": MODEL_PATH.exists()}


@app.post("/seed", response_model=SeedResponse)
def seed():
    farmers = load_farmers()
    buyers = load_buyers()
    prices = load_prices()
    if farmers.empty or buyers.empty or prices.empty:
        return SeedResponse(
            status="missing_data",
            farmers=len(farmers),
            buyers=len(buyers),
            prices=len(prices),
        )
    return SeedResponse(
        status="ok",
        farmers=len(farmers),
        buyers=len(buyers),
        prices=len(prices),
    )


@app.post("/train", response_model=TrainResponse)
def train():
    proc = subprocess.run(
        [sys.executable, "-m", "app.ml.train"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise HTTPException(
            status_code=500, detail=proc.stderr.strip() or "Training failed"
        )
    out = proc.stdout.strip().splitlines()[-1]
    try:
        d = eval(out)
        return TrainResponse(
            samples=d["samples"],
            features=d["features"],
            model_path=d["model_path"],
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Could not parse training output")


@app.get("/negotiate/{farmer_id}")
def negotiate_one(farmer_id: int, max_rounds: int = 5):
    farmers = load_farmers()
    buyers = load_buyers()
    prices = load_prices()

    row = farmers[farmers["farmer_id"] == farmer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Farmer {farmer_id} not found")

    f = row.iloc[0]
    # ✅ Safe mandi price fallback
    mandi_price_series = prices.loc[prices["crop"] == f["crop"], "mandi_price"]
    mandi_price = float(mandi_price_series.mean()) if not mandi_price_series.empty else 0.0

    farmer_agent = FarmerAgent(int(f["farmer_id"]), f["crop"], mandi_price)

    results = []
    for _, b in buyers.iterrows():
        # ✅ Skip buyers with mismatched crops early
        if f["crop"] != b["crop_preference"]:
            continue

        buyer_agent = BuyerAgent(int(b["buyer_id"]), b["crop_preference"], float(b["max_price"]))
        deal = negotiate(farmer_agent, buyer_agent, max_rounds=max_rounds)
        if deal and deal["status"] == "success":
            results.append(deal)

    return {
        "farmer_id": int(f["farmer_id"]),
        "crop": f["crop"],
        "mandi_price": mandi_price,
        "deals": results,
    }


@app.get("/negotiate_all")
def negotiate_all(max_rounds: int = 5):
    farmers = load_farmers()
    buyers = load_buyers()
    prices = load_prices()

    results = []
    for _, f in farmers.iterrows():
        mandi_price_series = prices.loc[prices["crop"] == f["crop"], "mandi_price"]
        mandi_price = float(mandi_price_series.mean()) if not mandi_price_series.empty else 0.0

        farmer_agent = FarmerAgent(int(f["farmer_id"]), f["crop"], mandi_price)

        farmer_deals = []
        for _, b in buyers.iterrows():
            if f["crop"] != b["crop_preference"]:
                continue

            buyer_agent = BuyerAgent(int(b["buyer_id"]), b["crop_preference"], float(b["max_price"]))
            deal = negotiate(farmer_agent, buyer_agent, max_rounds=max_rounds)
            if deal and deal["status"] == "success":
                farmer_deals.append(deal)

        results.append(
            {
                "farmer_id": int(f["farmer_id"]),
                "crop": f["crop"],
                "mandi_price": mandi_price,
                "deals": farmer_deals,
            }
        )

    return {"results": results}

