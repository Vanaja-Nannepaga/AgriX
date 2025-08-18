# Generates 50 realistic demo records for farmers, buyers, and mandi prices (India-centric).
import pandas as pd
import numpy as np
from pathlib import Path
import random

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(parents=True, exist_ok=True)

states = ["Maharashtra","Karnataka","Telangana","Andhra Pradesh","Gujarat","Punjab","Haryana","Uttar Pradesh","Madhya Pradesh","Rajasthan","West Bengal","Odisha","Bihar","Tamil Nadu"]
crops = ["wheat","rice","maize","cotton","soybean","tur","chana","groundnut","sugarcane","mustard"]

rng = np.random.default_rng(7)

# Prices (mandi baseline by state)
prices = []
for s in states:
    for c in crops:
        base = {
            "wheat": 2200, "rice": 2300, "maize": 1900, "cotton": 6500, "soybean": 4200,
            "tur": 7000, "chana": 5200, "groundnut": 6000, "sugarcane": 340, "mustard": 5600
        }.get(c, 3000)
        noise = rng.integers(-200, 200)
        price = max(200, base + int(noise))
        prices.append({"crop": c, "location_state": s, "mandi_price": price})
pd.DataFrame(prices).to_csv(DATA / "prices.csv", index=False)

# Farmers
farmers = []
for i in range(1, 51):
    c = random.choice(crops)
    s = random.choice(states)
    min_price = int(np.clip(
        [2200,2300,1900,6500,4200,7000,5200,6000,340,5600][crops.index(c)] + rng.integers(-150, 150),
        200, 9000
    ))
    qty = int(rng.integers(10, 120))
    farmers.append({
        "farmer_id": i, "crop": c, "location_district": f"D{i%20+1}", "location_state": s,
        "min_expected_price": min_price, "quantity": qty
    })
pd.DataFrame(farmers).to_csv(DATA / "farmers.csv", index=False)

# Buyers
buyers = []
for i in range(1, 51):
    c = random.choice(crops)
    s = random.choice(states)
    max_price = int(np.clip(
        [2200,2300,1900,6500,4200,7000,5200,6000,340,5600][crops.index(c)] + rng.integers(-250, 250),
        200, 10000
    ))
    trust = float(np.clip(rng.random() * 1.05, 0.0, 1.0))
    buyers.append({
        "buyer_id": i, "crop_preference": c, "max_price": max_price, "trust_score": round(trust, 3), "location_state": s
    })
pd.DataFrame(buyers).to_csv(DATA / "buyers.csv", index=False)

print("Seeded: farmers.csv, buyers.csv, prices.csv")
