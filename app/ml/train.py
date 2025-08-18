import pandas as pd
import numpy as np
from joblib import dump
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from ..data import load_farmers, load_buyers, load_prices, MODEL_PATH

RANDOM_SEED = 42
SYNTHETIC_SAMPLES = 5000

def _mandi_price_for_row(crop, state, prices):
    row = prices[(prices['crop']==crop) & (prices['location_state']==state)]
    if row.empty:
        med = prices[prices['crop']==crop]['mandi_price'].median()
        return float(med) if pd.notnull(med) else 0.0
    return float(row['mandi_price'].iloc[0])

def build_training(prices):
    # Build synthetic pairs from current CSVs
    farmers = load_farmers()
    buyers  = load_buyers()

    crops = sorted(set(prices['crop'].unique()) | set(farmers['crop'].unique()) | set(buyers['crop_preference'].unique()))
    states = sorted(set(prices['location_state'].unique()) | set(farmers['location_state'].unique()) | set(buyers['location_state'].unique()))

    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(SYNTHETIC_SAMPLES):
        # sample (farmer, buyer) with some chance of crop/state alignment
        if rng.random() < 0.7 and not farmers.empty:
            f = farmers.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
        else:
            # synthetic farmer
            f = pd.Series({
                "crop": rng.choice(crops) if crops else "wheat",
                "location_state": rng.choice(states) if states else "Maharashtra",
                "min_expected_price": float(rng.integers(1500, 3500)),
                "quantity": float(rng.integers(10, 100))
            })

        if rng.random() < 0.7 and not buyers.empty:
            b = buyers.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
        else:
            b = pd.Series({
                "crop_preference": f["crop"] if rng.random() < 0.8 else rng.choice(crops),
                "location_state": f["location_state"] if rng.random() < 0.6 else rng.choice(states),
                "max_price": float(rng.integers(1600, 3800)),
                "trust_score": float(rng.random())
            })

        mandi = _mandi_price_for_row(str(f["crop"]), str(f["location_state"]), prices)
        crop_match = 1.0 if str(b["crop_preference"]) == str(f["crop"]) else 0.0
        state_match = 1.0 if str(b["location_state"]) == str(f["location_state"]) else 0.0
        buyer_trust = float(b["trust_score"])
        farmer_min = float(f["min_expected_price"])
        buyer_max  = float(b["max_price"])

        # Label rule: success if crop matches AND price within ±10% of mandi AND trust >= 0.6
        within_10 = mandi*0.9 <= buyer_max <= mandi*1.1
        label = 1 if (crop_match > 0 and within_10 and buyer_trust >= 0.6 and buyer_max >= farmer_min and buyer_max >= mandi) else 0

        rows.append({
            "crop_match": crop_match,
            "state_match": state_match,
            "buyer_trust": buyer_trust,
            "farmer_min_price": farmer_min,
            "buyer_max_price": buyer_max,
            "mandi_price": mandi,
            "price_gap_to_mandi": buyer_max - mandi,
            "price_gap_to_farmer": buyer_max - farmer_min,
            "quantity": float(f["quantity"]),
            "y": label
        })

    df = pd.DataFrame(rows)
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    return X, y, df

def main():
    prices = load_prices()
    if prices.empty:
        raise SystemExit("prices.csv is empty. Run seeds/generate_synthetic.py first to create demo data.")
    X, y, df = build_training(prices)
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_SEED, class_weight="balanced"
    )
    clf.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, MODEL_PATH)
    print({"samples": int(len(df)), "features": int(X.shape[1]), "model_path": str(MODEL_PATH)})

if __name__ == "__main__":
    main()
