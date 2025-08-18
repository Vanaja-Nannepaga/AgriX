import numpy as np
import pandas as pd
from typing import List, Dict
from ..data import mandi_price_for, MODEL_PATH
from ..utils.explain import explain_match
from joblib import load

MANDI_FLOOR = 1.0   # must be >= 1.0 * mandi price (no farmer undercut)
PRICE_SLACK = 0.10  # allow suggestions up to +10% above mandi if buyer can pay

def _make_features(farmer_row, buyer_row, mandi_price):
    # Simple tabular features for demo ML + agent ranking
    return {
        "crop_match": 1.0 if buyer_row["crop_preference"] == farmer_row["crop"] else 0.0,
        "state_match": 1.0 if buyer_row["location_state"] == farmer_row["location_state"] else 0.0,
        "buyer_trust": float(buyer_row["trust_score"]),
        "farmer_min_price": float(farmer_row["min_expected_price"]),
        "buyer_max_price": float(buyer_row["max_price"]),
        "mandi_price": float(mandi_price),
        "price_gap_to_mandi": float(buyer_row["max_price"]) - float(mandi_price),
        "price_gap_to_farmer": float(buyer_row["max_price"]) - float(farmer_row["min_expected_price"]),
        "quantity": float(farmer_row["quantity"]),
    }

def _suggest_price(farmer_row, buyer_row, mandi_price):
    # Suggest a fair price: max(mandi, min(buyer_max, mandi*(1+PRICE_SLACK)))
    cap = min(buyer_row["max_price"], mandi_price * (1.0 + PRICE_SLACK))
    return max(mandi_price * MANDI_FLOOR, cap)

def load_model():
    if MODEL_PATH.exists():
        return load(MODEL_PATH)
    return None

def match_for_farmer(farmer_row, buyers_df, prices_df, top_k=3):
    crop = farmer_row["crop"]
    state = farmer_row["location_state"]
    mandi_price = mandi_price_for(crop, state, prices_df)

    model = load_model()

    # Candidate buyers: crop OR state match for recall; agent will filter
    cands = buyers_df[
        (buyers_df["crop_preference"] == crop) | (buyers_df["location_state"] == state)
    ].copy()

    scored = []
    for _, b in cands.iterrows():
        suggested_price = _suggest_price(farmer_row, b, mandi_price)
        # Hard fairness filter: never below mandi, and must cover farmer min
        if suggested_price < mandi_price or suggested_price < farmer_row["min_expected_price"]:
            continue
        feats = _make_features(farmer_row, b, mandi_price)
        X = np.array([list(feats.values())])
        prob = 0.5
        if model is not None:
            try:
                prob = float(model.predict_proba(X)[0,1])
            except Exception:
                prob = 0.5
        # Soft filter on trust; rank key balances price, prob, trust
        if b["trust_score"] < 0.4:
            continue
        rank_key = (suggested_price, prob, b["trust_score"], feats["crop_match"])
        scored.append((rank_key, b, prob, suggested_price))

    # Rank and prepare output
    scored.sort(key=lambda t: (t[2], t[1]["trust_score"], t[3]), reverse=True)
    suggestions = []
    for _, b, prob, suggested_price in scored[:top_k]:
        reasons = explain_match(farmer_row, b, mandi_price, prob, suggested_price)
        suggestions.append({
            "buyer_id": int(b["buyer_id"]),
            "suggested_price": float(round(suggested_price, 2)),
            "match_probability": float(round(prob, 4)),
            "reasons": reasons
        })

    return mandi_price, suggestions
