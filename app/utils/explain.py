def explain_match(farmer_row, buyer_row, mandi_price, prob, suggested_price):
    reasons = []
    if buyer_row['crop_preference'] == farmer_row['crop']:
        reasons.append("Crop preference matches")
    if suggested_price >= mandi_price:
        reasons.append(f"Offered ≥ mandi price (₹{suggested_price:.0f} ≥ ₹{mandi_price:.0f})")
    else:
        reasons.append(f"Below mandi price (₹{suggested_price:.0f} < ₹{mandi_price:.0f})")
    reasons.append(f"Buyer trust score {buyer_row['trust_score']:.2f}")
    reasons.append(f"ML match probability {prob:.2f}")
    return [{"text": r} for r in reasons]
