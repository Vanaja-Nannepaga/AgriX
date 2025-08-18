# app/agent/negotiation.py

import random

class FarmerAgent:
    def __init__(self, farmer_id, crop, mandi_price, min_acceptable_price=None):
        self.farmer_id = farmer_id
        self.crop = crop
        self.mandi_price = mandi_price
        # Farmer never goes below mandi price
        self.min_price = min_acceptable_price or mandi_price

    def propose_price(self, last_offer=None):
        """Farmer proposes a price (start at mandi or slightly higher)."""
        if last_offer is None:
            return self.mandi_price + random.randint(50, 200)
        # Move slightly towards buyer's last offer
        return max(self.min_price, int((last_offer + self.mandi_price) / 2))


class BuyerAgent:
    def __init__(self, buyer_id, crop_preference, max_price):
        self.buyer_id = buyer_id
        self.crop_preference = crop_preference
        self.max_price = max_price

    def counter_offer(self, last_offer):
        """Buyer counters with lower price if above max_price."""
        if last_offer <= self.max_price:
            return last_offer
        # Offer closer to max_price
        return int((last_offer + self.max_price) / 2)


def negotiate(farmer: FarmerAgent, buyer: BuyerAgent, max_rounds=5):
    """Simulate negotiation between one farmer and one buyer."""

    if farmer.crop != buyer.crop_preference:
        return None  # crop mismatch

    farmer_offer = farmer.propose_price()
    rounds = 0

    while rounds < max_rounds:
        rounds += 1

        if farmer_offer <= buyer.max_price and farmer_offer >= farmer.min_price:
            return {
                "farmer_id": farmer.farmer_id,
                "buyer_id": buyer.buyer_id,
                "agreed_price": farmer_offer,
                "rounds": rounds,
                "status": "success"
            }

        buyer_offer = buyer.counter_offer(farmer_offer)

        if buyer_offer < farmer.min_price:
            return {
                "farmer_id": farmer.farmer_id,
                "buyer_id": buyer.buyer_id,
                "status": "fail"
            }

        # Farmer responds to buyer's counter
        farmer_offer = farmer.propose_price(buyer_offer)

    return {
        "farmer_id": farmer.farmer_id,
        "buyer_id": buyer.buyer_id,
        "status": "fail"
    }

