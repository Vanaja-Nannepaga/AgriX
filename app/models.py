from pydantic import BaseModel, Field
from typing import List, Optional

class MatchReason(BaseModel):
    text: str

class BuyerSuggestion(BaseModel):
    buyer_id: int
    suggested_price: float
    match_probability: float = Field(ge=0.0, le=1.0)
    reasons: List[MatchReason]

class FarmerMatchResponse(BaseModel):
    farmer_id: int
    crop: str
    mandi_price: float
    suggestions: List[BuyerSuggestion]

class TrainResponse(BaseModel):
    samples: int
    features: int
    model_path: str

class SeedResponse(BaseModel):
    status: str
    farmers: int
    buyers: int
    prices: int
