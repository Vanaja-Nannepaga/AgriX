import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "ml" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FARMERS_CSV = DATA_DIR / "farmers.csv"
BUYERS_CSV  = DATA_DIR / "buyers.csv"
PRICES_CSV  = DATA_DIR / "prices.csv"
MODEL_PATH  = MODEL_DIR / "rf_match_model.joblib"

def load_farmers() -> pd.DataFrame:
    df = pd.read_csv(FARMERS_CSV)
    # Basic hygiene
    df['crop'] = df['crop'].astype(str).str.strip().str.lower()
    df['location_state'] = df['location_state'].astype(str).str.strip().str.title()
    return df

def load_buyers() -> pd.DataFrame:
    df = pd.read_csv(BUYERS_CSV)
    df['crop_preference'] = df['crop_preference'].astype(str).str.strip().str.lower()
    df['location_state'] = df['location_state'].astype(str).str.strip().str.title()
    return df

def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_CSV)
    df['crop'] = df['crop'].astype(str).str.strip().str.lower()
    df['location_state'] = df['location_state'].astype(str).str.strip().str.title()
    return df

def mandi_price_for(crop: str, state: str, prices_df: pd.DataFrame) -> float:
    row = prices_df[(prices_df['crop']==crop) & (prices_df['location_state']==state)]
    if row.empty:
        # fallback to state-agnostic median if present
        med = prices_df[prices_df['crop']==crop]['mandi_price'].median()
        return float(med) if pd.notnull(med) else 0.0
    return float(row['mandi_price'].iloc[0])
