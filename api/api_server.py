from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model
model = joblib.load("../export/sim_swap_fraud_rf_model.pkl")

app = FastAPI(
    title="SIM Swap Fraud Detector API",
    description="An API to predict SIM swap fraud risk based on enriched user behavior data.",
    version="1.0.0",
    contact={
        "name": "John Kiragu",
        "email": "kiragujohn@hotmail.com"
    }
)

class FraudInput(BaseModel):
    sim_swap_time_gap_minutes: int
    device_change_flag: int
    sim_type_change_flag: int
    imsi_change_flag: int
    iccid_change_flag: int
    otp_and_sim_change_geo_hash_length: float
    recent_sim_activation_days: int
    num_sim_changes_last_30d: int
    previous_sim_holder_tenure_days: int
    account_age_days: int
    ip_change_flag: int

def enrich_input(raw: pd.Series) -> pd.Series:
    enriched = raw.copy()
    enriched["time_urgency_score"] = 1 - min(raw["sim_swap_time_gap_minutes"], 10080) / 10080
    enriched["identity_shift_score"] = sum([
        raw["device_change_flag"],
        raw["sim_type_change_flag"],
        raw["imsi_change_flag"],
        raw["iccid_change_flag"]
    ])
    MAX_ALLOWED_DISTANCE = 10000
    normalized_geo = min(raw["otp_and_sim_change_geo_hash_length"] / MAX_ALLOWED_DISTANCE, 1.0)
    enriched["geo_risk_score"] = 1 - normalized_geo
    enriched["sim_activity_score"] = 1 - min(raw["recent_sim_activation_days"], 365) / 365
    enriched["account_risk_score"] = (
            raw["num_sim_changes_last_30d"] * 0.3 +
            (1 - min(raw["previous_sim_holder_tenure_days"], 3650) / 3650) * 0.3 +
            (1 - min(raw["account_age_days"], 3650) / 3650) * 0.3 +
            raw["ip_change_flag"] * 0.1
    )
    return enriched

def get_risk_level(score: float) -> str:
    if score < 0.2:
        return "Very Low"
    elif score < 0.4:
        return "Low"
    elif score < 0.6:
        return "Medium"
    elif score < 0.8:
        return "High"
    else:
        return "Very High"

@app.post("/predict/")
def predict(input_data: FraudInput):
    try:
        # Convert input Pydantic model to DataFrame
        data_dict = input_data.model_dump() if hasattr(input_data, "model_dump") else input_data.dict()
        raw_df = pd.DataFrame([data_dict])
        
        # Enrich features
        enriched_df = raw_df.apply(enrich_input, axis=1)

        # Include raw + engineered features expected by the model
        model_features = [
            "sim_swap_time_gap_minutes",
            "device_change_flag",
            "sim_type_change_flag",
            "imsi_change_flag",
            "iccid_change_flag",
            "otp_and_sim_change_geo_hash_length",
            "recent_sim_activation_days",
            "num_sim_changes_last_30d",
            "previous_sim_holder_tenure_days",
            "account_age_days",
            "ip_change_flag",
            # Engineered features
            "time_urgency_score",
            "identity_shift_score",
            "geo_risk_score",
            "sim_activity_score",
            "account_risk_score"
        ]

        # Select available model features safely
        available_features = [col for col in model_features if col in enriched_df.columns]
        X = enriched_df[available_features]

        # Predict probability
        prob = model.predict_proba(X)[:, 1][0]
        risk_level = get_risk_level(prob)

        return {
            "risk_level": risk_level,
            "risk_score": round(float(prob), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))