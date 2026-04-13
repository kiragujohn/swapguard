# file: api_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model
model = joblib.load("sim_swap_fraud_rf_model.pkl")

# Define FastAPI app
app = FastAPI()

# Define input schema
class FraudInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # add more features depending on your model input
    # e.g., age: int, transaction_amount: float, etc.

@app.post("/predict/")
def predict(input_data: FraudInput):
    try:
        df = pd.DataFrame([input_data.dict()])
        score = model.decision_function(df)
        flag = model.predict(df)
        return {
            "anomaly_score": float(score[0]),
            "anomaly_flag": int(flag[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
