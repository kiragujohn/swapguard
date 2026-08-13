import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# =====================================================================
# 1. LOAD MODEL BUNDLE AT STARTUP
# =====================================================================
MODEL_PATH = "models/swapguard_isolation_forest.joblib"

try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    s_min = model_bundle["s_min"]
    s_max = model_bundle["s_max"]
    feature_names = model_bundle["feature_names"]
    print(f"✅ Loaded Isolation Forest model bundle from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Could not load model from {MODEL_PATH}: {e}")
    model = None

# =====================================================================
# 2. PYDANTIC SCHEMAS (Matches Enterprise Contract)
# =====================================================================
class TargetAccountTelemetry(BaseModel):
    target_msisdn: str = Field(..., example="+12535550199")
    target_sim_activation_age_days: float = Field(..., ge=0, example=1825.0)
    target_home_registered_latitude: float = Field(..., ge=-90, le=90, example=47.2529)
    target_home_registered_longitude: float = Field(..., ge=-180, le=180, example=-122.4443)
    target_prior_completed_swap_timestamp: Optional[str] = Field(None, example="2026-05-10T10:15:00Z")

class RequestorTelemetry(BaseModel):
    requestor_originating_msisdn: str = Field(..., example="+12535550888")
    requestor_hardware_imei: str = Field(..., example="358249091234567")
    requestor_sim_activation_age_days: float = Field(..., ge=0, example=0.2)
    requestor_sim_is_prepaid: bool = Field(default=False)

class RequestorHardwareVelocityCounters(BaseModel):
    unique_imeis_used_by_requestor_msisdn_30d: int = Field(..., ge=0, example=4)
    unique_msisdns_used_by_requestor_imei_7d: int = Field(..., ge=0, example=6)
    requestor_swap_request_count_24h: int = Field(..., ge=0, example=3)

class RequestorNetworkLocation(BaseModel):
    requestor_cell_latitude: float = Field(..., ge=-90, le=90, example=21.3069)
    requestor_cell_longitude: float = Field(..., ge=-180, le=180, example=-157.8583)
    silent_network_auth_passed: bool = Field(default=False)

class ChannelAndSessionMetadata(BaseModel):
    originating_channel: str = Field(..., example="chatbot")
    automation_bot_score: float = Field(..., ge=0.0, le=1.0, example=0.82)
    fallback_to_kba_only: bool = Field(default=False)

class PreSwapRiskRequest(BaseModel):
    request_id: str = Field(..., example="req_9f8a7b6c5d")
    target_account_telemetry: TargetAccountTelemetry
    requestor_telemetry: RequestorTelemetry
    requestor_hardware_velocity_counters: RequestorHardwareVelocityCounters
    requestor_network_location: RequestorNetworkLocation
    channel_and_session_metadata: ChannelAndSessionMetadata

class PreSwapRiskResponse(BaseModel):
    request_id: str
    pre_swap_risk_score: float
    risk_band: str
    action_recommendation: str
    top_risk_drivers: List[str]
    processing_time_ms: float

# =====================================================================
# 3. FASTAPI APP INIT
# =====================================================================
app = FastAPI(
    title="SwapGuard Pre-Swap Risk Intelligence Engine",
    description="Enterprise API evaluating pre-provisioning SIM swap risks across telecom channels.",
    version="2.0.0",
    contact={
        "name": "John Kiragu Njoroge",
        "email": "kiragujohn@hotmail.com"
    }
)

# =====================================================================
# 4. HELPER LOGIC FOR RISK BANDS & DRIVERS
# =====================================================================
def evaluate_risk_band(score: float):
    if score >= 0.80:
        return "HIGH_RISK", "BLOCK_OR_STEP_UP"
    elif score >= 0.50:
        return "MEDIUM_RISK", "HOLD_FOR_MANUAL_REVIEW"
    elif score >= 0.30:
        return "ELEVATED_RISK", "MANDATE_OUT_OF_BAND_AUTH"
    else:
        return "LOW_RISK", "APPROVE_PROVISIONING"

def extract_risk_drivers(req: PreSwapRiskRequest) -> List[str]:
    drivers = []
    
    # 1. Burner SIM Check
    if req.requestor_telemetry.requestor_sim_activation_age_days < 3.0:
        drivers.append(f"NEW_BURNER_REQUESTOR_SIM_{req.requestor_telemetry.requestor_sim_activation_age_days:.1f}_DAYS")
        
    # 2. IMEI Farming Check
    if req.requestor_hardware_velocity_counters.unique_msisdns_used_by_requestor_imei_7d >= 3:
        cnt = req.requestor_hardware_velocity_counters.unique_msisdns_used_by_requestor_imei_7d
        drivers.append(f"IMEI_FARMING_HARVESTING_DETECTED_{cnt}_MSISDNS")
        
    # 3. Bot Risk Check
    if req.channel_and_session_metadata.automation_bot_score > 0.70:
        ch = req.channel_and_session_metadata.originating_channel.upper()
        drivers.append(f"AUTOMATED_CHANNEL_DETECTED_{ch}")
        
    # 4. Silent Network Auth Failure
    if not req.requestor_network_location.silent_network_auth_passed:
        drivers.append("SILENT_NETWORK_AUTHENTICATION_FAILED")

    return drivers

# =====================================================================
# 5. PREDICT ENDPOINT
# =====================================================================
@app.post("/v1/risk-score", response_model=PreSwapRiskResponse)
def predict_pre_swap_risk(payload: PreSwapRiskRequest):
    start_time = time.perf_counter()
    
    try:
        # Calculate Anomaly Score if model exists
        if model is not None:
            # Flatten payload for feature adapter processing
            flat_dict = {
                "target_sim_activation_age_days": payload.target_account_telemetry.target_sim_activation_age_days,
                "requestor_sim_activation_age_days": payload.requestor_telemetry.requestor_sim_activation_age_days,
                "unique_imeis_used_by_requestor_msisdn_30d": payload.requestor_hardware_velocity_counters.unique_imeis_used_by_requestor_msisdn_30d,
                "unique_msisdns_used_by_requestor_imei_7d": payload.requestor_hardware_velocity_counters.unique_msisdns_used_by_requestor_imei_7d,
                "automation_bot_score": payload.channel_and_session_metadata.automation_bot_score,
                "silent_network_auth_passed": int(payload.requestor_network_location.silent_network_auth_passed)
            }
            
            raw_df = pd.DataFrame([flat_dict])
            
            # Predict raw decision function score[cite: 1]
            raw_score = float(model.decision_function(raw_df)[0])[cite: 1]
            
            # Normalize to [0.0 - 1.0] continuous risk score[cite: 1]
            risk_score = round((s_max - raw_score) / (s_max - s_min), 4)[cite: 1]
        else:
            # Fallback score if model is not present
            risk_score = 0.5000

        risk_band, action_recommendation = evaluate_risk_band(risk_score)
        top_risk_drivers = extract_risk_drivers(payload)
        
        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return PreSwapRiskResponse(
            request_id=payload.request_id,
            pre_swap_risk_score=risk_score,
            risk_band=risk_band,
            action_recommendation=action_recommendation,
            top_risk_drivers=top_risk_drivers,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Risk evaluation error: {str(e)}")