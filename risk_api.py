from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from risk_engine import SwapGuardRiskEngine


app = FastAPI(
    title="SwapGuard Internal Risk Engine API",
    description="Internal ML inference API for SwapGuard pre-SIM-swap risk assessment.",
    version="0.1.0"
)


# ---------------------------------------------------------
# Load model once when FastAPI starts
# ---------------------------------------------------------

engine = SwapGuardRiskEngine(
    model_path="models/swapguard_hybrid_bundle.joblib"
)


# ---------------------------------------------------------
# API Request
# ---------------------------------------------------------

class PreSwapRiskRequest(BaseModel):

    tenureDaysSinceActivation: float = Field(ge=0)

    swapAgeHours: float = Field(ge=0)

    deviceVelocityCount7d: int = Field(ge=0)

    imeiSharedMsisdnCount: int = Field(ge=0)

    deviceIsNewImei: int = Field(ge=0, le=1)

    geoMnoToHomeDistanceKm: float = Field(ge=0)

    originatingChannel: str

    channelBotScore: float = Field(ge=0, le=1)

    profileDaysSinceContactChange: int = Field(ge=0)

    authSilentPossessionPassed: int = Field(ge=0, le=1)

    roamingIsActive: int = Field(ge=0, le=1)

    roamingCountryMismatch: int = Field(ge=0, le=1)


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "UP",
        "service": "SwapGuard Risk Engine",
        "modelVersion": "swapguard-community-hybrid-v0.1"
    }


# ---------------------------------------------------------
# Pre-Swap Risk Assessment
# ---------------------------------------------------------

@app.post("/internal/v1/risk/pre-swap")
def assess_pre_swap(request: PreSwapRiskRequest):

    try:

        # Convert Java/camelCase API fields into the feature names
        # expected by risk_engine.py.

        telemetry = {

            "tenure_days_since_activation":
                request.tenureDaysSinceActivation,

            "swap_age_hours":
                request.swapAgeHours,

            "device_velocity_count_7d":
                request.deviceVelocityCount7d,

            "imei_shared_msisdn_count":
                request.imeiSharedMsisdnCount,

            "device_is_new_imei":
                request.deviceIsNewImei,

            "geo_mno_to_home_distance_km":
                request.geoMnoToHomeDistanceKm,

            "originating_channel":
                request.originatingChannel,

            "channel_bot_score":
                request.channelBotScore,

            "profile_days_since_contact_change":
                request.profileDaysSinceContactChange,

            "auth_silent_possession_passed":
                request.authSilentPossessionPassed,

            "roaming_is_active":
                request.roamingIsActive,

            "roaming_country_mismatch":
                request.roamingCountryMismatch
        }

        return engine.evaluate(telemetry)

    except Exception as exc:

        print(f"SwapGuard risk-engine error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Risk assessment failed"
        )