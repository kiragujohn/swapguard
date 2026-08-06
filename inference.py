import joblib
import pandas as pd
from adapters.camara_preswap_adapter import PreSwapTelemetryAdapter

def main():
    print("🚀 Loading SwapGuard Inference Engine...")
    
    # 1. Load the trained model bundle
    try:
        model_bundle = joblib.load("models/swapguard_isolation_forest.joblib")
        model = model_bundle["model"]
        s_min = model_bundle["s_min"]
        s_max = model_bundle["s_max"]
    except FileNotFoundError:
        print("❌ Model bundle not found. Did you run model_training.py?")
        return

    # 2. Simulate an incoming real-time SIM swap request payload
    # (This represents a highly suspicious "burner phone" request)
    sample_request = {
        "request_id": "req_live_001",
        "target_msisdn": "+12535551234",
        "target_sim_activation_age_days": 450.5,
        "target_home_registered_latitude": 47.6062,
        "target_home_registered_longitude": -122.3321,
        "target_prior_completed_swap_timestamp": "2025-01-20T12:00:00Z", 
        "requestor_originating_msisdn": "+12535559999", # Different from target
        "requestor_hardware_imei": "351234567890123",
        "requestor_sim_activation_age_days": 1.2,       # Activated yesterday
        "requestor_sim_is_prepaid": True,
        "unique_imeis_used_by_requestor_msisdn_30d": 4, # SIM swapping behavior
        "unique_msisdns_used_by_requestor_imei_7d": 6,
        "requestor_swap_request_count_24h": 3,
        "requestor_cell_latitude": 34.0522,             # Geo-distance mismatch
        "requestor_cell_longitude": -118.2437,
        "silent_network_auth_passed": False,
        "originating_channel": "chatbot",
        "automation_bot_score": 0.88,                   # Likely a bot
        "fallback_to_kba_only": True
    }

    # 3. Convert JSON-like dict to DataFrame and process via adapter
    print("\n⚙️ Processing incoming request features...")
    df_live = pd.DataFrame([sample_request])
    X_live = PreSwapTelemetryAdapter.process_dataframe(df_live)

    # 4. Generate Risk Score
    raw_score = model.decision_function(X_live)[0]
    risk_score = 1.0 - ((raw_score - s_min) / (s_max - s_min))
    
    print(f"\n📊 Calculated Risk Score: {risk_score:.4f} / 1.0000")
    
    # 5. Apply Business Logic Thresholds
    print("\n🛡️ SwapGuard Decision:")
    if risk_score >= 0.80:
        print("   🔴 ACTION: BLOCK (High Fraud Probability)")
    elif risk_score >= 0.50:
        print("   🟡 ACTION: STEP-UP AUTHENTICATION (Require physical ID verification)")
    else:
        print("   🟢 ACTION: APPROVE (Low Risk)")

if __name__ == "__main__":
    main()