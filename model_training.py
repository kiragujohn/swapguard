import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from adapters.camara_preswap_adapter import PreSwapTelemetryAdapter

def main():
    print("🛡️ SwapGuard Community Edition — Anomaly Detection Training Pipeline")
    print("====================================================================")
    
    # Path to the generated dataset
    dataset_path = "datasets/sim_swap_request_dataset.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "sim_swap_request_dataset.csv"
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found at '{dataset_path}'. Run 'python generate_dataset.py' first.")
            return

    print(f"📥 Loading dataset: {dataset_path}")
    raw_df = pd.read_csv(dataset_path)
    print(f" • Total Records: {len(raw_df):,}")

    # Flexible label resolution to avoid KeyError
    if "is_fraud_label" in raw_df.columns:
        y_ground_truth = raw_df["is_fraud_label"]
    elif "is_fraud_ground_truth" in raw_df.columns:
        y_ground_truth = raw_df["is_fraud_ground_truth"]
    else:
        raise KeyError(f"Could not find a valid label column. Available columns: {list(raw_df.columns)}")

    print("\n⚙️ Processing CAMARA & MNO signals via PreSwapTelemetryAdapter...")
    X_features = PreSwapTelemetryAdapter.process_dataframe(raw_df)
    feature_names = list(X_features.columns)
    print(f" • Engineered Feature Matrix Shape: {X_features.shape} ({len(feature_names)} features)")

    # Train Isolation Forest (Using n_jobs=1 for Python 3.13 stability on macOS)
    print("\n🌲 Training Isolation Forest anomaly detection model...")
    model = IsolationForest(
        n_estimators=100,
        contamination=0.08,
        random_state=42,
        n_jobs=1 
    )
    model.fit(X_features)

    # Calculate Continuous Risk Scores [0.0, 1.0]
    # decision_function outputs negative values for anomalies, positive for normal
    raw_scores = model.decision_function(X_features)
    s_min, s_max = float(raw_scores.min()), float(raw_scores.max())
    
    # Corrected normalization: lower raw_score (anomaly) -> higher risk_score
    normalized_risk_scores = (s_max - raw_scores) / (s_max - s_min)

    # Evaluate Model Metrics
    auc = roc_auc_score(y_ground_truth, normalized_risk_scores)
    print(f"\n🎯 Model Evaluation Summary:")
    print(f" • ROC-AUC Score: {auc:.4f}")

    # Binary High-Risk Threshold
    y_pred_high_risk = (normalized_risk_scores >= 0.80).astype(int)
    
    print("\n📋 Classification Report (High-Risk Threshold >= 0.80):")
    print(classification_report(y_ground_truth, y_pred_high_risk, target_names=["Legitimate", "Fraud"]))

    # Save Model Artifact Bundle
    os.makedirs("models", exist_ok=True)
    model_bundle = {
        "model": model,
        "feature_names": feature_names,
        "s_min": s_min,
        "s_max": s_max,
    }
    
    model_path = "models/swapguard_isolation_forest.joblib"
    joblib.dump(model_bundle, model_path)
    print(f"\n✅ Model bundle successfully saved to: {model_path}")

if __name__ == "__main__":
    main()
