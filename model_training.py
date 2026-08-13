import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# SwapGuard Community Edition
# Three-model benchmark:
#   A. Isolation Forest (unsupervised)
#   B. Random Forest (supervised)
#   C. Hybrid fusion (unsupervised + supervised)
#
# IMPORTANT:
# - Training data fits the models.
# - Validation data selects thresholds and hybrid weight.
# - Test data is untouched until final evaluation.
# -----------------------------------------------------------------------------

DATASET_CANDIDATES = [
    "datasets/sim_swap_request_dataset.csv",
    "sim_swap_request_dataset.csv",
]

MODEL_PATH = "models/swapguard_hybrid_bundle.joblib"

RANDOM_STATE = 42
ISOLATION_CONTAMINATION = 0.08

# 70% train / 15% validation / 15% test
TEST_SIZE = 0.15
VALIDATION_SIZE_OF_REMAINDER = 0.15 / 0.85

FEATURE_COLUMNS: List[str] = [
    "tenure_days_since_activation",
    "tenure_is_new_line_30d",
    "swap_is_swapped_in_window",
    "swap_age_hours",
    "swap_is_very_recent_24h",
    "swap_is_recent_7d",
    "device_velocity_count_7d",
    "device_velocity_high_flag",
    "imei_shared_msisdn_count",
    "imei_is_shared_across_numbers",
    "device_is_new_imei",
    "geo_mno_to_home_distance_km",
    "geo_inconsistency_flag",
    "originating_channel",
    "channel_is_automated",
    "channel_bot_score",
    "channel_is_high_risk_bot",
    "profile_days_since_contact_change",
    "profile_contact_changed_recently_7d",
    "auth_silent_possession_passed",
    "auth_silent_possession_failed",
    "roaming_is_active",
    "roaming_country_mismatch",
]

CATEGORICAL_FEATURES = ["originating_channel"]


def resolve_dataset_path() -> str:
    for path in DATASET_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Dataset not found. Run `python generate_dataset.py` first."
    )


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            "Dataset is missing required features:\n"
            + "\n".join(f" - {name}" for name in missing)
        )
    if "is_fraud_label" not in df.columns:
        raise KeyError("Dataset must contain `is_fraud_label` for evaluation.")


def split_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/15/15 split.

    The test set is separated first and remains untouched during threshold
    and fusion-weight selection.
    """
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["is_fraud_label"],
    )

    train, validation = train_test_split(
        train_val,
        test_size=VALIDATION_SIZE_OF_REMAINDER,
        random_state=RANDOM_STATE,
        stratify=train_val["is_fraud_label"],
    )

    return train, validation, test


def build_feature_matrices(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Fit the categorical schema and missing-value treatment using training
    data only, then align validation and test to the same feature order.
    """
    X_train = train_df[FEATURE_COLUMNS].copy()
    X_val = validation_df[FEATURE_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()

    X_train = pd.get_dummies(
        X_train,
        columns=CATEGORICAL_FEATURES,
        prefix="channel",
        dtype=int,
    )
    X_val = pd.get_dummies(
        X_val,
        columns=CATEGORICAL_FEATURES,
        prefix="channel",
        dtype=int,
    )
    X_test = pd.get_dummies(
        X_test,
        columns=CATEGORICAL_FEATURES,
        prefix="channel",
        dtype=int,
    )

    feature_names = list(X_train.columns)

    X_val = X_val.reindex(columns=feature_names, fill_value=0)
    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    train_medians = X_train.median(numeric_only=True)

    X_train = X_train.fillna(train_medians).astype(float)
    X_val = X_val.fillna(train_medians).astype(float)
    X_test = X_test.fillna(train_medians).astype(float)

    return X_train, X_val, X_test, feature_names, train_medians.to_dict()


def fit_iforest_calibration(
    model: IsolationForest,
    X_train: pd.DataFrame,
) -> Dict[str, float]:
    raw = model.decision_function(X_train)

    q_low = float(np.quantile(raw, 0.01))
    q_high = float(np.quantile(raw, 0.99))

    if q_high <= q_low:
        raise ValueError("Invalid Isolation Forest calibration range.")

    return {
        "q_low": q_low,
        "q_high": q_high,
        "lower_quantile": 0.01,
        "upper_quantile": 0.99,
    }


def iforest_risk(
    model: IsolationForest,
    X: pd.DataFrame,
    calibration: Dict[str, float],
) -> np.ndarray:
    raw = model.decision_function(X)

    risk = 1.0 - (
        (raw - calibration["q_low"])
        / (calibration["q_high"] - calibration["q_low"])
    )

    return np.clip(risk, 0.0, 1.0)


def choose_f1_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Select a decision threshold using validation F1 only.

    This threshold is then frozen before final test evaluation.
    """
    best_threshold = 0.50
    best_f1 = -1.0

    # Search score thresholds without touching the final test labels.
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (scores >= threshold).astype(int)

        _, _, f1, _ = precision_recall_fscore_support(
            y_true,
            pred,
            average="binary",
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def choose_hybrid_weight(
    y_val: np.ndarray,
    anomaly_val: np.ndarray,
    supervised_val: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Select supervised/anomaly fusion weight and threshold using validation
    data only.

    hybrid = supervised_weight * supervised_probability
             + (1-supervised_weight) * anomaly_score
    """
    best_weight = 0.50
    best_threshold = 0.50
    best_f1 = -1.0

    for supervised_weight in np.linspace(0.0, 1.0, 21):
        hybrid_score = (
            supervised_weight * supervised_val
            + (1.0 - supervised_weight) * anomaly_val
        )

        threshold, f1 = choose_f1_threshold(y_val, hybrid_score)

        if f1 > best_f1:
            best_f1 = f1
            best_weight = float(supervised_weight)
            best_threshold = threshold

    return best_weight, best_threshold, best_f1


def evaluate(
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    pred = (scores >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        average="binary",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()

    result = {
        "Model": name,
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc),
        "Fraud Precision": float(precision),
        "Fraud Recall": float(recall),
        "Fraud F1": float(f1),
        "Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print(f"\n{'=' * 72}")
    print(name)
    print(f"{'=' * 72}")
    print(f"ROC-AUC:        {roc_auc:.4f}")
    print(f"PR-AUC:         {pr_auc:.4f}")
    print(f"Threshold:      {threshold:.4f}")
    print(f"Fraud Precision:{precision:>9.4f}")
    print(f"Fraud Recall:   {recall:>9.4f}")
    print(f"Fraud F1:       {f1:>9.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    return result


def main() -> None:
    print("🛡️ SwapGuard — Hybrid Supervised/Unsupervised Benchmark")
    print("=======================================================")

    dataset_path = resolve_dataset_path()
    df = pd.read_csv(dataset_path)
    validate_columns(df)

    train_df, validation_df, test_df = split_dataset(df)

    y_train = train_df["is_fraud_label"].astype(int).to_numpy()
    y_val = validation_df["is_fraud_label"].astype(int).to_numpy()
    y_test = test_df["is_fraud_label"].astype(int).to_numpy()

    print(f"\n📥 Dataset: {dataset_path}")
    print(f" • Total records:      {len(df):,}")
    print(f" • Training records:   {len(train_df):,}")
    print(f" • Validation records: {len(validation_df):,}")
    print(f" • Test records:       {len(test_df):,}")
    print(f" • Synthetic fraud rate: {df['is_fraud_label'].mean():.2%}")

    X_train, X_val, X_test, feature_names, train_medians = (
        build_feature_matrices(train_df, validation_df, test_df)
    )

    print(f" • Encoded model features: {len(feature_names)}")

    # ------------------------------------------------------------------
    # MODEL A — Isolation Forest
    # ------------------------------------------------------------------
    print("\n🌲 Training Model A: Isolation Forest...")

    isolation_forest = IsolationForest(
        n_estimators=250,
        contamination=ISOLATION_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    isolation_forest.fit(X_train)

    iforest_calibration = fit_iforest_calibration(
        isolation_forest,
        X_train,
    )

    anomaly_val = iforest_risk(
        isolation_forest,
        X_val,
        iforest_calibration,
    )
    anomaly_test = iforest_risk(
        isolation_forest,
        X_test,
        iforest_calibration,
    )

    iforest_threshold, iforest_validation_f1 = choose_f1_threshold(
        y_val,
        anomaly_val,
    )

    # ------------------------------------------------------------------
    # MODEL B — Random Forest
    # ------------------------------------------------------------------
    print("🌳 Training Model B: Random Forest...")

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    random_forest.fit(X_train, y_train)

    supervised_val = random_forest.predict_proba(X_val)[:, 1]
    supervised_test = random_forest.predict_proba(X_test)[:, 1]

    rf_threshold, rf_validation_f1 = choose_f1_threshold(
        y_val,
        supervised_val,
    )

    # ------------------------------------------------------------------
    # MODEL C — Hybrid
    # ------------------------------------------------------------------
    print("🧬 Selecting Model C: Hybrid fusion on validation set...")

    (
        supervised_weight,
        hybrid_threshold,
        hybrid_validation_f1,
    ) = choose_hybrid_weight(
        y_val,
        anomaly_val,
        supervised_val,
    )

    anomaly_weight = 1.0 - supervised_weight

    hybrid_test = (
        supervised_weight * supervised_test
        + anomaly_weight * anomaly_test
    )

    print("\n🔧 Validation-selected configuration")
    print(
        f" • Isolation Forest threshold: {iforest_threshold:.4f} "
        f"(validation F1={iforest_validation_f1:.4f})"
    )
    print(
        f" • Random Forest threshold:    {rf_threshold:.4f} "
        f"(validation F1={rf_validation_f1:.4f})"
    )
    print(
        f" • Hybrid supervised weight:   {supervised_weight:.2f}"
    )
    print(
        f" • Hybrid anomaly weight:      {anomaly_weight:.2f}"
    )
    print(
        f" • Hybrid threshold:           {hybrid_threshold:.4f} "
        f"(validation F1={hybrid_validation_f1:.4f})"
    )

    print(
        "\n🔒 Configuration frozen. "
        "Evaluating the untouched test partition..."
    )

    results = []

    results.append(
        evaluate(
            "Isolation Forest",
            y_test,
            anomaly_test,
            iforest_threshold,
        )
    )

    results.append(
        evaluate(
            "Random Forest",
            y_test,
            supervised_test,
            rf_threshold,
        )
    )

    results.append(
        evaluate(
            "SwapGuard Hybrid",
            y_test,
            hybrid_test,
            hybrid_threshold,
        )
    )

    results_df = pd.DataFrame(results)

    print("\n\n📊 FINAL HELD-OUT MODEL COMPARISON")
    print("===================================")

    display_columns = [
        "Model",
        "ROC-AUC",
        "PR-AUC",
        "Fraud Precision",
        "Fraud Recall",
        "Fraud F1",
        "FP",
        "FN",
        "TP",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Persist model bundle and benchmark CSV.
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    bundle = {
        "isolation_forest": isolation_forest,
        "random_forest": random_forest,
        "feature_names": feature_names,
        "source_feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_medians": train_medians,
        "iforest_calibration": iforest_calibration,
        "iforest_threshold": iforest_threshold,
        "random_forest_threshold": rf_threshold,
        "hybrid_supervised_weight": supervised_weight,
        "hybrid_anomaly_weight": anomaly_weight,
        "hybrid_threshold": hybrid_threshold,
        "random_state": RANDOM_STATE,
        "synthetic_fraud_rate": float(df["is_fraud_label"].mean()),
        "benchmark_results": results,
    }

    joblib.dump(bundle, MODEL_PATH)

    benchmark_path = "models/swapguard_benchmark_results.csv"
    results_df.to_csv(benchmark_path, index=False)

    print(f"\n✅ Hybrid model bundle saved to: {MODEL_PATH}")
    print(f"✅ Benchmark results saved to:   {benchmark_path}")
    print(
        "\n⚠️ These are synthetic held-out benchmark results. "
        "They must not be represented as production carrier performance."
    )


if __name__ == "__main__":
    main()