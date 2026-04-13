import random
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_record():
    sim_swap_time_gap_minutes = random.randint(0, 10080)  # up to 7 days
    device_change_flag = random.choice([0, 1])
    sim_type_change_flag = random.choice([0, 1])
    imsi_change_flag = random.choice([0, 1])
    iccid_change_flag = random.choice([0, 1])
    otp_and_sim_change_geo_hash_length = round(random.uniform(0.0, 1.0), 2)

    # New features
    recent_sim_activation_days = random.randint(0, 365)
    num_sim_changes_last_30d = random.randint(0, 5)
    previous_sim_holder_tenure_days = random.randint(0, 1095)  # Up to 3 years
    account_age_days = random.randint(1, 1825)  # Up to 5 years
    ip_change_flag = random.choice([0, 1])

    # Sim swap flag logic (simple heuristic)
    identity_shift_score = device_change_flag + sim_type_change_flag + imsi_change_flag + iccid_change_flag
    geo_risk_score = 1 - otp_and_sim_change_geo_hash_length
    time_urgency_score = 1 - min(sim_swap_time_gap_minutes, 10080) / 10080

    fraud_score = (
            0.3 * identity_shift_score +
            0.2 * geo_risk_score +
            0.2 * time_urgency_score +
            0.1 * (1 if recent_sim_activation_days < 7 else 0) +
            0.1 * (1 if num_sim_changes_last_30d >= 2 else 0) +
            0.1 * ip_change_flag
    )
    sim_swap_flag = 1 if fraud_score > 0.7 else 0

    return {
        "sim_swap_time_gap_minutes": sim_swap_time_gap_minutes,
        "device_change_flag": device_change_flag,
        "sim_type_change_flag": sim_type_change_flag,
        "imsi_change_flag": imsi_change_flag,
        "iccid_change_flag": iccid_change_flag,
        "otp_and_sim_change_geo_hash_length": otp_and_sim_change_geo_hash_length,
        "recent_sim_activation_days": recent_sim_activation_days,
        "num_sim_changes_last_30d": num_sim_changes_last_30d,
        "previous_sim_holder_tenure_days": previous_sim_holder_tenure_days,
        "account_age_days": account_age_days,
        "ip_change_flag": ip_change_flag,
        "sim_swap_flag": sim_swap_flag
    }


def generate_dataset(n=50000):
    return pd.DataFrame([generate_synthetic_record() for _ in range(n)])


if __name__ == "__main__":
    df = generate_dataset(50000)
    df.to_csv("sim_swap_fraud_dataset_50000.csv", index=False)
    print("Dataset with 50,000 records saved as 'sim_swap_fraud_dataset_50000.csv'")
