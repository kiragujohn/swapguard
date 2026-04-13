import csv
import random

# File path (current directory)
file_path = "sim_swap_fraud_dataset.csv"

# Define the CSV headers
headers = [
    "sim_swap_time_gap_minutes", "sim_swap_flag", "device_change_flag",
    "sim_type_change_flag", "imsi_change_flag", "iccid_change_flag",
    "otp_and_sim_change_geo_hash_length"
]

# Generate synthetic records
def generate_synthetic_data(num_records=10000):
    data = []
    for _ in range(num_records):
        gap = random.choice([1, 2, 5, 10, 30, 60, 1440, 10080])
        fraud = gap <= 10 and random.random() < 0.7
        device = random.random() < (0.5 if fraud else 0.1)
        sim_type = random.random() < (0.4 if fraud else 0.1)
        imsi = random.random() < (0.6 if fraud else 0.1)
        iccid = random.random() < (0.6 if fraud else 0.1)
        geo = random.randint(2, 5) if fraud else random.randint(6, 10)
        data.append([gap, fraud, device, sim_type, imsi, iccid, geo])
    return data

# Write data to CSV
with open(file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(generate_synthetic_data())

print(f"Dataset saved as {file_path}")