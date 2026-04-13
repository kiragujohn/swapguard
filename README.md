# sim-swap-fraud-detection-model
Data Set Definition 

## Dataset Field Descriptions

| Field Name                        | Description |
|----------------------------------|-------------|
| `sim_swap_time_gap_minutes`      | The time difference in minutes between the last SIM change and the current OTP request.<br> A **low value** (e.g., 1–2 mins) may indicate a fraud attempt, as fraudsters typically request OTPs shortly after swapping the SIM. |
| `sim_swap_flag`                  | A boolean label indicating whether the last network record is a **confirmed SIM swap**. Used as the ground truth for training/evaluation. |
| `device_change_flag`             | Indicates whether the **device (e.g., IMEI)** has changed compared to historical records.<br> `true` = suspicious if changed along with SIM. |
| `sim_type_change_flag`           | Indicates whether the **type of SIM** (e.g., physical to eSIM) changed.<br> May signal fraud or just a legitimate upgrade — needs correlation. |
| `imsi_change_flag`               | IMSI = International Mobile Subscriber Identity. A change indicates the user **swapped SIMs**. |
| `iccid_change_flag`              | ICCID = SIM card identifier. A change means the **SIM card was physically replaced**. |
| `otp_and_sim_change_geo_hash_length` | The **geohash prefix length** shared between the location of the SIM change and the OTP request.<br> A **shorter shared prefix** (e.g., 2–4) means the locations are far apart → suspicious.<br> A **longer match** (e.g., 9–10) means OTP was requested from the same/nearby location → normal. |

---

## Quick Interpretations

| Sample Row                                     | What's Going On |
|------------------------------------------------|-----------------|
| `2,true,true,true,false,true,3`               | SIM swap happened just 2 mins before OTP. New device and SIM type + changed ICCID. But IMSI same. Locations far apart (**geohash length 3**). ➤ **Very suspicious**. |
| `1440,false,false,false,false,false,9`        | 1 day since SIM change, no other changes. Locations match well. ➤ **Normal behavior**. |
| `1,true,true,true,true,true,2`                | **Extreme case**: 1-minute gap, all identity-related fields changed, and locations very far apart. ➤ **High fraud risk**. |
| `10080,false,false,false,false,false,10`      | SIM change occurred 1 week ago, no other changes, OTP came from same location. ➤ **Very safe behavior**. |


# how to run the code
To run the code, follow these steps:
navigate to the api directory and execute the following command:

```bash
python -m uvicorn api_server_v2:app --reload# swapguard-fraud-detection-model
# swapguard-fraud-detection-model
