# 🛡️ SwapGuard Community Edition
> **Pre-Swap SIM Fraud Anomaly Detection & Risk Scoring Engine for Telecom Operators**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![CAMARA Compliant](https://img.shields.io/badge/Standard-GSMA_CAMARA-orange.svg)](https://camara-project.info/)

SwapGuard Community Edition is an open-source, pre-swap risk scoring middleware designed for Mobile Network Operators (MNOs) and telecom providers. It evaluates line-modification requests **at the exact moment of initiation** (pre-swap) to block unauthorized SIM swap takeovers before execution.

By combining standardized GSMA Open Gateway / CAMARA API responses with internal MNO network signals, SwapGuard processes incoming telemetry in real-time using a hybrid semi-supervised Isolation Forest model.

---

## 🛠️ Architecture & Telemetry Data Flow

Standard CAMARA Open Gateway APIs restrict raw hardware counters and PII due to strict privacy regulations. SwapGuard solves this by synthesizing northbound CAMARA signals with internal MNO core telemetry:

```text
[ Customer SIM Swap Request ]
               │
               ▼
   [ MNO Policy Orchestrator ]
 (Merges CAMARA APIs + Core DBs)
               │
               ▼  HTTP POST /v1/risk-score
      [ SWAPGUARD ENGINE ]
               │
               ├──> PreSwapTelemetryAdapter (Normalizes ISO dates & computes deltas)
               ├──> Feature Pipeline (Logarithmic transformations & spatial distance)
               ├──> Isolation Forest Model (Scores pre-swap vector)
               │
               ▼  HTTP 200 OK (Risk Score & Action Recommendation)
   [ MNO Action Gateway ] ──► (ALLOW / STEP_UP / BLOCK)
```

---

## 🔬 SwapGuard Feature Engineering Specification

> 💡 **Crucial Architectural Concept: Target (Victim) vs. Requestor (Fraudster)**
> Standard telco fraud checks often fail because they query the *victim's* line history. Because the victim is typically a loyal, multi-year subscriber, their profile appears completely low-risk—allowing the transaction to pass.
>
> **SwapGuard flips this model:** It captures and analyzes the **Requestor's Line (`requestor_originating_msisdn`)** and **Requestor's Device Hardware (`requestor_hardware_imei`)** placing the call, chat, or API request. This actively exposes burner SIMs, SIM farming hardware, and rapid attack vectors.

### 📊 Feature Categories Overview

| Category | Features | Core Security Objective |
| :--- | :--- | :--- |
| **1. Channel Risk** | `originating_channel`, `automation_bot_score` | Detects non-human or automated attack vectors (Chatbot, IVR, RPA) that lack physical identity checks. |
| **2. Requestor Identity** | `requestor_originating_msisdn`, `requestor_hardware_imei` | Captures the phone number and hardware ID of the line **placing the request** (the fraudster's burner SIM). |
| **3. Geospatial Risk** | `requestor_to_target_distance_km`, `target_home_registered_latitude/longitude` | Measures physical distance between the target's registered home address and where the requestor's cell signal originates. |
| **4. Requestor Device History** | `unique_imeis_used_by_requestor_msisdn_30d` | Identifies a single requestor line hopping across multiple physical burner phones. |
| **5. Requestor MSISDN History** | `unique_msisdns_used_by_requestor_imei_7d` | Uncovers "SIM Farming" hardware cycling through dozens of bulk prepaid SIM cards on a single device. |
| **6. Requestor Account Age** | `requestor_sim_activation_age_days`, `requestor_sim_is_prepaid` | Flags newly activated burner SIMs used immediately for account takeover attacks. |
| **7. Requestor Velocity** | `requestor_swap_request_count_24h` | Measures repeated swap attempts initiated by the same requestor across short time windows. |

---

Logarithmic Tenure Decay ($\mathcal{T}_{\text{risk}}$):$$\mathcal{T}_{\text{risk}} = \frac{1}{1 + \ln(1 + t)}$$Multi-Tier Swap Recency ($\mathcal{S}_{\text{recency}}$):$$\mathcal{S}_{\text{recency}} = (0.20 \cdot \mathbb{I}_{t_{\text{swap}} < 30}) + (0.30 \cdot \mathbb{I}_{t_{\text{swap}} < 7})$$Haversine Spatial Distance Mismatch ($\mathcal{D}_{\text{geo}}$):$$\mathcal{D}_{\text{geo}} = 2 R \cdot \arcsin\left( \sqrt{ \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right) } \right)$$Normalized Pre-Swap Risk Proxy ($\mathcal{P}$):$$\mathcal{P} = 0.25 \cdot \mathcal{T}_{\text{risk}} + 0.20 \cdot \mathbb{I}_{\text{recent\_swap}} + 0.20 \cdot \mathbb{I}_{\text{device\_velocity}} + 0.15 \cdot \mathbb{I}_{\text{imei\_shared}} + 0.20 \cdot \mathbb{I}_{\text{channel\_risk}}$$Continuous Anomaly Score Normalization ($R$):$$R(x) = 1.0 - \frac{s(x) - s_{\min}}{s_{\max} - s_{\min}}$$

---

## 📊 Self-Explanatory Field Mapping & Feature Engineering Matrix

> 💡 **Understanding `target_prior_completed_swap_timestamp` in Pre-Swap Scoring:**
> In a pre-swap evaluation workflow, this field represents the timestamp of the **prior (most recent completed) SIM swap** on the target account, **not** the current swap request being evaluated.

| Payload Group | Raw Self-Explanatory Field | Engineered Feature | Meaning & Transformation |
| :--- | :--- | :--- | :--- |
| **Target Account** | `target_prior_completed_swap_timestamp` *(ISO 8601)* | `days_since_last_swap`<br>`recent_swap_flag` | **Prior Swap Recency:** Measures elapsed time since the subscriber's *previous* completed SIM change. Flagged `1` if $<30$ days. |
| **Target Account** | `target_sim_activation_age_days` | `target_tenure_risk` | **Target Account Longevity:** Evaluates account age. Long-standing accounts carry low target-side risk. |
| **Target Account** | `target_home_registered_lat/lon` | `geo_distance_km` | **Base Coordinates:** Used with requestor location for spatial Haversine distance calculations. |
| **Requestor Identity** | `requestor_sim_activation_age_days` | `burner_sim_flag` | **Burner SIM Originator Check:** Measures how recently the *requesting* line was activated. Flagged `1` if age $< 3$ days. |
| **Requestor Velocity** | `unique_imeis_used_by_requestor_msisdn_30d` | `high_device_velocity` | **Hardware Hopping:** Tracks IMEIs attached to the requesting line over 30 days. Flagged `1` if $\ge 3$. |
| **Requestor Velocity** | `unique_msisdns_used_by_requestor_imei_7d` | `imei_farming_flag` | **SIM Farming Detection:** Tracks MSISDNs attached to the requesting handset's IMEI over 7 days. Flagged `1` if $\ge 3$. |
| **Requestor Location** | `requestor_cell_latitude/longitude` | `geo_distance_km` | **Geographic Mismatch:** Calculates distance to target registered home address. Flagged `1` if $> 50	ext{ km}$. |
| **Channel Metadata** | `originating_channel`, `automation_bot_score` | `channel_risk` | **Automation Risk:** Evaluates origin (`chatbot`, `ivr`, `rpa_api`) and bot scores ($>0.70$). |

---

## 🔌 API Payload Specification

### Sample Ingestion Request (`POST /v1/risk-score`)

```json
{
  "request_id": "req_9f8a7b6c5d",
  
  "target_account_telemetry": {
    "target_msisdn": "+12535550199",
    "target_sim_activation_age_days": 1825.0,
    "target_home_registered_latitude": 47.2529,
    "target_home_registered_longitude": -122.4443,
    "target_prior_completed_swap_timestamp": "2026-05-10T10:15:00Z"
  },

  "requestor_telemetry": {
    "requestor_originating_msisdn": "+12535550888",
    "requestor_hardware_imei": "358249091234567",
    "requestor_sim_activation_age_days": 0.2,
    "requestor_sim_is_prepaid": true
  },

  "requestor_hardware_velocity_counters": {
    "unique_imeis_used_by_requestor_msisdn_30d": 4,
    "unique_msisdns_used_by_requestor_imei_7d": 6,
    "requestor_swap_request_count_24h": 3
  },

  "requestor_network_location": {
    "requestor_cell_latitude": 21.3069,
    "requestor_cell_longitude": -157.8583,
    "silent_network_auth_passed": false
  },

  "channel_and_session_metadata": {
    "originating_channel": "chatbot",
    "automation_bot_score": 0.82,
    "fallback_to_kba_only": true
  }
}
```

### Sample Risk Evaluation Response

```json
{
  "request_id": "req_9f8a7b6c5d",
  "pre_swap_risk_score": 0.9241,
  "risk_band": "HIGH_RISK",
  "action_recommendation": "BLOCK_OR_STEP_UP",
  "top_risk_drivers": [
    "AUTOMATED_CHANNEL_DETECTED_CHATBOT",
    "GEOGRAPHIC_DISTANCE_MISMATCH_4300KM",
    "IMEI_FARMING_HARVESTING_DETECTED_6_MSISDNS",
    "NEW_BURNER_REQUESTOR_SIM_0_2_DAYS"
  ],
  "processing_time_ms": 14.2
}
```

---

## 🟢 Risk Score Output Banding

MNO Policy Decision Points (PDP) map SwapGuard continuous risk scores (`0.00` – `1.00`) directly to authorization actions:

* 🔴 **High Risk Score (`0.80` – `1.00`)**
  > **Telemetry Profile:** Channel = `chatbot` | Distance = `4,300 km` | `unique_msisdns_used_by_requestor_imei_7d` = `6` | `requestor_sim_activation_age_days` = `0.2`
  * **Risk Assessment:** High probability of automated, credential-stuffing, or fraud-ring line takeover.
  * **Recommended Action:** **BLOCK** transaction or require biometric in-person ID verification.

* 🟠 **Medium Risk Score (`0.40` – `0.79`)**
  > **Telemetry Profile:** Channel = `web_portal` | Distance = `150 km` | `unique_imeis_used_by_requestor_msisdn_30d` = `3`
  * **Risk Assessment:** Moderate behavioral anomaly.
  * **Recommended Action:** **STEP-UP** authentication via SMS OTP or push notification.

* 🟢 **Low Risk Score (`0.00` – `0.39`)**
  > **Telemetry Profile:** Channel = `mobile_app` | Distance = `< 5 km` | `unique_msisdns_used_by_requestor_imei_7d` = `1` | `requestor_sim_activation_age_days` = `730`
  * **Risk Assessment:** Standard legitimate subscriber pattern.
  * **Recommended Action:** **ALLOW** automated execution.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/kiragujohn/swapguard-community.git
cd swapguard-community
pip install -r requirements.txt
```

### 2. Dataset Setup

The repository comes pre-packaged with a 10,000-row telemetry dataset at `datasets/sim_swap_request_dataset.csv`.

*(Optional)* If you wish to regenerate the dataset or modify parameters:
```bash
python generate_dataset.py
```

### 3. Train Isolation Forest & Evaluate

Train the unsupervised anomaly detector using the included dataset:

```bash
python model_training.py
```

---

## 📄 License & Standards Compliance

* **License:** [Apache License 2.0](LICENSE)
* **Copyright & Ownership:** Wahi Payment Systems LLC
* **API Compatibility:** GSMA Open Gateway / CAMARA SIM Swap API v1.0, Location Retrieval API v0.2, Number Verification API v1.0.
* **Author / Maintainer:** John Kiragu Njoroge (Wahi Payment Systems LLC)
