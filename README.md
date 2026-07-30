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

### Telemetry Data Source Breakdown

| Telemetry Type | Fields | Source / Provider |
|---|---|---|
| CAMARA APIs | `latestSimChange`, `devicePhoneNumberVerified`, `location` | GSMA Open Gateway / CAMARA Northbound APIs |
| MNO Core Network | `deviceCount`, `msisdnCount` | Equipment Identity Register (EIR) / HLR |
| MNO Billing / CRM | `registered_lat`, `registered_lon`, `daysSinceMsisdnActivation` | MNO Subscriber Management Database |
| Channel Metadata | `channel`, `bot_score`, `kba_only` | Frontend Application / API Gateway |

---

## 🔬 SwapGuard Feature Engineering Specification

> 💡 **Crucial Architectural Concept: Target (Victim) vs. Requestor (Fraudster)**
>
> Standard telco fraud checks often fail because they query the **victim's** line history. Because the victim is typically a loyal, multi-year subscriber, their profile appears completely low-risk — allowing the transaction to pass.
>
> SwapGuard flips this model: it captures and analyzes the **Requestor's Line** (`requestor_msisdn`) and **Requestor's Device Hardware** (`requestor_imei`) placing the call, chat, or API request. This actively exposes burner SIMs, SIM farming hardware, and rapid attack vectors.

### 📊 Feature Categories Overview

| Category | Features | Core Security Objective |
|---|---|---|
| 1. Channel Risk | `request_channel`, `automation_score` | Detects non-human or automated attack vectors (Chatbot, IVR, RPA) that lack physical identity checks. |
| 2. Requestor Identity | `requestor_msisdn`, `requestor_imei` | Captures the phone number and hardware ID of the line placing the request (the fraudster's burner SIM). |
| 3. Geospatial Risk | `requestor_to_victim_distance_km`, `victim_location_verified_flag` | Measures physical distance between the victim's registered home address and where the requestor's signal originates. |
| 4. Requestor Device History | `unique_devices_count_by_requestor` | Identifies a single requestor line hopping across multiple physical burner phones. |
| 5. Requestor MSISDN History | `unique_msisdns_count_by_device` | Uncovers "SIM Farming" — hardware cycling through dozens of bulk prepaid SIM cards on a single device. |
| 6. Requestor Account Age | `requestor_msisdn_activation_days`, `requestor_msisdn_prepaid_flag` | Flags newly activated burner SIMs used immediately for account takeover attacks. |
| 7. Requestor Velocity | `requestor_swap_requests_24h`, `requestor_swap_requests_7d` | Measures repeated swap attempts initiated by the same requestor across short time windows. |

### 🔍 Detailed Feature Definitions & Risk Logic

**1. Channel Risk Detection**
- `request_channel`: System interface used to submit the SIM swap request (`app`, `web`, `ivr`, `chatbot`, `live_agent`, `rpa_api`). Interfaces like `chatbot`, `ivr`, or `rpa_api` signal higher risk due to lack of human possession verification.
- `automation_score`: Machine-calculated probability (0.0 to 1.0) that the session is driven by an automated script or botnet. Scores **> 0.70** trigger high-risk flags.

**2. Requestor Identity Capture**
- `requestor_msisdn`: The actual phone number initiating the swap request (e.g., the burner phone calling customer care or chatting with the bot) — **not** the victim's phone number being targeted.
- `requestor_imei`: The unique hardware identity (IMEI) of the physical handset used by the requestor at the moment of the request. Serves as the primary key for tracking physical attack hardware across network logs.

**3. Geospatial Risk Detection**
- `requestor_to_victim_distance_km`: Great-circle spatial distance (Haversine formula) between the victim's verified home/billing address and the cell tower/location of the requesting phone.
  - *Example*: Victim's billing address is in Tacoma, WA. Request originates from a cell tower in Honolulu, HI (>4,000 km away) via Chatbot → **FLAGGED HIGH RISK**.
- `victim_location_verified_flag`: Indicates whether the victim's billing address on file has been recently verified via KYC or billing updates.

**4. Requestor Device History (Device-to-Requestor Ratio)**
- `unique_devices_count_by_requestor`: Count of distinct phone hardware IDs (IMEIs) associated with the requestor's line over the last 30 days (`COUNT(DISTINCT requestor_imei)`). Counts **> 3** signal a fraudster hopping across burner handsets to avoid device blacklisting.

**5. Requestor MSISDN History (Requestor-to-Device Ratio)**
- `unique_msisdns_count_by_device`: Count of distinct phone numbers (MSISDNs) inserted into the requestor's physical handset over the last 7 days (`COUNT(DISTINCT requestor_msisdn)`). Uncovers "SIM Farming," where fraudsters cycle through bulk prepaid SIMs on a single device.

**6. Requestor Account Age (Burner SIM Detection)**
- `requestor_msisdn_activation_days`: Elapsed time (in days) since the requesting line was activated on the network. Activation age **< 3 days** indicates a newly purchased burner SIM used specifically for takeover attacks.
- `requestor_msisdn_prepaid_flag`: Identifies whether the requestor's line is an unvetted prepaid SIM vs. an identity-verified postpaid contract.

**7. Requestor Velocity**
- `requestor_swap_requests_24h`: SIM swap requests initiated by the same requestor line within the last 24 hours. Counts **> 3** indicate an automated or organized fraud campaign.
- `requestor_swap_requests_7d`: Total swap request volume from the requestor line over a 7-day window (**> 5** triggers velocity flags).

---

## 🧮 Mathematical Methodology & Feature Formulations

SwapGuard transforms raw JSON telemetry into normalized mathematical vectors for real-time inference.

**1. Logarithmic Tenure Decay ($\mathcal{T}_{\text{risk}}$)**

Applies a non-linear logarithmic transformation to subscriber activation tenure ($t$ days). Risk decays rapidly after 30 days, creating a smooth scale for account age:

$$\mathcal{T}_{\text{risk}} = \frac{1}{1 + \ln(1 + t)}$$

**2. Multi-Tier Swap Recency ($\mathcal{S}_{\text{recency}}$)**

Evaluates prior SIM swap history using dual-resolution temporal flags (within 7 days vs. 30 days):

$$\mathcal{S}_{\text{recency}} = (0.20 \cdot \mathbb{I}_{t_{\text{swap}} < 30}) + (0.30 \cdot \mathbb{I}_{t_{\text{swap}} < 7})$$

**3. Haversine Spatial Distance Mismatch ($\mathcal{D}_{\text{geo}}$)**

Calculates physical distance in kilometers between the subscriber's registered address $(\phi_1, \lambda_1)$ and requesting device location $(\phi_2, \lambda_2)$:

$$\mathcal{D}_{\text{geo}} = 2 R \cdot \arcsin\left( \sqrt{ \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right) } \right)$$

**4. Normalized Pre-Swap Risk Proxy ($\mathcal{P}$)**

Combines weighted behavioral anomalies into a baseline risk index before machine learning inference:

P = 0.25 × T_risk
  + 0.20 × I[recent_swap]
  + 0.20 × I[device_velocity]
  + 0.15 × I[imei_shared]
  + 0.20 × I[channel_risk]

**5. Continuous Anomaly Score Normalization ($R$)**

Isolation Forest decision function outputs $s(x)$ are mapped to a bounded risk score within $[0.0, 1.0]$:

$$R(x) = 1.0 - \frac{s(x) - s_{\min}}{s_{\max} - s_{\min}}$$

---

## 📊 Raw Field Mapping & Feature Engineering Matrix

> 💡 **Understanding `latestSimChange` in Pre-Swap Scoring**
>
> In a pre-swap evaluation workflow, the `latestSimChange` field returned by the CAMARA SIM Swap API represents the timestamp of the **prior** (most recent completed) SIM swap — **not** the current swap request being evaluated.

| Source API / DB | Raw Field(s) | Engineered Feature | Meaning & Mathematical Transformation |
|---|---|---|---|
| CAMARA SIM Swap API | `latestSimChange` (ISO 8601) | `days_since_last_swap`, `recent_swap_flag`, `very_recent_swap_flag` | **Prior Swap Recency**: Measures elapsed time since the subscriber's previous SIM change. Flagged 1 if a swap occurred within the last 30 days (`recent_swap_flag`) or 7 days (`very_recent_swap_flag`). High recency indicates potential account compromise or fraud velocity. |
| CAMARA Location API | `area.center.latitude`, `area.center.longitude` | `geo_distance_km`, `geo_inconsistency_flag` | **Geographic Mismatch**: Measures physical distance (km) via the Haversine formula between the requesting device's current cell location and the subscriber's registered billing address. Flagged 1 if distance > 50 km. |
| CAMARA Silent Auth API | `devicePhoneNumberVerified` | `silent_auth_failed` | **Possession Check**: Inverts network-level Silent Network Authentication status (0 if carrier verified device possession via header enrichment, 1 if auth failed or was bypassed). |
| MNO CRM / Billing | `daysSinceMsisdnActivation` | `tenure_days`, `tenure_risk_score` | **Subscriber Tenure Risk**: Applies logarithmic decay $\frac{1}{1 + \ln(1 + t)}$ to line age ($t$). Newly activated SIM lines/accounts carry higher inherent risk than established long-term subscribers. |
| MNO EIR / HSS | `deviceCount` | `device_count_window`, `high_device_velocity` | **Hardware Switching Velocity**: Tracks the number of distinct IMEIs (handsets) associated with the line in the past 30 days. Flagged 1 if ≥ 3 devices were used, signaling SIM hopping. |
| MNO EIR / HSS | `msisdnCount` | `imei_sharing_count`, `imei_shared_flag` | **IMEI Farming / Multi-SIM Flag**: Tracks how many distinct MSISDNs (phone numbers) have been attached to the requesting handset's IMEI. Flagged 1 if shared across multiple lines, indicating automated fraud hardware. |
| Channel Metadata | `channel`, `bot_score`, `kba_only` | `request_channel`, `bot_score`, `kba_only_flag`, `channel_risk` | **Channel & Identity Risk**: Evaluates the entry point (chatbot, ivr, rpa, web_portal) and authentication strength. High bot scores (> 0.60) or reliance purely on Knowledge-Based Authentication (KBA) trigger high channel risk. |

---

## 🔌 API Payload Specification

### Sample Ingestion Request (`POST /v1/risk-score`)

```json
{
  "request_id": "req_9f8a7b6c5d",
  "target_msisdn": "+12535550199",
  "requestor_msisdn": "+12535550888",
  "requestor_imei": "358249091234567",
  "camara_telemetry": {
    "latestSimChange": "2026-05-10T10:15:00Z",
    "devicePhoneNumberVerified": false,
    "location": {
      "area": { "center": { "latitude": 21.3069, "longitude": -157.8583 } }
    }
  },
  "mno_internal_telemetry": {
    "daysSinceTargetMsisdnActivation": 1825.0,
    "daysSinceRequestingMsisdnActivation": 0.2,
    "deviceCount": 4,
    "msisdnCount": 6,
    "registered_lat": 47.2529,
    "registered_lon": -122.4443
  },
  "channel_data": {
    "channel": "chatbot",
    "bot_score": 0.82,
    "kba_only": 1
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
    "AUTOMATED_CHANNEL_DETECTED",
    "GEOGRAPHIC_DISTANCE_MISMATCH_4000KM",
    "SHARED_IMEI_HARVESTING_DETECTED",
    "NEW_BURNER_REQUESTOR_SIM_0_2_DAYS"
  ],
  "processing_time_ms": 14.2
}
```

---

## 🟢 Risk Score Output Banding

MNO Policy Decision Points (PDP) map SwapGuard continuous risk scores (0.00 – 1.00) directly to authorization actions:

### 🔴 High Risk Score (0.80 – 1.00)
- **Telemetry Profile**: Channel = Chatbot | Location = 4,000 km mismatch | `imei_shared_flag` = 1 | `requestor_sim_age_days` = 0.2
- **Risk Assessment**: High probability of automated, credential-stuffing, or fraud-ring line takeover.
- **Recommended Action**: `BLOCK` transaction or require biometric in-person ID verification.

### 🟠 Medium Risk Score (0.40 – 0.79)
- **Telemetry Profile**: Channel = Web Portal | Location = 150 km mismatch | `msisdn_device_count_30d` = 3 | `line_activation_age_days` = 90
- **Risk Assessment**: Moderate behavioral anomaly.
- **Recommended Action**: `STEP_UP` authentication via SMS OTP or push notification.

### 🟢 Low Risk Score (0.00 – 0.39)
- **Telemetry Profile**: Channel = Mobile App | Location = < 5 km mismatch | `imei_sharing_count` = 1 | `line_activation_age_days` = 730
- **Risk Assessment**: Standard legitimate subscriber pattern.
- **Recommended Action**: `ALLOW` automated execution.

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

(Optional) If you wish to regenerate the dataset or modify parameters:

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

- **License**: Apache License 2.0
- **Copyright & Ownership**: Wahi Payment Systems LLC
- **API Compatibility**: GSMA Open Gateway / CAMARA SIM Swap API v1.0, Location Retrieval API v0.2, Number Verification API v1.0
- **Author / Maintainer**: John Kiragu Njoroge (Wahi Payment Systems LLC)
