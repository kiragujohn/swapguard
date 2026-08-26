# 🛡️ SwapGuard Community Edition
> **Hybrid AI Pre-Swap SIM Fraud Risk Assessment Engine for Telecommunications Networks**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![CAMARA Aligned](https://img.shields.io/badge/Standard-GSMA_CAMARA_Aligned-orange.svg)](https://camara-project.info/)

SwapGuard Community Edition is an open-source, pre-swap risk-scoring middleware designed for Mobile Network Operators (MNOs) and telecom providers. It evaluates line-modification requests **at the point of initiation** (pre-swap) to support detection and mitigation of potentially unauthorized SIM-swap requests before execution.

SwapGuard combines standardized GSMA Open Gateway / CAMARA-aligned network signals with authorized MNO and request-context telemetry to produce explainable, real-time Pre-Swap risk assessments.

The current prototype combines three complementary decision layers:

1. **Isolation Forest anomaly detection** — detects behavioral patterns that deviate from learned baseline activity.
2. **Random Forest classification** — estimates whether telemetry resembles known fraud patterns represented in the training data.
3. **Telecommunications policy controls** — applies transparent security rules when combinations of high-risk signals converge.

The resulting assessment exposes the final risk score together with model-level evidence, human-readable risk drivers, triggered policy rules, and a recommended action.

> **Prototype notice:** Current model evaluation uses synthetic or authorized test telemetry. Reported metrics should not be interpreted as production carrier performance.

---

## 🛠️ Architecture & Telemetry Data Flow

Standardized CAMARA / Open Gateway APIs expose defined network capabilities rather than all internal carrier telemetry. SwapGuard's target architecture can combine applicable standardized signals with additional provider-authorized internal telemetry where available:

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
               ├──> Isolation Forest (Behavioral anomaly detection)
               ├──> Random Forest (Known-pattern fraud probability)
               ├──> Hybrid Fusion (Combines ML evidence)
               └──> Policy Engine (Transparent telecom-security controls)
               │
               ▼  HTTP 200 OK (Risk Score & Action Recommendation)
   [ MNO Action Gateway ] ──► (ALLOW / STEP_UP / BLOCK)
```

---

## 🔬 SwapGuard Feature Engineering Specification

> 💡 **Crucial Architectural Concept: Target Subscriber vs. Requestor Context**
> A target subscriber's historical account profile may appear low-risk even when the current SIM-swap request is being initiated through a different device, line, or channel. Evaluating only the target account's history can therefore miss risk indicators associated with the request itself.
>
> **SwapGuard expands the evaluation context:** Where those signals are available and authorized, it can analyze the **requestor's line (`requestor_originating_msisdn`)** and **requestor's device identifier (`requestor_hardware_imei`)** associated with the call, chat, or API request. These signals can help identify suspicious device reuse, recently activated lines, unusual request velocity, and other request-side anomalies.

### 📊 Feature Categories Overview

| Category | Features | Core Security Objective |
| :--- | :--- | :--- |
| **1. Channel Risk** | `originating_channel`, `automation_bot_score` | Detects non-human or automated attack vectors (Chatbot, IVR, RPA) that lack physical identity checks. |
| **2. Requestor Identity** | `requestor_originating_msisdn`, `requestor_hardware_imei` | Represents the line and device associated with the request, where available and authorized. |
| **3. Geospatial Risk** | `requestor_to_target_distance_km`, `target_home_registered_latitude/longitude` | Measures physical distance between the target's registered home address and where the requestor's cell signal originates. |
| **4. Requestor Device History** | `unique_imeis_used_by_requestor_msisdn_30d` | Measures recent device reuse associated with the requesting line and can flag unusually high device-association velocity. |
| **5. Requestor MSISDN History** | `unique_msisdns_used_by_requestor_imei_7d` | Measures how many mobile numbers were recently associated with the requesting device and can flag unusually high multi-line device reuse. |
| **6. Requestor Account Age** | `requestor_sim_activation_age_days`, `requestor_sim_is_prepaid` | Evaluates whether the requesting line is newly activated or otherwise exhibits account-age characteristics relevant to risk. |
| **7. Requestor Velocity** | `requestor_swap_request_count_24h` | Measures repeated swap attempts initiated by the same requestor across short time windows. |

---
## 🧠 Current Hybrid Model Feature Contract

The current `risk_engine.py` model contract uses the following core inputs. Supplemental Streamlit fields may also be retained for demonstration, logging, or future feature engineering.

| Feature | Category | Security Meaning |
| :--- | :--- | :--- |
| `tenure_days_since_activation` | Account Tenure | Age of the relevant subscriber/line relationship. |
| `swap_age_hours` | SIM Change History | Elapsed time since the prior SIM change. |
| `device_velocity_count_7d` | Device Behavior | Recent device-association velocity. |
| `imei_shared_msisdn_count` | Device Association | Number of MSISDNs associated with the relevant device/IMEI. |
| `device_is_new_imei` | Device Identity | Indicates a new or unrecognized device. |
| `geo_mno_to_home_distance_km` | Geospatial Risk | Geographic deviation from an expected/home reference. |
| `originating_channel` | Channel Risk | App, web, chatbot, IVR, agent, RPA/API, or other request channel. |
| `channel_bot_score` | Automation Risk | Evidence that the request may be automated. |
| `profile_days_since_contact_change` | Account Change Risk | Recency of contact/recovery-information changes. |
| `auth_silent_possession_passed` | Possession Verification | Network-level possession-verification result. |
| `roaming_is_active` | Network Context | Indicates current roaming state. |
| `roaming_country_mismatch` | Network/Location Risk | Indicates inconsistent roaming/geographic context. |

---

## 🧮 Mathematical Methodology & Feature Formulations

SwapGuard transforms raw JSON telemetry into normalized mathematical vectors for real-time inference.

### 1. Logarithmic Tenure Decay ($\mathcal{T}_{\text{risk}}$)
Applies a non-linear logarithmic transformation to subscriber activation tenure ($t$ days). Risk decays rapidly after 30 days, creating a smooth scale for account age:

$$\mathcal{T}_{\text{risk}} = \frac{1}{1 + \ln(1 + t)}$$

### 2. Multi-Tier Swap Recency ($\mathcal{S}_{\text{recency}}$)
Evaluates prior SIM swap history using dual-resolution temporal flags (within 7 days vs. 30 days):

$$\mathcal{S}_{\text{recency}} = (0.20 \cdot \mathbb{I}_{t_{\text{swap}} < 30}) + (0.30 \cdot \mathbb{I}_{t_{\text{swap}} < 7})$$

### 3. Haversine Spatial Distance Mismatch ($\mathcal{D}_{\text{geo}}$)
Calculates physical distance in kilometers between the target's registered address $(\phi_1, \lambda_1)$ and requesting cell tower location $(\phi_2, \lambda_2)$:

$$\mathcal{D}_{\text{geo}} = 2 R \cdot \arcsin\left( \sqrt{ \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right) } \right)$$

### 4. Normalized Pre-Swap Risk Proxy ($\mathcal{P}$)
Combines weighted behavioral anomalies into a baseline risk index before machine learning inference:

$$\mathcal{P} = 0.25\,\mathcal{T}_{\mathrm{risk}} + 0.20\,I_{\mathrm{recent\,swap}} + 0.20\,I_{\mathrm{device\,velocity}} + 0.15\,I_{\mathrm{imei\,shared}} + 0.20\,I_{\mathrm{channel\,risk}}$$

### 5. Continuous Anomaly Score Normalization ($R$)
Isolation Forest decision function outputs $s(x)$ are mapped to a bounded risk score within $[0.0, 1.0]$:

$$R(x) = 1.0 - \frac{s(x) - s_{\min}}{s_{\max} - s_{\min}}$$
---

### 6. Supervised Fraud Probability

The Random Forest component estimates:

$$S(x) = P(Y=\text{fraud}\mid x)$$

where $x$ is the engineered feature vector. This component captures fraud patterns represented in the supervised training examples.

### 7. Hybrid ML Risk Fusion

SwapGuard combines normalized anomaly evidence and supervised fraud probability:

$$H(x) = w_a A(x) + w_s S(x)$$

subject to:

$$w_a + w_s = 1$$

where $A(x)$ is the normalized Isolation Forest anomaly score and $S(x)$ is the Random Forest fraud probability.

Fusion parameters are selected using the validation partition and frozen before evaluation against the untouched test partition. Current experimental weights are therefore model-version-specific calibration parameters rather than universal SwapGuard constants.

### 8. Policy Risk Floor

Transparent telecom-security rules can establish a minimum risk level when strong security signals converge:

$$R_{\text{final}}(x) = \max(H(x), P_{\text{floor}}(x))$$

The policy layer is reported separately through `policy_rules_triggered`, preserving explainability between statistical model evidence and deterministic security controls.

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
| **Requestor Location** | `requestor_cell_latitude/longitude` | `geo_distance_km` | **Geographic Mismatch:** Calculates distance to target registered home address. Flagged `1` if $> 50\,\mathrm{km}$. |
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

### Sample Hybrid Risk Evaluation Response

```json
{
  "risk_score": 0.9156,
  "risk_band": "HIGH",
  "recommended_action": "HOLD_OR_MANUAL_REVIEW",
  "anomaly_score": 1.0,
  "supervised_fraud_probability": 0.789,
  "hybrid_ml_score": 0.9156,
  "policy_floor": 0.85,
  "risk_drivers": [
    "SILENT_AUTH_FAILED",
    "RECENT_PROFILE_CHANGE",
    "HIGH_AUTOMATION_SCORE",
    "VERY_RECENT_SIM_CHANGE",
    "NEW_DEVICE",
    "GEOGRAPHIC_INCONSISTENCY",
    "HIGH_DEVICE_VELOCITY",
    "MULTI_LINE_DEVICE_ASSOCIATION"
  ],
  "policy_rules_triggered": [
    "RULE_POSSESSION_FAILURE_PLUS_PROFILE_CHANGE",
    "RULE_POSSESSION_FAILURE_PLUS_HIGH_AUTOMATION",
    "RULE_RECENT_SWAP_NEW_DEVICE_GEO_MISMATCH"
  ],
  "model_version": "swapguard-community-hybrid-v0.1"
}
```

### Explainable Risk Output

The engine separates model evidence from the final policy-aware assessment:

| Output | Meaning |
| :--- | :--- |
| `anomaly_score` | Behavioral abnormality estimated by Isolation Forest. |
| `supervised_fraud_probability` | Known-pattern fraud probability estimated by Random Forest. |
| `hybrid_ml_score` | Fused supervised/unsupervised model score. |
| `policy_floor` | Minimum risk established by triggered transparent policy controls. |
| `risk_score` | Final policy-aware risk assessment. |
| `risk_drivers` | Human-readable security conditions contributing to the assessment. |
| `policy_rules_triggered` | Explicit deterministic rules activated during evaluation. |


---

## 🟢 Risk Score Output Banding

Illustrative MNO policy decision points can map SwapGuard continuous risk scores (`0.00` – `1.00`) to operator-defined review or authentication actions. The carrier retains authority over the final operational decision:

* 🔴 **High Risk Score (`0.80` – `1.00`)**
  > **Telemetry Profile:** Channel = `chatbot` | Distance = `4,300 km` | `unique_msisdns_used_by_requestor_imei_7d` = `6` | `requestor_sim_activation_age_days` = `0.2`
  * **Risk Assessment:** Elevated risk based on the demonstrated combination of behavioral, device, geographic, and policy indicators.
  * **Recommended Action:** **HOLD OR MANUAL REVIEW**, or require additional identity verification under the operator's approved controls.

* 🟠 **Medium Risk Score (`0.40` – `0.79`)**
  > **Telemetry Profile:** Channel = `web_portal` | Distance = `150 km` | `unique_imeis_used_by_requestor_msisdn_30d` = `3`
  * **Risk Assessment:** Moderate behavioral anomaly.
  * **Recommended Action:** **STEP-UP** authentication using an operator-approved additional verification method.

* 🟢 **Low Risk Score (`0.00` – `0.39`)**
  > **Telemetry Profile:** Channel = `mobile_app` | Distance = `< 5 km` | `unique_msisdns_used_by_requestor_imei_7d` = `1` | `requestor_sim_activation_age_days` = `730`
  * **Risk Assessment:** Standard legitimate subscriber pattern.
  * **Recommended Action:** **ALLOW WITH STANDARD CONTROLS**, subject to the operator's approved provisioning and fraud-control policies.

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

### 3. Train and Benchmark the Hybrid Models

```bash
python model_training.py
```

The current evaluation workflow uses:

```text
70% Training
15% Validation
15% Untouched Test
```

Training fits the Isolation Forest, Random Forest, and preprocessing components. Validation selects candidate thresholds and hybrid fusion parameters. The selected configuration is frozen before final evaluation on the untouched test partition.

### 4. Current Synthetic Held-Out Benchmark

| Model | ROC-AUC | PR-AUC | Fraud Precision | Fraud Recall | Fraud F1 | FP | FN | TP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Isolation Forest | 0.6589 | 0.1522 | 0.1895 | 0.1607 | 0.1739 | 77 | 94 | 18 |
| Random Forest | 0.6185 | 0.1340 | 0.1944 | 0.1875 | 0.1909 | 87 | 91 | 21 |
| **SwapGuard Hybrid** | **0.6563** | **0.1548** | **0.1985** | **0.2411** | **0.2177** | **109** | **85** | **27** |

In this synthetic held-out experiment, the Hybrid configuration achieved the highest fraud recall and fraud F1 of the three evaluated configurations.

> **Important:** These results come from a synthetic research dataset. They must not be interpreted as expected production fraud-detection performance. Validation and calibration against appropriately authorized telecommunications data are required before operational deployment.

### 5. Run the Streamlit Demonstration

```bash
streamlit run app.py
```

The Streamlit application demonstrates:

- interactive Pre-Swap telemetry simulation;
- final risk score and risk band;
- behavioral anomaly score;
- supervised known-pattern fraud probability;
- hybrid ML score;
- policy risk floor;
- explainable risk drivers;
- triggered telecom-security policy rules;
- recommended action;
- model/engine version;
- GSMA Open Gateway / CAMARA architecture documentation; and
- prototype scope and limitations.

The Streamlit interface is a research and engineering demonstration using simulated or authorized test telemetry. It does not access production carrier records, modify subscriber services, or execute controls in telecommunications, financial, healthcare, or government systems.

---

## 🏗️ Enterprise Integration Direction

The Community Edition focuses on the open-source Pre-Swap risk engine. The broader architecture is designed so the engine can sit behind an enterprise integration layer:

```text
MNO / Authorized Client
          │
          ▼
Spring Boot API Layer
          │
          ▼
Orchestration / Integration Layer
          │
          ├── CAMARA / Open Gateway
          ├── Authorized MNO telemetry
          └── Channel / session context
          │
          ▼
SwapGuard Risk Engine
          │
          ├── Isolation Forest
          ├── Random Forest
          └── Policy Engine
          │
          ▼
Explainable Risk Response
```

Apache Camel or comparable orchestration technology can later aggregate multiple authorized telemetry sources, normalize provider responses, apply resilience patterns, and route the resulting feature context to the risk engine.

---

## 📄 License & Standards Compliance

* **License:** [Apache License 2.0](LICENSE)
* **Copyright:** © John Kiragu Njoroge
* **Standards Direction:** GSMA Open Gateway / CAMARA-aligned network API integration. Relevant API specifications are implemented where applicable; formal certification is not claimed unless separately documented.
* **Model Architecture:** Isolation Forest + Random Forest + Hybrid Fusion + Policy Engine
* **Primary Scope:** Open-source Pre-Swap SIM/eSIM fraud risk assessment
* **Author / Maintainer:** John Kiragu Njoroge


---

## ⚠️ Disclaimer

SwapGuard Community Edition is a prototype research and engineering project.

Risk scores, thresholds, policy rules, synthetic datasets, model metrics, recommendations, and example workflows in this repository are provided for experimentation and demonstration. They should not be treated as production security controls without appropriate validation, governance, regulatory review, operational testing, and calibration using authorized data.
