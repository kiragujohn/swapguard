🚨 SWAPGUARD: SIM Swap Fraud Detection using Unsupervised Learning
📌 Overview
SWAPGUARD is an AI-driven fraud detection system designed to identify suspicious SIM swap activity in telecommunications networks.
It leverages:
•	Unsupervised machine learning 
•	Advanced feature engineering 
•	Heuristic validation 
to detect anomalies without requiring labeled fraud data.
The system supports:
•	⚡ Real-time fraud detection 
•	📊 Risk scoring 
•	👨‍💻 Analyst review workflows 
 
🧠 Key Features
•	🔍 Unsupervised anomaly detection using Isolation Forest 
•	🧮 Advanced feature engineering for fraud signals 
•	📊 Risk scoring and anomaly ranking 
•	⚖️ Hybrid validation (ML + heuristics) 
•	🔁 Semi-supervised learning with feedback loop 
•	🚀 API-ready deployment with FastAPI 
 
📂 Dataset
The model expects the following input features:
•	sim_swap_time_gap_minutes 
•	device_change_flag 
•	sim_type_change_flag 
•	imsi_change_flag 
•	iccid_change_flag 
•	otp_and_sim_change_geo_hash_length 
 
⚙️ Data Processing Pipeline
1️⃣ Data Cleaning
•	Handle missing values using median imputation 
•	Convert boolean-like fields to binary (0/1) 
2️⃣ Feature Scaling
•	Normalize numerical features using MinMaxScaler 
3️⃣ Feature Engineering
Derived features:
•	time_urgency_score
→ Measures how quickly OTP is requested after SIM swap 
•	identity_shift_score
→ Total number of identity-related changes 
•	geo_risk_score
→ Measures geographic inconsistency 
 
🤖 Model: Isolation Forest
The system uses Isolation Forest, an unsupervised algorithm that:
•	Detects anomalies via random partitioning 
•	Works efficiently on large datasets 
•	Requires no labeled training data 
Training Steps
1.	Select relevant features 
2.	Train the model 
3.	Predict anomaly scores and flags 
 
📊 Anomaly Scoring & Risk Ranking
Each record is assigned:
•	anomaly_score → Raw model score 
•	anomaly_flag → Binary indicator 
•	anomaly_rank → Severity ranking 
•	risk_level: 
o	🔴 High (Top 5%) 
o	🟠 Medium (Next 10%) 
o	🟢 Low (Remaining) 
 
🧩 Heuristic Validation Layer
To improve accuracy, domain-specific rules are applied.
Example Rules
•	OTP requested shortly after SIM swap 
•	Multiple identity changes 
•	High geographic mismatch 
Output
•	heuristic_fraud_flag → High-confidence fraud indicator 
 
📈 Visualization
•	📊 Anomaly distribution charts 
•	📦 Boxplots for feature comparison 
•	🔥 Correlation heatmaps 
These help analysts understand fraud patterns and model behavior.
 
🔁 Feedback Loop & Model Improvement
•	Pseudo-labeling using heuristic flags 
•	Analyst feedback integration 
•	Semi-supervised learning (Logistic Regression) 
•	Model tuning and feature importance analysis 
 
🚀 Deployment
Run API (FastAPI)
uvicorn main:app --reload
Supported Deployment Options
•	FastAPI / Flask → real-time inference 
•	Docker → containerization 
•	Streamlit → dashboards 
•	Airflow / Cron → batch processing 
•	MLflow / DVC → model tracking 
 
📡 API Example
POST /predict/
Request
{
  "sim_swap_time_gap_minutes": 100,
  "device_change_flag": 1,
  "sim_type_change_flag": 0,
  "imsi_change_flag": 0,
  "iccid_change_flag": 0,
  "otp_and_sim_change_geo_hash_length": 0.2
}
Response
{
  "anomaly_score": -0.45,
  "anomaly_flag": -1
}
 
🧪 Testing
•	Generate sample inputs 
•	Apply feature enrichment 
•	Predict anomaly score and risk level 
 
🧮 Rule-Based Risk Assessment
•	🔴 High Risk 
o	Identity shift ≥ 3 
o	OR high urgency + geo mismatch 
•	🟠 Medium Risk 
o	Flagged by model 
•	🟢 Low Risk 
o	Normal behavior 
 
📊 Outputs
Generated files:
•	scored_sim_swap_anomalies.csv 
•	heuristically_validated_anomalies.csv 
•	isolation_forest_model.pkl 
 
🛠️ Tech Stack
•	Python 
•	Pandas, NumPy 
•	Scikit-learn 
•	Matplotlib, Seaborn 
•	FastAPI 
•	Joblib 
 
📈 Future Improvements
•	Integrate deep learning models (Autoencoders) 
•	Add real-time streaming (Kafka) 
•	Build monitoring dashboards (Grafana, Power BI) 
•	Implement SHAP for explainability 
•	Automate retraining pipelines 
 
🤝 Contribution
Contributions are welcome!
Please open issues or submit pull requests.
