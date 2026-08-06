import math
from datetime import datetime
from typing import Any, Dict
import pandas as pd


class PreSwapTelemetryAdapter:
    """Transforms raw self-explanatory pre-swap telemetry payloads

    into normalized numerical feature vectors for ML model inference.
    """

    @staticmethod
    def calculate_haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculates Great-Circle distance between two coordinates in kilometers."""
        R = 6371.0  # Earth radius in km

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def calculate_log_tenure_risk(days: float) -> float:
        """Calculates Logarithmic Tenure Decay: T_risk = 1 / (1 + ln(1 + t))"""
        if days < 0:
            days = 0.0
        return 1.0 / (1.0 + math.log(1.0 + days))

    @classmethod
    def transform_payload_to_features(
        cls, payload: Dict[str, Any]
    ) -> Dict[str, float]:
        """Maps nested self-explanatory JSON telemetry into flat engineered features."""
        target = payload.get("target_account_telemetry", {})
        requestor = payload.get("requestor_telemetry", {})
        velocity = payload.get("requestor_hardware_velocity_counters", {})
        location = payload.get("requestor_network_location", {})
        channel = payload.get("channel_and_session_metadata", {})

        # --- 1. Target Account Longevity & Prior Swap Recency ---
        target_tenure_days = float(
            target.get("target_sim_activation_age_days", 365.0)
        )
        target_tenure_risk = cls.calculate_log_tenure_risk(target_tenure_days)

        # Parse prior swap timestamp
        prior_swap_ts_str = target.get("target_prior_completed_swap_timestamp")
        days_since_last_swap = 999.0
        recent_swap_flag = 0.0

        if prior_swap_ts_str:
            try:
                ts_clean = str(prior_swap_ts_str).rstrip("Z")
                prior_swap_dt = datetime.fromisoformat(ts_clean)
                current_dt = datetime(2026, 5, 20, 12, 0, 0)
                delta_days = (current_dt - prior_swap_dt).total_seconds() / 86400.0
                days_since_last_swap = max(0.0, delta_days)
                if days_since_last_swap < 30.0:
                    recent_swap_flag = 1.0
            except ValueError:
                pass

        # --- 2. Requestor Identity & Burner SIM Detection ---
        requestor_tenure_days = float(
            requestor.get("requestor_sim_activation_age_days", 365.0)
        )
        requestor_tenure_risk = cls.calculate_log_tenure_risk(
            requestor_tenure_days
        )
        burner_sim_flag = 1.0 if requestor_tenure_days < 3.0 else 0.0
        requestor_is_prepaid = (
            1.0 if requestor.get("requestor_sim_is_prepaid", False) else 0.0
        )

        # --- 3. Hardware Velocity & SIM Farming Counters ---
        unique_imeis_30d = float(
            velocity.get("unique_imeis_used_by_requestor_msisdn_30d", 1)
        )
        unique_msisdns_7d = float(
            velocity.get("unique_msisdns_used_by_requestor_imei_7d", 1)
        )
        swap_requests_24h = float(
            velocity.get("requestor_swap_request_count_24h", 1)
        )

        high_device_velocity = 1.0 if unique_imeis_30d >= 3 else 0.0
        imei_farming_flag = 1.0 if unique_msisdns_7d >= 3 else 0.0

        # --- 4. Geographic Distance Calculation ---
        t_lat = float(target.get("target_home_registered_latitude", 0.0))
        t_lon = float(target.get("target_home_registered_longitude", 0.0))
        r_lat = float(location.get("requestor_cell_latitude", 0.0))
        r_lon = float(location.get("requestor_cell_longitude", 0.0))

        geo_distance_km = cls.calculate_haversine_distance(
            t_lat, t_lon, r_lat, r_lon
        )
        geo_mismatch_flag = 1.0 if geo_distance_km > 50.0 else 0.0

        silent_auth_failed = (
            0.0 if location.get("silent_network_auth_passed", True) else 1.0
        )

        # --- 5. Channel Automation Risk ---
        ch_name = str(channel.get("originating_channel", "app")).lower()
        bot_score = float(channel.get("automation_bot_score", 0.0))
        kba_only = 1.0 if channel.get("fallback_to_kba_only", False) else 0.0

        automated_channel_flag = (
            1.0 if ch_name in ["chatbot", "ivr", "rpa_api"] else 0.0
        )
        high_bot_score_flag = 1.0 if bot_score > 0.70 else 0.0

        return {
            "target_tenure_risk": target_tenure_risk,
            "days_since_last_swap": days_since_last_swap,
            "recent_swap_flag": recent_swap_flag,
            "requestor_tenure_risk": requestor_tenure_risk,
            "burner_sim_flag": burner_sim_flag,
            "requestor_is_prepaid": requestor_is_prepaid,
            "unique_imeis_used_by_requestor_msisdn_30d": unique_imeis_30d,
            "unique_msisdns_used_by_requestor_imei_7d": unique_msisdns_7d,
            "high_device_velocity": high_device_velocity,
            "imei_farming_flag": imei_farming_flag,
            "requestor_swap_request_count_24h": swap_requests_24h,
            "geo_distance_km": geo_distance_km,
            "geo_mismatch_flag": geo_mismatch_flag,
            "silent_auth_failed": silent_auth_failed,
            "automation_bot_score": bot_score,
            "automated_channel_flag": automated_channel_flag,
            "high_bot_score_flag": high_bot_score_flag,
            "fallback_to_kba_only": kba_only,
        }

    @classmethod
    def process_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Processes a flat CSV DataFrame of raw telemetry into an engineered feature DataFrame."""
        features_list = []

        for _, row in df.iterrows():
            payload = {
                "target_account_telemetry": {
                    "target_msisdn": row.get("target_msisdn"),
                    "target_sim_activation_age_days": row.get(
                        "target_sim_activation_age_days"
                    ),
                    "target_home_registered_latitude": row.get(
                        "target_home_registered_latitude"
                    ),
                    "target_home_registered_longitude": row.get(
                        "target_home_registered_longitude"
                    ),
                    "target_prior_completed_swap_timestamp": row.get(
                        "target_prior_completed_swap_timestamp"
                    ),
                },
                "requestor_telemetry": {
                    "requestor_originating_msisdn": row.get(
                        "requestor_originating_msisdn"
                    ),
                    "requestor_hardware_imei": row.get(
                        "requestor_hardware_imei"
                    ),
                    "requestor_sim_activation_age_days": row.get(
                        "requestor_sim_activation_age_days"
                    ),
                    "requestor_sim_is_prepaid": row.get(
                        "requestor_sim_is_prepaid"
                    ),
                },
                "requestor_hardware_velocity_counters": {
                    "unique_imeis_used_by_requestor_msisdn_30d": row.get(
                        "unique_imeis_used_by_requestor_msisdn_30d"
                    ),
                    "unique_msisdns_used_by_requestor_imei_7d": row.get(
                        "unique_msisdns_used_by_requestor_imei_7d"
                    ),
                    "requestor_swap_request_count_24h": row.get(
                        "requestor_swap_request_count_24h"
                    ),
                },
                "requestor_network_location": {
                    "requestor_cell_latitude": row.get(
                        "requestor_cell_latitude"
                    ),
                    "requestor_cell_longitude": row.get(
                        "requestor_cell_longitude"
                    ),
                    "silent_network_auth_passed": row.get(
                        "silent_network_auth_passed"
                    ),
                },
                "channel_and_session_metadata": {
                    "originating_channel": row.get("originating_channel"),
                    "automation_bot_score": row.get("automation_bot_score"),
                    "fallback_to_kba_only": row.get("fallback_to_kba_only"),
                },
            }

            features_list.append(cls.transform_payload_to_features(payload))

        return pd.DataFrame(features_list)