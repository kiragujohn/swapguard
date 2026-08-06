"""
SwapGuard Adapters Module
Contains telemetry parsing and feature normalization for CAMARA and MNO APIs.
"""

from .camara_preswap_adapter import PreSwapTelemetryAdapter

__all__ = ["PreSwapTelemetryAdapter"]