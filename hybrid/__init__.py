"""
Hybrid Prediction System
========================

Combines XGBoost expected goals (xG) generation with Poisson probability mathematics
for superior accuracy with guaranteed quality through confidence gating.

Components:
- hybrid_predictor: Main orchestrator combining XGBoost xG with Poisson probabilities
- confidence_gate: Quality control for XGBoost predictions with automatic fallback
- ensemble: Optional weighted blending of XGBoost and Poisson predictions
"""

from .hybrid_predictor import predict_match_hybrid, Team
from .confidence_gate import evaluate_xgb_confidence, should_use_xgb_prediction
from .ensemble import blend_xg_predictions

__all__ = [
    'predict_match_hybrid',
    'Team',
    'evaluate_xgb_confidence',
    'should_use_xgb_prediction',
    'blend_xg_predictions'
]
