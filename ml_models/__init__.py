"""
ML Models Package for Football Prediction System
=================================================

This package contains machine learning models that extend the base Poisson
prediction system. Models are only used in production if they match or exceed
the accuracy of the base Poisson model.

Modules:
    feature_builder: Convert Team objects to ML feature vectors
    xgboost_model: Primary ML model using XGBoost
    neural_net: Optional MLP model (backup)
    train_ml: Training pipeline
    evaluate_ml: Evaluation and comparison metrics
    calibration_optimizer: Data-driven parameter optimization
"""

__version__ = "1.0.0"
__all__ = [
    "feature_builder",
    "xgboost_model",
    "train_ml",
    "evaluate_ml",
]
