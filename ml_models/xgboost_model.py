"""
XGBoost Model Implementation for Football Prediction
===================================================

Implements a multi-objective XGBoost predictor that predicts:
1. Match Outcome (Home/Draw/Away) - Classification
2. Expected Goals (Home/Away) - Regression

This model is designed to be safer and more accurate than the base Poisson model,
using the same input data but capturing non-linear relationships.
"""

import xgboost as xgb
import numpy as np
import joblib
import json
import os
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from .feature_builder import FeatureBuilder

class XGBoostPredictor:
    """
    Multi-objective predictor using XGBoost.
    Wraps three separate models:
    - Outcome Classifier (Win/Draw/Loss)
    - Home Goals Regressor
    - Away Goals Regressor
    """
    
    def __init__(self, model_dir: str = 'ml_models/model_artifacts'):
        """
        Initialize predictor.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        # outcome_model: Multi-class classification (0: Away, 1: Draw, 2: Home)
        self.outcome_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # goal_models: Regression for xG
        self.home_goals_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.away_goals_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.feature_builder = FeatureBuilder()
        self.is_trained = False
        
    def train(self, X: np.ndarray, y_outcome: np.ndarray, y_home_goals: np.ndarray, y_away_goals: np.ndarray):
        """
        Train all models.
        
        Args:
            X: Feature matrix
            y_outcome: Match outcomes (0: Away, 1: Draw, 2: Home)
            y_home_goals: Actual home goals
            y_away_goals: Actual away goals
        """
        print("Training Outcome Model...")
        self.outcome_model.fit(X, y_outcome)
        
        print("Training Home Goals Model...")
        self.home_goals_model.fit(X, y_home_goals)
        
        print("Training Away Goals Model...")
        self.away_goals_model.fit(X, y_away_goals)
        
        self.is_trained = True
        
    def predict(self, features: np.ndarray) -> Dict:
        """
        Generate predictions for a single match or batch.
        
        Args:
            features: Feature vector(s)
            
        Returns:
            Dictionary with probabilities and xG
        """
        if not self.is_trained:
            raise ValueError("Models not trained!")
            
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Get outcome probabilities
        probs = self.outcome_model.predict_proba(features)
        
        # Get expected goals
        xg_home = self.home_goals_model.predict(features)
        xg_away = self.away_goals_model.predict(features)
        
        # Clip negative xG (physically impossible)
        xg_home = np.maximum(0, xg_home)
        xg_away = np.maximum(0, xg_away)
        
        return {
            'outcome_probs': probs,  # [Away, Draw, Home]
            'xg_home': xg_home,
            'xg_away': xg_away
        }
    
    def save(self):
        """Save trained models to disk."""
        if not self.is_trained:
            print("Warning: Saving untrained models")
            
        joblib.dump(self.outcome_model, self.model_dir / 'xgb_outcome.joblib')
        joblib.dump(self.home_goals_model, self.model_dir / 'xgb_home_goals.joblib')
        joblib.dump(self.away_goals_model, self.model_dir / 'xgb_away_goals.joblib')
        print(f"Models saved to {self.model_dir}")
        
    def load(self):
        """Load trained models from disk."""
        try:
            self.outcome_model = joblib.load(self.model_dir / 'xgb_outcome.joblib')
            self.home_goals_model = joblib.load(self.model_dir / 'xgb_home_goals.joblib')
            self.away_goals_model = joblib.load(self.model_dir / 'xgb_away_goals.joblib')
            self.is_trained = True
            print(f"Models loaded from {self.model_dir}")
            return True
        except FileNotFoundError:
            print(f"No saved models found in {self.model_dir}")
            return False

    def predict_match(self, home_team, away_team, league_params=None) -> Dict:
        """
        Wrapper to predict directly from Team objects.
        matches the interface needed for the comparison script.
        """
        features = self.feature_builder.build_features_for_match(
            home_team, away_team, league_params
        )
        
        result = self.predict(features)
        
        # Format for compatibility with existing system
        probs = result['outcome_probs'][0] # [Away, Draw, Home]
        
        return {
            'match_outcome': {
                'home_win': float(probs[2]),
                'draw': float(probs[1]),
                'away_win': float(probs[0])
            },
            'expected_goals': {
                'home': float(result['xg_home'][0]),
                'away': float(result['xg_away'][0])
            }
        }
