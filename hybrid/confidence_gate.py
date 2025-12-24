"""
Confidence Gate for XGBoost Predictions
========================================

Evaluates the quality and reliability of XGBoost expected goals predictions.
Automatically falls back to Poisson if confidence is below threshold.

Confidence Metrics:
1. Physical Bounds Check: λ values must be in [0.0, 6.0]
2. Prediction Stability: Variance estimation through ensemble predictions
3. Feature Quality: Ensures all required features are present
"""

import numpy as np
import json
import os
from typing import Dict, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> Dict:
    """Load hybrid configuration."""
    config_path = 'hybrid_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            "confidence_threshold": 50,
            "xg_bounds": {"min": 0.0, "max": 6.0, "warn_threshold": 5.0}
        }


def check_physical_bounds(xg_home: float, xg_away: float, config: Dict) -> Tuple[bool, float]:
    """
    Check if expected goals are within physically reasonable bounds.
    
    Args:
        xg_home: Home team expected goals
        xg_away: Away team expected goals
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, confidence_score)
    """
    bounds = config.get('xg_bounds', {})
    min_xg = bounds.get('min', 0.0)
    max_xg = bounds.get('max', 6.0)
    warn_threshold = bounds.get('warn_threshold', 5.0)
    
    # Hard bounds check
    if xg_home < min_xg or xg_home > max_xg:
        logger.warning(f"Home xG out of bounds: {xg_home:.2f}")
        return False, 0.0
    
    if xg_away < min_xg or xg_away > max_xg:
        logger.warning(f"Away xG out of bounds: {xg_away:.2f}")
        return False, 0.0
    
    # Calculate confidence based on how far from extreme values
    # Values near warn_threshold get lower confidence
    home_distance = min(abs(xg_home - min_xg), abs(xg_home - warn_threshold))
    away_distance = min(abs(xg_away - min_xg), abs(xg_away - warn_threshold))
    
    # Normalize to 0-100 scale
    home_conf = min(100, (home_distance / warn_threshold) * 100)
    away_conf = min(100, (away_distance / warn_threshold) * 100)
    
    avg_confidence = (home_conf + away_conf) / 2
    
    return True, avg_confidence


def check_feature_quality(features: np.ndarray) -> Tuple[bool, float]:
    """
    Check if features are complete and valid.
    
    Args:
        features: Feature vector used for prediction
        
    Returns:
        Tuple of (is_valid, confidence_score)
    """
    # Check for NaN or infinite values
    if np.isnan(features).any():
        logger.warning("Features contain NaN values")
        return False, 0.0
    
    if np.isinf(features).any():
        logger.warning("Features contain infinite values")
        return False, 0.0
    
    # Check for missing features (all zeros might indicate missing data)
    zero_ratio = np.sum(features == 0) / len(features)
    if zero_ratio > 0.5:
        logger.warning(f"High proportion of zero features: {zero_ratio:.2%}")
        confidence = max(0, 100 * (1 - zero_ratio))
        return True, confidence
    
    return True, 100.0


def estimate_prediction_variance(model, features: np.ndarray, n_samples: int = 5) -> Tuple[float, float]:
    """
    Estimate prediction variance through multiple forward passes.
    
    For models with dropout or ensemble components, this provides
    a measure of prediction uncertainty.
    
    Args:
        model: XGBoost predictor model
        features: Feature vector
        n_samples: Number of predictions to make
        
    Returns:
        Tuple of (variance_home, variance_away)
    """
    # XGBoost is deterministic, so we can't get true variance
    # Instead, we check prediction stability across slightly perturbed features
    # This is a simplified approach - in production, you might use:
    # - Bayesian XGBoost
    # - Ensemble of models
    # - Quantile regression for uncertainty
    
    predictions_home = []
    predictions_away = []
    
    for _ in range(n_samples):
        # Add small noise to features (< 1% perturbation)
        noise = np.random.normal(0, 0.01, features.shape)
        perturbed_features = features + noise
        
        # Get prediction
        try:
            pred = model.predict(perturbed_features)
            predictions_home.append(pred['xg_home'][0] if isinstance(pred['xg_home'], np.ndarray) else pred['xg_home'])
            predictions_away.append(pred['xg_away'][0] if isinstance(pred['xg_away'], np.ndarray) else pred['xg_away'])
        except Exception as e:
            logger.warning(f"Error in variance estimation: {e}")
            return 0.5, 0.5  # Default medium variance
    
    var_home = np.var(predictions_home) if len(predictions_home) > 0 else 0.5
    var_away = np.var(predictions_away) if len(predictions_away) > 0 else 0.5
    
    return var_home, var_away


def variance_to_confidence(variance: float, threshold: float = 0.2) -> float:
    """
    Convert prediction variance to confidence score.
    
    Args:
        variance: Prediction variance
        threshold: Variance threshold for full confidence
        
    Returns:
        Confidence score (0-100)
    """
    # Low variance = high confidence
    # High variance = low confidence
    if variance <= threshold:
        return 100.0
    else:
        # Exponential decay
        confidence = 100 * np.exp(-5 * (variance - threshold))
        return max(0, min(100, confidence))


def evaluate_xgb_confidence(xg_home: float, 
                           xg_away: float, 
                           features: np.ndarray = None,
                           model = None,
                           config: Dict = None) -> Dict:
    """
    Evaluate confidence in XGBoost predictions.
    
    Args:
        xg_home: Predicted home expected goals
        xg_away: Predicted away expected goals
        features: Feature vector used for prediction (optional)
        model: XGBoost model (optional, for variance estimation)
        config: Configuration dictionary (optional)
        
    Returns:
        Dictionary with confidence metrics
    """
    if config is None:
        config = load_config()
    
    confidence_scores = []
    
    # 1. Physical bounds check
    bounds_valid, bounds_conf = check_physical_bounds(xg_home, xg_away, config)
    if not bounds_valid:
        return {
            'overall_confidence': 0.0,
            'use_xgb': False,
            'reason': 'Physical bounds violated',
            'bounds_check': 0.0,
            'feature_quality': 0.0,
            'prediction_stability': 0.0
        }
    confidence_scores.append(bounds_conf)
    
    # 2. Feature quality check (if features provided)
    feature_conf = 100.0
    if features is not None:
        feature_valid, feature_conf = check_feature_quality(features)
        if not feature_valid:
            return {
                'overall_confidence': 0.0,
                'use_xgb': False,
                'reason': 'Invalid features',
                'bounds_check': bounds_conf,
                'feature_quality': 0.0,
                'prediction_stability': 0.0
            }
        confidence_scores.append(feature_conf)
    
    # 3. Prediction stability check (if model provided)
    stability_conf = 100.0
    if model is not None and features is not None:
        try:
            var_home, var_away = estimate_prediction_variance(model, features, n_samples=3)
            home_conf = variance_to_confidence(var_home)
            away_conf = variance_to_confidence(var_away)
            stability_conf = (home_conf + away_conf) / 2
            confidence_scores.append(stability_conf)
        except Exception as e:
            logger.warning(f"Could not estimate variance: {e}")
            stability_conf = 80.0  # Assume reasonable stability if we can't measure
            confidence_scores.append(stability_conf)
    
    # Calculate overall confidence (average of all checks)
    overall_confidence = np.mean(confidence_scores)
    
    # Decision: use XGBoost if confidence above threshold
    threshold = config.get('confidence_threshold', 50)
    use_xgb = overall_confidence >= threshold
    
    result = {
        'overall_confidence': round(overall_confidence, 1),
        'use_xgb': use_xgb,
        'threshold': threshold,
        'bounds_check': round(bounds_conf, 1),
        'feature_quality': round(feature_conf, 1),
        'prediction_stability': round(stability_conf, 1)
    }
    
    if not use_xgb:
        result['reason'] = f'Confidence {overall_confidence:.1f}% below threshold {threshold}%'
        logger.info(f"[FALLBACK] Using Poisson xG (XGBoost confidence: {overall_confidence:.1f}%)")
    
    return result


def should_use_xgb_prediction(xg_home: float, xg_away: float, **kwargs) -> bool:
    """
    Simple wrapper to determine if XGBoost prediction should be used.
    
    Args:
        xg_home: Home expected goals
        xg_away: Away expected goals
        **kwargs: Additional arguments for evaluate_xgb_confidence
        
    Returns:
        True if XGBoost should be used, False to fall back to Poisson
    """
    result = evaluate_xgb_confidence(xg_home, xg_away, **kwargs)
    return result['use_xgb']


# Example usage
if __name__ == '__main__':
    # Test confidence gate
    print("Testing Confidence Gate")
    print("=" * 60)
    
    # Test 1: Normal prediction
    result = evaluate_xgb_confidence(1.5, 1.2)
    print(f"\nTest 1: Normal prediction (1.5, 1.2)")
    print(f"Overall Confidence: {result['overall_confidence']:.1f}%")
    print(f"Use XGBoost: {result['use_xgb']}")
    
    # Test 2: High xG prediction
    result = evaluate_xgb_confidence(4.5, 3.8)
    print(f"\nTest 2: High xG (4.5, 3.8)")
    print(f"Overall Confidence: {result['overall_confidence']:.1f}%")
    print(f"Use XGBoost: {result['use_xgb']}")
    
    # Test 3: Out of bounds
    result = evaluate_xgb_confidence(7.0, 1.5)
    print(f"\nTest 3: Out of bounds (7.0, 1.5)")
    print(f"Overall Confidence: {result['overall_confidence']:.1f}%")
    print(f"Use XGBoost: {result['use_xgb']}")
    print(f"Reason: {result.get('reason', 'N/A')}")
    
    # Test 4: With features
    features = np.random.randn(40)
    result = evaluate_xgb_confidence(1.8, 1.3, features=features)
    print(f"\nTest 4: With valid features (1.8, 1.3)")
    print(f"Overall Confidence: {result['overall_confidence']:.1f}%")
    print(f"Feature Quality: {result['feature_quality']:.1f}%")
    print(f"Use XGBoost: {result['use_xgb']}")
    
    print("\n✅ Confidence gate tests completed")
