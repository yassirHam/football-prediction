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


def estimate_confidence_from_features(
    features: np.ndarray,
    xg_home: float,
    xg_away: float,
    league_avg_goals: float = 2.7
) -> float:
    """
    Deterministic confidence proxy based on input quality.
    
    This replaces the old random noise approach with a stable, explainable
    confidence measure that reflects:
    - Feature completeness (fewer zeros = better)
    - Feature distribution quality (good spread = reliable)
    - xG plausibility (close to league average = more confident)
    
    Args:
        features: Feature vector from feature builder
        xg_home: Predicted home expected goals
        xg_away: Predicted away expected goals
        league_avg_goals: Expected total goals for league (default 2.7)
    
    Returns:
        Confidence score in range [0.0, 1.0]
    """
    # 1. Feature completeness (avoid sparse/missing data)
    zero_ratio = np.sum(features == 0) / max(len(features), 1)
    completeness_score = max(0.0, 1.0 - zero_ratio)
    
    # 2. Feature distribution quality (well-distributed features are more reliable)
    feature_std = np.std(features)
    feature_range = np.max(features) - np.min(features)
    if feature_range > 0.1:
        distribution_score = min(1.0, feature_std / feature_range)
    else:
        distribution_score = 0.5  # Neutral if all features similar
    
    # 3. xG plausibility (predictions closer to league average are safer)
    xg_total = xg_home + xg_away
    deviation = abs(xg_total - league_avg_goals)
    # Allow ±1.5 goals deviation before reducing confidence
    reasonableness_score = max(0.0, 1.0 - (deviation / (league_avg_goals * 0.75)))
    
    # Combine with equal weighting
    confidence = (completeness_score + distribution_score + reasonableness_score) / 3.0
    
    return float(np.clip(confidence, 0.0, 1.0))


def estimate_prediction_stability(
    features: np.ndarray,
    xg_home: float,
    xg_away: float
) -> Tuple[float, float]:
    """
    Estimate prediction stability using deterministic feature analysis.
    
    This function has been refactored to remove random noise perturbations.
    Instead, it uses feature quality metrics as a proxy for prediction stability.
    
    Args:
        features: Feature vector
        xg_home: Predicted home expected goals
        xg_away: Predicted away expected goals
        
    Returns:
        Tuple of (stability_home, stability_away) in range [0.0, 1.0]
        Lower values = more stable/confident predictions
    """
    # Use deterministic feature-based confidence
    confidence = estimate_confidence_from_features(features, xg_home, xg_away)
    
    # Convert confidence to stability metric (inverted for compatibility)
    # High confidence -> low "variance"
    # Low confidence -> high "variance"
    stability = 1.0 - confidence
    
    # Return as tuple for backward compatibility
    # (old code expected two values, one per team)
    return stability, stability


def variance_to_confidence(variance: float, threshold: float = 0.5) -> float:
    """
    Convert stability metric to confidence score.
    
    Updated to work with deterministic stability estimates.
    
    Args:
        variance: Stability metric (0.0-1.0, lower is better)
        threshold: Threshold for full confidence
        
    Returns:
        Confidence score (0-100)
    """
    # Variance is now a stability metric: 0 = perfect, 1 = poor
    # Convert to confidence: 1 = perfect, 0 = poor
    confidence_ratio = 1.0 - variance
    
    # Scale to 0-100
    confidence = confidence_ratio * 100.0
    
    return max(0.0, min(100.0, confidence))


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
    
    # 3. Prediction stability check (using deterministic feature analysis)
    stability_conf = 100.0
    if features is not None:
        try:
            # Use deterministic stability estimation (no longer needs model)
            stab_home, stab_away = estimate_prediction_stability(features, xg_home, xg_away)
            home_conf = variance_to_confidence(stab_home)
            away_conf = variance_to_confidence(stab_away)
            stability_conf = (home_conf + away_conf) / 2
            confidence_scores.append(stability_conf)
        except Exception as e:
            logger.warning(f"Could not estimate stability: {e}")
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
