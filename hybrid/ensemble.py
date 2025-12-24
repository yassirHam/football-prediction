"""
Ensemble Predictor for Weighted XGBoost-Poisson Blending
=========================================================

Provides weighted averaging of XGBoost and Poisson expected goals predictions.
Weights can be learned per-league or use global defaults.

This is an optional, safer mode that provides gradual integration of ML
by blending predictions rather than hard switching.
"""

import json
import os
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def load_ensemble_weights(league_code: str = None) -> Dict[str, float]:
    """
    Load ensemble weights for a specific league.
    
    Args:
        league_code: League identifier (e.g., 'E0', 'SP1'). Uses DEFAULT if None.
        
    Returns:
        Dictionary with 'xgb' and 'poisson' weights that sum to 1.0
    """
    config_path = 'hybrid_config.json'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        ensemble_weights = config.get('ensemble_weights', {})
        
        # Try to get league-specific weights
        if league_code and league_code in ensemble_weights:
            weights = ensemble_weights[league_code]
            logger.info(f"Using league-specific weights for {league_code}: {weights}")
            return weights
        
        # Fall back to default
        if 'DEFAULT' in ensemble_weights:
            weights = ensemble_weights['DEFAULT']
            logger.info(f"Using default ensemble weights: {weights}")
            return weights
    
    # Hard-coded fallback if config not found
    default_weights = {'xgb': 0.6, 'poisson': 0.4}
    logger.warning(f"No ensemble weights found, using fallback: {default_weights}")
    return default_weights


def blend_xg_predictions(xgb_xg_home: float,
                        xgb_xg_away: float,
                        poisson_xg_home: float,
                        poisson_xg_away: float,
                        league_code: str = None,
                        custom_weights: Dict[str, float] = None) -> Tuple[float, float]:
    """
    Blend XGBoost and Poisson expected goals predictions.
    
    Args:
        xgb_xg_home: XGBoost predicted home xG
        xgb_xg_away: XGBoost predicted away xG
        poisson_xg_home: Poisson calculated home xG
        poisson_xg_away: Poisson calculated away xG
        league_code: League identifier for league-specific weights
        custom_weights: Optional custom weights dict {'xgb': w1, 'poisson': w2}
        
    Returns:
        Tuple of (blended_xg_home, blended_xg_away)
    """
    # Get weights
    if custom_weights is not None:
        weights = custom_weights
    else:
        weights = load_ensemble_weights(league_code)
    
    xgb_weight = weights.get('xgb', 0.6)
    poisson_weight = weights.get('poisson', 0.4)
    
    # Normalize weights to sum to 1.0
    total_weight = xgb_weight + poisson_weight
    xgb_weight /= total_weight
    poisson_weight /= total_weight
    
    # Blend predictions
    blended_home = xgb_weight * xgb_xg_home + poisson_weight * poisson_xg_home
    blended_away = xgb_weight * xgb_xg_away + poisson_weight * poisson_xg_away
    
    logger.debug(f"Blending xG: XGB({xgb_xg_home:.2f}, {xgb_xg_away:.2f}) + "
                f"Poisson({poisson_xg_home:.2f}, {poisson_xg_away:.2f}) → "
                f"Blended({blended_home:.2f}, {blended_away:.2f}) "
                f"[weights: XGB={xgb_weight:.2f}, Poisson={poisson_weight:.2f}]")
    
    return blended_home, blended_away


def validate_ensemble_on_data(test_data: list, 
                               weight_range: np.ndarray = None) -> Dict[str, float]:
    """
    Find optimal ensemble weights by testing on validation data.
    
    Args:
        test_data: List of dictionaries with keys:
                  {'xgb_home', 'xgb_away', 'poisson_home', 'poisson_away', 
                   'actual_home', 'actual_away'}
        weight_range: Array of XGBoost weights to test (e.g., [0.5, 0.6, 0.7, 0.8])
        
    Returns:
        Dictionary with optimal weights and performance metrics
    """
    if weight_range is None:
        weight_range = np.arange(0.0, 1.1, 0.1)
    
    best_mae = float('inf')
    best_xgb_weight = 0.5
    
    results = []
    
    for xgb_w in weight_range:
        poisson_w = 1.0 - xgb_w
        
        # Calculate MAE for this weight combination
        errors = []
        for match in test_data:
            blended_home = xgb_w * match['xgb_home'] + poisson_w * match['poisson_home']
            blended_away = xgb_w * match['xgb_away'] + poisson_w * match['poisson_away']
            
            error_home = abs(blended_home - match['actual_home'])
            error_away = abs(blended_away - match['actual_away'])
            
            errors.append((error_home + error_away) / 2)
        
        mae = np.mean(errors)
        results.append({
            'xgb_weight': xgb_w,
            'poisson_weight': poisson_w,
            'mae': mae
        })
        
        if mae < best_mae:
            best_mae = mae
            best_xgb_weight = xgb_w
    
    logger.info(f"Optimal ensemble weights: XGB={best_xgb_weight:.2f}, "
                f"Poisson={1-best_xgb_weight:.2f} (MAE={best_mae:.4f})")
    
    return {
        'optimal_weights': {
            'xgb': round(best_xgb_weight, 2),
            'poisson': round(1 - best_xgb_weight, 2)
        },
        'mae': round(best_mae, 4),
        'all_results': results
    }


# Example usage
if __name__ == '__main__':
    print("Testing Ensemble Predictor")
    print("=" * 60)
    
    # Test 1: Basic blending
    print("\nTest 1: Basic blending")
    xgb_home, xgb_away = 1.8, 1.2
    poisson_home, poisson_away = 1.5, 1.4
    
    blended_home, blended_away = blend_xg_predictions(
        xgb_home, xgb_away,
        poisson_home, poisson_away
    )
    
    print(f"XGBoost: ({xgb_home}, {xgb_away})")
    print(f"Poisson: ({poisson_home}, {poisson_away})")
    print(f"Blended: ({blended_home:.2f}, {blended_away:.2f})")
    
    # Test 2: Custom weights
    print("\nTest 2: Custom weights (70% XGB, 30% Poisson)")
    blended_home, blended_away = blend_xg_predictions(
        xgb_home, xgb_away,
        poisson_home, poisson_away,
        custom_weights={'xgb': 0.7, 'poisson': 0.3}
    )
    print(f"Blended: ({blended_home:.2f}, {blended_away:.2f})")
    
    # Test 3: Weight optimization
    print("\nTest 3: Weight optimization on sample data")
    
    # Generate synthetic test data
    np.random.seed(42)
    test_data = []
    for _ in range(50):
        actual_home = np.random.poisson(1.5)
        actual_away = np.random.poisson(1.3)
        
        # XGBoost predictions (slightly better)
        xgb_h = actual_home + np.random.normal(0, 0.8)
        xgb_a = actual_away + np.random.normal(0, 0.8)
        
        # Poisson predictions (slightly worse)
        pois_h = actual_home + np.random.normal(0, 1.0)
        pois_a = actual_away + np.random.normal(0, 1.0)
        
        test_data.append({
            'xgb_home': max(0, xgb_h),
            'xgb_away': max(0, xgb_a),
            'poisson_home': max(0, pois_h),
            'poisson_away': max(0, pois_a),
            'actual_home': actual_home,
            'actual_away': actual_away
        })
    
    result = validate_ensemble_on_data(test_data)
    print(f"Optimal weights: {result['optimal_weights']}")
    print(f"Best MAE: {result['mae']}")
    
    print("\n✅ Ensemble predictor tests completed")
