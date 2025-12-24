"""
Hybrid Prediction System - Main Orchestrator
=============================================

Combines XGBoost expected goals generation with Poisson probability calculations.

Architecture:
1. XGBoost predicts λ_home and λ_away (expected goals)
2. Confidence gate validates XGBoost predictions
3. If confidence is high, use XGBoost xG; else fall back to Poisson
4. Poisson engine calculates ALL probabilities from the chosen xG values

This ensures we get improved goal predictions while maintaining
probability quality and mathematical interpretability.
"""

import os
import sys
import json
import logging
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from football_predictor import (
    Team,
    calculate_expected_goals,
    predict_score_probabilities,
    calculate_match_outcome_probabilities,
    calculate_total_goals_probabilities,
    poisson_probability,
    get_top_n_scores,
    calculate_first_half_xg,
    calculate_confidence,
    calculate_strength_multiplier
)

# Import hybrid components - handle both relative and absolute imports
try:
    from .confidence_gate import evaluate_xgb_confidence, load_config
    from .ensemble import blend_xg_predictions
except ImportError:
    # Running as standalone script
    from confidence_gate import evaluate_xgb_confidence, load_config
    from ensemble import blend_xg_predictions

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_xgboost_model():
    """
    Load XGBoost model for xG prediction.
    
    Returns:
        XGBoostPredictor instance or None if model not available
    """
    try:
        from ml_models.xgboost_model import XGBoostPredictor
        
        model = XGBoostPredictor()
        if model.load():
            logger.info("✅ XGBoost model loaded successfully")
            return model
        else:
            logger.warning("⚠️  XGBoost model not found, will use Poisson only")
            return None
    except Exception as e:
        logger.warning(f"⚠️  Could not load XGBoost model: {e}")
        return None


# Global model instance (loaded once)
_XGBOOST_MODEL = None
_CONFIG = None


def get_xgboost_model():
    """Get or load XGBoost model (singleton pattern)."""
    global _XGBOOST_MODEL
    if _XGBOOST_MODEL is None:
        _XGBOOST_MODEL = load_xgboost_model()
    return _XGBOOST_MODEL


def get_config():
    """Get or load configuration (singleton pattern)."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def get_expected_goals_hybrid(home_team: Team, 
                               away_team: Team, 
                               neutral_venue: bool = False) -> Tuple[float, float, Dict]:
    """
    Get expected goals using hybrid XGBoost-Poisson approach with confidence gating.
    
    Args:
        home_team: Home team object
        away_team: Away team object
        neutral_venue: Whether match is at neutral venue
        
    Returns:
        Tuple of (xg_home, xg_away, metadata_dict)
        metadata_dict contains: {
            'source': 'xgboost' | 'poisson' | 'ensemble',
            'confidence': float,
            'confidence_details': dict
        }
    """
    config = get_config()
    model = get_xgboost_model()
    
    # Always calculate Poisson xG as fallback
    poisson_xg_home, poisson_xg_away = calculate_expected_goals(home_team, away_team, neutral_venue)
    
    # If ML is disabled or model not available, use Poisson
    if not config.get('enable_ml', True) or model is None:
        logger.info("Using Poisson xG (ML disabled or model unavailable)")
        return poisson_xg_home, poisson_xg_away, {
            'source': 'poisson',
            'confidence': 100.0,
            'reason': 'ML disabled or model not available'
        }
    
    # Try to get XGBoost prediction
    try:
        # Build features and get XGBoost prediction
        xgb_prediction = model.predict_match(home_team, away_team)
        xgb_xg_home = xgb_prediction['expected_goals']['home']
        xgb_xg_away = xgb_prediction['expected_goals']['away']
        
        # Evaluate confidence
        confidence_result = evaluate_xgb_confidence(
            xgb_xg_home, 
            xgb_xg_away,
            features=None,  # Could pass features here for more thorough checks
            model=model,
            config=config
        )
        
        # Decision based on confidence gate
        if confidence_result['use_xgb']:
            # High confidence: use XGBoost
            if config.get('use_ensemble', False):
                # Ensemble mode: blend predictions
                final_xg_home, final_xg_away = blend_xg_predictions(
                    xgb_xg_home, xgb_xg_away,
                    poisson_xg_home, poisson_xg_away,
                    league_code=home_team.league
                )
                logger.info(f"Using ensemble xG: ({final_xg_home:.2f}, {final_xg_away:.2f})")
                return final_xg_home, final_xg_away, {
                    'source': 'ensemble',
                    'confidence': confidence_result['overall_confidence'],
                    'confidence_details': confidence_result,
                    'xgb_xg': (xgb_xg_home, xgb_xg_away),
                    'poisson_xg': (poisson_xg_home, poisson_xg_away)
                }
            else:
                # Pure XGBoost mode
                logger.info(f"Using XGBoost xG: ({xgb_xg_home:.2f}, {xgb_xg_away:.2f}) "
                          f"[confidence: {confidence_result['overall_confidence']:.1f}%]")
                return xgb_xg_home, xgb_xg_away, {
                    'source': 'xgboost',
                    'confidence': confidence_result['overall_confidence'],
                    'confidence_details': confidence_result
                }
        else:
            # Low confidence: fall back to Poisson
            reason = confidence_result.get('reason', 'Low confidence')
            logger.warning(f"[FALLBACK] Using Poisson xG. Reason: {reason}")
            return poisson_xg_home, poisson_xg_away, {
                'source': 'poisson',
                'confidence': confidence_result['overall_confidence'],
                'confidence_details': confidence_result,
                'reason': reason,
                'attempted_xgb_xg': (xgb_xg_home, xgb_xg_away)
            }
            
    except Exception as e:
        # Error in XGBoost prediction: fall back to Poisson
        logger.error(f"Error in XGBoost prediction: {e}. Falling back to Poisson.")
        return poisson_xg_home, poisson_xg_away, {
            'source': 'poisson',
            'confidence': 100.0,
            'reason': f'XGBoost error: {str(e)}'
        }


def predict_match_hybrid(home_team: Team, 
                         away_team: Team, 
                         neutral_venue: bool = False, 
                         is_cup: bool = False) -> Dict:
    """
    Main hybrid prediction function combining XGBoost xG with Poisson probabilities.
    
    This function maintains the same interface as the original predict_match()
    from football_predictor.py but uses hybrid xG generation.
    
    Args:
        home_team: Home team object (or first team if neutral venue)
        away_team: Away team object (or second team if neutral venue)
        neutral_venue: If True, match is at neutral venue (no home advantage)
        is_cup: If True, uses International/Cup ranking logic
        
    Returns:
        Dictionary containing all predictions and insights, identical format to
        the original predict_match() function
    """
    # 1. Get expected goals using hybrid approach
    xg_home, xg_away, xg_metadata = get_expected_goals_hybrid(home_team, away_team, neutral_venue)
    
    # 2. Apply ranking multiplier (same as original)
    home_strength, away_strength = calculate_strength_multiplier(home_team, away_team)
    xg_home *= home_strength
    xg_away *= away_strength
    
    # 3. Use Poisson for ALL probability calculations
    full_probabilities = predict_score_probabilities(xg_home, xg_away)
    top_full_scores = get_top_n_scores(full_probabilities, n=5)
    
    # First half predictions
    xg_home_ht = calculate_first_half_xg(home_team, xg_home)
    xg_away_ht = calculate_first_half_xg(away_team, xg_away)
    ht_probabilities = predict_score_probabilities(xg_home_ht, xg_away_ht)
    top_ht_scores = get_top_n_scores(ht_probabilities, n=5)
    
    # Match outcome probabilities
    match_outcomes = calculate_match_outcome_probabilities(full_probabilities)
    
    # Total goals
    total_goals_probs = calculate_total_goals_probabilities(full_probabilities)
    
    # Both teams to score
    prob_home_scores = 1 - poisson_probability(0, xg_home)
    prob_away_scores = 1 - poisson_probability(0, xg_away)
    both_score_prob = prob_home_scores * prob_away_scores
    
    # Clean sheet probabilities
    home_clean_sheet = poisson_probability(0, xg_away)
    away_clean_sheet = poisson_probability(0, xg_home)
    
    # Summary insights
    expected_total = xg_home + xg_away
    tempo = "HIGH" if expected_total > 3.0 else "MEDIUM" if expected_total > 2.0 else "LOW"
    
    # Early goal likelihood
    early_goal_prob = 1 - poisson_probability(0, xg_home_ht) * poisson_probability(0, xg_away_ht)
    early_goal = "HIGH" if early_goal_prob > 0.70 else "MEDIUM" if early_goal_prob > 0.50 else "LOW"
    
    # Enhanced confidence calculation
    home_conf = calculate_confidence(home_team, xg_home)
    away_conf = calculate_confidence(away_team, xg_away)
    overall_confidence_score = (home_conf["overall"] + away_conf["overall"]) / 2
    
    if overall_confidence_score >= 75:
        overall_confidence = "HIGH"
    elif overall_confidence_score >= 50:
        overall_confidence = "MEDIUM"
    else:
        overall_confidence = "LOW"
    
    # Import advanced stats if available
    try:
        import advanced_statistics
        home_momentum, home_trend_stability = advanced_statistics.calculate_trend_score(
            [float(g) for g in home_team.goals_scored]
        )
        away_momentum, away_trend_stability = advanced_statistics.calculate_trend_score(
            [float(g) for g in away_team.goals_scored]
        )
        match_predictability = advanced_statistics.calculate_match_predictability(home_conf, away_conf)
        home_interval = advanced_statistics.calculate_prediction_interval(xg_home, 0.90)
        away_interval = advanced_statistics.calculate_prediction_interval(xg_away, 0.90)
        over_under_probs = advanced_statistics.calculate_over_under_probabilities(full_probabilities)
        home_model_fit = advanced_statistics.poisson_goodness_of_fit(home_team.goals_scored, xg_home)
        away_model_fit = advanced_statistics.poisson_goodness_of_fit(away_team.goals_scored, xg_away)
    except:
        home_momentum = "STABLE"
        away_momentum = "STABLE"
        match_predictability = overall_confidence
        home_interval = (max(0, int(xg_home) - 1), int(xg_home) + 2)
        away_interval = (max(0, int(xg_away) - 1), int(xg_away) + 2)
        over_under_probs = {}
        home_model_fit = 0.5
        away_model_fit = 0.5
    
    # Build result dictionary (same format as original predict_match)
    result = {
        "first_half_predictions": top_ht_scores,
        "full_match_predictions": top_full_scores,
        "match_outcome": match_outcomes,
        "total_goals": total_goals_probs,
        "betting_insights": over_under_probs,
        "expected_goals": {"home": xg_home, "away": xg_away},
        "both_teams_score": both_score_prob,
        "clean_sheet": {
            "home": home_clean_sheet,
            "away": away_clean_sheet
        },
        "confidence_score": round(overall_confidence_score, 1),
        "confidence_breakdown": {
            "home": home_conf,
            "away": away_conf,
            "overall": round(overall_confidence_score, 1)
        },
        "prediction_intervals": {
            "home_goals_90": home_interval,
            "away_goals_90": away_interval
        },
        "team_insights": {
            "home_momentum": home_momentum,
            "away_momentum": away_momentum,
            "match_predictability": match_predictability
        },
        "model_quality": {
            "poisson_fit_home": round(home_model_fit, 3),
            "poisson_fit_away": round(away_model_fit, 3),
            "overall_reliability": "HIGH" if min(home_model_fit, away_model_fit) > 0.7 else "MEDIUM" if min(home_model_fit, away_model_fit) > 0.35 else "LOW"
        },
        "insights": {
            "tempo": tempo,
            "early_goal_likelihood": early_goal,
            "confidence": overall_confidence,
            "neutral_venue": neutral_venue
        },
        # Additional hybrid-specific metadata
        "hybrid_metadata": xg_metadata
    }
    
    return result


# Re-export Team for convenience
__all__ = ['predict_match_hybrid', 'Team', 'get_expected_goals_hybrid']


# Example usage and testing
if __name__ == '__main__':
    print("Testing Hybrid Predictor")
    print("=" * 60)
    
    # Create test teams
    man_city = Team(
        name="Manchester City",
        goals_scored=[3, 2, 4, 1, 2],
        goals_conceded=[1, 0, 1, 1, 2],
        first_half_goals=[2, 1, 2, 0, 1],
        league="E0"
    )
    
    liverpool = Team(
        name="Liverpool",
        goals_scored=[2, 3, 1, 2, 3],
        goals_conceded=[1, 2, 0, 1, 1],
        first_half_goals=[1, 2, 0, 1, 2],
        league="E0"
    )
    
    # Run hybrid prediction
    print("\n🔮 Predicting: Manchester City vs Liverpool")
    print("-" * 60)
    
    result = predict_match_hybrid(man_city, liverpool)
    
    # Display results
    print(f"\n{'EXPECTED GOALS (xG):'}")
    print(f"  Manchester City: {result['expected_goals']['home']:.2f}")
    print(f"  Liverpool: {result['expected_goals']['away']:.2f}")
    print(f"  Source: {result['hybrid_metadata']['source'].upper()}")
    print(f"  Confidence: {result['hybrid_metadata']['confidence']:.1f}%")
    
    print(f"\n{'MATCH OUTCOME:'}")
    print(f"  Home Win: {result['match_outcome']['home_win']*100:.1f}%")
    print(f"  Draw: {result['match_outcome']['draw']*100:.1f}%")
    print(f"  Away Win: {result['match_outcome']['away_win']*100:.1f}%")
    
    print(f"\n{'TOP 3 SCORELINES:'}")
    for i, ((h, a), prob) in enumerate(result['full_match_predictions'][:3], 1):
        print(f"  {i}. {h}-{a}: {prob*100:.1f}%")
    
    print(f"\n{'BETTING INSIGHTS:'}")
    print(f"  Over 2.5 goals: {result['total_goals']['over_2.5']*100:.1f}%")
    print(f"  Both teams to score: {result['both_teams_score']*100:.1f}%")
    
    print("\n✅ Hybrid predictor test completed")
