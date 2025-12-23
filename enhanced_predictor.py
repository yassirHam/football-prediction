"""
Enhanced Football Predictor with Advanced Features
===================================================
This module wraps the base football_predictor with enhanced features:
- Shots on target integration for better xG estimation
- Home/away form separation
- League-specific calibration support

Compatible with existing code - uses enhanced Team data when available,
falls back to base prediction otherwise.
"""

from football_predictor import (
    Team, predict_match as base_predict_match,
    calculate_offensive_strength, calculate_defensive_weakness,
    LEAGUE_AVG_GOALS, HOME_ADVANTAGE, AWAY_PENALTY, FORM_WEIGHTS,
    weighted_average
)
from typing import Dict, Tuple
import json
import os

try:
    from advanced_statistics import (
        exponential_weighted_average, 
        bayesian_adjustment,
        calculate_trend_score,
        calculate_prediction_interval,
        calculate_match_predictability,
        calculate_over_under_probabilities,
        poisson_goodness_of_fit
    )
    ADVANCED_STATS_AVAILABLE = True
except:
    ADVANCED_STATS_AVAILABLE = False


def get_league_parameters(league: str) -> Dict[str, float]:
    """
    Load league-specific parameters if available.
    
    Args:
        league: League identifier (e.g., "E0", "D1", "SP1")
    
    Returns:
        Dict with league_avg_goals, home_advantage, away_penalty
    """
    # Import the new get_league_params function from football_predictor
    from football_predictor import get_league_params
    
    # Use the new unified parameter system
    return get_league_params(league)


def enhanced_offensive_strength(team: Team, is_home: bool, league_params: Dict) -> float:
    """
    Calculate offensive strength with enhanced features.
    
    Uses shots on target and home/away form when available.
    """
    league_avg = league_params['league_avg_goals']
    
    # Use home/away specific data if available
    if is_home and team.home_goals_scored is not None and len(team.home_goals_scored) >= 3:
        goals_data = team.home_goals_scored
    elif not is_home and team.away_goals_scored is not None and len(team.away_goals_scored) >= 3:
        goals_data = team.away_goals_scored
    else:
        goals_data = team.goals_scored
    
    # Calculate average goals
    if ADVANCED_STATS_AVAILABLE and len(goals_data) > 0:
        avg_goals = exponential_weighted_average([float(g) for g in goals_data])
        avg_goals = bayesian_adjustment(avg_goals, league_avg, len(goals_data))
    else:
        avg_goals = weighted_average(goals_data, FORM_WEIGHTS[:len(goals_data)])
    
    # Apply shots on target boost if available
    if team.shots_on_target is not None and len(team.shots_on_target) > 0:
        avg_sot = sum(team.shots_on_target) / len(team.shots_on_target)
        # Quality boost: teams with 5+ SOT per game get up to 5% boost
        sot_boost = 1.0 + min(0.05, max(0, (avg_sot - 4) * 0.01))
        avg_goals *= sot_boost
    
    # Apply home advantage
    multiplier = league_params['home_advantage'] if is_home else 1.0
    return (avg_goals * multiplier) / league_avg


def enhanced_defensive_weakness(team: Team, is_home: bool, league_params: Dict) -> float:
    """
    Calculate defensive weakness with enhanced features.
    
    Uses shots on target conceded and home/away form when available.
    """
    league_avg = league_params['league_avg_goals']
    
    # Use home/away specific data if available
    if is_home and team.home_goals_conceded is not None and len(team.home_goals_conceded) >= 3:
        conceded_data = team.home_goals_conceded
    elif not is_home and team.away_goals_conceded is not None and len(team.away_goals_conceded) >= 3:
        conceded_data = team.away_goals_conceded
    else:
        conceded_data = team.goals_conceded
    
    # Calculate average conceded
    if ADVANCED_STATS_AVAILABLE and len(conceded_data) > 0:
        avg_conceded = exponential_weighted_average([float(g) for g in conceded_data])
        avg_conceded = bayesian_adjustment(avg_conceded, league_avg, len(conceded_data))
    else:
        avg_conceded = weighted_average(conceded_data, FORM_WEIGHTS[:len(conceded_data)])
    
    # Apply shots on target conceded penalty if available
    if team.shots_on_target_conceded is not None and len(team.shots_on_target_conceded) > 0:
        avg_sot_conceded = sum(team.shots_on_target_conceded) / len(team.shots_on_target_conceded)
        # Teams conceding 6+ SOT are more vulnerable (up to 5% penalty)
        sot_penalty = 1.0 + min(0.05, max(0, (avg_sot_conceded - 5) * 0.01))
        avg_conceded *= sot_penalty
    
    return avg_conceded / league_avg


def enhanced_predict_match(home_team: Team, away_team: Team, neutral_venue: bool = False, is_cup: bool = False) -> Dict:
    """
    Enhanced prediction with advanced features.
    
    Uses shots on target, home/away form, and league-specific parameters when available.
    Falls back to base prediction if enhanced data not available.
    
    Args:
        home_team: Home team with optional enhanced data
        away_team: Away team with optional enhanced data
        neutral_venue: Whether match is at neutral venue
        is_cup: Whether this is a cup/international match
    
    Returns:
        Prediction dictionary (same format as base predictor)
    """
    # Determine league (use home team's league, or DEFAULT if not set)
    league = getattr(home_team, 'league', 'DEFAULT')
    if league == 'DEFAULT' and hasattr(away_team, 'league'):
        league = away_team.league
    
    # Get league-specific parameters
    league_params = get_league_parameters(league)
    
    # Calculate expected goals with enhanced features
    home_osi = enhanced_offensive_strength(home_team, is_home=(not neutral_venue), league_params=league_params)
    away_osi = enhanced_offensive_strength(away_team, is_home=False, league_params=league_params)
    
    home_dwi = enhanced_defensive_weakness(home_team, is_home=(not neutral_venue), league_params=league_params)
    away_dwi = enhanced_defensive_weakness(away_team, is_home=False, league_params=league_params)
    
    league_avg = league_params['league_avg_goals']
    xg_home = home_osi * away_dwi * league_avg
    
    # Apply away penalty
    away_penalty = 1.0 if neutral_venue else league_params['away_penalty']
    xg_away = away_osi * home_dwi * league_avg * away_penalty
    
    # Apply ranking multiplier based on competition type and tournament stage
    from football_predictor import calculate_strength_multiplier
    home_strength, away_strength = calculate_strength_multiplier(home_team, away_team)
    xg_home *= home_strength
    xg_away *= away_strength
    
    # Use base predictor for probability calculations (reuses existing logic)
    # Temporarily modify team objects to use calculated xG
    # This is a bit hacky but avoids duplicating all the probability code
    from football_predictor import (
        predict_score_probabilities,
        get_top_n_scores,
        calculate_first_half_xg,
        calculate_match_outcome_probabilities,
        calculate_total_goals_probabilities,
        poisson_probability,
        calculate_confidence
    )
    
    full_probabilities = predict_score_probabilities(xg_home, xg_away)
    top_full_scores = get_top_n_scores(full_probabilities, n=5)
    
    # First half predictions
    xg_home_ht = calculate_first_half_xg(home_team, xg_home)
    xg_away_ht = calculate_first_half_xg(away_team, xg_away)
    ht_probabilities = predict_score_probabilities(xg_home_ht, xg_away_ht)
    top_ht_scores = get_top_n_scores(ht_probabilities, n=5)
    
    # Match outcomes
    match_outcomes = calculate_match_outcome_probabilities(full_probabilities)
    total_goals_probs = calculate_total_goals_probabilities(full_probabilities)
    
    # Both teams to score (using corrected formula)
    prob_home_scores = 1 - poisson_probability(0, xg_home)
    prob_away_scores = 1 - poisson_probability(0, xg_away)
    both_score_prob = prob_home_scores * prob_away_scores
    
    # Clean sheets
    home_clean_sheet = poisson_probability(0, xg_away)
    away_clean_sheet = poisson_probability(0, xg_home)
    
    # Confidence
    home_conf = calculate_confidence(home_team, xg_home)
    away_conf = calculate_confidence(away_team, xg_away)
    overall_confidence_score = (home_conf["overall"] + away_conf["overall"]) / 2
    
    if overall_confidence_score >= 75:
        overall_confidence = "HIGH"
    elif overall_confidence_score >= 50:
        overall_confidence = "MEDIUM"
    else:
        overall_confidence = "LOW"
    
    # Build result (same format as base predictor)
    expected_total = xg_home + xg_away
    tempo = "HIGH" if expected_total > 3.0 else "MEDIUM" if expected_total > 2.0 else "LOW"
    
    early_goal_prob = 1 - poisson_probability(0, xg_home_ht) * poisson_probability(0, xg_away_ht)
    early_goal = "HIGH" if early_goal_prob > 0.70 else "MEDIUM" if early_goal_prob > 0.50 else "LOW"
    
    # Team momentum and trends
    if ADVANCED_STATS_AVAILABLE:
        home_momentum, home_trend_stability = calculate_trend_score([float(g) for g in home_team.goals_scored])
        away_momentum, away_trend_stability = calculate_trend_score([float(g) for g in away_team.goals_scored])
        match_predictability = calculate_match_predictability(home_conf, away_conf)
        
        # Prediction intervals
        home_interval = calculate_prediction_interval(xg_home, 0.90)
        away_interval = calculate_prediction_interval(xg_away, 0.90)
        
        # Over/Under Probabilities (New Betting Feature)
        over_under_probs = calculate_over_under_probabilities(full_probabilities)
        
        # Model quality
        home_model_fit = poisson_goodness_of_fit(home_team.goals_scored, xg_home)
        away_model_fit = poisson_goodness_of_fit(away_team.goals_scored, xg_away)
    else:
        home_momentum = "STABLE"
        away_momentum = "STABLE"
        match_predictability = overall_confidence
        home_interval = (max(0, int(xg_home) - 1), int(xg_home) + 2)
        away_interval = (max(0, int(xg_away) - 1), int(xg_away) + 2)
        over_under_probs = {}
        home_model_fit = 0.5
        away_model_fit = 0.5

    return {
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
            "neutral_venue": neutral_venue,
            "league": league,
            "enhanced_features_used": {
                "shots_on_target": home_team.shots_on_target is not None or away_team.shots_on_target is not None,
                "home_away_form":home_team.home_goals_scored is not None or away_team.away_goals_scored is not None,
                "league_specific_params": league != "DEFAULT"
            }
        }
    }


# Export both versions
predict_match = enhanced_predict_match  # Use enhanced by default

__all__ = ['enhanced_predict_match', 'predict_match', 'get_league_parameters']
