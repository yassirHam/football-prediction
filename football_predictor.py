"""
Football Match Prediction System - Enhanced Version
====================================================
An advanced statistical prediction engine based on recent team performance.

This implementation uses sophisticated statistical methods:
- Exponential Weighted Moving Average (EWMA) for trend detection
- Bayesian adjustment with league priors
- Multi-factor confidence scoring
- Dixon-Coles correlation for low-scoring games
- Prediction intervals and goodness-of-fit testing

Author: Football Analytics System
Version: 2.0 - Enhanced
"""

import math
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Import advanced statistics module
try:
    import advanced_statistics
    from advanced_statistics import (
        exponential_weighted_average,
        calculate_coefficient_of_variation,
        bayesian_adjustment,
        calculate_trend_score,
        poisson_goodness_of_fit,
        calculate_multi_factor_confidence,
        dixon_coles_tau,
        calculate_prediction_interval,
        calculate_match_predictability,
        calculate_over_under_probabilities
    )
    ADVANCED_STATS_AVAILABLE = True
except ImportError:
    ADVANCED_STATS_AVAILABLE = False
    print("Warning: Advanced statistics module not available. Using basic methods.")

# Import feature engineering module
try:
    from feature_engineering import (
        create_team_form_profile,
        get_best_form_estimate,
        apply_streak_boost,
        apply_trend_adjustment
    )
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    print("Note: Feature engineering module not available. Using standard form calculation.")

# Constants - Load calibrated parameters if available, otherwise use defaults
LEAGUE_PARAMS = {}  # Dictionary to store league-specific parameters
try:
    import json
    # Load league-specific parameters
    with open('calibrated_params_by_league.json', 'r') as f:
        league_data = json.load(f)
        LEAGUE_PARAMS = league_data.get('leagues', {})
        global_params = league_data.get('global', {})
        
        LEAGUE_AVG_GOALS = global_params.get('league_avg_goals', 1.40)
        HOME_ADVANTAGE = global_params.get('home_advantage', 1.125)
        AWAY_PENALTY = global_params.get('away_penalty', 0.875)
        
        print(f"✅ Loaded league-specific parameters for {len(LEAGUE_PARAMS)} leagues")
        print(f"   Global fallback: AVG={LEAGUE_AVG_GOALS:.3f}, HOME={HOME_ADVANTAGE:.3f}, AWAY={AWAY_PENALTY:.3f}")
        
except (FileNotFoundError, json.JSONDecodeError, KeyError):
    # Fall back to global calibrated parameters
    try:
        with open('calibrated_params.json', 'r') as f:
            CALIBRATED_PARAMS = json.load(f)
            LEAGUE_AVG_GOALS = CALIBRATED_PARAMS.get('league_avg_goals', 1.40)
            HOME_ADVANTAGE = CALIBRATED_PARAMS.get('home_advantage', 1.125)
            AWAY_PENALTY = CALIBRATED_PARAMS.get('away_penalty', 0.875)
            print(f"✅ Loaded global calibrated parameters: AVG={LEAGUE_AVG_GOALS:.3f}, HOME={HOME_ADVANTAGE:.3f}, AWAY={AWAY_PENALTY:.3f}")
    except (FileNotFoundError, json.JSONDecodeError):
        # Use default values if calibration not available
        LEAGUE_AVG_GOALS = 1.40  # Updated from 22k match analysis
        HOME_ADVANTAGE = 1.125   # Calculated from 22k matches
        AWAY_PENALTY = 0.875     # Calculated from 22k matches
    
FORM_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]  # Most recent to oldest (fallback)


def get_league_params(league_code: str) -> Dict:
    """
    Get league-specific parameters if available, otherwise use global defaults.
    
    Args:
        league_code: League identifier (e.g., 'E0', 'SP1', 'SC0')
        
    Returns:
        Dictionary with league_avg_goals, home_advantage, away_penalty
    """
    if league_code in LEAGUE_PARAMS:
        return LEAGUE_PARAMS[league_code]
    else:
        # Return global fallback
        return {
            "league_avg_goals": LEAGUE_AVG_GOALS,
            "home_advantage": HOME_ADVANTAGE,
            "away_penalty": AWAY_PENALTY,
            "dixon_coles_rho": -0.08
        }



@dataclass
class Team:
    """Represents a football team with recent performance data."""
    name: str
    goals_scored: List[int]  # Last 5 matches
    goals_conceded: List[int]  # Last 5 matches
    first_half_goals: List[int]  # First half goals in last 5 matches
    league_position: int = 10  # Default to mid-table if not provided
    
    # Advanced features for improved accuracy
    shots_on_target: List[int] = None  # Shots on target in last 5 matches (optional)
    shots_on_target_conceded: List[int] = None  # SOT conceded (optional)
    home_goals_scored: List[int] = None  # Goals scored at home (optional)
    away_goals_scored: List[int] = None  # Goals scored away (optional) 
    home_goals_conceded: List[int] = None  # Goals conceded at home (optional)
    away_goals_conceded: List[int] = None  # Goals conceded away (optional)
    league: str = "DEFAULT"  # League identifier for league-specific calibration
    
    # Tournament context (for cup/international matches)
    competition_type: str = "LEAGUE"  # "LEAGUE", "CLUB_CUP", "INTERNATIONAL"
    tournament_stage: str = "GROUP"  # "GROUP", "R16", "QUARTER", "SEMI", "FINAL"
    fifa_ranking: int = None  # For international teams (1-210)

def calculate_strength_multiplier(home_team: Team, away_team: Team) -> Tuple[float, float]:
    """
    Calculate xG multipliers based on ranking difference and tournament context.
    
    Handles:
    - League matches: Standard league position logic (1-20)
    - International cups: FIFA rankings (1-210) with tournament stage pressure
    - Club cups: League positions with tournament stage pressure
    
    Args:
        home_team: Home team (or team 1 if neutral venue)
        away_team: Away team (or team 2 if neutral venue)
        
    Returns:
        Tuple of (home_boost, away_boost)
    """
    competition_type = home_team.competition_type
    tournament_stage = home_team.tournament_stage
    
    # Get rankings based on competition type
    if competition_type == "INTERNATIONAL":
        # Use FIFA rankings for international matches
        home_rank = home_team.fifa_ranking if home_team.fifa_ranking else home_team.league_position
        away_rank = away_team.fifa_ranking if away_team.fifa_ranking else away_team.league_position
        ranking_range = 210  # FIFA rankings are 1-210
        factor = 0.0025  # 0.25% per rank (smaller effect due to large range)
        effective_cap = 50  # Cap ranking difference effect
    elif competition_type == "CLUB_CUP":
        # Use league positions for club cups
        home_rank = home_team.league_position
        away_rank = away_team.league_position
        ranking_range = 20  # Assume standard league size
        factor = 0.015  # 1.5% per rank (stronger effect for smaller range)
        effective_cap = 15  # Cap at 15 positions
    else:  # LEAGUE
        # Standard league match
        home_rank = home_team.league_position
        away_rank = away_team.league_position
        ranking_range = 20
        factor = 0.015  # 1.5% per rank
        effective_cap = None  # No cap for league matches
    
    # Calculate ranking difference (positive if home is better)
    diff = away_rank - home_rank
    
    # Apply cap if in cup competition
    if effective_cap:
        diff = max(-effective_cap, min(effective_cap, diff))
    
    # Apply tournament stage pressure modifier
    # Later stages = higher pressure = smaller upsets, form matters more
    stage_modifier = get_tournament_stage_modifier(tournament_stage, competition_type)
    
    # Calculate base multipliers
    home_boost = 1 + (diff * factor * stage_modifier)
    away_boost = 1 - (diff * factor * stage_modifier)
    
    # Clamp values to reasonable range
    return max(0.6, min(1.5, home_boost)), max(0.6, min(1.5, away_boost))


def get_tournament_stage_modifier(stage: str, comp_type: str) -> float:
    """
    Get tournament stage modifier for ranking-based adjustments.
    
    - Early stages (Group, R16): Higher volatility, upsets more likely → Lower modifier
    - Later stages (Semi, Final): Pressure increases, form matters more → Higher modifier
    
    Args:
        stage: Tournament stage (GROUP, R16, QUARTER, SEMI, FINAL)
        comp_type: Competition type (INTERNATIONAL, CLUB_CUP, LEAGUE)
        
    Returns:
        Modifier multiplier (0.7 to 1.3)
    """
    if comp_type == "LEAGUE":
        return 1.0  # No tournament pressure in league
    
    # Tournament stage progression modifiers
    stage_modifiers = {
        "GROUP": 0.8,      # Group stage: Lower pressure, more upsets
        "R16": 0.9,        # Round of 16: Starting knockout
        "QUARTER": 1.0,    # Quarter-finals: Standard
        "SEMI": 1.1,       # Semi-finals: High pressure, form crucial
        "FINAL": 1.2       # Final: Maximum pressure, best teams
    }
    
    return stage_modifiers.get(stage, 1.0)



def weighted_average(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted average of values.
    
    Args:
        values: List of numerical values
        weights: List of weights (same length as values)
    
    Returns:
        Weighted average
    """
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def calculate_offensive_strength(team: Team, is_home: bool, league_params: Dict = None) -> float:
    """
    Calculate offensive strength index with advanced statistics and feature engineering.
    
    Uses EWMA, Bayesian adjustment, and adaptive multi-window averaging.
    
    Args:
        team: Team object with performance data
        is_home: Whether team is playing at home
        league_params: Optional league-specific parameters
    
    Returns:
        Offensive strength index (OSI)
    """
    # Get league-specific parameters if not provided
    if league_params is None:
        league_params = get_league_params(team.league)
    
    league_avg = league_params['league_avg_goals']
    home_mult = league_params['home_advantage']
    
    # Use feature engineering if available
    if FEATURE_ENGINEERING_AVAILABLE and len(team.goals_scored) >= 5:
        # Create form profile for adaptive averaging
        form_profile = create_team_form_profile(team.goals_scored, team.goals_conceded)
        avg_goals = get_best_form_estimate(form_profile)
        
        # Apply Bayesian adjustment
        if ADVANCED_STATS_AVAILABLE:
            avg_goals = bayesian_adjustment(avg_goals, league_avg, len(team.goals_scored))
    elif ADVANCED_STATS_AVAILABLE:
        # Use EWMA for more responsive trend detection
        avg_goals = exponential_weighted_average([float(g) for g in team.goals_scored])
        # Apply Bayesian adjustment
        avg_goals = bayesian_adjustment(avg_goals, league_avg, len(team.goals_scored))
    else:
        # Fallback to simple weighted average
        avg_goals = weighted_average(team.goals_scored, FORM_WEIGHTS)
    
    multiplier = home_mult if is_home else 1.0
    return (avg_goals * multiplier) / league_avg


def calculate_defensive_weakness(team: Team, league_params: Dict = None) -> float:
    """
    Calculate defensive weakness index with advanced statistics.
    
    Uses EWMA and Bayesian adjustment for more accurate assessment.
    
    Args:
        team: Team object with performance data
        league_params: Optional league-specific parameters
    
    Returns:
        Defensive weakness index (DWI)
    """
    # Get league-specific parameters if not provided
    if league_params is None:
        league_params = get_league_params(team.league)
    
    league_avg = league_params['league_avg_goals']
    
    if ADVANCED_STATS_AVAILABLE:
        # Use EWMA for better trend detection
        avg_conceded = exponential_weighted_average([float(g) for g in team.goals_conceded])
        # Apply Bayesian adjustment
        avg_conceded = bayesian_adjustment(avg_conceded, league_avg, len(team.goals_conceded))
    else:
        # Fallback to simple weighted average
        avg_conceded = weighted_average(team.goals_conceded, FORM_WEIGHTS)
    
    return avg_conceded / league_avg


def calculate_expected_goals(home_team: Team, away_team: Team, neutral_venue: bool = False) -> Tuple[float, float]:
    """
    Calculate expected goals for both teams using league-specific parameters and feature engineering.
    
    Args:
        home_team: Home team object (or first team if neutral venue)
        away_team: Away team object (or second team if neutral venue)
        neutral_venue: If True, no home advantage applied
    
    Returns:
        Tuple of (xG_home, xG_away)
    """
    # Get league-specific parameters (both teams should be from same league)
    league_params = get_league_params(home_team.league)
    league_avg = league_params['league_avg_goals']
    away_penalty = 1.0 if neutral_venue else league_params['away_penalty']
    
    # Apply home advantage only if not neutral venue
    home_osi = calculate_offensive_strength(home_team, is_home=(not neutral_venue), league_params=league_params)
    away_osi = calculate_offensive_strength(away_team, is_home=False, league_params=league_params)
    
    home_dwi = calculate_defensive_weakness(home_team, league_params=league_params)
    away_dwi = calculate_defensive_weakness(away_team, league_params=league_params)
    
    xg_home = home_osi * away_dwi * league_avg
    xg_away = away_osi * home_dwi * league_avg * away_penalty
    
    # Apply feature engineering enhancements if available
    if FEATURE_ENGINEERING_AVAILABLE and len(home_team.goals_scored) >= 5:
        # Create form profiles
        home_profile = create_team_form_profile(home_team.goals_scored, home_team.goals_conceded)
        away_profile = create_team_form_profile(away_team.goals_scored, away_team.goals_conceded)
        
        # Apply streak boosts
        xg_home = apply_streak_boost(xg_home, home_profile)
        xg_away = apply_streak_boost(xg_away, away_profile)
        
        # Apply trend adjustments
        xg_home = apply_trend_adjustment(xg_home, home_profile)
        xg_away = apply_trend_adjustment(xg_away, away_profile)
    
    return xg_home, xg_away


def poisson_probability(k: int, lambda_: float) -> float:
    """
    Calculate Poisson probability P(X = k).
    
    Args:
        k: Number of goals
        lambda_: Expected value (xG)
    
    Returns:
        Probability of exactly k goals
    """
    return (lambda_ ** k * math.exp(-lambda_)) / math.factorial(k)


def predict_score_probabilities(xg_home: float, xg_away: float, 
                                max_goals: int = 6) -> Dict[Tuple[int, int], float]:
    """
    Generate probabilities for all score combinations with Dixon-Coles adjustment.
    
    Applies correlation adjustment for low-scoring games to improve accuracy.
    
    Args:
        xg_home: Expected goals for home team
        xg_away: Expected goals for away team
        max_goals: Maximum goals to consider
    
    Returns:
        Dictionary mapping (home_goals, away_goals) to probability
    """
    probabilities = {}
    
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob_home = poisson_probability(home_goals, xg_home)
            prob_away = poisson_probability(away_goals, xg_away)
            base_prob = prob_home * prob_away
            
            # Apply Dixon-Coles correlation adjustment for low-scoring games
            if ADVANCED_STATS_AVAILABLE:
                tau = dixon_coles_tau(home_goals, away_goals, xg_home, xg_away)
                probabilities[(home_goals, away_goals)] = base_prob * tau
            else:
                probabilities[(home_goals, away_goals)] = base_prob
    
    # Normalize probabilities to sum to 1.0 (Critical fix for Dixon-Coles)
    total_prob = sum(probabilities.values())
    if total_prob > 0:
        for key in probabilities:
            probabilities[key] /= total_prob
            
    return probabilities


def get_top_n_scores(probabilities: Dict[Tuple[int, int], float], 
                     n: int = 3) -> List[Tuple[Tuple[int, int], float]]:
    """
    Get top N most likely scores with normalized probabilities.
    
    Args:
        probabilities: Score probabilities dictionary
        n: Number of top scores to return
    
    Returns:
        List of ((home_goals, away_goals), probability) tuples
    """
    sorted_scores = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_n = sorted_scores[:n]
    
    # Normalize
    total_prob = sum(prob for _, prob in top_n)
    normalized = [((score, prob / total_prob)) for score, prob in top_n]
    
    return normalized


def calculate_first_half_xg(team: Team, full_xg: float) -> float:
    """
    Adjust expected goals for first half.
    
    Args:
        team: Team object with first half data
        full_xg: Full match expected goals
    
    Returns:
        First half expected goals
    """
    avg_total_goals = weighted_average(team.goals_scored, FORM_WEIGHTS)
    avg_first_half = weighted_average(team.first_half_goals, FORM_WEIGHTS)
    
    if avg_total_goals > 0:
        first_half_ratio = avg_first_half / avg_total_goals
    else:
        first_half_ratio = 0.45  # Default 45% of goals in first half
    
    return full_xg * first_half_ratio


def calculate_match_outcome_probabilities(probabilities: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    """
    Calculate win/draw/loss probabilities.
    
    Args:
        probabilities: Score probabilities dictionary
    
    Returns:
        Dictionary with home_win, draw, away_win probabilities
    """
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    
    for (home_goals, away_goals), prob in probabilities.items():
        if home_goals > away_goals:
            home_win += prob
        elif home_goals == away_goals:
            draw += prob
        else:
            away_win += prob
    
    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win
    }


def calculate_total_goals_probabilities(probabilities: Dict[Tuple[int, int], float]) -> Dict:
    """
    Calculate over/under probabilities.
    
    Args:
        probabilities: Score probabilities dictionary
    
    Returns:
        Dictionary with over/under probabilities, most likely total, and full distribution
    """
    total_probs = {}
    
    # Calculate P(total = n) for each possible total
    totals = {}
    for (home, away), prob in probabilities.items():
        total = home + away
        totals[total] = totals.get(total, 0) + prob
    
    # Calculate cumulative probabilities
    under_1_5 = sum(totals.get(i, 0) for i in range(0, 2))  # 0, 1 goals
    under_2_5 = sum(totals.get(i, 0) for i in range(0, 3))  # 0, 1, 2 goals
    over_2_5 = 1 - under_2_5
    over_3_5 = 1 - sum(totals.get(i, 0) for i in range(0, 4))
    
    # Find most likely total
    most_likely_total = max(totals.items(), key=lambda x: x[1])[0] if totals else 2
    
    return {
        "under_1.5": under_1_5,
        "under_2.5": under_2_5,
        "over_2.5": over_2_5,
        "over_3.5": over_3_5,
        "most_likely_total": most_likely_total,
        "total_distribution": totals
    }


def calculate_confidence(team: Team, xg_value: float = None) -> Dict:
    """
    Calculate advanced multi-factor confidence score.
    
    Considers consistency, sample size, trend stability, and model fit.
    
    Args:
        team: Team object
        xg_value: Expected goals for validation (optional)
    
    Returns:
        Dictionary with numerical score (0-100) and breakdown
    """
    if ADVANCED_STATS_AVAILABLE and xg_value is not None:
        # Use advanced multi-factor confidence calculation
        confidence = calculate_multi_factor_confidence(
            team.goals_scored,
            team.goals_conceded,
            xg_value,
            LEAGUE_AVG_GOALS
        )
        return confidence
    else:
        # Fallback to simple variance-based calculation
        mean = weighted_average(team.goals_scored, FORM_WEIGHTS)
        variance = weighted_average(
            [(g - mean)**2 for g in team.goals_scored], 
            FORM_WEIGHTS
        )
        
        if variance < 0.5:
            score = 85.0
        elif variance < 1.5:
            score = 65.0
        else:
            score = 40.0
        
        return {
            "overall": score,
            "consistency": score,
            "sample_size": len(team.goals_scored) * 20,
            "trend_stability": 50.0,
            "model_fit": 50.0
        }


def predict_match(home_team: Team, away_team: Team, neutral_venue: bool = False, is_cup: bool = False) -> Dict:
    """
    Main prediction function with enhanced statistical analysis.
    
    Args:
        home_team: Home team object (or first team if neutral venue)
        away_team: Away team object (or second team if neutral venue)
        neutral_venue: If True, match is played at neutral venue (no home advantage)
        is_cup: If True, uses International/Cup ranking logic
    
    Returns:
        Dictionary containing all predictions and insights with enhanced confidence scoring
    """
    # Full match predictions
    xg_home, xg_away = calculate_expected_goals(home_team, away_team, neutral_venue)
    
    # Apply ranking multiplier based on competition type and tournament stage
    home_strength, away_strength = calculate_strength_multiplier(home_team, away_team)
    xg_home *= home_strength
    xg_away *= away_strength
    # --------------------------------------------
    
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
    # P(both score) = P(home scores ≥1) × P(away scores ≥1)
    prob_home_scores = 1 - poisson_probability(0, xg_home)
    prob_away_scores = 1 - poisson_probability(0, xg_away)
    both_score_prob = prob_home_scores * prob_away_scores
    
    # Clean sheet probabilities
    home_clean_sheet = poisson_probability(0, xg_away)
    away_clean_sheet = poisson_probability(0, xg_home)
    
    # Summary insights
    expected_total = xg_home + xg_away
    tempo = "HIGH" if expected_total > 3.0 else "MEDIUM" if expected_total > 2.0 else "LOW"
    
    # Early goal likelihood (at least 1 goal in first half)
    early_goal_prob = 1 - poisson_probability(0, xg_home_ht) * poisson_probability(0, xg_away_ht)
    early_goal = "HIGH" if early_goal_prob > 0.70 else "MEDIUM" if early_goal_prob > 0.50 else "LOW"
    
    # Enhanced confidence calculation
    home_conf = calculate_confidence(home_team, xg_home)
    away_conf = calculate_confidence(away_team, xg_away)
    
    # Calculate overall confidence score
    overall_confidence_score = (home_conf["overall"] + away_conf["overall"]) / 2
    
    # Determine confidence level for backward compatibility
    if overall_confidence_score >= 75:
        overall_confidence = "HIGH"
    elif overall_confidence_score >= 50:
        overall_confidence = "MEDIUM"
    else:
        overall_confidence = "LOW"
    
    # Team momentum and trends
    if ADVANCED_STATS_AVAILABLE:
        home_momentum, home_trend_stability = calculate_trend_score([float(g) for g in home_team.goals_scored])
        away_momentum, away_trend_stability = calculate_trend_score([float(g) for g in away_team.goals_scored])
        match_predictability = calculate_match_predictability(home_conf, away_conf)
        
        # Prediction intervals
        home_interval = calculate_prediction_interval(xg_home, 0.90)
        away_interval = calculate_prediction_interval(xg_away, 0.90)
        
        # Over/Under Probabilities (New Betting Feature)
        over_under_probs = advanced_statistics.calculate_over_under_probabilities(full_probabilities)
        
        # Model quality
        home_model_fit = poisson_goodness_of_fit(home_team.goals_scored, xg_home)
        away_model_fit = poisson_goodness_of_fit(away_team.goals_scored, xg_away)
    else:
        home_momentum = "STABLE"
        away_momentum = "STABLE"
        match_predictability = overall_confidence
        home_interval = (max(0, int(xg_home) - 1), int(xg_home) + 2)
        away_interval = (max(0, int(xg_away) - 1), int(xg_away) + 2)
        over_under_probs = {}  # Basic model unavailable
        home_model_fit = 0.5
        away_model_fit = 0.5
    
    return {
        "first_half_predictions": top_ht_scores,
        "full_match_predictions": top_full_scores,
        "match_outcome": match_outcomes,
        "total_goals": total_goals_probs,  # Legacy 0-5+ probs
        "betting_insights": over_under_probs,  # New specific O/U probs
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
        }
    }


def format_predictions(result: Dict, home_name: str, away_name: str) -> str:
    """
    Format prediction results for display.
    
    Args:
        result: Prediction results dictionary
        home_name: Home team name
        away_name: Away team name
    
    Returns:
        Formatted string for display
    """
    output = f"\n{'='*60}\n"
    output += f"MATCH PREDICTION: {home_name} vs {away_name}\n"
    output += f"{'='*60}\n\n"
    
    output += "EXPECTED GOALS (xG):\n"
    output += f"  {home_name}: {result['expected_goals']['home']:.2f}\n"
    output += f"  {away_name}: {result['expected_goals']['away']:.2f}\n\n"
    
    output += "FIRST HALF PREDICTIONS:\n"
    for (home, away), prob in result['first_half_predictions']:
        output += f"  {home}–{away}: {prob*100:.1f}%\n"
    output += "\n"
    
    output += "FULL MATCH PREDICTIONS:\n"
    for (home, away), prob in result['full_match_predictions']:
        output += f"  {home}–{away}: {prob*100:.1f}%\n"
    output += "\n"
    
    output += "TOTAL GOALS PROBABILITIES:\n"
    output += f"  Under 1.5 goals: {result['total_goals']['under_1.5']*100:.1f}%\n"
    output += f"  Under 2.5 goals: {result['total_goals']['under_2.5']*100:.1f}%\n"
    output += f"  Over 2.5 goals: {result['total_goals']['over_2.5']*100:.1f}%\n"
    output += f"  Over 3.5 goals: {result['total_goals']['over_3.5']*100:.1f}%\n\n"
    
    output += "MATCH INSIGHTS:\n"
    output += f"  Expected Tempo: {result['insights']['tempo']}\n"
    output += f"  Early Goal Likelihood: {result['insights']['early_goal_likelihood']}\n"
    output += f"  Prediction Confidence: {result['insights']['confidence']}\n"
    
    return output


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Example: Manchester City (Home) vs Liverpool (Away)
    man_city = Team(
        name="Manchester City",
        goals_scored=[3, 2, 4, 1, 2],        # Last 5 matches (most recent first)
        goals_conceded=[1, 0, 1, 1, 2],      # Last 5 matches
        first_half_goals=[2, 1, 2, 0, 1]     # First half goals in last 5
    )
    
    liverpool = Team(
        name="Liverpool",
        goals_scored=[2, 3, 1, 2, 3],
        goals_conceded=[1, 2, 0, 1, 1],
        first_half_goals=[1, 2, 0, 1, 2]
    )
    
    # Run prediction
    result = predict_match(man_city, liverpool)
    output = format_predictions(result, "Manchester City", "Liverpool")
    print(output)
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Low-Scoring Match")
    print("="*60 + "\n")
    
    # Example 2: Defensive teams
    team_a = Team(
        name="Atletico Madrid",
        goals_scored=[1, 0, 1, 2, 1],
        goals_conceded=[0, 1, 0, 1, 0],
        first_half_goals=[0, 0, 1, 1, 0]
    )
    
    team_b = Team(
        name="Barcelona",
        goals_scored=[1, 1, 0, 1, 2],
        goals_conceded=[1, 0, 1, 2, 1],
        first_half_goals=[1, 0, 0, 0, 1]
    )
    
    result2 = predict_match(team_a, team_b)
    output2 = format_predictions(result2, "Atletico Madrid", "Barcelona")
    print(output2)
