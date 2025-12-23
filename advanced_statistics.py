"""
Advanced Statistics Module for Football Prediction
===================================================
Provides enhanced statistical methods for more accurate predictions:
- Exponential Weighted Moving Average (EWMA)
- Bayesian adjustment with league priors
- Multi-factor confidence scoring
- Dixon-Coles correlation adjustment
- Poisson goodness-of-fit testing
- Prediction intervals

Author: Football Analytics System
Version: 2.0 - Enhanced
"""

import math
from typing import List, Tuple, Dict
from scipy import stats
import numpy as np


# Enhanced Constants
EWMA_ALPHA = 0.30  # Optimized for large dataset (was 0.35)
BAYESIAN_CONFIDENCE_FACTOR = 5.0  # Increased robust usage (was 3.0)
DIXON_COLES_RHO = -0.08  # Gentler correlation for better exact score accuracy (was -0.15)
USE_DIXON_COLES = True  # Enabled for better total goals/Over-Under accuracy


def exponential_weighted_average(values: List[float], alpha: float = EWMA_ALPHA) -> float:
    """
    Calculate Exponential Weighted Moving Average.
    
    Gives exponentially decreasing weights to older values.
    More responsive to recent changes than simple weighted average.
    
    Args:
        values: List of values (most recent first)
        alpha: Smoothing factor (0 < alpha <= 1), higher = more weight to recent
    
    Returns:
        EWMA value
    
    Formula:
        EWMA(t) = α·value(t) + (1-α)·EWMA(t-1)
    """
    if not values:
        return 0.0
    
    ewma = values[0]  # Start with most recent value
    for value in values[1:]:
        ewma = alpha * value + (1 - alpha) * ewma
    
    return ewma


def calculate_coefficient_of_variation(values: List[float], weights: List[float] = None) -> float:
    """
    Calculate coefficient of variation (CV = σ/μ).
    
    Measures relative variability - lower CV means more consistent performance.
    
    Args:
        values: List of numerical values
        weights: Optional weights for weighted calculation
    
    Returns:
        Coefficient of variation (0 to infinity, lower is better)
    """
    if not values or len(values) == 0:
        return float('inf')
    
    if weights is None:
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
    else:
        # Weighted calculation
        total_weight = sum(weights)
        mean = sum(v * w for v, w in zip(values, weights)) / total_weight
        variance = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total_weight
    
    if mean == 0:
        return float('inf')
    
    std_dev = math.sqrt(variance)
    return std_dev / mean


def bayesian_adjustment(raw_value: float, league_avg: float, sample_size: int, 
                       confidence_factor: float = BAYESIAN_CONFIDENCE_FACTOR) -> float:
    """
    Apply Bayesian adjustment using league average as prior.
    
    Shrinks extreme values toward league average based on sample size.
    With more data, relies more on observed value; with less data, relies more on prior.
    
    Args:
        raw_value: Observed/calculated value
        league_avg: League-wide average (prior)
        sample_size: Number of observations
        confidence_factor: Strength of prior (higher = more shrinkage)
    
    Returns:
        Adjusted value
    
    Formula:
        Adjusted = w·raw + (1-w)·prior
        where w = n/(n+k), n = sample size, k = confidence factor
    """
    weight = sample_size / (sample_size + confidence_factor)
    return weight * raw_value + (1 - weight) * league_avg


def calculate_trend_score(values: List[float]) -> Tuple[str, float]:
    """
    Detect performance trend and calculate stability score.
    
    Args:
        values: List of values (most recent first)
    
    Returns:
        Tuple of (trend_label, stability_score)
        trend_label: "IMPROVING", "STABLE", or "DECLINING"
        stability_score: 0-100, higher = more stable trend
    """
    if len(values) < 3:
        return "STABLE", 50.0
    
    # Reverse for chronological order
    chronological = list(reversed(values))
    
    # Calculate linear regression slope
    x = list(range(len(chronological)))
    n = len(x)
    
    x_mean = sum(x) / n
    y_mean = sum(chronological) / n
    
    numerator = sum((x[i] - x_mean) * (chronological[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator
    
    # Determine trend
    if slope > 0.3:
        trend = "IMPROVING"
    elif slope < -0.3:
        trend = "DECLINING"
    else:
        trend = "STABLE"
    
    # Calculate R-squared for stability
    if denominator == 0:
        r_squared = 0
    else:
        y_pred = [x_mean + slope * (xi - x_mean) for xi in x]
        ss_res = sum((chronological[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y - y_mean) ** 2 for y in chronological)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    stability_score = max(0, min(100, r_squared * 100))
    
    return trend, stability_score


def poisson_goodness_of_fit(observed_values: List[int], expected_lambda: float) -> float:
    """
    Test how well observed goals fit Poisson distribution.
    
    Uses Chi-squared goodness-of-fit test.
    
    Args:
        observed_values: List of actual goal counts
        expected_lambda: Expected value (xG)
    
    Returns:
        P-value (0-1): higher = better fit
        > 0.05 generally indicates good fit
    """
    if len(observed_values) < 3:
        return 0.1  # Insufficient data, return low confidence to avoid false 'HIGH' reliability
    
    # Count occurrences of each goal count
    max_goals = max(observed_values) if observed_values else 3
    observed_freq = {}
    for goals in range(max_goals + 1):
        observed_freq[goals] = observed_values.count(goals)
    
    # Expected frequencies from Poisson distribution
    n = len(observed_values)
    expected_freq = {}
    for goals in range(max_goals + 1):
        prob = (expected_lambda ** goals * math.exp(-expected_lambda)) / math.factorial(goals)
        expected_freq[goals] = n * prob
    
    # Chi-squared test
    chi_squared = 0
    for goals in range(max_goals + 1):
        obs = observed_freq.get(goals, 0)
        exp = expected_freq.get(goals, 0.1)  # Avoid division by zero
        if exp > 0:
            chi_squared += ((obs - exp) ** 2) / exp
    
    # Degrees of freedom: categories - 1 - estimated parameters
    df = max(1, max_goals - 1)
    
    # Calculate p-value
    try:
        p_value = 1 - stats.chi2.cdf(chi_squared, df)
    except:
        p_value = 0.1  # Default to low if calculation fails
    
    return max(0, min(1, p_value))


def calculate_multi_factor_confidence(
    goals_scored: List[int],
    goals_conceded: List[int],
    xg_value: float,
    league_avg: float = 1.4
) -> Dict[str, float]:
    """
    Calculate comprehensive confidence score using multiple factors.
    
    Factors:
    1. Consistency (40%): Based on coefficient of variation
    2. Sample Size (20%): More matches = higher confidence
    3. Trend Stability (20%): Stable trends = higher confidence
    4. Model Fit (20%): How well data fits Poisson distribution
    
    Args:
        goals_scored: Recent goals scored
        goals_conceded: Recent goals conceded
        xg_value: Calculated expected goals
        league_avg: League average for comparison
    
    Returns:
        Dictionary with overall score (0-100) and breakdown
    """
    # Factor 1: Consistency (lower CV = higher score)
    cv = calculate_coefficient_of_variation(goals_scored)
    consistency_score = max(0, min(100, 100 - (cv * 50)))
    
    # Factor 2: Sample Size (more observations = higher confidence)
    sample_size = len(goals_scored)
    sample_size_score = min(100, sample_size * 20)  # 5 matches = 100
    
    # Factor 3: Trend Stability
    _, trend_stability = calculate_trend_score(goals_scored)
    
    # Factor 4: Model Fit (Poisson goodness-of-fit)
    p_value = poisson_goodness_of_fit(goals_scored, xg_value)
    model_fit_score = p_value * 100
    
    # Weighted combination
    overall_score = (
        0.40 * consistency_score +
        0.20 * sample_size_score +
        0.20 * trend_stability +
        0.20 * model_fit_score
    )
    
    return {
        "overall": round(overall_score, 1),
        "consistency": round(consistency_score, 1),
        "sample_size": round(sample_size_score, 1),
        "trend_stability": round(trend_stability, 1),
        "model_fit": round(model_fit_score, 1)
    }


def dixon_coles_tau(home_goals: int, away_goals: int, lambda_home: float, 
                    lambda_away: float, rho: float = DIXON_COLES_RHO, 
                    use_adjustment: bool = USE_DIXON_COLES) -> float:
    """
    Calculate Dixon-Coles correlation adjustment factor.
    
    Adjusts probabilities for low-scoring outcomes (0-0, 1-0, 0-1, 1-1)
    to account for correlation between home and away goals.
    
    NOTE: This can improve total goals accuracy but may reduce exact score accuracy.
    Set use_adjustment=False to disable (recommended for exact scores).
    
    Args:
        home_goals: Home team goals in this outcome
        away_goals: Away team goals in this outcome
        lambda_home: Expected goals for home team
        lambda_away: Expected goals for away team
        rho: Correlation parameter (typically -0.05 to -0.10, lower = gentler adjustment)
        use_adjustment: Whether to apply Dixon-Coles adjustment (default: False)
    
    Returns:
        Adjustment factor tau (multiply base probability by this)
    """
    # Return 1.0 (no adjustment) if disabled or not a low-scoring outcome
    if not use_adjustment or (home_goals, away_goals) not in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        return 1.0
    
    # Dixon-Coles formula (only applied if use_adjustment=True)
    if home_goals == 0 and away_goals == 0:
        tau = 1 - lambda_home * lambda_away * rho
    elif home_goals == 1 and away_goals == 0:
        tau = 1 + lambda_away * rho
    elif home_goals == 0 and away_goals == 1:
        tau = 1 + lambda_home * rho
    elif home_goals == 1 and away_goals == 1:
        tau = 1 - rho
    else:
        tau = 1.0
    
    return tau


def calculate_prediction_interval(xg: float, confidence_level: float = 0.90) -> Tuple[int, int]:
    """
    Calculate prediction interval for goals.
    
    Returns range of goals with given confidence level.
    
    Args:
        xg: Expected goals
        confidence_level: Confidence level (e.g., 0.90 for 90%)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Use Poisson distribution to find interval
    alpha = (1 - confidence_level) / 2  # Two-tailed
    
    # Find lower bound
    lower = 0
    cumulative = 0
    while cumulative < alpha and lower < 20:
        prob = (xg ** lower * math.exp(-xg)) / math.factorial(lower)
        cumulative += prob
        if cumulative >= alpha:
            break
        lower += 1
    
    # Find upper bound
    upper = 0
    cumulative = 0
    target = 1 - alpha
    while cumulative < target and upper < 20:
        prob = (xg ** upper * math.exp(-xg)) / math.factorial(upper)
        cumulative += prob
        upper += 1
    
    return max(0, lower), max(0, upper - 1)


def calculate_match_predictability(home_confidence: Dict, away_confidence: Dict) -> str:
    """
    Determine overall match predictability based on both teams' confidence.
    
    Args:
        home_confidence: Home team confidence breakdown
        away_confidence: Away team confidence breakdown
    
    Returns:
        "HIGH", "MEDIUM", or "LOW" predictability
    """
    avg_confidence = (home_confidence["overall"] + away_confidence["overall"]) / 2
    
    if avg_confidence >= 75:
        return "HIGH"
    elif avg_confidence >= 50:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_over_under_probabilities(full_probabilities: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    """
    Calculate precise probabilities for Over/Under markets.
    
    Args:
        full_probabilities: Dictionary of (home_goals, away_goals) -> probability
    
    Returns:
        Dictionary with probabilities for Over/Under 1.5, 2.5, 3.5
    """
    markets = {
        "over_1_5": 0.0, "under_1_5": 0.0,
        "over_2_5": 0.0, "under_2_5": 0.0,
        "over_3_5": 0.0, "under_3_5": 0.0
    }
    
    for (h, a), prob in full_probabilities.items():
        total = h + a
        
        # 1.5 Goals
        if total > 1.5: markets["over_1_5"] += prob
        else: markets["under_1_5"] += prob
        
        # 2.5 Goals
        if total > 2.5: markets["over_2_5"] += prob
        else: markets["under_2_5"] += prob
        
        # 3.5 Goals
        if total > 3.5: markets["over_3_5"] += prob
        else: markets["under_3_5"] += prob
        
    return {k: round(v * 100, 1) for k, v in markets.items()}
