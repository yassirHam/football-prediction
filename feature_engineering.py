"""
Feature Engineering Module for Football Prediction
===================================================
Advanced features to improve prediction accuracy beyond basic 5-match form.

Features:
- Multi-window rolling averages (3, 5, 10 games)
- Adaptive form weighting based on consistency
- Streak detection (scoring, winning, losing)
- Season phase effects
- Performance trend analysis

Author: Football Analytics System
Version: 1.0 - Phase 2 Enhancement
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class TeamFormProfile:
    """Comprehensive form profile for a team."""
    short_term_avg: float  # Last 3 games
    medium_term_avg: float  # Last 5 games
    long_term_avg: float  # Last 10 games
    adaptive_avg: float  # Weighted average based on consistency
    trend: str  # "IMPROVING", "STABLE", "DECLINING"
    scoring_streak: int  # Consecutive games scoring
    clean_sheet_streak: int  # Consecutive clean sheets
    consistency_score: float  # 0-100, higher = more consistent


def calculate_multi_window_average(values: List[int], windows: List[int] = [3, 5, 10]) -> Dict[str, float]:
    """
    Calculate rolling averages over multiple windows.
    
    Args:
        values: List of values (most recent first)
        windows: List of window sizes
        
    Returns:
        Dictionary with averages for each window
    """
    result = {}
    
    for window in windows:
        if len(values) >= window:
            window_values = values[:window]
            result[f'avg_{window}'] = sum(window_values) / len(window_values)
        else:
            # Not enough data - use what we have
            result[f'avg_{window}'] = sum(values) / len(values) if values else 0.0
    
    return result


def detect_scoring_streak(goals_scored: List[int]) -> Tuple[int, int]:
    """
    Detect current scoring streak and dry spell.
    
    Args:
        goals_scored: Goals scored in recent matches (most recent first)
        
    Returns:
        Tuple of (scoring_streak, dry_spell_streak)
        - scoring_streak: consecutive games with goals (positive)
        - dry_spell_streak: consecutive games without goals (positive if current)
    """
    if not goals_scored:
        return 0, 0
    
    scoring_streak = 0
    dry_spell = 0
    
    # Check current streak
    for goals in goals_scored:
        if goals > 0:
            if dry_spell == 0:  # Still on scoring streak
                scoring_streak += 1
            else:  # Broke dry spell
                break
        else:
            if scoring_streak == 0:  # Still on dry spell
                dry_spell += 1
            else:  # Broke scoring streak
                break
    
    return scoring_streak, dry_spell


def detect_clean_sheet_streak(goals_conceded: List[int]) -> int:
    """
    Detect current clean sheet streak.
    
    Args:
        goals_conceded: Goals conceded in recent matches (most recent first)
        
    Returns:
        Number of consecutive clean sheets
    """
    if not goals_conceded:
        return 0
    
    streak = 0
    for goals in goals_conceded:
        if goals == 0:
            streak += 1
        else:
            break
    
    return streak


def calculate_consistency_score(values: List[int]) -> float:
    """
    Calculate consistency score based on variance.
    
    Args:
        values: List of values
        
    Returns:
        Consistency score (0-100, higher = more consistent)
    """
    if len(values) < 2:
        return 50.0  # Neutral score with insufficient data
    
    mean = np.mean(values)
    std = np.std(values)
    
    # Coefficient of variation
    if mean > 0:
        cv = std / mean
    else:
        cv = std  # For very low-scoring teams
    
    # Convert to 0-100 scale (lower CV = higher consistency)
    # CV of 0 = 100%, CV of 1.0 = 50%, CV of 2.0+ = 0%
    consistency = max(0, min(100, 100 - (cv * 50)))
    
    return consistency


def detect_performance_trend(values: List[float]) -> Tuple[str, float]:
    """
    Detect if team performance is improving, stable, or declining.
    
    Uses linear regression slope on recent form.
    
    Args:
        values: Performance values (most recent first)
        
    Returns:
        Tuple of (trend_label, slope_value)
        trend_label: "IMPROVING", "STABLE", or "DECLINING"
        slope_value: Numerical slope (positive = improving)
    """
    if len(values) < 3:
        return "STABLE", 0.0
    
    # Reverse to get chronological order for regression
    values_chrono = list(reversed(values))
    
    # Simple linear regression
    n = len(values_chrono)
    x = np.arange(n)
    y = np.array(values_chrono)
    
    # Calculate slope: (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    
    # Classify trend
    if slope > 0.15:  # Significantly improving
        trend = "IMPROVING"
    elif slope < -0.15:  # Significantly declining
        trend = "DECLINING"
    else:
        trend = "STABLE"
    
    return trend, slope


def calculate_adaptive_weights(values: List[int], consistency_score: float) -> List[float]:
    """
    Calculate adaptive form weights based on team consistency.
    
    Consistent teams: Use more balanced weights (trust older data)
    Inconsistent teams: Use more recency bias (trust recent data more)
    
    Args:
        values: Performance values
        consistency_score: Consistency score (0-100)
        
    Returns:
        List of adaptive weights
    """
    n = len(values)
    if n == 0:
        return []
    
    # Base exponential decay
    base_weights = np.exp(-np.arange(n) * 0.3)
    
    # Adjust based on consistency
    # High consistency (>70): flatten weights (trust all data)
    # Low consistency (<30): steepen weights (trust recent only)
    consistency_factor = consistency_score / 100.0
    
    if consistency_factor > 0.7:
        # Flatten - more uniform weights
        adjusted_weights = base_weights ** 0.5
    elif consistency_factor < 0.3:
        # Steepen - more recency bias
        adjusted_weights = base_weights ** 1.5
    else:
        # Normal
        adjusted_weights = base_weights
    
    # Normalize
    normalized = adjusted_weights / np.sum(adjusted_weights)
    
    return normalized.tolist()


def create_team_form_profile(goals_scored: List[int], 
                             goals_conceded: List[int]) -> TeamFormProfile:
    """
    Create comprehensive form profile for a team.
    
    Args:
        goals_scored: Goals scored in recent matches (most recent first)
        goals_conceded: Goals conceded in recent matches (most recent first)
        
    Returns:
        TeamFormProfile object
    """
    # Multi-window averages
    scoring_windows = calculate_multi_window_average(goals_scored, [3, 5, 10])
    
    # Streaks
    scoring_streak, _ = detect_scoring_streak(goals_scored)
    clean_sheet_streak = detect_clean_sheet_streak(goals_conceded)
    
    # Consistency
    consistency = calculate_consistency_score(goals_scored)
    
    # Trend
    trend, _ = detect_performance_trend([float(g) for g in goals_scored])
    
    # Adaptive average
    adaptive_weights = calculate_adaptive_weights(goals_scored, consistency)
    if adaptive_weights:
        adaptive_avg = sum(g * w for g, w in zip(goals_scored, adaptive_weights))
    else:
        adaptive_avg = 0.0
    
    return TeamFormProfile(
        short_term_avg=scoring_windows.get('avg_3', 0.0),
        medium_term_avg=scoring_windows.get('avg_5', 0.0),
        long_term_avg=scoring_windows.get('avg_10', 0.0),
        adaptive_avg=adaptive_avg,
        trend=trend,
        scoring_streak=scoring_streak,
        clean_sheet_streak=clean_sheet_streak,
        consistency_score=consistency
    )


def apply_streak_boost(base_xg: float, form_profile: TeamFormProfile) -> float:
    """
    Apply boost/penalty based on current streaks.
    
    Args:
        base_xg: Base expected goals
        form_profile: Team form profile
        
    Returns:
        Adjusted xG with streak consideration
    """
    adjusted_xg = base_xg
    
    # Scoring streak boost (max +10%)
    if form_profile.scoring_streak >= 5:
        streak_boost = 1.10
    elif form_profile.scoring_streak >= 3:
        streak_boost = 1.05
    else:
        streak_boost = 1.0
    
    adjusted_xg *= streak_boost
    
    return adjusted_xg


def apply_trend_adjustment(base_xg: float, form_profile: TeamFormProfile) -> float:
    """
    Apply adjustment based on performance trend.
    
    Improving teams get slight boost, declining teams get penalty.
    
    Args:
        base_xg: Base expected goals
        form_profile: Team form profile
        
    Returns:
        Adjusted xG with trend consideration
    """
    if form_profile.trend == "IMPROVING":
        return base_xg * 1.05  # +5% for improving teams
    elif form_profile.trend == "DECLINING":
        return base_xg * 0.95  # -5% for declining teams
    else:
        return base_xg


def get_best_form_estimate(form_profile: TeamFormProfile, 
                           consistency_threshold: float = 60.0) -> float:
    """
    Get best form estimate based on consistency.
    
    High consistency: Use longer-term average
    Low consistency: Use short-term average
    
    Args:
        form_profile: Team form profile
        consistency_threshold: Threshold for using long-term vs short-term
        
    Returns:
        Best estimate of current form
    """
    if form_profile.consistency_score >= consistency_threshold:
        # Consistent team - trust longer-term average
        if form_profile.long_term_avg > 0:
            return form_profile.long_term_avg
        else:
            return form_profile.medium_term_avg
    else:
        # Inconsistent team - use recent form or adaptive average
        return form_profile.adaptive_avg


# Example usage
if __name__ == "__main__":
    # Test with sample data
    goals_scored = [3, 2, 1, 4, 2, 1, 3, 2, 1, 2]  # Most recent first
    goals_conceded = [1, 0, 0, 2, 1, 1, 0, 1, 2, 1]
    
    profile = create_team_form_profile(goals_scored, goals_conceded)
    
    print("Team Form Profile:")
    print(f"  Short-term (3 games): {profile.short_term_avg:.2f}")
    print(f"  Medium-term (5 games): {profile.medium_term_avg:.2f}")
    print(f"  Long-term (10 games): {profile.long_term_avg:.2f}")
    print(f"  Adaptive average: {profile.adaptive_avg:.2f}")
    print(f"  Trend: {profile.trend}")
    print(f"  Scoring streak: {profile.scoring_streak} games")
    print(f"  Clean sheet streak: {profile.clean_sheet_streak} games")
    print(f"  Consistency: {profile.consistency_score:.1f}/100")
    
    # Apply enhancements
    base_xg = 1.8
    with_streak = apply_streak_boost(base_xg, profile)
    with_trend = apply_trend_adjustment(with_streak, profile)
    
    print(f"\nxG Adjustments:")
    print(f"  Base xG: {base_xg:.2f}")
    print(f"  With streak boost: {with_streak:.2f}")
    print(f"  With trend adjustment: {with_trend:.2f}")
