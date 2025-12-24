"""
Feature Builder for Machine Learning Models
============================================

Converts Team objects from the Poisson predictor into feature vectors
suitable for machine learning models. Features are added incrementally
and validated for usefulness.

Feature Categories:
    1. Rolling Form: Goals scored/conceded over 3, 5, 10 match windows
    2. Home/Away Splits: Separate home and away performance metrics
    3. Goal Trends: Recent goal difference momentum
    4. Head-to-Head: Historical performance against specific opponent
    5. League Context: League-specific features and encodings
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    use_rolling_form: bool = True
    use_home_away_splits: bool = True
    use_goal_trends: bool = True
    use_h2h: bool = False  # Disabled by default, enable after validation
    rolling_windows: List[int] = None
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [3, 5, 10]


class FeatureBuilder:
    """Build ML features from Team objects."""
    
    def __init__(self, config: FeatureConfig = None):
        """
        Initialize feature builder.
        
        Args:
            config: Feature configuration (uses defaults if None)
        """
        self.config = config if config is not None else FeatureConfig()
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Generate feature names for reference."""
        features = []
        
        # Rolling form features
        if self.config.use_rolling_form:
            for window in self.config.rolling_windows:
                features.extend([
                    f'goals_scored_avg_{window}',
                    f'goals_conceded_avg_{window}',
                    f'goal_diff_avg_{window}',
                    f'clean_sheets_{window}',
                    f'goals_per_match_std_{window}'
                ])
        
        # Home/Away split features
        if self.config.use_home_away_splits:
            features.extend([
                'home_goals_scored_avg',
                'home_goals_conceded_avg',
                'away_goals_scored_avg',
                'away_goals_conceded_avg',
                'home_away_diff_scored',
                'home_away_diff_conceded'
            ])
        
        # Goal trend features
        if self.config.use_goal_trends:
            features.extend([
                'recent_form_trend',  # Last 3 vs previous 3
                'goal_momentum',       # Weighted recent performance
                'defensive_stability'  # Variance in goals conceded
            ])
        
        # Basic match context
        features.extend([
            'is_home',
            'league_avg_goals',
            'home_advantage_multiplier'
        ])
        
        self.feature_names = features
    
    def build_features_for_match(self, 
                                 home_team, 
                                 away_team,
                                 league_params: Dict = None,
                                 h2h_matches: List[Dict] = None) -> np.ndarray:
        """
        Build feature vector for a match.
        
        Args:
            home_team: Home team object (from football_predictor)
            away_team: Away team object
            league_params: League-specific parameters
            h2h_matches: Historical head-to-head matches (optional)
            
        Returns:
            Feature vector as numpy array [home_features, away_features]
        """
        # Default league params
        if league_params is None:
            league_params = {
                'league_avg_goals': 1.387,
                'home_advantage': 1.146,
                'away_penalty': 0.854
            }
        
        # Build features for each team
        home_features = self._build_team_features(
            home_team, 
            is_home=True, 
            league_params=league_params
        )
        
        away_features = self._build_team_features(
            away_team, 
            is_home=False, 
            league_params=league_params
        )
        
        # Combine features
        combined_features = np.concatenate([home_features, away_features])
        
        return combined_features
    
    def _build_team_features(self, 
                            team, 
                            is_home: bool,
                            league_params: Dict) -> np.ndarray:
        """Build features for a single team."""
        features = []
        
        # 1. Rolling form features
        if self.config.use_rolling_form:
            for window in self.config.rolling_windows:
                # Get last N matches (limited by available data)
                n_matches = min(window, len(team.goals_scored))
                recent_scored = team.goals_scored[:n_matches]
                recent_conceded = team.goals_conceded[:n_matches]
                
                # Average goals
                avg_scored = np.mean(recent_scored) if recent_scored else 0.0
                avg_conceded = np.mean(recent_conceded) if recent_conceded else 0.0
                avg_diff = avg_scored - avg_conceded
                
                # Clean sheets
                clean_sheets = sum(1 for x in recent_conceded if x == 0)
                clean_sheet_rate = clean_sheets / n_matches if n_matches > 0 else 0.0
                
                # Variability
                std_scored = np.std(recent_scored) if len(recent_scored) > 1 else 0.5
                
                features.extend([
                    avg_scored,
                    avg_conceded,
                    avg_diff,
                    clean_sheet_rate,
                    std_scored
                ])
        
        # 2. Home/Away splits
        if self.config.use_home_away_splits:
            # Use if available, otherwise use overall stats
            if hasattr(team, 'home_goals_scored') and team.home_goals_scored:
                home_scored_avg = np.mean(team.home_goals_scored)
                home_conceded_avg = np.mean(team.home_goals_conceded) if team.home_goals_conceded else 0.0
            else:
                home_scored_avg = np.mean(team.goals_scored) * 1.15 if is_home else np.mean(team.goals_scored)
                home_conceded_avg = np.mean(team.goals_conceded)
            
            if hasattr(team, 'away_goals_scored') and team.away_goals_scored:
                away_scored_avg = np.mean(team.away_goals_scored)
                away_conceded_avg = np.mean(team.away_goals_conceded) if team.away_goals_conceded else 0.0
            else:
                away_scored_avg = np.mean(team.goals_scored) * 0.85 if not is_home else np.mean(team.goals_scored)
                away_conceded_avg = np.mean(team.goals_conceded)
            
            home_away_diff_scored = home_scored_avg - away_scored_avg
            home_away_diff_conceded = home_conceded_avg - away_conceded_avg
            
            features.extend([
                home_scored_avg,
                home_conceded_avg,
                away_scored_avg,
                away_conceded_avg,
                home_away_diff_scored,
                home_away_diff_conceded
            ])
        
        # 3. Goal trends
        if self.config.use_goal_trends:
            # Recent trend (last 3 vs previous matches)
            if len(team.goals_scored) >= 6:
                recent_3 = np.mean(team.goals_scored[:3])
                previous_3 = np.mean(team.goals_scored[3:6])
                form_trend = recent_3 - previous_3
            else:
                form_trend = 0.0
            
            # Goal momentum (exponentially weighted)
            weights = np.array([0.4, 0.3, 0.2, 0.1])[:len(team.goals_scored)]
            weights = weights / weights.sum()
            goal_momentum = np.average(team.goals_scored[:len(weights)], weights=weights)
            
            # Defensive stability
            defensive_stability = 1.0 / (1.0 + np.std(team.goals_conceded)) if len(team.goals_conceded) > 1 else 0.5
            
            features.extend([
                form_trend,
                goal_momentum,
                defensive_stability
            ])
        
        # 4. Match context
        features.extend([
            1.0 if is_home else 0.0,
            league_params.get('league_avg_goals', 1.387),
            league_params.get('home_advantage', 1.146) if is_home else league_params.get('away_penalty', 0.854)
        ])
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_count(self) -> int:
        """Get total number of features per team."""
        return len(self.feature_names)
    
    def get_total_feature_count(self) -> int:
        """Get total features for both teams combined."""
        return len(self.feature_names) * 2


# Utility functions for loading historical data

def calculate_h2h_features(home_team_name: str, 
                          away_team_name: str,
                          match_history: pd.DataFrame,
                          n_matches: int = 5) -> Dict[str, float]:
    """
    Calculate head-to-head features from historical matches.
    
    Args:
        home_team_name: Name of home team
        away_team_name: Name of away team
        match_history: DataFrame with all historical matches
        n_matches: Number of recent H2H matches to consider
        
    Returns:
        Dictionary of H2H features
    """
    # Filter for matches between these two teams
    h2h = match_history[
        ((match_history['HomeTeam'] == home_team_name) & (match_history['AwayTeam'] == away_team_name)) |
        ((match_history['HomeTeam'] == away_team_name) & (match_history['AwayTeam'] == home_team_name))
    ].tail(n_matches)
    
    if len(h2h) == 0:
        return {
            'h2h_home_win_rate': 0.33,
            'h2h_draw_rate': 0.33,
            'h2h_avg_total_goals': 2.5,
            'h2h_home_avg_scored': 1.25,
            'h2h_away_avg_scored': 1.25,
            'h2h_matches_available': 0
        }
    
    # Calculate statistics
    home_wins = 0
    draws = 0
    total_goals = []
    home_goals = []
    away_goals = []
    
    for _, match in h2h.iterrows():
        if match['HomeTeam'] == home_team_name:
            # Home team is actually home in this H2H match
            home_goals.append(match['FTHG'])
            away_goals.append(match['FTAG'])
            if match['FTHG'] > match['FTAG']:
                home_wins += 1
            elif match['FTHG'] == match['FTAG']:
                draws += 1
        else:
            # Home team was away in this H2H match
            home_goals.append(match['FTAG'])
            away_goals.append(match['FTHG'])
            if match['FTAG'] > match['FTHG']:
                home_wins += 1
            elif match['FTAG'] == match['FTHG']:
                draws += 1
        
        total_goals.append(match['FTHG'] + match['FTAG'])
    
    return {
        'h2h_home_win_rate': home_wins / len(h2h),
        'h2h_draw_rate': draws / len(h2h),
        'h2h_avg_total_goals': np.mean(total_goals),
        'h2h_home_avg_scored': np.mean(home_goals),
        'h2h_away_avg_scored': np.mean(away_goals),
        'h2h_matches_available': len(h2h)
    }


if __name__ == '__main__':
    # Test feature builder
    from football_predictor import Team
    
    # Create sample teams
    home = Team(
        name="Manchester City",
        goals_scored=[3, 2, 4, 1, 2],
        goals_conceded=[1, 0, 1, 1, 2],
        first_half_goals=[2, 1, 2, 0, 1],
        home_goals_scored=[3, 4, 2],
        home_goals_conceded=[1, 1, 2],
        away_goals_scored=[2, 1],
        away_goals_conceded=[0, 1]
    )
    
    away = Team(
        name="Liverpool",
        goals_scored=[2, 3, 1, 2, 3],
        goals_conceded=[1, 2, 0, 1, 1],
        first_half_goals=[1, 2, 0, 1, 2],
        away_goals_scored=[2, 1, 2],
        away_goals_conceded=[1, 0, 1]
    )
    
    # Build features
    builder = FeatureBuilder()
    features = builder.build_features_for_match(home, away)
    
    print(f"Feature vector shape: {features.shape}")
    print(f"Total features: {builder.get_total_feature_count()}")
    print(f"Feature names ({len(builder.feature_names)}):")
    for i, name in enumerate(builder.feature_names):
        if i < len(features) // 2:
            print(f"  {name}: {features[i]:.3f}")
    
    print("\n✅ Feature builder test successful!")
