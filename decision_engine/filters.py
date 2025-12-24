"""
Decision Filters for Hybrid Prediction System
==============================================

Filters predictions to determine when to act on them.
Does NOT block predictions, only flags them as STRONG/PASS/WEAK.
"""

import json
from typing import Dict, Optional
from pathlib import Path


class PredictionFilters:
    """
    Applies filters to predictions to determine signal strength.
    
    Filters:
    - Minimum confidence threshold
    - Total goals distance from league average
    - Optional odds comparison (stub)
    """
    
    def __init__(self, config_file: str = 'decision_engine/filter_config.json'):
        """
        Initialize prediction filters.
        
        Args:
            config_file: Path to filter configuration
        """
        self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load filter configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'confidence_threshold': 30,
                'goals_distance_threshold': 0.4,
                'league_averages': {
                    'E0': 2.7,   # Premier League
                    'SP1': 2.6,  # La Liga
                    'D1': 2.9,   # Bundesliga
                    'DEFAULT': 2.7
                }
            }
    
    def apply_confidence_filter(self, confidence: float) -> str:
        """
        Apply confidence filter.
        
        Args:
            confidence: Confidence score (0-100)
            
        Returns:
            'PASS', 'WEAK', or 'STRONG'
        """
        threshold = self.config.get('confidence_threshold', 30)
        
        if confidence >= threshold + 10:
            return 'STRONG'
        elif confidence >= threshold:
            return 'PASS'
        else:
            return 'WEAK'
    
    def apply_goals_distance_filter(self, predicted_total_goals: float, league: str = 'DEFAULT') -> str:
        """
        Apply total goals distance filter.
        
        Args:
            predicted_total_goals: Predicted total goals (xG_home + xG_away)
            league: League code
            
        Returns:
            'PASS', 'WEAK', or 'STRONG'
        """
        league_avgs = self.config.get('league_averages', {})
        league_avg = league_avgs.get(league, league_avgs.get('DEFAULT', 2.7))
        threshold = self.config.get('goals_distance_threshold', 0.4)
        
        distance = abs(predicted_total_goals - league_avg)
        
        if distance >= threshold * 1.5:
            return 'STRONG'  # Significant deviation
        elif distance >= threshold:
            return 'PASS'
        else:
            return 'WEAK'  # Too close to average, less interesting
    
    def apply_odds_filter(self, predicted_prob: float, market_odds: Optional[float] = None) -> str:
        """
        Apply odds comparison filter (optional, stub for future use).
        
        Args:
            predicted_prob: Predicted probability (0-1)
            market_odds: Market odds (decimal, e.g., 2.5) - optional
            
        Returns:
            'PASS', 'WEAK', or 'STRONG'
        """
        if market_odds is None:
            return 'PASS'  # No odds available, neutral
        
        # Implied probability from odds
        implied_prob = 1 / market_odds
        
        # Value threshold (predicted prob > implied prob + margin)
        margin = 0.05
        
        if predicted_prob > implied_prob + margin:
            return 'STRONG'  # Value bet
        elif predicted_prob > implied_prob:
            return 'PASS'  # Slight edge
        else:
            return 'WEAK'  # No value
    
    def apply_all_filters(self, 
                         confidence: float,
                         xg_home: float,
                         xg_away: float,
                         league: str = 'DEFAULT',
                         market_odds: Optional[float] = None) -> Dict:
        """
        Apply all filters and return results.
        
        Args:
            confidence: Confidence score (0-100)
            xg_home: Predicted home xG
            xg_away: Predicted away xG
            league: League code
            market_odds: Optional market odds
            
        Returns:
            Dictionary with filter results
        """
        total_goals = xg_home + xg_away
        
        confidence_result = self.apply_confidence_filter(confidence)
        goals_result = self.apply_goals_distance_filter(total_goals, league)
        odds_result = self.apply_odds_filter(max(xg_home, xg_away) / total_goals if total_goals > 0 else 0.5, market_odds)
        
        # Aggregate results
        results = [confidence_result, goals_result, odds_result]
        strong_count = results.count('STRONG')
        weak_count = results.count('WEAK')
        
        # Overall signal
        if strong_count >= 2:
            overall = 'STRONG'
        elif weak_count >= 2:
            overall = 'WEAK'
        else:
            overall = 'PASS'
        
        return {
            'filters': {
                'confidence': confidence_result,
                'goals_distance': goals_result,
                'odds': odds_result
            },
            'overall_signal': overall,
            'details': {
                'confidence_value': confidence,
                'total_goals': round(total_goals, 2),
                'league_avg': self.config['league_averages'].get(league, 2.7)
            }
        }


# Example usage
if __name__ == '__main__':
    filters = PredictionFilters()
    
    # Test with high confidence, high goals match
    result = filters.apply_all_filters(
        confidence=38,
        xg_home=2.1,
        xg_away=1.8,
        league='E0'
    )
    
    print("Test 1: High confidence, high goals")
    print(f"Overall Signal: {result['overall_signal']}")
    print(f"Filters: {result['filters']}")
    print(f"Details: {result['details']}")
    
    # Test with low confidence
    result2 = filters.apply_all_filters(
        confidence=24,
        xg_home=1.4,
        xg_away=1.3,
        league='SP1'
    )
    
    print("\nTest 2: Low confidence, average goals")
    print(f"Overall Signal: {result2['overall_signal']}")
    print(f"Filters: {result2['filters']}")
    
    print("\n✅ Prediction filters test completed")
