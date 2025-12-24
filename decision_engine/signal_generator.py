"""
Signal Generator for Decision Support
======================================

Generates actionable signals from predictions and filter results.
Combines multiple factors to provide decision recommendations.
"""

from typing import Dict

try:
    from .filters import PredictionFilters
except ImportError:
    from filters import PredictionFilters


class SignalGenerator:
    """
    Generates actionable signals from predictions.
    
    Combines:
    - Prediction confidence
    - Filter results
    - Match characteristics
    
    Output: STRONG / PASS / WEAK signal with reasoning
    """
    
    def __init__(self, filters: PredictionFilters = None):
        """
        Initialize signal generator.
        
        Args:
            filters: PredictionFilters instance (creates new if None)
        """
        self.filters = filters if filters else PredictionFilters()
    
    def generate_signal(self, prediction: Dict, league: str = 'DEFAULT') -> Dict:
        """
        Generate signal from prediction.
        
        Args:
            prediction: Prediction dictionary from hybrid predictor
            league: League code
            
        Returns:
            Dictionary with signal and reasoning
        """
        # Extract data
        hybrid_meta = prediction.get('hybrid_metadata', {})
        confidence = hybrid_meta.get('confidence', 0)
        
        xg_home = prediction['expected_goals']['home']
        xg_away = prediction['expected_goals']['away']
        
        match_outcome = prediction['match_outcome']
        prob_home = match_outcome['home_win']
        prob_draw = match_outcome['draw']
        prob_away = match_outcome['away_win']
        
        # Apply filters
        filter_results = self.filters.apply_all_filters(
            confidence=confidence,
            xg_home=xg_home,
            xg_away=xg_away,
            league=league
        )
        
        # Get overall signal
        signal = filter_results['overall_signal']
        
        # Determine recommended bets/actions
        recommendations = []
        
        if signal == 'STRONG':
            # High confidence recommendations
            if prob_home > 0.5:
                recommendations.append(f"Home Win ({prob_home*100:.1f}%)")
            elif prob_away > 0.5:
                recommendations.append(f"Away Win ({prob_away*100:.1f}%)")
            
            total_goals = xg_home + xg_away
            if total_goals > 3.0:
                recommendations.append(f"Over 2.5 Goals ({total_goals:.1f} xG)")
            elif total_goals < 2.0:
                recommendations.append(f"Under 2.5 Goals ({total_goals:.1f} xG)")
        
        # Build reasoning
        reasoning = []
        if filter_results['filters']['confidence'] == 'STRONG':
            reasoning.append(f"High confidence ({confidence:.1f}%)")
        elif filter_results['filters']['confidence'] == 'WEAK':
            reasoning.append(f"Low confidence ({confidence:.1f}%)")
        
        if filter_results['filters']['goals_distance'] == 'STRONG':
            reasoning.append("Significant deviation from league average")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'filter_results': filter_results,
            'recommendations': recommendations,
            'reasoning': reasoning,
            'match_summary': {
                'predicted_score': f"{xg_home:.1f} - {xg_away:.1f}",
                'most_likely_outcome': max(
                    [('Home', prob_home), ('Draw', prob_draw), ('Away', prob_away)],
                    key=lambda x: x[1]
                )[0],
                'total_goals': round(xg_home + xg_away, 1)
            }
        }
    
    def print_signal(self, signal_result: Dict, home_team: str, away_team: str):
        """
        Print signal in human-readable format.
        
        Args:
            signal_result: Result from generate_signal()
            home_team: Home team name
            away_team: Away team name
        """
        print(f"\n{'='*70}")
        print(f"DECISION SIGNAL: {home_team} vs {away_team}")
        print(f"{'='*70}")
        
        # Signal strength
        signal = signal_result['signal']
        confidence = signal_result['confidence']
        
        emoji = '🟢' if signal == 'STRONG' else '🟡' if signal == 'PASS' else '🔴'
        print(f"\n{emoji} SIGNAL: {signal} (Confidence: {confidence:.1f}%)")
        
        # Match summary
        summary = signal_result['match_summary']
        print(f"\nPredicted Score: {summary['predicted_score']}")
        print(f"Most Likely: {summary['most_likely_outcome']}")
        print(f"Total Goals: {summary['total_goals']}")
        
        # Recommendations
        if signal_result['recommendations']:
            print(f"\nRecommendations:")
            for rec in signal_result['recommendations']:
                print(f"  • {rec}")
        
        # Reasoning
        if signal_result['reasoning']:
            print(f"\nReasoning:")
            for reason in signal_result['reasoning']:
                print(f"  • {reason}")
        
        print(f"\n{'='*70}\n")


# Example usage
if __name__ == '__main__':
    from hybrid.hybrid_predictor import predict_match_hybrid, Team
    
    # Create example teams
    home = Team("Manchester City", [3,2,4,1,2], [1,0,1,1,2], [2,1,2,0,1], league="E0")
    away = Team("Liverpool", [2,3,1,2,3], [1,2,0,1,1], [1,2,0,1,2], league="E0")
    
    # Get prediction
    prediction = predict_match_hybrid(home, away)
    
    # Generate signal
    generator = SignalGenerator()
    signal = generator.generate_signal(prediction, league='E0')
    
    # Print signal
    generator.print_signal(signal, "Manchester City", "Liverpool")
    
    print("✅ Signal generator test completed")
