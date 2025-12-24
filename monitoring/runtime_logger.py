"""
Runtime Logger for Hybrid Prediction System
============================================

Logs every prediction made by the system for monitoring and analysis.
Tracks prediction source, confidence, xG values, and results.
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path


class RuntimeLogger:
    """
    Logs predictions in real-time for monitoring and analysis.
    
    Outputs:
    - JSON logs for detailed analysis
    - CSV summary for quick review
    - Console summaries on demand
    """
    
    def __init__(self, log_dir: str = 'logs'):
        """
        Initialize runtime logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # File paths
        self.json_log_path = self.log_dir / 'predictions.jsonl'
        self.csv_log_path = self.log_dir / 'predictions_summary.csv'
        
        # Initialize CSV if it doesn't exist
        if not self.csv_log_path.exists():
            self._initialize_csv()
    
    def _initialize_csv(self):
        """Create CSV file with headers."""
        headers = [
            'timestamp', 'home_team', 'away_team',
            'source', 'confidence', 
            'xg_home', 'xg_away', 'total_xg',
            'prob_home_win', 'prob_draw', 'prob_away_win',
            'actual_home_goals', 'actual_away_goals',
            'prediction_correct', 'goal_error'
        ]
        
        with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_prediction(self, 
                       home_team: str,
                       away_team: str,
                       prediction: Dict,
                       actual_result: Optional[Dict] = None):
        """
        Log a single prediction.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            prediction: Prediction dictionary from hybrid predictor
            actual_result: Optional dict with {'home_goals': int, 'away_goals': int}
        """
        timestamp = datetime.now().isoformat()
        
        # Extract prediction data
        hybrid_meta = prediction.get('hybrid_metadata', {})
        source = hybrid_meta.get('source', 'unknown')
        confidence = hybrid_meta.get('confidence', 0)
        
        xg_home = prediction['expected_goals']['home']
        xg_away = prediction['expected_goals']['away']
        total_xg = xg_home + xg_away
        
        match_outcome = prediction['match_outcome']
        prob_home_win = match_outcome['home_win']
        prob_draw = match_outcome['draw']
        prob_away_win = match_outcome['away_win']
        
        # Prepare log entry
        log_entry = {
            'timestamp': timestamp,
            'home_team': home_team,
            'away_team': away_team,
            'prediction': {
                'source': source,
                'confidence': confidence,
                'expected_goals': {
                    'home': xg_home,
                    'away': xg_away,
                    'total': total_xg
                },
                'probabilities': {
                    'home_win': prob_home_win,
                    'draw': prob_draw,
                    'away_win': prob_away_win
                }
            }
        }
        
        # Add actual result if provided
        if actual_result:
            actual_home = actual_result['home_goals']
            actual_away = actual_result['away_goals']
            
            # Calculate if prediction was correct
            if prob_home_win > prob_draw and prob_home_win > prob_away_win:
                predicted_outcome = 'home'
            elif prob_away_win > prob_draw and prob_away_win > prob_home_win:
                predicted_outcome = 'away'
            else:
                predicted_outcome = 'draw'
            
            if actual_home > actual_away:
                actual_outcome = 'home'
            elif actual_away > actual_home:
                actual_outcome = 'away'
            else:
                actual_outcome = 'draw'
            
            prediction_correct = predicted_outcome == actual_outcome
            goal_error = abs(xg_home - actual_home) + abs(xg_away - actual_away)
            
            log_entry['actual_result'] = {
                'home_goals': actual_home,
                'away_goals': actual_away,
                'outcome': actual_outcome
            }
            log_entry['evaluation'] = {
                'prediction_correct': prediction_correct,
                'goal_error': goal_error / 2  # Average error
            }
        else:
            actual_home = actual_away = None
            prediction_correct = None
            goal_error = None
        
        # Write JSON log (append)
        with open(self.json_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Write CSV log (append)
        with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, home_team, away_team,
                source, confidence,
                f"{xg_home:.2f}", f"{xg_away:.2f}", f"{total_xg:.2f}",
                f"{prob_home_win:.3f}", f"{prob_draw:.3f}", f"{prob_away_win:.3f}",
                actual_home if actual_home is not None else '',
                actual_away if actual_away is not None else '',
                prediction_correct if prediction_correct is not None else '',
                f"{goal_error:.2f}" if goal_error is not None else ''
            ])
    
    def get_recent_predictions(self, n: int = 10) -> list:
        """
        Get the N most recent predictions.
        
        Args:
            n: Number of recent predictions to retrieve
            
        Returns:
            List of prediction dictionaries
        """
        if not self.json_log_path.exists():
            return []
        
        predictions = []
        with open(self.json_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                predictions.append(json.loads(line))
        
        return predictions[-n:]
    
    def print_summary(self, n: int = 50):
        """
        Print a console summary of recent predictions.
        
        Args:
            n: Number of recent predictions to summarize
        """
        recent = self.get_recent_predictions(n)
        
        if not recent:
            print("No predictions logged yet.")
            return
        
        print(f"\n{'='*70}")
        print(f"PREDICTION LOG SUMMARY (Last {len(recent)} predictions)")
        print(f"{'='*70}\n")
        
        # Source breakdown
        sources = {}
        for pred in recent:
            source = pred['prediction']['source']
            sources[source] = sources.get(source, 0) + 1
        
        print("Prediction Sources:")
        for source, count in sources.items():
            print(f"  {source.capitalize()}: {count} ({count/len(recent)*100:.1f}%)")
        
        # Average confidence
        confidences = [p['prediction']['confidence'] for p in recent]
        avg_confidence = sum(confidences) / len(confidences)
        print(f"\nAverage Confidence: {avg_confidence:.1f}%")
        
        # Evaluated predictions
        evaluated = [p for p in recent if 'evaluation' in p]
        if evaluated:
            correct = sum(1 for p in evaluated if p['evaluation']['prediction_correct'])
            accuracy = correct / len(evaluated) * 100
            
            avg_goal_error = sum(p['evaluation']['goal_error'] for p in evaluated) / len(evaluated)
            
            print(f"\nEvaluated Predictions: {len(evaluated)}")
            print(f"  Accuracy: {accuracy:.1f}%")
            print(f"  Average Goal Error: {avg_goal_error:.2f}")
        
        print(f"\n{'='*70}\n")


# Example usage
if __name__ == '__main__':
    logger = RuntimeLogger()
    
    # Example prediction
    example_prediction = {
        'expected_goals': {'home': 1.8, 'away': 1.2},
        'match_outcome': {'home_win': 0.48, 'draw': 0.25, 'away_win': 0.27},
        'hybrid_metadata': {'source': 'xgboost', 'confidence': 72.5}
    }
    
    logger.log_prediction(
        home_team="Team A",
        away_team="Team B",
        prediction=example_prediction,
        actual_result={'home_goals': 2, 'away_goals': 1}
    )
    
    logger.print_summary()
    print("✅ Runtime logger test completed")
