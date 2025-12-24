"""
Metrics Aggregator for Hybrid Prediction System
================================================

Aggregates prediction metrics over rolling windows for monitoring and analysis.
Calculates XGBoost usage %, average confidence, MAE, accuracy, and log loss.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics import log_loss, mean_absolute_error, accuracy_score


class MetricsAggregator:
    """
    Aggregates prediction metrics over rolling windows.
    
    Tracks:
    - XGBoost usage percentage
    - Average confidence
    - Goal MAE
    - Match outcome accuracy
    - Log loss
    """
    
    def __init__(self, log_file: str = 'logs/predictions.jsonl'):
        """
        Initialize metrics aggregator.
        
        Args:
            log_file: Path to JSON log file
        """
        self.log_file = Path(log_file)
    
    def load_predictions(self, n: Optional[int] = None) -> List[Dict]:
        """
        Load predictions from log file.
        
        Args:
            n: Number of recent predictions to load (None = all)
            
        Returns:
            List of prediction dictionaries
        """
        if not self.log_file.exists():
            return []
        
        predictions = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                predictions.append(json.loads(line))
        
        if n is not None:
            return predictions[-n:]
        return predictions
    
    def calculate_metrics(self, window_size: int = 50) -> Dict:
        """
        Calculate aggregated metrics over a rolling window.
        
        Args:
            window_size: Number of recent predictions to analyze
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.load_predictions(n=window_size)
        
        if not predictions:
            return {
                'window_size': 0,
                'error': 'No predictions found'
            }
        
        # Source breakdown
        sources = {}
        for pred in predictions:
            source = pred['prediction']['source']
            sources[source] = sources.get(source, 0) + 1
        
        xgboost_count = sources.get('xgboost', 0)
        poisson_count = sources.get('poisson', 0)
        total = len(predictions)
        
        xgboost_percentage = (xgboost_count / total * 100) if total > 0 else 0
        
        # Average confidence
        confidences = [p['prediction']['confidence'] for p in predictions]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Metrics requiring actual results
        evaluated = [p for p in predictions if 'actual_result' in p]
        
        metrics = {
            'window_size': len(predictions),
            'source_breakdown': {
                'xgboost': xgboost_count,
                'poisson': poisson_count,
                'xgboost_percentage': round(xgboost_percentage, 1)
            },
            'confidence': {
                'average': round(avg_confidence, 1),
                'min': round(min(confidences) if confidences else 0, 1),
                'max': round(max(confidences) if confidences else 0, 1)
            }
        }
        
        if evaluated:
            # Extract actual and predicted values
            actual_home_goals = [p['actual_result']['home_goals'] for p in evaluated]
            actual_away_goals = [p['actual_result']['away_goals'] for p in evaluated]
            predicted_home_xg = [p['prediction']['expected_goals']['home'] for p in evaluated]
            predicted_away_xg = [p['prediction']['expected_goals']['away'] for p in evaluated]
            
            # Goal MAE
            mae_home = mean_absolute_error(actual_home_goals, predicted_home_xg)
            mae_away = mean_absolute_error(actual_away_goals, predicted_away_xg)
            mae_avg = (mae_home + mae_away) / 2
            
            # Match outcome accuracy
            actual_outcomes = []
            predicted_probs = []
            
            for p in evaluated:
                # Actual outcome (0: away, 1: draw, 2: home)
                h = p['actual_result']['home_goals']
                a = p['actual_result']['away_goals']
                if h > a:
                    actual_outcomes.append(2)
                elif h < a:
                    actual_outcomes.append(0)
                else:
                    actual_outcomes.append(1)
                
                # Predicted probabilities [away, draw, home]
                probs = p['prediction']['probabilities']
                predicted_probs.append([
                    probs['away_win'],
                    probs['draw'],
                    probs['home_win']
                ])
            
            # Accuracy
            predicted_outcomes = [np.argmax(p) for p in predicted_probs]
            accuracy = accuracy_score(actual_outcomes, predicted_outcomes) * 100
            
            # Log loss
            ll = log_loss(actual_outcomes, predicted_probs)
            
            metrics['performance'] = {
                'evaluated_predictions': len(evaluated),
                'goal_mae': {
                    'home': round(mae_home, 4),
                    'away': round(mae_away, 4),
                    'average': round(mae_avg, 4)
                },
                'match_accuracy': round(accuracy, 2),
                'log_loss': round(ll, 4)
            }
        else:
            metrics['performance'] = {
                'evaluated_predictions': 0,
                'note': 'No actual results available for evaluation'
            }
        
        return metrics
    
    def print_metrics(self, window_size: int = 50):
        """
        Print metrics to console.
        
        Args:
            window_size: Number of recent predictions to analyze
        """
        metrics = self.calculate_metrics(window_size)
        
        print(f"\n{'='*70}")
        print(f"AGGREGATED METRICS (Last {metrics['window_size']} predictions)")
        print(f"{'='*70}\n")
        
        # Source breakdown
        src = metrics['source_breakdown']
        print("Prediction Sources:")
        print(f"  XGBoost: {src['xgboost']} ({src['xgboost_percentage']}%)")
        print(f"  Poisson: {src['poisson']}")
        
        # Confidence
        conf = metrics['confidence']
        print(f"\nConfidence Statistics:")
        print(f"  Average: {conf['average']}%")
        print(f"  Range: {conf['min']}% - {conf['max']}%")
        
        # Performance (if available)
        if 'goal_mae' in metrics.get('performance', {}):
            perf = metrics['performance']
            print(f"\nPerformance Metrics ({perf['evaluated_predictions']} evaluated):")
            print(f"  Goal MAE: {perf['goal_mae']['average']}")
            print(f"    Home: {perf['goal_mae']['home']}")
            print(f"    Away: {perf['goal_mae']['away']}")
            print(f"  Match Accuracy: {perf['match_accuracy']}%")
            print(f"  Log Loss: {perf['log_loss']}")
        else:
            print(f"\nPerformance Metrics: {metrics['performance']['note']}")
        
        print(f"\n{'='*70}\n")
    
    def export_summary(self, window_sizes: List[int] = [50, 100], output_file: str = 'logs/metrics_summary.json'):
        """
        Export metrics summary to JSON file.
        
        Args:
            window_sizes: List of window sizes to calculate
            output_file: Path to output file
        """
        summary = {
            'timestamp': pd.Timestamp.now().isoformat() if 'pd' in dir() else str(datetime.now()),
            'windows': {}
        }
        
        for size in window_sizes:
            metrics = self.calculate_metrics(size)
            summary['windows'][f'last_{size}'] = metrics
        
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Metrics summary exported to {output_path}")


# Example usage
if __name__ == '__main__':
    aggregator = MetricsAggregator()
    
    # Print metrics for last 50 predictions
    aggregator.print_metrics(window_size=50)
    
    # Export summary
    aggregator.export_summary([50, 100])
    
    print("✅ Metrics aggregator test completed")
