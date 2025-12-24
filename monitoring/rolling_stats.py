"""
Rolling Statistics Tracker
===========================

Maintains and tracks rolling window statistics for the hybrid prediction system.
Provides real-time monitoring of system performance without modifying predictions.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, List


class RollingStats:
    """
    Tracks rolling window statistics for predictions.
    
    Maintains in-memory rolling windows for:
    - XGBoost usage percentage
    - Average confidence
    - Goal MAE
    - Match accuracy
    - Log loss
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize rolling statistics tracker.
        
        Args:
            window_size: Number of predictions to track in rolling window
        """
        self.window_size = window_size
        
        # Rolling windows (deque for efficient O(1) append and pop)
        self.sources = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.xg_home = deque(maxlen=window_size)
        self.xg_away = deque(maxlen=window_size)
        self.actual_home = deque(maxlen=window_size)
       self.actual_away = deque(maxlen=window_size)
        self.outcomes_actual = deque(maxlen=window_size)
        self.outcomes_predicted = deque(maxlen=window_size)
        self.log_loss_contrib = deque(maxlen=window_size)
    
    def add_prediction(self, 
                      source: str,
                      confidence: float,
                      xg_home: float,
                      xg_away: float,
                      prob_home: float,
                      prob_draw: float,
                      prob_away: float,
                      actual_home_goals: Optional[int] = None,
                      actual_away_goals: Optional[int] = None):
        """
        Add a prediction to the rolling window.
        
        Args:
            source: Prediction source ('xgboost' or 'poisson')
            confidence: Confidence score (0-100)
            xg_home: Predicted home xG
            xg_away: Predicted away xG
            prob_home: Probability of home win
            prob_draw: Probability of draw
            prob_away: Probability of away win
            actual_home_goals: Actual home goals (if known)
            actual_away_goals: Actual away goals (if known)
        """
        self.sources.append(source)
        self.confidences.append(confidence)
        self.xg_home.append(xg_home)
        self.xg_away.append(xg_away)
        
        if actual_home_goals is not None and actual_away_goals is not None:
            self.actual_home.append(actual_home_goals)
            self.actual_away.append(actual_away_goals)
            
            # Determine actual outcome
            if actual_home_goals > actual_away_goals:
                actual_outcome = 2  # Home win
            elif actual_away_goals > actual_home_goals:
                actual_outcome = 0  # Away win
            else:
                actual_outcome = 1  # Draw
            
            self.outcomes_actual.append(actual_outcome)
            
            # Predicted outcome (argmax)
            probs = [prob_away, prob_draw, prob_home]
            predicted_outcome = np.argmax(probs)
            self.outcomes_predicted.append(predicted_outcome)
            
            # Log loss contribution
            prob_actual = probs[actual_outcome]
            log_loss_val = -np.log(max(prob_actual, 1e-15))  # Avoid log(0)
            self.log_loss_contrib.append(log_loss_val)
    
    def get_xgboost_usage(self) -> float:
        """
        Calculate XGBoost usage percentage.
        
        Returns:
            Percentage of predictions using XGBoost
        """
        if not self.sources:
            return 0.0
        
        xgb_count = sum(1 for s in self.sources if s == 'xgboost')
        return (xgb_count / len(self.sources)) * 100
    
    def get_average_confidence(self) -> float:
        """
        Calculate average confidence.
        
        Returns:
            Average confidence across window
        """
        if not self.confidences:
            return 0.0
        return np.mean(self.confidences)
    
    def get_goal_mae(self) -> Optional[Dict[str, float]]:
        """
        Calculate goal MAE.
        
        Returns:
            Dictionary with home, away, and average MAE, or None if no data
        """
        if not self.actual_home or not self.actual_away:
            return None
        
        mae_home = np.mean([abs(a - p) for a, p in zip(self.actual_home, self.xg_home[-len(self.actual_home):])])
        mae_away = np.mean([abs(a - p) for a, p in zip(self.actual_away, self.xg_away[-len(self.actual_away):])])
        
        return {
            'home': mae_home,
            'away': mae_away,
            'average': (mae_home + mae_away) / 2
        }
    
    def get_match_accuracy(self) -> Optional[float]:
        """
        Calculate match outcome accuracy.
        
        Returns:
            Accuracy percentage, or None if no data
        """
        if not self.outcomes_actual or not self.outcomes_predicted:
            return None
        
        correct = sum(1 for a, p in zip(self.outcomes_actual, self.outcomes_predicted) if a == p)
        return (correct / len(self.outcomes_actual)) * 100
    
    def get_log_loss(self) -> Optional[float]:
        """
        Calculate log loss.
        
        Returns:
            Log loss value, or None if no data
        """
        if not self.log_loss_contrib:
            return None
        
        return np.mean(self.log_loss_contrib)
    
    def get_all_stats(self) -> Dict:
        """
        Get all statistics as a dictionary.
        
        Returns:
            Dictionary with all current statistics
        """
        mae = self.get_goal_mae()
        accuracy = self.get_match_accuracy()
        ll = self.get_log_loss()
        
        stats = {
            'window_size': len(self.sources),
            'max_window': self.window_size,
            'xgboost_usage': round(self.get_xgboost_usage(), 1),
            'average_confidence': round(self.get_average_confidence(), 1)
        }
        
        if mae:
            stats['goal_mae'] = {
                'home': round(mae['home'], 4),
                'away': round(mae['away'], 4),
                'average': round(mae['average'], 4)
            }
        
        if accuracy is not None:
            stats['match_accuracy'] = round(accuracy, 2)
        
        if ll is not None:
            stats['log_loss'] = round(ll, 4)
        
        return stats
    
    def print_stats(self):
        """Print current statistics to console."""
        stats = self.get_all_stats()
        
        print(f"\n{'='*60}")
        print(f"ROLLING STATISTICS (Window: {stats['window_size']}/{stats['max_window']})")
        print(f"{'='*60}")
        print(f"XGBoost Usage: {stats['xgboost_usage']}%")
        print(f"Average Confidence: {stats['average_confidence']}%")
        
        if 'goal_mae' in stats:
            print(f"\nGoal MAE: {stats['goal_mae']['average']}")
            print(f"  Home: {stats['goal_mae']['home']}")
            print(f"  Away: {stats['goal_mae']['away']}")
        
        if 'match_accuracy' in stats:
            print(f"\nMatch Accuracy: {stats['match_accuracy']}%")
        
        if 'log_loss' in stats:
            print(f"Log Loss: {stats['log_loss']}")
        
        print(f"{'='*60}\n")


# Example usage
if __name__ == '__main__':
    tracker = RollingStats(window_size=50)
    
    # Simulate some predictions
    print("Simulating predictions...")
    for i in range(50):
        source = 'xgboost' if i % 4 != 0 else 'poisson'  # 75% XGBoost
        confidence = np.random.uniform(20, 35) if source == 'xgboost' else 100
        xg_h = np.random.uniform(0.8, 2.5)
        xg_a = np.random.uniform(0.8, 2.5)
        
        # Simulate actual result (with some noise from prediction)
        actual_h = max(0, int(xg_h + np.random.normal(0, 0.8)))
        actual_a = max(0, int(xg_a + np.random.normal(0, 0.8)))
        
        # Probabilities based on xG (simplified)
        total_xg = xg_h + xg_a
        prob_h = min(0.7, xg_h / max(total_xg, 0.1))
        prob_a = min(0.7, xg_a / max(total_xg, 0.1))
        prob_d = 1 - prob_h - prob_a
        
        tracker.add_prediction(
            source=source,
            confidence=confidence,
            xg_home=xg_h,
            xg_away=xg_a,
            prob_home=prob_h,
            prob_draw=prob_d,
            prob_away=prob_a,
            actual_home_goals=actual_h,
            actual_away_goals=actual_a
        )
    
    # Print statistics
    tracker.print_stats()
    
    print("✅ Rolling statistics test completed")
