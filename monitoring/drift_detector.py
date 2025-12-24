"""
Drift Detector for Hybrid Prediction System
============================================

Detects performance degradation and triggers retraining alerts.
Monitors rolling MAE, fallback rate, and confidence over time.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque


class DriftDetector:
    """
    Detects performance drift and triggers alerts.
    
    Monitors:
    - Rolling MAE (14-day window) for > 5% degradation
    - XGBoost fallback rate > 35%
    - Average confidence < 22%
    """
    
    def __init__(self, 
                 log_file: str = 'logs/predictions.jsonl',
                 baseline_mae: float = 0.97,
                 baseline_confidence: float = 27.0):
        """
        Initialize drift detector.
        
        Args:
            log_file: Path to predictions log
            baseline_mae: Baseline MAE from initial validation
            baseline_confidence: Baseline confidence from initial validation
        """
        self.log_file = Path(log_file)
        self.baseline_mae = baseline_mae
        self.baseline_confidence = baseline_confidence
        
        # Thresholds
        self.mae_degradation_threshold = 0.05  # 5% worse
        self.fallback_rate_threshold = 0.35    # 35%
        self.confidence_threshold = 22.0       # Below 22%
        
        # Alert tracking
        self.alerts = []
    
    def load_recent_predictions(self, days: int = 14) -> List[Dict]:
        """
        Load predictions from the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of prediction dictionaries
        """
        if not self.log_file.exists():
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_predictions = []
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                pred = json.loads(line)
                pred_date = datetime.fromisoformat(pred['timestamp'])
                if pred_date >= cutoff_date:
                    recent_predictions.append(pred)
        
        return recent_predictions
    
    def calculate_rolling_mae(self, predictions: List[Dict]) -> Optional[float]:
        """
        Calculate MAE from evaluated predictions.
        
        Args:
            predictions: List of predictions
            
        Returns:
            MAE value or None if insufficient data
        """
        evaluated = [p for p in predictions if 'evaluation' in p]
        
        if not evaluated:
            return None
        
        goal_errors = [p['evaluation']['goal_error'] for p in evaluated]
        return np.mean(goal_errors)
    
    def calculate_fallback_rate(self, predictions: List[Dict]) -> float:
        """
        Calculate XGBoost fallback rate.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Fallback rate (0-1)
        """
        if not predictions:
            return 0.0
        
        poisson_count = sum(1 for p in predictions if p['prediction']['source'] == 'poisson')
        return poisson_count / len(predictions)
    
    def calculate_average_confidence(self, predictions: List[Dict]) -> float:
        """
        Calculate average confidence.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Average confidence
        """
        if not predictions:
            return 0.0
        
        confidences = [p['prediction']['confidence'] for p in predictions]
        return np.mean(confidences)
    
    def check_for_drift(self, days: int = 14) -> Dict:
        """
        Check for performance drift.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with drift status and details
        """
        predictions = self.load_recent_predictions(days)
        
        if len(predictions) < 10:
            return {
                'status': 'INSUFFICIENT_DATA',
                'message': f'Only {len(predictions)} predictions in last {days} days',
                'alerts': []
            }
        
        # Calculate metrics
        current_mae = self.calculate_rolling_mae(predictions)
        fallback_rate = self.calculate_fallback_rate(predictions)
        avg_confidence = self.calculate_average_confidence(predictions)
        
        # Check for drift
        alerts = []
        alert_level = 'OK'
        
        # 1. MAE degradation
        if current_mae is not None:
            mae_change = (current_mae - self.baseline_mae) / self.baseline_mae
            if mae_change > self.mae_degradation_threshold:
                alerts.append({
                    'type': 'MAE_DEGRADATION',
                    'severity': 'HIGH',
                    'message': f'MAE increased {mae_change*100:.1f}% (baseline: {self.baseline_mae:.4f}, current: {current_mae:.4f})',
                    'recommendation': 'Consider retraining XGBoost model'
                })
                alert_level = 'HIGH'
        
        # 2. High fallback rate
        if fallback_rate > self.fallback_rate_threshold:
            alerts.append({
                'type': 'HIGH_FALLBACK_RATE',
                'severity': 'MEDIUM',
                'message': f'XGBoost fallback rate: {fallback_rate*100:.1f}% (threshold: {self.fallback_rate_threshold*100:.0f}%)',
                'recommendation': 'Lower confidence threshold or retrain model'
            })
            if alert_level == 'OK':
                alert_level = 'MEDIUM'
        
        # 3. Low confidence
        if avg_confidence < self.confidence_threshold:
            alerts.append({
                'type': 'LOW_CONFIDENCE',
                'severity': 'MEDIUM',
                'message': f'Average confidence: {avg_confidence:.1f}% (threshold: {self.confidence_threshold:.0f}%)',
                'recommendation': 'Review model or lower confidence threshold'
            })
            if alert_level == 'OK':
                alert_level = 'MEDIUM'
        
        # Store alerts
        self.alerts.extend(alerts)
        
        return {
            'status': alert_level,
            'predictions_analyzed': len(predictions),
            'evaluated_predictions': sum(1 for p in predictions if 'evaluation' in p),
            'metrics': {
                'mae': current_mae,
                'fallback_rate': round(fallback_rate * 100, 1),
                'confidence': round(avg_confidence, 1)
            },
            'baselines': {
                'mae': self.baseline_mae,
                'confidence': self.baseline_confidence
            },
            'alerts': alerts
        }
    
    def print_drift_report(self, days: int = 14):
        """
        Print drift detection report.
        
        Args:
            days: Number of days to analyze
        """
        result = self.check_for_drift(days)
        
        print(f"\n{'='*70}")
        print(f"DRIFT DETECTION REPORT ({days}-day window)")
        print(f"{'='*70}")
        
        status_emoji = '🟢' if result['status'] == 'OK' else '🟡' if result['status'] == 'MEDIUM' else '🔴'
        print(f"\nStatus: {status_emoji} {result['status']}")
        print(f"Predictions Analyzed: {result['predictions_analyzed']}")
        print(f"Evaluated: {result['evaluated_predictions']}")
        
        if 'metrics' in result:
            metrics = result['metrics']
            baselines = result['baselines']
            
            print(f"\nCurrent Metrics:")
            if metrics['mae'] is not None:
                mae_change = ((metrics['mae'] - baselines['mae']) / baselines['mae'] * 100)
                print(f"  MAE: {metrics['mae']:.4f} (baseline: {baselines['mae']:.4f}, {mae_change:+.1f}%)")
            print(f"  Fallback Rate: {metrics['fallback_rate']}%")
            conf_change = metrics['confidence'] - baselines['confidence']
            print(f"  Confidence: {metrics['confidence']}% (baseline: {baselines['confidence']:.1f}%, {conf_change:+.1f}%)")
        
        if result.get('alerts'):
            print(f"\n⚠️  ALERTS ({len(result['alerts'])}):")
            for alert in result['alerts']:
                severity_emoji = '🔴' if alert['severity'] == 'HIGH' else '🟡'
                print(f"\n{severity_emoji} {alert['type']} ({alert['severity']})")
                print(f"  {alert['message']}")
                print(f"  → {alert['recommendation']}")
        else:
            print(f"\n✅ No alerts - system performing normally")
        
        print(f"\n{'='*70}\n")
    
    def send_alert(self, alert: Dict):
        """
        Send alert (stub for email/webhook integration).
        
        Args:
            alert: Alert dictionary
        """
        # Console warning
        print(f"\n⚠️  ALERT: {alert['type']} - {alert['message']}")
        
        # Log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'alert': alert
        }
        
        alert_log = Path('logs/alerts.jsonl')
        alert_log.parent.mkdir(exist_ok=True)
        
        with open(alert_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # TODO: Email/webhook integration
        # send_email(alert)
        # send_webhook(alert)


# Example usage
if __name__ == '__main__':
    detector = DriftDetector()
    
    # Check for drift
    detector.print_drift_report(days=14)
    
    print("✅ Drift detector test completed")
