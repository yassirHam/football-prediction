# Post-Deployment Operations Guide

## Executive Summary

This document explains the operational features added to the hybrid prediction system **without modifying the prediction models themselves**. These features focus on monitoring, decision quality, and drift detection to make the system more trustworthy and controllable in production.

---

## Why Models Are Frozen 🔒

### The Stability Principle

Once a prediction model is validated and deployed, **changing it introduces risk**:

1. **Regression Risk**: New features or retraining may degrade performance
2. **Reproducibility**: Frozen models ensure consistent predictions
3. **Trust**: Stakeholders can rely on documented, tested behavior
4. **Debugging**: Issues are easier to trace when the core logic is stable

### What's Frozen

❌ **Model architecture** (XGBoost + Poisson hybrid)  
❌ **Feature engineering** (rolling windows, home/away splits)  
❌ **Confidence thresholds** (25% for XGBoost usage)  
❌ **Training data processing** (unless retraining is triggered)

### What's Not Frozen

✅ **Monitoring and logging**  
✅ **Decision filters**  
✅ **Lineup adjustments (post-prediction)**  
✅ **Alert thresholds**  
✅ **Configuration parameters**

---

## Monitoring Infrastructure 📊

### Components

| Module | Purpose | Output |
|--------|---------|--------|
| `runtime_logger.py` | Logs every prediction | JSON + CSV files |
| `metrics_aggregator.py` | Calculates rolling statistics | Console + JSON summaries |
| `rolling_stats.py` | Real-time performance tracking | In-memory windows |
| `drift_detector.py` | Detects performance degradation | Alerts + logs |

### What Gets Monitored

**Per Prediction:**
- Prediction source (XGBoost or Poisson)
- Confidence score
- Predicted xG (home/away)
- Match outcome probabilities
- Actual result (when available)
- Goal prediction error
- Classification accuracy

**Aggregated (50/100-match windows):**
- % XGBoost usage
- Average confidence
- Goal MAE
- Match outcome accuracy
- Log loss

### Usage Example

```python
from monitoring.runtime_logger import RuntimeLogger
from monitoring.drift_detector import DriftDetector

# Log predictions
logger = RuntimeLogger()
logger.log_prediction(
    home_team="Team A",
    away_team="Team B",
    prediction=prediction_dict,
    actual_result={'home_goals': 2, 'away_goals': 1}
)

# Check drift
detector = DriftDetector()
detector.print_drift_report(days=14)
```

### Accessing Logs

- **JSON logs**: `logs/predictions.jsonl` (detailed, for analysis)
- **CSV summary**: `logs/predictions_summary.csv` (quick review)  
- **Metrics**: `logs/metrics_summary.json` (aggregated stats)
- **Alerts**: `logs/alerts.jsonl` (drift warnings)

---

## Decision Filters 🎯

### Purpose

**Decision filters don't block predictions** - they flag them as:
- 🟢 **STRONG**: High confidence, act on this
- 🟡 **PASS**: Neutral, monitor
- 🔴 **WEAK**: Low confidence, skip

### Available Filters

#### 1. Confidence Filter

**Threshold**: 30% (configurable)

```python
if confidence >= 40:  # 30 + 10
    return 'STRONG'
elif confidence >= 30:
    return 'PASS'
else:
    return 'WEAK'
```

#### 2. Goals Distance Filter

**Threshold**: 0.4 goals from league average

```python
distance = abs(predicted_total_goals - league_avg)
if distance >= 0.6:  # 0.4 * 1.5
    return 'STRONG'  # Unusual match
```

#### 3. Odds Comparison (Optional)

**Threshold**: 5% value margin

```python
if predicted_prob > implied_prob + 0.05:
    return 'STRONG'  # Value bet
```

### Usage

```python
from decision_engine.signal_generator import SignalGenerator

generator = SignalGenerator()
signal = generator.generate_signal(prediction, league='E0')

generator.print_signal(signal, "Home Team", "Away Team")

# Check signal
if signal['signal'] == 'STRONG':
    print("High conviction prediction!")
    for rec in signal['recommendations']:
        print(f"  • {rec}")
```

### Configuration

Edit `decision_engine/filter_config.json`:

```json
{
  "confidence_threshold": 30,
  "goals_distance_threshold": 0.4,
  "league_averages": {
    "E0": 2.7,
    "SP1": 2.6,
    "D1": 2.9
  }
}
```

---

## Lineup Adjustments ⚽

### Rule-Based Post-Prediction Modifications

**IMPORTANT**: These adjustments are applied **AFTER** the hybrid prediction but **BEFORE** final Poisson probability calculations.

### Adjustment Rules

| Absence Type | xG Impact | Example |
|-------------|-----------|---------|
| Key Attacker | -0.15 xG | Missing top scorer |
| Key Defender | +0.12 conceded xG | Opponent scores more |
| Goalkeeper | +0.20 conceded xG | Backup keeper |
| Multiple absences | 1.5x multiplier | Compounding effect |
| **Maximum adjustment** | **±0.3 xG** | Safety cap |

### Transparency

All adjustments are:
- ✅ **Logged with reasoning**
- ✅ **Reversible** (original xG preserved)
- ✅ **Capped** to prevent excessive changes

### Usage

```python
from adjustments.lineup_adjuster import LineupAdjuster

adjuster = LineupAdjuster()

adjusted_xg_home, adjusted_xg_away, details = adjuster.adjust_for_absences(
    xg_home=1.8,
    xg_away=1.2,
    home_absences=['key_attacker', 'key_defender'],
    away_absences=[]
)

adjuster.print_adjustments(details)
# Shows: Original xG: 1.80 → Final xG: 1.49 (-0.31, capped at -0.30)
```

### Configuration

Edit `adjustments/rules.json`:

```json
{
  "adjustments": {
    "key_attacker_missing": -0.15,
    "key_defender_missing": 0.12,
    "max_adjustment_per_team": 0.3
  }
}
```

---

## Drift Detection & Retraining Alerts 🚨

### What is Drift?

**Drift** = degradation in model performance over time due to:
- Changing team tactics
- New players/transfers
- Rule changes
- Seasonal variations

### Detection Criteria

The system alerts if:

| Metric | Threshold | Action |
|--------|-----------|--------|
| **Rolling MAE** | >5% worse than baseline | 🔴 HIGH alert |
| **Fallback rate** | >35% Poisson usage | 🟡 MEDIUM alert |
| **Average confidence** | <22% | 🟡 MEDIUM alert |

### Monitoring Window

- **Default**: 14 days
- **Minimum data**: 10 predictions to trigger analysis

### Alert Outputs

1. **Console warning**: Immediate visibility
2. **Log entry**: `logs/alerts.jsonl`
3. **Email/webhook**: (Stub - integrate as needed)

### Usage

```python
from monitoring.drift_detector import DriftDetector

detector = DriftDetector(
    baseline_mae=0.97,
    baseline_confidence=27.0
)

# Check for drift
result = detector.check_for_drift(days=14)

if result['status'] == 'HIGH':
    print("⚠️ Performance degradation detected!")
    for alert in result['alerts']:
        print(f"  • {alert['recommendation']}")
```

### When to Retrain

✅ **Retrain if**:
- Rolling MAE increases >5% for 14+ days
- Multiple HIGH alerts in a month
- New season starts (strategic timing)

❌ **Don't retrain if**:
- Single bad prediction
- Short-term variance (<7 days)
- Alerts are MEDIUM with stable MAE

---

## Complete Workflow Example 🔄

### 1. Make Prediction

```python
from hybrid.hybrid_predictor import predict_match_hybrid, Team
from monitoring.runtime_logger import RuntimeLogger
from decision_engine.signal_generator import SignalGenerator
from adjustments.lineup_adjuster import LineupAdjuster

# Create teams
home = Team("Man City", [3,2,4], [1,0,1], [2,1,2], league="E0")
away = Team("Liverpool", [2,3,1], [1,2,0], [1,2,0], league="E0")

# Get prediction
prediction = predict_match_hybrid(home, away)
```

### 2. Apply Lineup Adjustments (if needed)

```python
adjuster = LineupAdjuster()

# Adjust for missing attacker
xg_h, xg_a, adj_details = adjuster.adjust_for_absences(
    xg_home=prediction['expected_goals']['home'],
    xg_away=prediction['expected_goals']['away'],
    home_absences=['key_attacker']
)

# Update prediction with adjusted xG
from football_predictor import predict_score_probabilities
adjusted_probs = predict_score_probabilities(xg_h, xg_a)
# ... recalculate outcome probabilities
```

### 3. Generate Decision Signal

```python
generator = SignalGenerator()
signal = generator.generate_signal(prediction, league='E0')

print(f"Signal: {signal['signal']}")
print(f"Recommendations: {signal['recommendations']}")
```

### 4. Log Prediction

```python
logger = RuntimeLogger()
logger.log_prediction(
    home_team="Man City",
    away_team="Liverpool",
    prediction=prediction,
    actual_result=None  # Add later when match is played
)
```

### 5. Monitor for Drift (Periodic)

```python
# Run daily or weekly
from monitoring.drift_detector import DriftDetector

detector = DriftDetector()
detector.print_drift_report(days=14)
```

---

## Maintenance Schedule 📅

### Daily
- ✅ Check console for drift warnings
- ✅ Review prediction logs for anomalies

### Weekly
- ✅ Run `metrics_aggregator.print_metrics(100)`
- ✅ Review decision signal distribution
- ✅ Check `logs/alerts.jsonl` for patterns

### Monthly
- ✅ Full drift analysis (`drift_detector.py`)
- ✅ Review filter thresholds
- ✅ Update lineup adjustment rules if needed

### Quarterly
- ⚠️ **Retraining evaluation** (if drift detected)
- ⚠️ Recalibrate league averages
- ⚠️ Update configuration parameters

---

## Configuration Reference 📝

### Key Files

| File | Purpose |
|------|---------|
| `hybrid_config.json` | ML enable/disable, confidence threshold |
| `decision_engine/filter_config.json` | Decision filter thresholds |
| `adjustments/rules.json` | Lineup adjustment rules |

### Safe to Modify

✅ Filter thresholds (confidence, goals distance)  
✅ League averages for decision filters  
✅ Lineup adjustment values (within ±0.3 cap)  
✅ Alert thresholds for drift detection  

### Do NOT Modify

❌ Model architecture in `hybrid_predictor.py`  
❌ Feature builder in `ml_models/feature_builder.py`  
❌ XGBoost training logic (unless retraining)  
❌ Confidence gate logic in `hybrid/confidence_gate.py`

---

## Integration with Web App 🌐

### Automatic Monitoring

To enable automatic logging in the web app, add to `app.py`:

```python
from monitoring.runtime_logger import RuntimeLogger

logger = RuntimeLogger()

# In prediction endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    # ... get prediction ...
    prediction = predict_match_hybrid(home_team, away_team)
    
    # Log prediction
    logger.log_prediction(
        home_team=data['home_team']['name'],
        away_team=data['away_team']['name'],
        prediction=prediction
    )
    
    return jsonify(prediction)
```

### Adding Decision Signals to API

```python
from decision_engine.signal_generator import SignalGenerator

generator = SignalGenerator()

# In prediction endpoint
signal = generator.generate_signal(prediction, league='E0')

response = {
    **prediction,  # Original prediction
    'decision_signal': signature['signal'],
    'signal_details': signal
}
```

---

## Troubleshooting 🔧

### Issue: High Fallback Rate (>35%)

**Causes:**
1. Confidence threshold too high
2. Recent data shift
3. Model needs retraining

**Solutions:**
1. Lower `confidence_threshold` in `hybrid_config.json` (e.g., 25 → 20)
2. Check recent predictions for patterns
3. Run drift detector to confirm degradation

### Issue: Decision Filters Too Strict/Lenient

**Symptom:** Too many WEAK or too many STRONG signals

**Solution:**
Adjust `decision_engine/filter_config.json`:

```json
{
  "confidence_threshold": 35,  // Raise for stricter
  "goals_distance_threshold": 0.3  // Lower for more signals
}
```

### Issue: Lineup Adjustments Too Aggressive

**Symptom:** Adjusted xG always hits ±0.3 cap

**Solution:**
Reduce adjustment values in `adjustments/rules.json`:

```json
{
  "key_attacker_missing": -0.10,  // Was -0.15
  "key_defender_missing": 0.08    // Was 0.12
}
```

---

## FAQs ❓

### Why not retrain automatically when drift is detected?

**Answer**: Retraining should be a deliberate decision. Automatic retraining risks:
- Model degradation without human review
- Computational cost at inappropriate times
- Loss of reproducibility

The system **alerts** when retraining may be needed, but a human decides **when** and **how**.

### Can I use the decision filters to block predictions?

**Answer**: No. Filters are for **decision support**, not gatekeeping. All predictions are valid - filters just flag confidence level.

### What if I want to experiment with different confidence thresholds?

**Answer**: Safe to modify `hybrid_config.json` - this doesn't change the model, just when XGBoost vs Poisson is used.

### How do lineup adjustments affect probabilities?

**Answer**: Adjustments modify xG values, then Poisson recalculates probabilities. The Poisson logic itself is unchanged.

---

## Summary Checklist ✅

Before deploying to production with post-deployment features:

- [ ] Monitoring is enabled (`RuntimeLogger` integrated)
- [ ] Baseline metrics are set (`DriftDetector` configured)
- [ ] Decision filter thresholds are appropriate for your use case
- [ ] Lineup adjustment rules are reviewed and approved
- [ ] Alert destinations are configured (console/log/email)
- [ ] Maintenance schedule is documented
- [ ] Team understands what triggers retraining

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-24  
**Status:** Production Ready

**Remember**: The goal is **operational excellence**, not model perfection. These tools make the system observable, debuggable, and controllable without risking the core prediction quality.
