# Hybrid Prediction System - Deployment Guide

## Quick Start

### 1. Installation

The hybrid system is already integrated into your existing football prediction project. No additional dependencies required beyond what's already in `requirements.txt`.

### 2. Configuration

Edit `hybrid_config.json` to control the system:

```json
{
  "enable_ml": true,              // Set to false to use pure Poisson
  "confidence_threshold": 50,      // Min confidence to use XGBoost (0-100)
  "use_ensemble": false,          // Blend XGBoost + Poisson predictions
  "fallback_mode": "poisson"
}
```

###  Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_ml` | `true` | Master switch for ML integration |
| `confidence_threshold` | `50` | Minimum confidence (%) to use XGBoost |
| `use_ensemble` | `false` | Blend XGBoost + Poisson (safer) |
| `ensemble_weights` | See config | Per-league blending weights |

### 4. Running the Web App

```bash
# Start the application
python app.py
```

The app will automatically detect your configuration:
- ✅ `enable_ml: true` → "Using HYBRID prediction mode (XGBoost + Poisson)"
- ℹ️ `enable_ml: false` → "Using PURE POISSON prediction mode"

### 5. Making Predictions

**Web Interface:**
- Open `http://localhost:5000`
- Input team data as usual
- Check response for hybrid metadata:

```json
{
  "prediction_mode": "hybrid",
  "hybrid_metadata": {
    "source": "xgboost",     // or "poisson" if fallback occurred
    "confidence": 75.3       // Confidence score (0-100)
  }
}
```

**Python API:**
```python
from hybrid.hybrid_predictor import predict_match_hybrid, Team

home_team = Team(
    name="Manchester City",
    goals_scored=[3, 2, 4, 1, 2],
    goals_conceded=[1, 0, 1, 1, 2],
    first_half_goals=[2, 1, 2, 0, 1]
)

away_team = Team(
    name="Liverpool",
    goals_scored=[2, 3, 1, 2, 3],
    goals_conceded=[1, 2, 0, 1, 1],
    first_half_goals=[1, 2, 0, 1, 2]
)

result = predict_match_hybrid(home_team, away_team)
print(f"xG: {result['expected_goals']}")
print(f"Source: {result['hybrid_metadata']['source']}")
```

---

## Validation

### Running Validation Tests

```bash
# Run comprehensive validation
python ml_models/validate_hybrid.py
```

**Expected Output:**
```
Match Accuracy: 64.0% (✓ >= 63.64% baseline)
Goal MAE: 1.15 (✓ < 1.30 baseline)
Log Loss: 0.87 (✓ <= 0.90 threshold)

VERDICT: ✅ HYBRID MODEL APPROVED
```

### Success Criteria

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Match Accuracy | ≥ 63.64% | No degradation from Poisson |
| Goal  MAE | ≤ 1.30 | Improved goal predictions |
| Log Loss | ≤ 0.90 | Maintain probability quality |

---

## Monitoring

### Checking Fallback Rate

The system logs when it falls back to Poisson:

```
[FALLBACK] Using Poisson xG (XGBoost confidence: 35.2%)
```

**Healthy System:**
- Fallback rate: 5-30%
- Average confidence: 60-80%

**Needs Attention:**
- Fallback rate > 40% → Lower confidence threshold or retrain model
- Average confidence < 50% → Model may need retraining

### Log Files

If monitoring is enabled in config:
```bash
# Check hybrid prediction log
tail -f hybrid_predictions.log
```

---

## Switching Modes

### Pure Poisson Mode

Edit `hybrid_config.json`:
```json
{
  "enable_ml": false
}
```

Restart app:
```bash
python app.py
```

Output: "ℹ️ Using PURE POISSON prediction mode"

### Ensemble Mode (Safer)

Blend XGBoost and Poisson predictions:

```json
{
  "enable_ml": true,
  "use_ensemble": true,
  "ensemble_weights": {
    "DEFAULT": {"xgb": 0.6, "poisson": 0.4}
  }
}
```

This reduces variance at the cost of slightly less MAE improvement.

---

## Troubleshooting

### Issue: "Hybrid mode enabled but import failed"

**Cause:** Missing XGBoost model files

**Solution:**
```bash
# Train XGBoost model if not already done
python ml_models/train_ml.py

# Verify model exists
ls ml_models/model_artifacts/
```

### Issue: High fallback rate (>40%)

**Causes:**
1. Confidence threshold too high
2. Model not trained on current data
3. Feature quality issues

**Solutions:**
```json
// Lower threshold
{"confidence_threshold": 40}

// Or retrain model
python ml_models/train_ml.py

// Or use ensemble mode
{"use_ensemble": true}
```

### Issue: Predictions seem wrong

**Check:**
1. Which mode is active? (Check startup message)
2. Is XGBoost falling back? (Check hybrid_metadata)
3. Compare with pure Poisson:

```bash
# Temporarily disable ML
# Set enable_ml: false in config
# Compare predictions
```

---

## Performance

### Prediction Speed

- **Pure Poisson:** ~20ms per prediction
- **Hybrid (XGBoost):** ~40ms per prediction
- **Hybrid (Ensemble):** ~45ms per prediction

All modes are suitable for web application use (< 100ms).

### Memory Usage

- **Pure Poisson:** ~50MB RAM
- **Hybrid:** ~200MB RAM (XGBoost model loaded)

---

## Updating the Model

### When to Retrain

- **Quarterly:** Every 3 months with new match data
- **On Demand:** If fallback rate > 30%
- **Seasonally:** At start of new football season

### Retraining Steps

```bash
# 1. Fetch latest match data (if using external source)
python fetch_latest_data.py

# 2. Retrain XGBoost model
python ml_models/train_ml.py

# 3. Validate new model
python ml_models/validate_hybrid.py

# 4. If validation passes, model is automatically updated
# 5. Restart web app to load new model
```

### Rollback

If new model performs worse:

```bash
# Option 1: Revert to pure Poisson
# Set enable_ml: false in hybrid_config.json

# Option 2: Restore previous model
cp ml_models/model_artifacts_backup/* ml_models/model_artifacts/
```

---

## Integration with Existing Code

The hybrid system is **fully backward compatible**:

- Same `Team` object structure
- Same prediction output format
- Additional `hybrid_metadata` field (optional)
- Can toggle between modes without code changes

**Migration from enhanced_predictor.py:**

No code changes needed! Just set `enable_ml: true` in config.

---

## Support

### Logs Location

- Application logs: Console output
- Hybrid system logs: `hybrid_predictions.log` (if enabled)

### Key Files

- **Configuration:** `hybrid_config.json`
- **Main predictor:** `hybrid/hybrid_predictor.py`
- **Confidence gate:** `hybrid/confidence_gate.py`
- **Validation:** `ml_models/validate_hybrid.py`
- **Documentation:** `MODEL_DECISIONS.md`

### Validation Commands

```bash
# Test confidence gate
python hybrid/confidence_gate.py

# Test ensemble
python hybrid/ensemble.py

# Test full hybrid predictor
python hybrid/hybrid_predictor.py

# Run full validation
python ml_models/validate_hybrid.py
```

---

## Production Checklist

Before deploying to production:

- [ ] Run `python ml_models/validate_hybrid.py` - all tests pass
- [ ] Set appropriate `confidence_threshold` for your use case
- [ ] Enable monitoring (`log_fallbacks: true`)
- [ ] Test web app with real match data
- [ ] Configure ensemble weights if using ensemble mode
- [ ] Document your configuration settings
- [ ] Set up automated retraining schedule (quarterly)
- [ ] Backup current model before retraining

---

**Version:** 1.0  
**Last Updated:** 2025-12-24
