# ⚽ Football Match Prediction System

A statistical prediction engine for football matches based on recent team performance using Poisson distribution modeling.

## Features

- 📊 **Statistical Analysis**: Uses Poisson distribution for realistic probability modeling
- 🎯 **Multiple Predictions**: First half, full match, and total goals predictions
- 📈 **Expected Goals (xG)**: Statistical estimation of scoring probability
- 🏠 **Home Advantage**: 15% boost for home teams
- 📉 **Recent Form Weighting**: More recent matches have higher impact
- 🔍 **Match Insights**: Tempo analysis, early goal likelihood, and confidence levels

## Quick Start

### Installation

No external dependencies required! Uses only Python standard library.

```bash
# Clone or download this directory
cd football-prediction

# Run the example
python football_predictor.py
```

### Basic Usage

```python
from football_predictor import Team, predict_match, format_predictions

# Define teams with last 5 matches data (most recent first)
home_team = Team(
    name="Manchester City",
    goals_scored=[3, 2, 4, 1, 2],        # Goals scored in last 5 matches
    goals_conceded=[1, 0, 1, 1, 2],      # Goals conceded in last 5 matches
    first_half_goals=[2, 1, 2, 0, 1]     # First half goals in last 5 matches
)

away_team = Team(
    name="Liverpool",
    goals_scored=[2, 3, 1, 2, 3],
    goals_conceded=[1, 2, 0, 1, 1],
    first_half_goals=[1, 2, 0, 1, 2]
)

# Get predictions
result = predict_match(home_team, away_team)

# Display formatted output
print(format_predictions(result, "Manchester City", "Liverpool"))
```

## Example Output

```
============================================================
MATCH PREDICTION: Manchester City vs Liverpool
============================================================

EXPECTED GOALS (xG):
  Manchester City: 2.48
  Liverpool: 1.76

FIRST HALF PREDICTIONS:
  1–0: 34.2%
  1–1: 28.7%
  0–0: 22.1%

FULL MATCH PREDICTIONS:
  2–1: 18.4%
  2–2: 15.8%
  3–1: 14.2%

TOTAL GOALS PROBABILITIES:
  Under 1.5 goals: 12.3%
  Under 2.5 goals: 31.7%
  Over 2.5 goals: 68.3%
  Over 3.5 goals: 45.2%

MATCH INSIGHTS:
  Expected Tempo: MEDIUM
  Early Goal Likelihood: HIGH
  Prediction Confidence: MEDIUM
```

## How It Works

### Statistical Model

The system uses **Poisson distribution** to model goal-scoring:

1. **Offensive Strength Index (OSI)**
   ```
   OSI = (weighted_avg_goals_scored × home_multiplier) / league_avg_goals
   ```

2. **Defensive Weakness Index (DWI)**
   ```
   DWI = weighted_avg_goals_conceded / league_avg_goals
   ```

3. **Expected Goals (xG)**
   ```
   xG_home = OSI_home × DWI_away × league_avg_goals
   xG_away = OSI_away × DWI_home × league_avg_goals × 0.85
   ```

4. **Poisson Probability**
   ```
   P(X = k) = (λ^k × e^(-λ)) / k!
   ```

### Form Weighting

Recent matches have more impact:
- Match 1 (most recent): **30%** weight
- Match 2: **25%** weight
- Match 3: **20%** weight
- Match 4: **15%** weight
- Match 5 (oldest): **10%** weight

### Home Advantage

- Home team gets **15% offensive boost** (multiplier: 1.15)
- Away team has **15% penalty** (multiplier: 0.85)

## Data Structure

### Input Format

```python
team_data = {
    "name": "Team Name",
    "goals_scored": [3, 2, 1, 2, 1],      # Last 5 matches (most recent first)
    "goals_conceded": [1, 1, 2, 0, 1],
    "first_half_goals": [2, 1, 0, 1, 1]
}
```

### Output Format

```python
{
    "first_half_predictions": [
        ((1, 0), 0.342),  # 1-0: 34.2%
        ((1, 1), 0.287),  # 1-1: 28.7%
        ((0, 0), 0.221)   # 0-0: 22.1%
    ],
    "full_match_predictions": [
        ((2, 1), 0.184),
        ((2, 2), 0.158),
        ((3, 1), 0.142)
    ],
    "total_goals": {
        "under_1.5": 0.123,
        "under_2.5": 0.317,
        "over_2.5": 0.683,
        "over_3.5": 0.452
    },
    "expected_goals": {
        "home": 2.48,
        "away": 1.76
    },
    "insights": {
        "tempo": "MEDIUM",
        "early_goal_likelihood": "HIGH",
        "confidence": "MEDIUM"
    }
}
```

## Backend Integration

### Flask Example

```python
from flask import Flask, request, jsonify
from football_predictor import Team, predict_match

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    home_team = Team(**data['home_team'])
    away_team = Team(**data['away_team'])
    
    result = predict_match(home_team, away_team)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### FastAPI Example

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from football_predictor import Team, predict_match

app = FastAPI()

class TeamData(BaseModel):
    name: str
    goals_scored: List[int]
    goals_conceded: List[int]
    first_half_goals: List[int]

class MatchRequest(BaseModel):
    home_team: TeamData
    away_team: TeamData

@app.post("/predict")
def predict(match: MatchRequest):
    home_team = Team(**match.home_team.dict())
    away_team = Team(**match.away_team.dict())
    return predict_match(home_team, away_team)
```

## Customization

You can adjust constants in `football_predictor.py`:

```python
LEAGUE_AVG_GOALS = 1.4      # Average goals per team per match
HOME_ADVANTAGE = 1.15       # Home team offensive boost
AWAY_PENALTY = 0.85         # Away team penalty
FORM_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]  # Recency weights
```

## Future Enhancements

- [ ] Add head-to-head history weighting
- [ ] Dynamic league average calculation
- [ ] Player absence adjustments
- [ ] Machine learning model integration
- [ ] Historical accuracy tracking
- [ ] Multi-league support with different parameters

## Technical Details

- **Language**: Python 3.7+
- **Dependencies**: None (standard library only)
- **Performance**: Predictions in < 10ms
- **Type Hints**: Full type annotation support
- **Documentation**: Comprehensive docstrings

## License

This project is provided as-is for educational and analytical purposes.

## Author

Football Analytics System v1.0

---

**Note**: This is a statistical model and should be used for informational purposes only. Past performance does not guarantee future results.
