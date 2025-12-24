"""
Flask Web Application for Football Match Prediction
====================================================
A web interface for football predictions with hybrid XGBoost-Poisson architecture.
Supports both hybrid mode (ML + Poisson) and pure Poisson mode via configuration.
"""

from flask import Flask, render_template, request, jsonify
import os
import json
from datetime import datetime
from decision_engine.signal_generator import SignalGenerator

# Load configuration to determine which predictor to use
USE_HYBRID_MODE = True  # Can be toggled via config
try:
    if os.path.exists('hybrid_config.json'):
        with open('hybrid_config.json', 'r') as f:
            config = json.load(f)
            USE_HYBRID_MODE = config.get('enable_ml', True)
except:
    pass

# Import appropriate predictor based on configuration
if USE_HYBRID_MODE:
    try:
        from hybrid.hybrid_predictor import Team, predict_match_hybrid as predict_match
        print("✅ Using HYBRID prediction mode (XGBoost + Poisson)")
    except ImportError as e:
        print(f"⚠️  Hybrid mode enabled but import failed: {e}")
        print("   Falling back to enhanced Poisson predictor")
        from enhanced_predictor import Team, enhanced_predict_match as predict_match
        USE_HYBRID_MODE = False
else:
    from enhanced_predictor import Team, enhanced_predict_match as predict_match
    print("ℹ️  Using PURE POISSON prediction mode")

app = Flask(__name__)
signal_generator = SignalGenerator()

HISTORY_FILE = 'prediction_history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(entry):
    history = load_history()
    history.insert(0, entry)  # Add new entry at start
    # Keep only last 50 entries
    history = history[:50]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def normalize_rank(rank, total_teams):
    """Normalize rank to a 1-20 scale standard."""
    # Ensure reasonable bounds
    if total_teams < 2: total_teams = 20
    if rank < 1: rank = 1
    
    # Formula: (Rank / Total) * 20
    return (rank / total_teams) * 20

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.json
        
        # Get neutral venue flag
        neutral_venue = data.get('neutral_venue', False)
        
        # Helper for safer input parsing
        def safe_int(val, default=0):
            try:
                if val is None or val == "":
                    return default
                return int(val)
            except (ValueError, TypeError):
                return default

        # Normalize Ranks
        home_pos = safe_int(data['home_team'].get('league_position'), 10)
        away_pos = safe_int(data['away_team'].get('league_position'), 10)
        
        # Get total teams (default to 20 if not provided for backward compat)
        home_league_size = safe_int(data['home_team'].get('league_size'), 20)
        away_league_size = safe_int(data['away_team'].get('league_size'), 20)
        
        home_rank_norm = normalize_rank(home_pos, home_league_size)
        away_rank_norm = normalize_rank(away_pos, away_league_size)

        # Create home team
        home_team = Team(
            name=data['home_team']['name'],
            goals_scored=[safe_int(x) for x in data['home_team']['goals_scored']],
            goals_conceded=[safe_int(x) for x in data['home_team']['goals_conceded']],
            first_half_goals=[safe_int(x) for x in data['home_team']['first_half_goals']],
            league_position=home_rank_norm
        )
        
        # Create away team
        away_team = Team(
            name=data['away_team']['name'],
            goals_scored=[safe_int(x) for x in data['away_team']['goals_scored']],
            goals_conceded=[safe_int(x) for x in data['away_team']['goals_conceded']],
            first_half_goals=[safe_int(x) for x in data['away_team']['first_half_goals']],
            league_position=away_rank_norm
        )
        
        # Get predictions
        is_cup = data.get('is_cup', False)
        result = predict_match(home_team, away_team, neutral_venue, is_cup=is_cup)
        
        # Save to History
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "home_team": data['home_team']['name'],
            "away_team": data['away_team']['name'],
            "home_rank": f"{home_pos}/{home_league_size}",
            "away_rank": f"{away_pos}/{away_league_size}",
            "prediction": {
                "home_win": round(result['match_outcome']['home_win'] * 100, 1),
                "draw": round(result['match_outcome']['draw'] * 100, 1),
                "away_win": round(result['match_outcome']['away_win'] * 100, 1),
                "confidence": result['insights']['confidence']
            }
        }
        save_history(history_entry)
        
        # Format response with enhanced data
        response = {
            "success": True,
            "home_team": data['home_team']['name'],
            "away_team": data['away_team']['name'],
            "neutral_venue": neutral_venue,
            "match_outcome": {
                "home_win": round(result['match_outcome']['home_win'] * 100, 1),
                "draw": round(result['match_outcome']['draw'] * 100, 1),
                "away_win": round(result['match_outcome']['away_win'] * 100, 1)
            },
            "first_half_predictions": [
                {"score": f"{h}-{a}", "probability": round(prob * 100, 1)}
                for (h, a), prob in result['first_half_predictions']
            ],
            "full_match_predictions": [
                {"score": f"{h}-{a}", "probability": round(prob * 100, 1)}
                for (h, a), prob in result['full_match_predictions']
            ],
            "total_goals": {
                key: round(val * 100, 1) 
                for key, val in result['total_goals'].items()
                if isinstance(val, float)
            },
            "expected_goals": {
                "home": round(result['expected_goals']['home'], 2),
                "away": round(result['expected_goals']['away'], 2)
            },
            "both_teams_score": round(result['both_teams_score'] * 100, 1),
            "clean_sheet": {
                "home": round(result['clean_sheet']['home'] * 100, 1),
                "away": round(result['clean_sheet']['away'] * 100, 1)
            },
            "confidence_score": result.get('confidence_score', 50.0),
            "confidence_breakdown": result.get('confidence_breakdown', {}),
            "prediction_intervals": result.get('prediction_intervals', {}),
            "team_insights": result.get('team_insights', {}),
            "model_quality": result.get('model_quality', {}),
            "betting_insights": result.get('betting_insights', {}),
            "insights": result['insights'],
            "insights": result['insights'],
            "prediction_mode": "hybrid" if USE_HYBRID_MODE else "poisson",
            "hybrid_metadata": {
                "source": result.get('hybrid_metadata', {}).get('source', 'poisson'),
                "confidence": float(result.get('hybrid_metadata', {}).get('confidence', 0))
            },
            "decision_signal": signal_generator.generate_signal(result, league=data.get('home_team', {}).get('competition_type', 'DEFAULT'))
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/api/history')
def get_history():
    """Return prediction history."""
    return jsonify(load_history())

@app.route('/examples')
def examples():
    """Return example team data."""
    # User requested to remove all examples
    examples_data = {}
    return jsonify(examples_data)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Save actual match result with prediction data to learn from it."""
    try:
        data = request.json
        
        # File path for learned matches
        file_path = "data/learned_matches.csv"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Check if file exists to write header
        is_new_file = not os.path.exists(file_path)
        
        # Calculate error metrics if prediction data provided
        home_goals_actual = int(data.get('home_goals', 0))
        away_goals_actual = int(data.get('away_goals', 0))
        total_goals_actual = home_goals_actual + away_goals_actual
        
        # Get prediction data (if user provided prediction before submitting feedback)
        pred_home_xg = float(data.get('predicted_home_xg', 0))
        pred_away_xg = float(data.get('predicted_away_xg', 0))
        pred_total_xg = pred_home_xg + pred_away_xg
        
        # Most likely score from prediction
        pred_home_score = data.get('predicted_home_score', '')
        pred_away_score = data.get('predicted_away_score', '')
        
        # Calculate errors
        if pred_home_xg > 0:  # Only if prediction was made
            xg_error_home = abs(home_goals_actual - pred_home_xg)
            xg_error_away = abs(away_goals_actual - pred_away_xg)
            total_goals_error = abs(total_goals_actual - pred_total_xg)
            
            # Score prediction accuracy
            if pred_home_score and pred_away_score:
                score_exact = 1 if (int(pred_home_score) == home_goals_actual and 
                                   int(pred_away_score) == away_goals_actual) else 0
                score_error = abs(int(pred_home_score) - home_goals_actual) + abs(int(pred_away_score) - away_goals_actual)
            else:
                score_exact = 0
                score_error = 0
        else:
            xg_error_home = 0
            xg_error_away = 0
            total_goals_error = 0
            score_exact = 0
            score_error = 0
        
        # Outcome prediction accuracy
        actual_outcome = 'H' if home_goals_actual > away_goals_actual else ('A' if away_goals_actual > home_goals_actual else 'D')
        predicted_outcome = data.get('predicted_outcome', '')
        outcome_correct = 1 if actual_outcome == predicted_outcome else 0
        
        # BTTS (Both Teams To Score)
        btts_actual = 1 if (home_goals_actual > 0 and away_goals_actual > 0) else 0
        btts_predicted = data.get('predicted_btts', 0)
        btts_correct = 1 if btts_actual == btts_predicted else 0
        
        # Confidence score (from prediction)
        confidence = data.get('confidence_score', 50)
        
        # Get HT scores if provided
        ht_home_goals = data.get('ht_home_goals')
        ht_away_goals = data.get('ht_away_goals')
        
        # Prepare row data
        row_data = {
            'Date': data.get('date', datetime.now().strftime('%Y-%m-%d')),
            'HomeTeam': data['home_team'],
            'AwayTeam': data['away_team'],
            'FTHG': home_goals_actual, # Changed to FTHG to match old header
            'FTAG': away_goals_actual, # Changed to FTAG to match old header
            'Competition': data.get('competition', 'UserFeedback'), # Added competition
            # Add HT Scores
            'HTHomeGoals': ht_home_goals if ht_home_goals is not None else '',
            'HTAwayGoals': ht_away_goals if ht_away_goals is not None else '',
            # Prediction Data
            'PredictedHomeXG': round(pred_home_xg, 2),
            'PredictedAwayXG': round(pred_away_xg, 2),
            'PredictedHomeScore': pred_home_score,
            'PredictedAwayScore': pred_away_score,
            # Error Metrics
            'XGErrorHome': round(xg_error_home, 2),
            'XGErrorAway': round(xg_error_away, 2),
            'TotalGoalsError': round(total_goals_error, 2), # Ensure this is rounded
            'ScoreExactMatch': score_exact, # Use the calculated score_exact
            'ScoreError': score_error,
            'ActualOutcome': actual_outcome,
            'PredictedOutcome': predicted_outcome, # Use the calculated predicted_outcome
            'OutcomeCorrect': outcome_correct, # Use the calculated outcome_correct
            # BTTS
            'BTTSActual': btts_actual, # Use the calculated btts_actual
            'BTTSPredicted': btts_predicted, # Use the calculated btts_predicted
            'BTTSCorrect': btts_correct, # Use the calculated btts_correct
            'Confidence': confidence
        }

        # Define columns (Updated with HT)
        columns = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Competition', # Added Competition
            'HTHomeGoals', 'HTAwayGoals', # New Columns
            'PredictedHomeXG', 'PredictedAwayXG', 'PredictedHomeScore', 'PredictedAwayScore',
            'XGErrorHome', 'XGErrorAway', 'TotalGoalsError', 'ScoreExactMatch', 'ScoreError',
            'ActualOutcome', 'PredictedOutcome', 'OutcomeCorrect',
            'BTTSActual', 'BTTSPredicted', 'BTTSCorrect', 'Confidence'
        ]
        
        # Check if we need to migrate existing file (add new headers)
        if not is_new_file:
            import pandas as pd
            try:
                df = pd.read_csv(file_path)
                # Check for missing columns and add them
                missing_cols = [col for col in columns if col not in df.columns]
                if missing_cols:
                    for col in missing_cols:
                        df[col] = '' # Add with empty string as default
                    # Simple migration: overwrite file with new columns
                    print(f"Migrating learned_matches.csv to include new columns: {', '.join(missing_cols)}...")
                    df.to_csv(file_path, index=False)
            except Exception as e:
                print(f"Error migrating CSV: {e}")

        with open(file_path, 'a', newline='', encoding="utf-8") as f: # Added encoding
            if is_new_file:
                f.write(','.join(columns) + '\n')
            
            # Create line carefully matching columns order
            line = []
            for col in columns:
                line.append(str(row_data.get(col, '')))
            f.write(','.join(line) + '\n')
            
        return jsonify({
            "success": True,
            "error_metrics": {
                "xg_error_home": round(xg_error_home, 2),
                "xg_error_away": round(xg_error_away, 2),
                "total_goals_error": round(total_goals_error, 2),
                "score_exact_match": score_exact,
                "outcome_correct": outcome_correct,
                "btts_correct": btts_correct
            }
        })
        
    except Exception as e:
        print(f"Error saving feedback: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Football Prediction Web App...")
    print("📍 Open your browser at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
