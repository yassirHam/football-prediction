"""
Model Training and Validation Script
=====================================
Tests the prediction model against real historical match data
and provides accuracy metrics and parameter calibration.

Usage:
    python train_model.py
"""

import pandas as pd
import glob
from football_predictor import Team, predict_match
from collections import defaultdict
import numpy as np

def load_all_datasets():
    """Load all CSV files in the current directory."""
    csv_files = glob.glob("*.csv")
    
    datasets = {}
    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')  # Handle BOM
            
            # Clean column names (remove BOM and whitespace)
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
            
            # Validation
            required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Date']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️  Skipping {file}: Missing columns. Found: {list(df.columns)}")
                continue

            league_code = file.replace('.csv', '')
            datasets[league_code] = df
            print(f"✅ Loaded {league_code}: {len(df)} matches")
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
    
    return datasets


def extract_team_form(df, team_name, date, is_home=True, matches=5):
    """
    Extract last N matches for a team before a given date.
    
    Returns: goals_scored, goals_conceded, first_half_goals
    """
    try:
        # Filter matches before the given date
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        before_match = df[df['Date'] < date].copy()
        
        # Get team's matches (both home and away)
        home_matches = before_match[before_match['HomeTeam'] == team_name].copy()
        away_matches = before_match[before_match['AwayTeam'] == team_name].copy()
        
        # Combine and sort by date
        home_matches['TeamGoals'] = pd.to_numeric(home_matches['FTHG'], errors='coerce')
        home_matches['TeamConceded'] = pd.to_numeric(home_matches['FTAG'], errors='coerce')
        home_matches['TeamHTGoals'] = pd.to_numeric(home_matches['HTHG'], errors='coerce')
        
        away_matches['TeamGoals'] = pd.to_numeric(away_matches['FTAG'], errors='coerce')
        away_matches['TeamConceded'] = pd.to_numeric(away_matches['FTHG'], errors='coerce')
        away_matches['TeamHTGoals'] = pd.to_numeric(away_matches['HTAG'], errors='coerce')
        
        all_matches = pd.concat([home_matches, away_matches])
        all_matches = all_matches.sort_values('Date', ascending=False)
        
        # Get last N matches
        recent = all_matches.head(matches)
        
        if len(recent) < matches:
            return None  # Not enough data
        
        # Drop any matches with NaN values
        recent = recent.dropna(subset=['TeamGoals', 'TeamConceded', 'TeamHTGoals'])
        
        if len(recent) < matches:
            return None
        
        goals_scored = recent['TeamGoals'].tolist()[:matches]
        goals_conceded = recent['TeamConceded'].tolist()[:matches]
        ht_goals = recent['TeamHTGoals'].tolist()[:matches]
        
        # Convert to int
        goals_scored = [int(x) for x in goals_scored]
        goals_conceded = [int(x) for x in goals_conceded]
        ht_goals = [int(x) for x in ht_goals]
        
        return goals_scored, goals_conceded, ht_goals
    except Exception as e:
        # Silently skip problematic matches
        return None


def test_single_match(df, match_idx):
    """Test prediction for a single match."""
    match = df.iloc[match_idx]
    
    home_team = match['HomeTeam']
    away_team = match['AwayTeam']
    actual_home_goals = match['FTHG']
    actual_away_goals = match['FTAG']
    match_date = match['Date']
    
    # Get team form
    home_form = extract_team_form(df, home_team, match_date, is_home=True)
    away_form = extract_team_form(df, away_team, match_date, is_home=False)
    
    if home_form is None or away_form is None:
        return None  # Skip if not enough data
    
    # Create team objects
    home_team_obj = Team(
        name=home_team,
        goals_scored=home_form[0],
        goals_conceded=home_form[1],
        first_half_goals=home_form[2]
    )
    
    away_team_obj = Team(
        name=away_team,
        goals_scored=away_form[0],
        goals_conceded=away_form[1],
        first_half_goals=away_form[2]
    )
    
    # Get prediction
    try:
        prediction = predict_match(home_team_obj, away_team_obj)
        
        # Extract predicted outcome
        predicted_score = prediction['full_match_predictions'][0][0]  # Top prediction
        predicted_prob = prediction['full_match_predictions'][0][1]  # Probability
        
        # Determine actual outcome
        if actual_home_goals > actual_away_goals:
            actual_outcome = 'home_win'
        elif actual_home_goals < actual_away_goals:
            actual_outcome = 'away_win'
        else:
            actual_outcome = 'draw'
        
        # Determine predicted outcome
        outcome_probs = prediction['match_outcome']
        if outcome_probs['home_win'] > max(outcome_probs['draw'], outcome_probs['away_win']):
            predicted_outcome = 'home_win'
        elif outcome_probs['away_win'] > max(outcome_probs['draw'], outcome_probs['home_win']):
            predicted_outcome = 'away_win'
        else:
            predicted_outcome = 'draw'
        
        actual_score_str = f"{actual_home_goals}-{actual_away_goals}"
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'actual_score': actual_score_str,
            'predicted_score': f"{predicted_score[0]}-{predicted_score[1]}",
            'score_prob': predicted_prob,
            'actual_outcome': actual_outcome,
            'predicted_outcome': predicted_outcome,
            'outcome_prob': outcome_probs[predicted_outcome],
            'confidence': prediction.get('confidence_score', 0),
            'xg_home': prediction['expected_goals']['home'],
            'xg_away': prediction['expected_goals']['away'],
            'actual_total': actual_home_goals + actual_away_goals,
            'predicted_total': predicted_score[0] + predicted_score[1]
        }
    except Exception as e:
        print(f"Error predicting {home_team} vs {away_team}: {e}")
        return None


def validate_model(max_matches_per_league=100):
    """
    Validate model against all datasets.
    
    Args:
        max_matches_per_league: Maximum matches to test per league (for speed)
    """
    print("\n" + "="*60)
    print("🔬 MODEL VALIDATION STARTING")
    print("="*60 + "\n")
    
    datasets = load_all_datasets()
    
    if not datasets:
        print("❌ No datasets found! Add CSV files to the directory.")
        return
    
    all_results = []
    
    for league_code, df in datasets.items():
        print(f"\n📊 Testing {league_code}...")
        
        # Start from match that has enough history
        start_idx = 50  # Skip first 50 to ensure teams have history
        end_idx = min(len(df), start_idx + max_matches_per_league)
        
        league_results = []
        for idx in range(start_idx, end_idx):
            result = test_single_match(df, idx)
            if result:
                league_results.append(result)
        
        print(f"   ✅ Tested {len(league_results)} matches")
        all_results.extend(league_results)
    
    if not all_results:
        print("\n❌ No valid predictions generated.")
        return
    
    # Calculate metrics
    print("\n" + "="*60)
    print("📈 RESULTS")
    print("="*60 + "\n")
    
    # Exact score accuracy
    exact_scores = sum(1 for r in all_results if r['actual_score'] == r['predicted_score'])
    exact_score_accuracy = exact_scores / len(all_results) * 100
    
    # Outcome accuracy
    correct_outcomes = sum(1 for r in all_results if r['actual_outcome'] == r['predicted_outcome'])
    outcome_accuracy = correct_outcomes / len(all_results) * 100
    
    # Total goals accuracy (within 1)
    total_goals_close = sum(1 for r in all_results 
                           if abs(r['actual_total'] - r['predicted_total']) <= 1)
    total_goals_accuracy = total_goals_close / len(all_results) * 100
    
    # Average confidence for correct vs incorrect
    correct_predictions = [r for r in all_results if r['actual_outcome'] == r['predicted_outcome']]
    incorrect_predictions = [r for r in all_results if r['actual_outcome'] != r['predicted_outcome']]
    
    avg_confidence_correct = np.mean([r['confidence'] for r in correct_predictions]) if correct_predictions else 0
    avg_confidence_incorrect = np.mean([r['confidence'] for r in incorrect_predictions]) if incorrect_predictions else 0
    
    print(f"📊 Total Matches Tested: {len(all_results)}")
    print(f"\n🎯 EXACT SCORE ACCURACY: {exact_score_accuracy:.1f}%")
    print(f"   ({exact_scores}/{len(all_results)} correct)")
    
    print(f"\n✅ OUTCOME ACCURACY (Win/Draw/Loss): {outcome_accuracy:.1f}%")
    print(f"   ({correct_outcomes}/{len(all_results)} correct)")
    
    print(f"\n⚽ TOTAL GOALS (±1 accuracy): {total_goals_accuracy:.1f}%")
    print(f"   ({total_goals_close}/{len(all_results)} within 1 goal)")
    
    calibration_msg = ""
    if avg_confidence_correct > avg_confidence_incorrect + 10:
        calibration_msg = "✅ Good calibration - model is more confident when correct!"
    else:
        calibration_msg = "⚠️ Poor calibration - confidence scores need improvement"

    report = f"""
============================================================
📈 TRAINING & VALIDATION REPORT
============================================================
📊 Total Matches Tested: {len(all_results)}

🎯 EXACT SCORE ACCURACY: {exact_score_accuracy:.1f}%
   ({exact_scores}/{len(all_results)} correct)

✅ OUTCOME ACCURACY (Win/Draw/Loss): {outcome_accuracy:.1f}%
   ({correct_outcomes}/{len(all_results)} correct)

⚽ TOTAL GOALS (±1 accuracy): {total_goals_accuracy:.1f}%
   ({total_goals_close}/{len(all_results)} within 1 goal)

📈 CONFIDENCE CALIBRATION:
   Correct Predictions: {avg_confidence_correct:.1f}/100
   Incorrect Predictions: {avg_confidence_incorrect:.1f}/100
   Separation: {avg_confidence_correct - avg_confidence_incorrect:.1f} points
   {calibration_msg}

============================================================
"""
    print(report)
    try:
        with open("training_results.txt", "w", encoding="utf-8") as f:
            f.write(report)
    except Exception as e:
        print(f"Error saving report: {e}")
        
    return all_results


if __name__ == "__main__":
    results = validate_model(max_matches_per_league=100)
    
    if results:
        print("\n✅ Validation complete!")
        print(f"💾 Tested {len(results)} total matches across all leagues")
    else:
        print("\n❌ Validation failed - check CSV files and data format")
