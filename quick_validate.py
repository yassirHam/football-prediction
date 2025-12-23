"""
Quick Model Validation Script
==============================
Simpler version that tests model predictions against real data.
"""

import pandas as pd
import glob
from football_predictor import Team, predict_match
import numpy as np
import sys

print("\n" + "="*60)
print("FOOTBALL PREDICTION MODEL VALIDATION")
print("="*60 + "\n")

# Load first CSV file
csv_files = glob.glob("*.csv")
if not csv_files:
    print("[ERROR] No CSV files found!")
    sys.exit(1)

print(f"Found {len(csv_files)} CSV files")
print(f"Testing with: {csv_files[0]}\n")

# Load data
df = pd.read_csv(csv_files[0], encoding='utf-8-sig')
df.columns = df.columns.str.replace('ï»¿', '').str.strip()

print(f"[OK] Loaded {len(df)} matches from {csv_files[0]}\n")

# Sample some matches to test
test_matches = []
tested = 0
skipped = 0

for idx in range(100, min(200, len(df))):
    row = df.iloc[idx]
    
    # Create simple team data (using just current match for demo)
    # In reality, you'd want historical data, but this shows the concept
    home_team = Team(
        name=str(row['HomeTeam']),
        goals_scored=[2, 1, 3, 1, 2],  # Placeholder - would use real history
        goals_conceded=[1, 1, 0, 2, 1],
        first_half_goals=[1, 0, 2, 0, 1]
    )
    
    away_team = Team(
        name=str(row['AwayTeam']),
        goals_scored=[1, 2, 1, 2, 3],  # Placeholder
        goals_conceded=[2, 1, 1, 1, 0],
        first_half_goals=[0, 1, 1, 1, 2]
    )
    
    try:
        # Get prediction
        pred = predict_match(home_team, away_team)
        
        # Get actual result
        actual_home = int(row['FTHG'])
        actual_away = int(row['FTAG'])
        
        # Determine actual outcome
        if actual_home > actual_away:
            actual_outcome = 'home_win'
        elif actual_home < actual_away:
            actual_outcome = 'away_win'
        else:
            actual_outcome = 'draw'
        
        # Determine predicted outcome
        outcome_probs = pred['match_outcome']
        if outcome_probs['home_win'] > max(outcome_probs['draw'], outcome_probs['away_win']):
            predicted_outcome = 'home_win'
        elif outcome_probs['away_win'] > max(outcome_probs['draw'], outcome_probs['home_win']):
            predicted_outcome = 'away_win'
        else:
            predicted_outcome = 'draw'
        
        test_matches.append({
            'actual_outcome': actual_outcome,
            'predicted_outcome': predicted_outcome,
            'correct': actual_outcome == predicted_outcome,
            'home': row['HomeTeam'],
            'away': row['AwayTeam'],
            'score': f"{actual_home}-{actual_away}",
            'confidence': pred.get('confidence_score', 50)
        })
        
        tested += 1
        
    except Exception as e:
        skipped += 1
        continue

print(f"[TEST] Tested: {tested} matches")
print(f"[SKIP] Skipped: {skipped} matches\n")

if test_matches:
    # Calculate accuracy
    correct = sum(1 for m in test_matches if m['correct'])
    accuracy = (correct / len(test_matches)) * 100
    
    print("="*60)
    print("RESULTS")
    print("="*60 + "\n")
    print(f"[ACCURACY] Outcome Accuracy (Win/Draw/Loss): {accuracy:.1f}%")
    print(f"           ({correct}/{len(test_matches)} correct)\n")
    
    # Show samples
    print("[CORRECT] CORRECT PREDICTIONS (sample):")
    correct_samples = [m for m in test_matches if m['correct']][:3]
    for m in correct_samples:
        print(f"   {m['home']} vs {m['away']}: {m['score']}")
        print(f"   Predicted: {m['predicted_outcome']}, Confidence: {m['confidence']:.1f}\n")
    
    print("[INCORRECT] INCORRECT PREDICTIONS (sample):")
    incorrect_samples = [m for m in test_matches if not m['correct']][:3]
    for m in incorrect_samples:
        print(f"   {m['home']} vs {m['away']}: {m['score']}")
        print(f"   Predicted: {m['predicted_outcome']} (Actual: {m['actual_outcome']})")
        print(f"   Confidence: {m['confidence']:.1f}\n")
    
    print("="*60)
    print("[DONE] Validation complete!")
    print("="*60)
else:
    print("[ERROR] No matches could be tested")
