"""
Analyze prediction accuracy issues in the football prediction system.

This script will:
1. Load all available match data including learned_matches.csv
2. Re-predict each match using current model
3. Analyze accuracy of:
   - Total goals (especially off-by-1 errors)
   - Exact score predictions  
   - Both teams to score predictions
4. Identify patterns and root causes
"""

import pandas as pd
import glob
from football_predictor import Team, predict_match
import numpy as np

def load_all_matches():
    """Load all match data from CSV files."""
    csv_files = glob.glob("data/*.csv")
    all_matches = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
            
            # Keep only matches with required columns
            if all(col in df.columns for col in ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']):
                all_matches.append(df)
                print(f"Loaded {len(df)} matches from {csv_file}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    # Load learned matches if exists
    try:
        learned_df = pd.read_csv("data/learned_matches.csv", encoding='utf-8-sig')
        learned_df.columns = learned_df.columns.str.replace('ï»¿', '').str.strip()
        if len(learned_df) > 0:
            all_matches.append(learned_df)
            print(f"Loaded {len(learned_df)} learned matches")
    except:
        print("No learned matches found")
    
    if all_matches:
        return pd.concat(all_matches, ignore_index=True)
    return pd.DataFrame()

def create_team_from_history(team_name, df, match_date_idx, is_home=True):
    """Create Team object from historical match data."""
    team_col = 'HomeTeam' if is_home else 'AwayTeam'
    goals_col = 'FTHG' if is_home else 'FTAG'
    conceded_col = 'FTAG' if is_home else 'FTHG'
    
    # Get previous 5 matches for this team
    team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)]
    team_matches = team_matches.iloc[:match_date_idx]
    team_matches = team_matches.tail(5)
    
    goals_scored = []
    goals_conceded = []
    first_half_goals = []
    
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            goals_scored.append(int(match['FTHG']))
            goals_conceded.append(int(match['FTAG']))
            # Estimate first half goals (if not available, use 45% heuristic)
            if 'HTHG' in match:
                first_half_goals.append(int(match['HTHG']))
            else:
                first_half_goals.append(int(match['FTHG'] * 0.45))
        else:
            goals_scored.append(int(match['FTAG']))
            goals_conceded.append(int(match['FTHG']))
            if 'HTAG' in match:
                first_half_goals.append(int(match['HTAG']))
            else:
                first_half_goals.append(int(match['FTAG'] * 0.45))
    
    # If we don't have 5 matches, pad with league average
    while len(goals_scored) < 5:
        goals_scored.insert(0, 1)
        goals_conceded.insert(0, 1)
        first_half_goals.insert(0, 0)
    
    return Team(
        name=team_name,
        goals_scored=goals_scored[-5:],
        goals_conceded=goals_conceded[-5:],
        first_half_goals=first_half_goals[-5:],
        league_position=10
    )

def analyze_accuracy():
    """Main analysis function."""
    print("="*70)
    print("FOOTBALL PREDICTION ACCURACY ANALYSIS")
    print("="*70)
    
    df = load_all_matches()
    if df.empty:
        print("No match data found!")
        return
    
    print(f"\nTotal matches available: {len(df)}")
    
    # Stats trackers
    total_goals_errors = []
    exact_score_correct = 0
    exact_score_total = 0
    btts_correct = 0
    btts_total = 0
    off_by_one_count = 0
    
    # Detailed error tracking
    total_goals_predicted = []
    total_goals_actual = []
    
    # Only analyze matches where we have enough history (after match 10)
    for idx in range(10, min(len(df), 100)):  # Limit to 100 for speed
        match = df.iloc[idx]
        
        try:
            home_team = create_team_from_history(match['HomeTeam'], df, idx, is_home=True)
            away_team = create_team_from_history(match['AwayTeam'], df, idx, is_home=False)
            
            # Get prediction
            prediction = predict_match(home_team, away_team)
            
            # Actual results
            actual_home = int(match['FTHG'])
            actual_away = int(match['FTAG'])
            actual_total = actual_home + actual_away
            actual_btts = (actual_home > 0) and (actual_away > 0)
            
            # Predicted results
            predicted_score = prediction['full_match_predictions'][0][0]  # Most likely score
            predicted_total = predicted_score[0] + predicted_score[1]
            predicted_btts = prediction['both_teams_score'] > 0.5
            
            # Track total goals error
            total_error = abs(predicted_total - actual_total)
            total_goals_errors.append(total_error)
            total_goals_predicted.append(predicted_total)
            total_goals_actual.append(actual_total)
            
            if total_error == 1:
                off_by_one_count += 1
            
            # Track exact score
            exact_score_total += 1
            if predicted_score == (actual_home, actual_away):
                exact_score_correct += 1
            
            # Track BTTS
            btts_total += 1
            if predicted_btts == actual_btts:
                btts_correct += 1
                
        except Exception as e:
            print(f"Error analyzing match {idx}: {e}")
            continue
    
    # Print results
    print("\n" + "="*70)
    print("ACCURACY RESULTS")
    print("="*70)
    
    print(f"\nMatches Analyzed: {len(total_goals_errors)}")
    
    print("\n### TOTAL GOALS ANALYSIS ###")
    print(f"Average Total Goals Error: {np.mean(total_goals_errors):.2f} goals")
    print(f"Off by 1 goal: {off_by_one_count} ({off_by_one_count/len(total_goals_errors)*100:.1f}%)")
    print(f"Off by 2+ goals: {sum(1 for e in total_goals_errors if e >= 2)} ({sum(1 for e in total_goals_errors if e >= 2)/len(total_goals_errors)*100:.1f}%)")
    print(f"Exact total: {sum(1 for e in total_goals_errors if e == 0)} ({sum(1 for e in total_goals_errors if e == 0)/len(total_goals_errors)*100:.1f}%)")
    
    # Bias analysis
    avg_predicted = np.mean(total_goals_predicted)
    avg_actual = np.mean(total_goals_actual)
    print(f"\nAverage Predicted Total: {avg_predicted:.2f}")
    print(f"Average Actual Total: {avg_actual:.2f}")
    print(f"Bias: {avg_predicted - avg_actual:+.2f} (negative = under-predicting)")
    
    print("\n### EXACT SCORE ANALYSIS ###")
    print(f"Exact Score Accuracy: {exact_score_correct}/{exact_score_total} ({exact_score_correct/exact_score_total*100:.1f}%)")
    
    print("\n### BOTH TEAMS TO SCORE ANALYSIS ###")
    print(f"BTTS Accuracy: {btts_correct}/{btts_total} ({btts_correct/btts_total*100:.1f}%)")
    
    print("\n" + "="*70)
    print("IDENTIFIED ISSUES")
    print("="*70)
    
    issues = []
    if off_by_one_count / len(total_goals_errors) > 0.35:
        issues.append(f"⚠️  HIGH off-by-1 error rate: {off_by_one_count/len(total_goals_errors)*100:.1f}%")
    
    if abs(avg_predicted - avg_actual) > 0.3:
        if avg_predicted > avg_actual:
            issues.append(f"⚠️  Model is OVER-predicting goals by {avg_predicted - avg_actual:.2f}")
        else:
            issues.append(f"⚠️  Model is UNDER-predicting goals by {avg_actual - avg_predicted:.2f}")
    
    if exact_score_correct / exact_score_total < 0.15:
        issues.append(f"⚠️  LOW exact score accuracy: {exact_score_correct/exact_score_total*100:.1f}%")
    
    if btts_correct / btts_total < 0.65:
        issues.append(f"⚠️  LOW BTTS accuracy: {btts_correct/btts_total*100:.1f}%")
    
    for issue in issues:
        print(issue)
    
    if not issues:
        print("✅ No major issues detected!")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    analyze_accuracy()
