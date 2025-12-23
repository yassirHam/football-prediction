"""
Model Parameter Optimization Script
====================================
Grid search for best model parameters using historical data.
"""

import pandas as pd
import glob
import numpy as np
from itertools import product
import football_predictor
import advanced_statistics
from football_predictor import Team, predict_match
import sys

# Constants to optimize
PARAM_GRID = {
    'ewma_alpha': [0.2, 0.3, 0.4, 0.5],
    'bayesian_factor': [2.0, 3.0, 5.0],
    # 'dixon_coles_rho': [-0.05, -0.10, -0.13]  # Uncomment to test enabling Dixon-Coles
}

def load_data():
    """Load and clean data from CSVs."""
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("[ERROR] No CSV files found!")
        sys.exit(1)
        
    all_dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
            df['League'] = file.replace('.csv', '')
            all_dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to load {file}: {e}")
            
    if not all_dfs:
        sys.exit(1)
        
    return pd.concat(all_dfs, ignore_index=True)

def extract_history(df, team, date, matches=5):
    """Extract recent form for a team."""
    # Convert date if needed
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date, dayfirst=True, errors='coerce')
        
    # Filter past matches
    past = df[pd.to_datetime(df['Date'], dayfirst=True, errors='coerce') < date].copy()
    
    # Get team matches
    team_matches = past[(past['HomeTeam'] == team) | (past['AwayTeam'] == team)].copy()
    
    # Sort by date descending
    team_matches['DateObj'] = pd.to_datetime(team_matches['Date'], dayfirst=True, errors='coerce')
    team_matches = team_matches.sort_values('DateObj', ascending=False).head(matches)
    
    if len(team_matches) < 3: # Need at least 3 matches
        return None
        
    goals_scored = []
    goals_conceded = []
    first_half = [] # Dummy for now or extract if available
    
    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team:
            goals_scored.append(int(row['FTHG']))
            goals_conceded.append(int(row['FTAG']))
            ht_goals = int(row['HTHG']) if 'HTHG' in row else 0 # Fallback
        else:
            goals_scored.append(int(row['FTAG']))
            goals_conceded.append(int(row['FTHG']))
            ht_goals = int(row['HTAG']) if 'HTAG' in row else 0
            
        first_half.append(ht_goals)
        
    return {
        'scored': goals_scored, 
        'conceded': goals_conceded,
        'ht': first_half
    }

def evaluate_params(df, alpha, bayesian_k):
    """Run validation on recent matches with specific parameters."""
    
    # Set parameters
    advanced_statistics.EWMA_ALPHA = alpha
    advanced_statistics.BAYESIAN_CONFIDENCE_FACTOR = bayesian_k
    football_predictor.advanced_statistics = advanced_statistics
    
    correct_exact = 0
    correct_outcome = 0
    total_tested = 0
    
    # Focus on last 500 matches only (most relevant for modern trends)
    # Sort by date
    df['DateObj'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    recent_matches = df.sort_values('DateObj', ascending=False).head(500)
    
    # Sample 100 from these 500
    sample = recent_matches.sample(n=min(100, len(recent_matches)), random_state=42)
    
    for _, row in sample.iterrows():
        try:
            date = row['DateObj']
            home = row['HomeTeam']
            away = row['AwayTeam']
            
            home_hist = extract_history(df, home, date)
            away_hist = extract_history(df, away, date)
            
            if not home_hist or not away_hist:
                continue
                
            # Create Team objects
            home_obj = Team(home, home_hist['scored'], home_hist['conceded'], home_hist['ht'])
            away_obj = Team(away, away_hist['scored'], away_hist['conceded'], away_hist['ht'])
            
            pred = predict_match(home_obj, away_obj)
            
            # Check Exact Score
            top_score = pred['full_match_predictions'][0][0] # (H, A)
            actual_score = (int(row['FTHG']), int(row['FTAG']))
            
            if top_score == actual_score:
                correct_exact += 1
                
            # Check Outcome
            if actual_score[0] > actual_score[1]: outcome = 'home_win'
            elif actual_score[0] < actual_score[1]: outcome = 'away_win'
            else: outcome = 'draw'
            
            probs = pred['match_outcome']
            pred_outcome = max(probs, key=probs.get)
            
            if pred_outcome == outcome:
                correct_outcome += 1
                
            total_tested += 1
        except Exception:
            continue
            
    if total_tested == 0:
        return 0, 0
        
    return (correct_exact / total_tested * 100), (correct_outcome / total_tested * 100)

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} matches. Optimizing parameters on recent data...")
    print("-" * 60)
    print(f"{'ALPHA':<10} | {'BAYES':<10} | {'EXACT %':<10} | {'OUTCOME %':<10} | {'SCORE':<10}")
    print("-" * 60)
    
    best_score = 0
    best_params = None
    
    # Grid Search
    # Wider range for Alpha
    for alpha in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for k in [2.0, 3.0, 4.0, 5.0]:
            exact_acc, outcome_acc = evaluate_params(df, alpha, k)
            
            # Weighted score: 60% Exact Accuracy + 40% Outcome Accuracy
            # Emphasizing Exact Score as requested
            score = (exact_acc * 0.6) + (outcome_acc * 0.4)
            
            print(f"{alpha:<10} | {k:<10} | {exact_acc:<10.1f} | {outcome_acc:<10.1f} | {score:<10.1f}")
            
            if score > best_score:
                best_score = score
                best_params = (alpha, k)
                
    print("-" * 60)
    print(f"BEST CONFIGURATION (Score: {best_score:.1f})")
    print(f"EWMA_ALPHA = {best_params[0]}")
    print(f"BAYESIAN_CONFIDENCE_FACTOR = {best_params[1]}")
    print("-" * 60)
