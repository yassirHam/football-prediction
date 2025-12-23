"""
Debug Model Script
==================
Tests a single match in detail to debug data extraction and prediction.
"""
import pandas as pd
import glob
from football_predictor import Team, predict_match

# Load one file
csv_file = glob.glob("E0.csv")[0]
df = pd.read_csv(csv_file, encoding='utf-8-sig')
df.columns = df.columns.str.replace('ï»¿', '').str.strip()
print(f"Loaded {len(df)} matches from {csv_file}")

# Pick a match deep in the season
idx = 100
row = df.iloc[idx]
print(f"\nAnalyzing Match {idx}: {row['HomeTeam']} vs {row['AwayTeam']} on {row['Date']}")

# Extract history manually
date = pd.to_datetime(row['Date'], dayfirst=True, errors='coerce')
past = df[pd.to_datetime(df['Date'], dayfirst=True, errors='coerce') < date]

print(f"Found {len(past)} matches before this date")

home = row['HomeTeam']
home_matches = past[(past['HomeTeam'] == home) | (past['AwayTeam'] == home)].sort_values('Date', ascending=False).head(5)
print(f"\nHome Team ({home}) Recent Matches: {len(home_matches)}")
for _, m in home_matches.iterrows():
    print(f"  {m['Date']}: {m['HomeTeam']} {m['FTHG']}-{m['FTAG']} {m['AwayTeam']}")

goals_scored = []
for _, m in home_matches.iterrows():
    if m['HomeTeam'] == home: goals_scored.append(int(m['FTHG']))
    else: goals_scored.append(int(m['FTAG']))
print(f"  Goals Scored: {goals_scored}")

# Away team
away = row['AwayTeam']
away_matches = past[(past['HomeTeam'] == away) | (past['AwayTeam'] == away)].sort_values('Date', ascending=False).head(5)
print(f"\nAway Team ({away}) Recent Matches: {len(away_matches)}")
goals_scored_away = []
for _, m in away_matches.iterrows():
    if m['HomeTeam'] == away: goals_scored_away.append(int(m['FTHG']))
    else: goals_scored_away.append(int(m['FTAG']))
print(f"  Goals Scored: {goals_scored_away}")

# Prediction
if len(home_matches) >= 3 and len(away_matches) >= 3:
    home_obj = Team(home, goals_scored, [1]*len(goals_scored), [0]*len(goals_scored)) # Dummy conceded/ht
    away_obj = Team(away, goals_scored_away, [1]*len(goals_scored_away), [0]*len(goals_scored_away))
    
    pred = predict_match(home_obj, away_obj)
    print("\nPrediction:")
    print(f"  Confidence: {pred['confidence_score']}")
    print(f"  Outcome Probs: {pred['match_outcome']}")
    print(f"  Top Score: {pred['full_match_predictions'][0]}")
    
    actual_home = int(row['FTHG'])
    actual_away = int(row['FTAG'])
    print(f"\nActual Result: {actual_home}-{actual_away}")
