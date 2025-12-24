"""
Training Script for Football Prediction ML Models
=================================================

Handles data loading, feature engineering, and model training.
Includes cross-validation to ensure robustness.
"""

import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, mean_absolute_error, accuracy_score, brier_score_loss
from football_predictor import Team
from ml_models.feature_builder import FeatureBuilder
from ml_models.xgboost_model import XGBoostPredictor
import json

def load_and_prepare_data(data_dir: str = 'data'):
    """
    Load raw match data and convert to ML training examples.
    
    Returns:
        X: Feature matrix
        y_outcome: Outcome labels (0:A, 1:D, 2:H)
        y_home: Home goals
        y_away: Away goals
    """
    print("Loading data...")
    csv_files = glob.glob(f"{data_dir}/*.csv")
    dfs = []
    
    for f in csv_files:
        if 'learned_matches' in f: continue
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            # Standardize columns
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
            
            # Extract league from filename
            league = Path(f).stem
            df['League'] = league
            
            if {'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Date'}.issubset(df.columns):
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['Date']).sort_values('Date')
                dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not dfs:
        raise ValueError("No data found!")
        
    full_df = pd.concat(dfs, ignore_index=True).sort_values('Date')
    print(f"Total matches loaded: {len(full_df)}")
    
    # ⚠️ TIME-SERIES SAFETY: Verify chronological ordering
    # This assertion prevents future data from leaking into past predictions
    assert full_df['Date'].is_monotonic_increasing, "CRITICAL: Data must be chronologically sorted!"
    print("✅ Chronological ordering verified")
    
    # Initialize Feature Builder
    builder = FeatureBuilder()
    
    X_list = []
    y_outcome_list = []
    y_home_list = []
    y_away_list = []
    
    # We need history to build features, so we iterate through matches
    # This is slow but necessary to build accurate point-in-time features
    print("Building features (this may take a minute)...")
    
    # Optimized Feature Generation (O(N) instead of O(N^2))
    print("Building features (optimized)...")
    
    # Track team history state incrementally
    # team_name -> {scored: [], conceded: []}
    team_history = {}
    
    # Diagnostic counters
    total_matches = 0
    skipped_no_history = 0
    skipped_min_history = 0
    training_examples_added = 0
    
    # Process all matches chronologically
    for i, match in full_df.iterrows():
        home = match['HomeTeam']
        away = match['AwayTeam']
        total_matches += 1
        
        # 1. Initialize teams FIRST (before any checks)
        if home not in team_history:
            team_history[home] = {'scored': [], 'conceded': []}
        if away not in team_history:
            team_history[away] = {'scored': [], 'conceded': []}
        
        # 2. Build features using current history
        # Check if we have minimum history required (3 matches each)
        h_hist = team_history[home]
        a_hist = team_history[away]
        
        if len(h_hist['scored']) >= 3 and len(a_hist['scored']) >= 3:
            # Create Team objects from state (use last 10 matches max)
            home_team = Team(
                name=home,
                goals_scored=h_hist['scored'][-10:],
                goals_conceded=h_hist['conceded'][-10:],
                first_half_goals=[0]*5
            )
            away_team = Team(
                name=away,
                goals_scored=a_hist['scored'][-10:],
                goals_conceded=a_hist['conceded'][-10:],
                first_half_goals=[0]*5
            )
            
            # Set league for params lookup
            home_team.league = match.get('League', 'DEFAULT')
            away_team.league = match.get('League', 'DEFAULT')
            
            # Params (ideally dynamic per league)
            params = {
                'league_avg_goals': 1.4,
                'home_advantage': 1.15,
                'away_penalty': 0.85
            }
            
            try:
                features = builder.build_features_for_match(home_team, away_team, params)
                
                # Targets
                h_goals = int(match['FTHG'])
                a_goals = int(match['FTAG'])
                
                if h_goals > a_goals:
                    outcome = 2  # Home win
                elif h_goals == a_goals:
                    outcome = 1  # Draw
                else:
                    outcome = 0  # Away win
                
                X_list.append(features)
                y_outcome_list.append(outcome)
                y_home_list.append(h_goals)
                y_away_list.append(a_goals)
                training_examples_added += 1
            except Exception as e:
                # Feature engineering failed
                pass
        else:
            # Not enough history for at least one team
            skipped_min_history += 1

        # 3. Update history AFTER using it (chronological integrity)
        team_history[home]['scored'].append(match['FTHG'])
        team_history[home]['conceded'].append(match['FTAG'])
        
        team_history[away]['scored'].append(match['FTAG'])
        team_history[away]['conceded'].append(match['FTHG'])
    
    # Diagnostic output
    print(f"\n📊 Training Data Statistics:")
    print(f"  Total matches processed: {total_matches}")
    print(f"  Training examples created: {training_examples_added}")
    print(f"  Skipped (min history < 3): {skipped_min_history}")
    print(f"  Data utilization: {100 * training_examples_added / total_matches:.1f}%")
    print(f"  Unique teams tracked: {len(team_history)}")

            
    return np.array(X_list), np.array(y_outcome_list), np.array(y_home_list), np.array(y_away_list)

def _create_team_snapshot(team_name, history_df):
    """Helper to create team object from dataframe history."""
    # Kept for compatibility but not used in main loop anymore
    pass

def train_and_validate():
    """Main training loop with validation."""
    X, y_outcome, y_home, y_away = load_and_prepare_data()
    print(f"Training dataset size: {len(X)}")
    
    # Time Series Split for validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    predictor = XGBoostPredictor()
    
    fold = 1
    metrics = []
    
    for train_index, test_index in tscv.split(X):
        print(f"\nFold {fold}/3")
        
        # ⚠️ TIME-SERIES SAFETY: Verify no training data from future
        # This assertion ensures all training indices come before test indices
        assert train_index.max() < test_index.min(), \
            f"TIME-SERIES LEAKAGE DETECTED! Train max ({train_index.max()}) >= Test min ({test_index.min()})"
        print(f"  ✅ Time-series integrity verified (train < test)")
        
        X_train, X_test = X[train_index], X[test_index]
        y_out_train, y_out_test = y_outcome[train_index], y_outcome[test_index]
        y_home_train, y_home_test = y_home[train_index], y_home[test_index]
        y_away_train, y_away_test = y_away[train_index], y_away[test_index]
        
        # Train
        predictor.train(X_train, y_out_train, y_home_train, y_away_train)
        
        # Evaluate
        preds = predictor.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_out_test, np.argmax(preds['outcome_probs'], axis=1))
        ll = log_loss(y_out_test, preds['outcome_probs'])
        mae_home = mean_absolute_error(y_home_test, preds['xg_home'])
        mae_away = mean_absolute_error(y_away_test, preds['xg_away'])
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Log Loss: {ll:.4f}")
        print(f"  Home MAE: {mae_home:.4f}")
        print(f"  Away MAE: {mae_away:.4f}")
        
        metrics.append({
            'fold': fold,
            'accuracy': acc,
            'log_loss': ll,
            'mae_home': mae_home
        })
        fold += 1
        
    # Retrain on full data
    print("\nTraining final model on full dataset...")
    predictor.train(X, y_outcome, y_home, y_away)
    predictor.save()
    
    # Save metrics
    with open('ml_models/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("\nDone!")

if __name__ == "__main__":
    train_and_validate()
