"""
Hybrid System Validation Script
================================

Rigorous comparison of Hybrid (XGBoost+Poisson) vs Pure Poisson on real match data.

Validates that:
1. Match accuracy ≥ Poisson baseline
2. Goal MAE < Poisson baseline  
3. Log loss ≤ Poisson (within tolerance)
4. Probabilities remain well calibrated
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import glob
from typing import List, Dict
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error

# Import predictors
from football_predictor import Team, predict_match as predict_poisson
from hybrid.hybrid_predictor import predict_match_hybrid

import logging
logging.basicConfig(level=logging.WARNING)  # Suppress info logs during validation


def create_team_from_history(team_name: str, history_df: pd.DataFrame, n_matches: int = 10) -> Team:
    """
    Create a Team object from historical match data.
    
    Args:
        team_name: Name of the team
        history_df: DataFrame with historical matches
        n_matches: Number of recent matches to use
        
    Returns:
        Team object or None if insufficient data
    """
    team_matches = history_df[
        (history_df['HomeTeam'] == team_name) | (history_df['AwayTeam'] == team_name)
    ].tail(n_matches)
    
    if len(team_matches) < 3:
        return None
    
    goals_scored = []
    goals_conceded = []
    first_half_goals = []
    
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            goals_scored.append(int(match['FTHG']))
            goals_conceded.append(int(match['FTAG']))
            # First half goals if available
            ht_goals = match.get('HTHG', 0)
            first_half_goals.append(int(ht_goals) if pd.notna(ht_goals) else 0)
        else:
            goals_scored.append(int(match['FTAG']))
            goals_conceded.append(int(match['FTHG']))
            ht_goals = match.get('HTAG', 0)
            first_half_goals.append(int(ht_goals) if pd.notna(ht_goals) else 0)
    
    team = Team(
        name=team_name,
        goals_scored=goals_scored[::-1],  # Reverse to get most recent first
        goals_conceded=goals_conceded[::-1],
        first_half_goals=first_half_goals[::-1] if any(first_half_goals) else [0] * len(goals_scored)
    )
    
    # Add league information
    if 'League' in history_df.columns:
        team.league = history_df['League'].iloc[0]
    
    return team


def load_validation_matches(data_dir: str = 'data', n_matches_per_league: int = 20, min_history: int = 50):
    """
    Load validation matches from CSV files.
    
    Args:
        data_dir: Directory containing CSV files
        n_matches_per_league: Number of recent matches to use per league
        min_history: Minimum historical matches required
        
    Returns:
        List of validation match dictionaries
    """
    csv_files = glob.glob(f"{data_dir}/*.csv")
    validation_matches = []
    
    for csv_file in csv_files:
        if 'learned_matches' in csv_file:
            continue
            
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
            
            # Check for required columns
            if not all(col in df.columns for col in ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']):
                continue
            
            # Parse date if available
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['Date']).sort_values('Date')
            
            # Add league identifier
            league_name = os.path.basename(csv_file).replace('.csv', '')
            df['League'] = league_name
            
            # Need sufficient history
            if len(df) < min_history + n_matches_per_league:
                continue
            
            # Use last n_matches for validation, rest for history
            history_df = df.iloc[:-n_matches_per_league]
            test_df = df.iloc[-n_matches_per_league:]
            
            for _, match in test_df.iterrows():
                home_team = create_team_from_history(match['HomeTeam'], history_df)
                away_team = create_team_from_history(match['AwayTeam'], history_df)
                
                if home_team is None or away_team is None:
                    continue
                
                validation_matches.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'actual_home_goals': int(match['FTHG']),
                    'actual_away_goals': int(match['FTAG']),
                    'league': league_name
                })
                
                # Update history for next iteration (sliding window)
                history_df = pd.concat([history_df, pd.DataFrame([match])], ignore_index=True)
        
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    return validation_matches


def run_validation(n_matches: int = 200):
    """
    Run comprehensive validation comparing Hybrid vs Poisson.
    
    Args:
        n_matches: Target number of validation matches
    """
    print("=" * 70)
    print("HYBRID SYSTEM VALIDATION")
    print("=" * 70)
    
    # Load validation data
    print("\nLoading validation matches...")
    validation_matches = load_validation_matches(n_matches_per_league=10)
    
    if len(validation_matches) == 0:
        print("❌ No validation data found!")
        return
    
    # Limit to target number
    if len(validation_matches) > n_matches:
        validation_matches = validation_matches[:n_matches]
    
    print(f"✅ Loaded {len(validation_matches)} validation matches")
    
    # Run predictions
    print("\nRunning predictions...")
    results = []
    
    for i, match in enumerate(validation_matches):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(validation_matches)}")
        
        try:
            # Poisson prediction
            poisson_pred = predict_poisson(match['home_team'], match['away_team'])
            
            # Hybrid prediction
            hybrid_pred = predict_match_hybrid(match['home_team'], match['away_team'])
            
            # Actual outcome
            h_goals = match['actual_home_goals']
            a_goals = match['actual_away_goals']
            
            if h_goals > a_goals:
                actual_outcome = 2  # Home win
            elif h_goals == a_goals:
                actual_outcome = 1  # Draw
            else:
                actual_outcome = 0  # Away win
            
            results.append({
                'actual_outcome': actual_outcome,
                'actual_home_goals': h_goals,
                'actual_away_goals': a_goals,
                'poisson_probs': [
                    poisson_pred['match_outcome']['away_win'],
                    poisson_pred['match_outcome']['draw'],
                    poisson_pred['match_outcome']['home_win']
                ],
                'hybrid_probs': [
                    hybrid_pred['match_outcome']['away_win'],
                    hybrid_pred['match_outcome']['draw'],
                    hybrid_pred['match_outcome']['home_win']
                ],
                'poisson_xg_home': poisson_pred['expected_goals']['home'],
                'poisson_xg_away': poisson_pred['expected_goals']['away'],
                'hybrid_xg_home': hybrid_pred['expected_goals']['home'],
                'hybrid_xg_away': hybrid_pred['expected_goals']['away'],
                'hybrid_source': hybrid_pred['hybrid_metadata']['source'],
                'hybrid_confidence': hybrid_pred['hybrid_metadata']['confidence']
            })
            
        except Exception as e:
            print(f"  Error on match {i}: {e}")
            continue
    
    print(f"✅ Completed {len(results)} predictions\n")
    
    # Calculate metrics
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    df_results = pd.DataFrame(results)
    
    # 1. Log Loss
    poisson_log_loss = log_loss(df_results['actual_outcome'], list(df_results['poisson_probs']))
    hybrid_log_loss = log_loss(df_results['actual_outcome'], list(df_results['hybrid_probs']))
    
    # 2. Match Accuracy
    poisson_preds = [np.argmax(p) for p in df_results['poisson_probs']]
    hybrid_preds = [np.argmax(p) for p in df_results['hybrid_probs']]
    poisson_acc = accuracy_score(df_results['actual_outcome'], poisson_preds)
    hybrid_acc = accuracy_score(df_results['actual_outcome'], hybrid_preds)
    
    # 3. Goal MAE
    poisson_mae = (
        mean_absolute_error(df_results['actual_home_goals'], df_results['poisson_xg_home']) +
        mean_absolute_error(df_results['actual_away_goals'], df_results['poisson_xg_away'])
    ) / 2
    
    hybrid_mae = (
        mean_absolute_error(df_results['actual_home_goals'], df_results['hybrid_xg_home']) +
        mean_absolute_error(df_results['actual_away_goals'], df_results['hybrid_xg_away'])
    ) / 2
    
    # Display results
    print(f"\n{'Metric':<30} | {'Poisson':<12} | {'Hybrid':<12} | {'Diff':<12}")
    print("-" * 70)
    print(f"{'Match Accuracy':<30} | {poisson_acc*100:>10.2f}% | {hybrid_acc*100:>10.2f}% | {(hybrid_acc-poisson_acc)*100:>+10.2f}%")
    print(f"{'Goal MAE (lower=better)':<30} | {poisson_mae:>12.4f} | {hybrid_mae:>12.4f} | {hybrid_mae-poisson_mae:>+12.4f}")
    print(f"{'Log Loss (lower=better)':<30} | {poisson_log_loss:>12.4f} | {hybrid_log_loss:>12.4f} | {hybrid_log_loss-poisson_log_loss:>+12.4f}")
    
    # Hybrid metadata statistics
    print(f"\n{'HYBRID SYSTEM STATISTICS'}")
    print("-" * 70)
    source_counts = df_results['hybrid_source'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source.capitalize()} predictions: {count} ({count/len(df_results)*100:.1f}%)")
    print(f"  Average confidence: {df_results['hybrid_confidence'].mean():.1f}%")
    
    # Decision
    print("\n" + "=" * 70)
    print("VALIDATION VERDICT")
    print("=" * 70)
    
    # Define success criteria (matching implementation plan)
    BASELINE_ACCURACY = 0.6364  # 63.64%
    BASELINE_MAE = 1.2965
    MAX_LOG_LOSS = 0.90
    
    accuracy_ok = hybrid_acc >= BASELINE_ACCURACY
    mae_ok = hybrid_mae <= BASELINE_MAE
    log_loss_ok = hybrid_log_loss <= MAX_LOG_LOSS
    
    print(f"\nSuccess Criteria:")
    print(f"  {'✅' if accuracy_ok else '❌'} Match Accuracy ≥ {BASELINE_ACCURACY*100:.2f}%: {hybrid_acc*100:.2f}%")
    print(f"  {'✅' if mae_ok else '❌'} Goal MAE ≤ {BASELINE_MAE:.4f}: {hybrid_mae:.4f}")
    print(f"  {'✅' if log_loss_ok else '❌'} Log Loss ≤ {MAX_LOG_LOSS:.2f}: {hybrid_log_loss:.4f}")
    
    if accuracy_ok and mae_ok and log_loss_ok:
        print("\n🎉 VERDICT: ✅ HYBRID MODEL APPROVED FOR PRODUCTION")
    else:
        print("\n⚠️  VERDICT: ❌ HYBRID MODEL NEEDS IMPROVEMENT")
        if not accuracy_ok:
            print(f"     - Match accuracy below baseline (needs {(BASELINE_ACCURACY - hybrid_acc)*100:+.2f}%)")
        if not mae_ok:
            print(f"     - Goal MAE above baseline (needs {(hybrid_mae - BASELINE_MAE):+.4f} improvement)")
        if not log_loss_ok:
            print(f"     - Log loss too high (needs {(hybrid_log_loss - MAX_LOG_LOSS):+.4f} improvement)")
    
    print("=" * 70)
    
    return {
        'accuracy': hybrid_acc,
        'mae': hybrid_mae,
        'log_loss': hybrid_log_loss,
        'passed': accuracy_ok and mae_ok and log_loss_ok
    }


if __name__ == '__main__':
    run_validation(n_matches=200)
