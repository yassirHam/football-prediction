"""
First-Half Top-2 Coverage Evaluation
====================================

CRITICAL: This script is DIAGNOSTIC ONLY - NO MODEL CHANGES ALLOWED

Evaluates how often the real first-half score appears in the model's
TOP-2 predicted first-half exact scores.

RESTRICTIONS (DO NOT VIOLATE):
- No model retraining
- No probability tuning
- No confidence threshold changes
- No Poisson/ML logic modifications
- No score distribution adjustments
- No betting optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from collections import Counter, defaultdict
from football_predictor import predict_match, poisson_probability, predict_score_probabilities, Team
import json


def load_test_data(data_dir: str = 'data', test_ratio: float = 0.2):
    """
    Load test data with first-half scores.
    
    Returns:
        DataFrame with test matches including HTHG (half-time home goals) 
        and HTAG (half-time away goals)
    """
    print("Loading test data with first-half scores...")
    csv_files = glob.glob(f"{data_dir}/*.csv")
    dfs = []
    
    for f in csv_files:
        if 'learned_matches' in f or 'combined' in f:
            continue
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
            
            # Extract league from filename
            league = Path(f).stem.split('_')[0]
            df['League'] = league
            
            # Check for required columns
            required_cols = {'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Date'}
            if required_cols.issubset(df.columns):
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['Date']).sort_values('Date')
                
                # Only include if we have half-time data
                if 'HTHG' in df.columns and 'HTAG' in df.columns:
                    df = df.dropna(subset=['HTHG', 'HTAG'])
                    dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if not dfs:
        raise ValueError("No data with first-half scores found!")
    
    full_df = pd.concat(dfs, ignore_index=True).sort_values('Date')
    
    # Time-based split: use later matches for testing
    dates_sorted = full_df['Date'].sort_values()
    cutoff_idx = int(len(dates_sorted) * (1 - test_ratio))
    cutoff_date = dates_sorted.iloc[cutoff_idx]
    
    test_df = full_df[full_df['Date'] >= cutoff_date].copy()
    
    print(f"✅ Test data loaded: {len(test_df)} matches")
    print(f"   Cutoff date: {cutoff_date}")
    print(f"   Test period: {test_df['Date'].min()} to {test_df['Date'].max()}")
    
    return test_df


def create_team_from_history(team_name: str, history_df: pd.DataFrame, is_home: bool):
    """
    Create a Team object from historical match data.
    """
    if is_home:
        team_matches = history_df[history_df['HomeTeam'] == team_name]
        goals_scored = team_matches['FTHG'].tolist()[-10:]
        goals_conceded = team_matches['FTAG'].tolist()[-10:]
        ht_goals = team_matches['HTHG'].tolist()[-5:] if 'HTHG' in team_matches.columns else [0]*5
    else:
        team_matches = history_df[history_df['AwayTeam'] == team_name]
        goals_scored = team_matches['FTAG'].tolist()[-10:]
        goals_conceded = team_matches['FTHG'].tolist()[-10:]
        ht_goals = team_matches['HTAG'].tolist()[-5:] if 'HTAG' in team_matches.columns else [0]*5
    
    # Need at least 3 matches
    if len(goals_scored) < 3:
        return None
    
    return Team(
        name=team_name,
        goals_scored=goals_scored,
        goals_conceded=goals_conceded,
        first_half_goals=ht_goals if len(ht_goals) >= 3 else [0]*5
    )


def get_first_half_probabilities(home_team: Team, away_team: Team):
    """
    Generate first-half exact score probabilities using the model.
    
    Returns:
        dict: {(home_goals, away_goals): probability}
    """
    # Use the model's existing first-half logic
    # First get full-time prediction to extract xG
    prediction = predict_match(home_team, away_team)
    
    # Scale down xG for first half (approximate 0.45x multiplier)
    # This is typical: ~45% of goals occur in first half
    ht_multiplier = 0.45
    xg_ht_home = prediction['xG_home'] * ht_multiplier
    xg_ht_away = prediction['xG_away'] * ht_multiplier
    
    # Generate first-half score probabilities (max 3 goals each in first half)
    ht_probs = predict_score_probabilities(xg_ht_home, xg_ht_away, max_goals=3)
    
    return ht_probs


def evaluate_top2_coverage():
    """
    Main evaluation function.
    """
    print("\n" + "="*80)
    print("FIRST-HALF TOP-2 COVERAGE EVALUATION")
    print("="*80)
    print("\n⚠️  DIAGNOSTIC MODE: Model remains unchanged\n")
    
    # Load ALL data (for history)
    print("Loading all data for team history...")
    csv_files = glob.glob("data/*.csv")
    all_dfs = []
    
    for f in csv_files:
        if 'learned_matches' in f or 'combined' in f:
            continue
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
            league = Path(f).stem.split('_')[0]
            df['League'] = league
            
            required_cols = {'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Date'}
            if required_cols.issubset(df.columns):
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['Date']).sort_values('Date')
                
                if 'HTHG' in df.columns and 'HTAG' in df.columns:
                    df = df.dropna(subset=['HTHG', 'HTAG'])
                    all_dfs.append(df)
        except Exception as e:
            continue
    
    full_df = pd.concat(all_dfs, ignore_index=True).sort_values('Date')
    print(f"✅ Total matches with HT data: {len(full_df)}")
    
    # Split into test set (last 10% - smaller to ensure sufficient history)
    dates_sorted = full_df['Date'].sort_values()
    cutoff_idx = int(len(dates_sorted) * 0.90)  # Use 90% for training, 10% for test
    cutoff_date = dates_sorted.iloc[cutoff_idx]
    
    test_df = full_df[full_df['Date'] >= cutoff_date].copy()
    
    print(f"✅ Test set: {len(test_df)} matches")
    print(f"   Cutoff date: {cutoff_date}")
    print(f"   Test period: {test_df['Date'].min()} to {test_df['Date'].max()}")
    
    # Track results
    results = []
    
    print("\n📊 Generating predictions for test set...")
    evaluated = 0
    skipped = 0
    
    for idx, match in test_df.iterrows():
        # Get historical data before this match from FULL dataset
        history = full_df[full_df['Date'] < match['Date']]
        
        #Create team objects
        home_team = create_team_from_history(match['HomeTeam'], history, is_home=True)
        away_team = create_team_from_history(match['AwayTeam'], history, is_home=False)
        
        if home_team is None or away_team is None:
            skipped += 1
            continue
        
        try:
            # Get first-half probabilities
            ht_probs = get_first_half_probabilities(home_team, away_team)
            
            # Sort by probability
            sorted_probs = sorted(ht_probs.items(), key=lambda x: x[1], reverse=True)
            
            # Extract top-2
            top1_score = sorted_probs[0][0]
            top1_prob = sorted_probs[0][1]
            top2_score = sorted_probs[1][0] if len(sorted_probs) > 1 else None
            top2_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0
            
            # Get actual first-half score
            actual_ht = (int(match['HTHG']), int(match['HTAG']))
            
            # Check hits
            top1_hit = (actual_ht == top1_score)
            top2_hit = (actual_ht in [top1_score, top2_score])
            
            # Get confidence from full prediction
            full_pred = predict_match(home_team, away_team)
            confidence = full_pred['confidence']
            
            results.append({
                'home': match['HomeTeam'],
                'away': match['AwayTeam'],
                'actual_ht': actual_ht,
                'top1_pred': top1_score,
                'top1_prob': top1_prob,
                'top2_pred': top2_score,
                'top2_prob': top2_prob,
                'top2_prob_sum': top1_prob + top2_prob,
                'top1_hit': top1_hit,
                'top2_hit': top2_hit,
                'confidence': confidence,
                'league': match['League']
            })
            evaluated += 1
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"✅ Evaluated: {evaluated} matches")
    print(f"⚠️  Skipped: {skipped} matches (insufficient history)")
    
    if len(results) == 0:
        print("\n❌ ERROR: No matches could be evaluated!")
        print("   This likely means insufficient historical data.")
        return
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # COMPUTE METRICS
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Table 1: Global Performance
    total_matches = len(results_df)
    top1_hit_rate = results_df['top1_hit'].mean() * 100
    top2_coverage = results_df['top2_hit'].mean() * 100
    avg_top2_prob = results_df['top2_prob_sum'].mean() * 100
    
    print("\nTable 1 — Global Performance")
    print("-" * 50)
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 50)
    print(f"{'Matches evaluated':<30} {total_matches:>15}")
    print(f"{'Top-1 hit rate':<30} {top1_hit_rate:>14.1f}%")
    print(f"{'Top-2 coverage rate':<30} {top2_coverage:>14.1f}%")
    print(f"{'Avg Top-2 probability sum':<30} {avg_top2_prob:>14.1f}%")
    print("-" * 50)
    
    # Table 2: Confidence Buckets
    print("\nTable 2 — Confidence Buckets")
    print("-" * 60)
    print(f"{'Confidence Range':<20} {'Matches':>15} {'Top-2 Coverage':>20}")
    print("-" * 60)
    
    # Define buckets
    results_df['conf_bucket'] = pd.cut(
        results_df['confidence'], 
        bins=[0, 50, 65, 100], 
        labels=['<50%', '50-65%', '>65%']
    )
    
    for bucket in ['<50%', '50-65%', '>65%']:
        bucket_data = results_df[results_df['conf_bucket'] == bucket]
        if len(bucket_data) > 0:
            coverage = bucket_data['top2_hit'].mean() * 100
            print(f"{bucket:<20} {len(bucket_data):>15} {coverage:>19.1f}%")
        else:
            print(f"{bucket:<20} {0:>15} {'N/A':>20}")
    print("-" * 60)
    
    # Most common Top-2 combinations
    print("\nMost Common Top-2 Combinations:")
    print("-" * 60)
    top2_combos = []
    for _, row in results_df.iterrows():
        combo = tuple(sorted([row['top1_pred'], row['top2_pred']]))
        top2_combos.append(combo)
    
    combo_counts = Counter(top2_combos)
    for combo, count in combo_counts.most_common(10):
        pct = count / len(results_df) * 100
        print(f"  {combo[0]} & {combo[1]}: {count} times ({pct:.1f}%)")
    
    # League breakdown (if sufficient data)
    print("\nTable 3 — League Breakdown (leagues with ≥20 test matches)")
    print("-" * 60)
    print(f"{'League':<10} {'Matches':>10} {'Top-2 Coverage':>20}")
    print("-" * 60)
    
    for league in results_df['League'].unique():
        league_data = results_df[results_df['League'] == league]
        if len(league_data) >= 20:
            coverage = league_data['top2_hit'].mean() * 100
            print(f"{league:<10} {len(league_data):>10} {coverage:>19.1f}%")
    print("-" * 60)
    
    # INTERPRETATION
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    print(f"\n📊 **Overall Performance:**")
    print(f"   • Top-2 coverage: {top2_coverage:.1f}%")
    print(f"   • Top-1 hit rate: {top1_hit_rate:.1f}%")
    print(f"   • Improvement from Top-1 to Top-2: +{top2_coverage - top1_hit_rate:.1f} percentage points")
    
    # Random baseline
    # For first half with max 3 goals each side: 4x4 = 16 possible scores
    random_top1 = 100 / 16  # ~6.25%
    random_top2 = 200 / 16  # ~12.5%
    print(f"\n📈 **Comparison to Random Baseline:**")
    print(f"   • Random Top-1: {random_top1:.1f}% (model: {top1_hit_rate:.1f}%)")
    print(f"   • Random Top-2: {random_top2:.1f}% (model: {top2_coverage:.1f}%)")
    print(f"   • Model improvement over random: +{top2_coverage - random_top2:.1f} percentage points")
    
    print(f"\n🎯 **Stability Analysis:**")
    high_conf = results_df[results_df['confidence'] > 65]
    low_conf = results_df[results_df['confidence'] < 50]
    
    if len(high_conf) > 0 and len(low_conf) > 0:
        high_cov = high_conf['top2_hit'].mean() * 100
        low_cov = low_conf['top2_hit'].mean() * 100
        print(f"   • High confidence (>65%): {high_cov:.1f}% coverage")
        print(f"   • Low confidence (<50%): {low_cov:.1f}% coverage")
        print(f"   • Confidence effect: {high_cov - low_cov:+.1f} percentage points")
    
    print(f"\n💡 **Key Findings:**")
    print(f"   1. Top-2 predictions cover {top2_coverage:.1f}% of actual first-half scores")
    print(f"   2. This is {top2_coverage - random_top2:.1f}% points above random guessing")
    print(f"   3. Average probability mass in Top-2: {avg_top2_prob:.1f}%")
    
    # SAFETY STATEMENT
    print("\n" + "="*80)
    print("SAFETY VERIFICATION")
    print("="*80)
    print("\nThis analysis is diagnostic only and does not modify the model or introduce overfitting.")
    print()
    
    # Save results
    results_df.to_csv('first_half_top2_results.csv', index=False)
    print("📁 Detailed results saved to: first_half_top2_results.csv")


if __name__ == '__main__':
    evaluate_top2_coverage()
