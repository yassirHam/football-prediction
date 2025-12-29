"""
Simplified First-Half Top-2 Coverage Evaluation
===============================================

DIAGNOSTIC ONLY - evaluates a sample of recent matches
"""

import pandas as pd
import numpy as np
from football_predictor import predict_match, predict_score_probabilities, Team
from pathlib import Path
import glob

print("\n" + "="*80)
print("FIRST-HALF TOP-2 COVERAGE EVALUATION (Simplified)")
print("="*80)
print("\n⚠️  DIAGNOSTIC MODE: Model remains unchanged\n")

# Load sample data file with half-time scores
sample_file = 'data/E0_2425.csv'  # Recent season with HT data

try:
    df = pd.read_csv(sample_file, encoding='utf-8-sig')
    df.columns = df.columns.str.replace('ï»¿', '').str.strip()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date', 'HTHG', 'HTAG']).sort_values('Date')
    
    print(f"✅ Loaded {len(df)} matches from {sample_file}")
    
    # Use last 50 matches as test set
    test_df = df.tail(50).copy()
    history_df = df.head(len(df) - 50).copy()
    
    print(f"  Training period: {history_df['Date'].min()} to {history_df['Date'].max()}")
    print(f"  Test period: {test_df['Date'].min()} to {test_df['Date'].max()}")
    
    results = []
    
    print("\n📊 Evaluating predictions...")
    for idx, match in test_df.iterrows():
        # Simple team creation from last 5 matches
        home = match['HomeTeam']
        away = match['AwayTeam']
        
        h_home_matches = history_df[history_df['HomeTeam'] == home].tail(5)
        a_away_matches = history_df[history_df['AwayTeam'] == away].tail(5)
        
        if len(h_home_matches) < 3 or len(a_away_matches) < 3:
            continue
        
        home_team = Team(
            name=home,
            goals_scored=h_home_matches['FTHG'].tolist(),
            goals_conceded=h_home_matches['FTAG'].tolist(),
            first_half_goals=h_home_matches['HTHG'].tolist()
        )
        
        away_team = Team(
            name=away,
            goals_scored=a_away_matches['FTAG'].tolist(),
            goals_conceded=a_away_matches['FTHG'].tolist(),
            first_half_goals=a_away_matches['HTAG'].tolist()
        )
        
        # Get prediction
        pred = predict_match(home_team, away_team)
        
        # Get first-half probabilities (model already provides these!)
        ht_probs_list = pred.get('first_half_predictions', [])
        
        # Convert list format to dict
        ht_probs = {score: prob for score, prob in ht_probs_list}
        
        if not ht_probs:
            continue
            
        sorted_probs = sorted(ht_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Top-2
        top1 = sorted_probs[0]
        top2 = sorted_probs[1]
        
        # Actual
        actual = (int(match['HTHG']), int(match['HTAG']))
        
        # Check
        top1_hit = (actual == top1[0])
        top2_hit = (actual in [top1[0], top2[0]])
        
        results.append({
            'actual': actual,
            'top1': top1[0],
            'top1_prob': top1[1],
            'top2': top2[0],
            'top2_prob': top2[1],
            'top1_hit': top1_hit,
            'top2_hit': top2_hit,
            'confidence': pred.get('prediction_quality', 75.0)  # Use quality as proxy for confidence
        })
    
    # RESULTS
    results_df = pd.DataFrame(results)
    n = len(results_df)
    
    print(f"✅ Evaluated {n} matches\n")
    
    print("="*80)
    print("RESULTS")
    print("="*80)
    
    # Table 1: Global Performance
    top1_rate = results_df['top1_hit'].mean() * 100
    top2_rate = results_df['top2_hit'].mean() * 100
    avg_prob_sum = (results_df['top1_prob'] + results_df['top2_prob']).mean() * 100
    
    print("\nTable 1 — Global Performance")
    print("-" * 50)
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 50)
    print(f"{'Matches evaluated':<30} {n:>15}")
    print(f"{'Top-1 hit rate':<30} {top1_rate:>14.1f}%")
    print(f"{'Top-2 coverage rate':<30} {top2_rate:>14.1f}%")
    print(f"{'Avg Top-2 probability sum':<30} {avg_prob_sum:>14.1f}%")
    print("-" * 50)
    
    # Table 2: Confidence Buckets
    results_df['conf_bucket'] = pd.cut(results_df['confidence'], bins=[0, 50, 65, 100], labels=['<50%', '50-65%', '>65%'])
    
    print("\nTable 2 — Confidence Buckets")
    print("-" * 60)
    print(f"{'Confidence Range':<20} {'Matches':>15} {'Top-2 Coverage':>20}")
    print("-" * 60)
    
    for bucket in ['<50%', '50-65%', '>65%']:
        subset = results_df[results_df['conf_bucket'] == bucket]
        if len(subset) > 0:
            cov = subset['top2_hit'].mean() * 100
            print(f"{bucket:<20} {len(subset):>15} {cov:>19.1f}%")
        else:
            print(f"{bucket:<20} {0:>15} {'N/A':>20}")
    print("-" * 60)
    
    # INTERPRETATION
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    random_top1 = 100 / 16  # 16 possible scores
    random_top2 = 200 / 16
    
    print(f"\n📊 **Overall Performance:**")
    print(f"   • Top-2 coverage: {top2_rate:.1f}%")
    print(f"   • Top-1 hit rate: {top1_rate:.1f}%")
    print(f"   • Improvement from Top-1 to Top-2: +{top2_rate - top1_rate:.1f} percentage points")
    
    print(f"\n📈 **Comparison to Random Baseline:**")
    print(f"   • Random Top-1: {random_top1:.1f}% (model: {top1_rate:.1f}%)")
    print(f"   • Random Top-2: {random_top2:.1f}% (model: {top2_rate:.1f}%)")
    print(f"   • Model improvement over random: +{top2_rate - random_top2:.1f} percentage points")
    
    print(f"\n💡 **Key Findings:**")
    print(f"   1. Top-2 predictions cover {top2_rate:.1f}% of actual first-half scores")
    print(f"   2. This is {top2_rate - random_top2:.1f}% points above random guessing")
    print(f"   3. Average probability mass in Top-2: {avg_prob_sum:.1f}%")
    
    # SAFETY STATEMENT
    print("\n" + "="*80)
    print("SAFETY VERIFICATION")
    print("="*80)
    print("\nThis analysis is diagnostic only and does not modify the model or introduce overfitting.")
    
    # Save
    results_df.to_csv('first_half_top2_results.csv', index=False)
    print("\n📁 Results saved to: first_half_top2_results.csv\n")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
