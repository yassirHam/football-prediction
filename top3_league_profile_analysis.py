"""
Top-3 Coverage Analysis: League & Match Profile Evaluation
==========================================================

DIAGNOSTIC ONLY - analyzes where and when Top-3 predictions work best

Part A: League-by-league Top-3 coverage
Part B: xG-based match profile slicing
"""

import pandas as pd
import numpy as np
from football_predictor import predict_match, Team
from pathlib import Path
import glob

print("\n" + "="*80)
print("TOP-3 COVERAGE: LEAGUE & MATCH PROFILE ANALYSIS")
print("="*80)
print("\n⚠️  DIAGNOSTIC MODE: Model remains unchanged\n")

# Load ALL leagues with half-time data
print("Loading multi-league test data...")
csv_files = glob.glob('data/*.csv')
all_dfs = []

for f in csv_files:
    if 'learned_matches' in f or 'combined' in f:
        continue
    try:
        df = pd.read_csv(f, encoding='utf-8-sig')
        df.columns = df.columns.str.replace('ï»¿', '').str.strip()
        
        # Extract league from filename
        league = Path(f).stem.split('_')[0]
        df['League'] = league
        
        if 'HTHG' in df.columns and 'HTAG' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date', 'HTHG', 'HTAG']).sort_values('Date')
            all_dfs.append(df)
    except Exception:
        continue

if not all_dfs:
    print("❌ No data with first-half scores found!")
    exit(1)

full_df = pd.concat(all_dfs, ignore_index=True).sort_values('Date')
print(f"✅ Loaded {len(full_df)} total matches from {len(all_dfs)} leagues")

# Use last 10% as test set (ensures sufficient history)
dates_sorted = full_df['Date'].sort_values()
cutoff_idx = int(len(dates_sorted) * 0.90)
cutoff_date = dates_sorted.iloc[cutoff_idx]

test_df = full_df[full_df['Date'] >= cutoff_date].copy()
print(f"✅ Test set: {len(test_df)} matches")
print(f"   Cutoff: {cutoff_date}")
print(f"   Period: {test_df['Date'].min()} to {test_df['Date'].max()}")

# Generate predictions
print("\n📊 Generating predictions...")
results = []

for idx, match in test_df.iterrows():
    home = match['HomeTeam']
    away = match['AwayTeam']
    
    # Get history for this team from full dataset
    history = full_df[full_df['Date'] < match['Date']]
    h_home = history[history['HomeTeam'] == home].tail(5)
    a_away = history[history['AwayTeam'] == away].tail(5)
    
    if len(h_home) < 3 or len(a_away) < 3:
        continue
    
    home_team = Team(
        name=home,
        goals_scored=h_home['FTHG'].tolist(),
        goals_conceded=h_home['FTAG'].tolist(),
        first_half_goals=h_home['HTHG'].tolist()
    )
    
    away_team = Team(
        name=away,
        goals_scored=a_away['FTAG'].tolist(),
        goals_conceded=a_away['FTHG'].tolist(),
        first_half_goals=a_away['HTAG'].tolist()
    )
    
    try:
        pred = predict_match(home_team, away_team)
        
        # Get first-half probabilities
        ht_probs_list = pred.get('first_half_predictions', [])
        ht_probs = {score: prob for score, prob in ht_probs_list}
        
        if not ht_probs:
            continue
        
        sorted_probs = sorted(ht_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Top-3
        top1 = sorted_probs[0]
        top2 = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0)
        top3 = sorted_probs[2] if len(sorted_probs) > 2 else (None, 0)
        
        actual = (int(match['HTHG']), int(match['HTAG']))
        
        # Hits
        top1_hit = (actual == top1[0])
        top3_hit = (actual in [top1[0], top2[0], top3[0]])
        
        # Extract xG values for slicing
        xg_home = pred.get('home_xG', 1.5)  # Fallback if key missing
        xg_away = pred.get('away_xG', 1.5)
        
        # Calculate match characteristics
        xg_total = xg_home + xg_away
        xg_diff = abs(xg_home - xg_away)
        
        results.append({
            'league': match['League'],
            'actual': actual,
            'top1': top1[0],
            'top1_prob': top1[1],
            'top2_prob': top2[1],
            'top3_prob': top3[1],
            'top1_hit': top1_hit,
            'top3_hit': top3_hit,
            'top3_prob_sum': top1[1] + top2[1] + top3[1],
            'xg_home': xg_home,
            'xg_away': xg_away,
            'xg_total': xg_total,
            'xg_diff': xg_diff
        })
    except Exception:
        continue

results_df = pd.DataFrame(results)
print(f"✅ Evaluated {len(results_df)} matches\n")

# ============================================================================
# PART A — LEAGUE-BY-LEAGUE TOP-3 COVERAGE
# ============================================================================

print("="*80)
print("PART A — LEAGUE-BY-LEAGUE TOP-3 COVERAGE")
print("="*80)

print("\nTable A — Top-3 Coverage by League (≥20 test matches)")
print("-" * 75)
print(f"{'League':<10} {'Matches':>10} {'Top-1 %':>10} {'Top-3 %':>10} {'Avg Top-3 Prob':>20}")
print("-" * 75)

league_stats = []
for league in sorted(results_df['league'].unique()):
    league_data = results_df[results_df['league'] == league]
    
    if len(league_data) >= 20:
        matches = len(league_data)
        top1_rate = league_data['top1_hit'].mean() * 100
        top3_rate = league_data['top3_hit'].mean() * 100
        avg_prob = league_data['top3_prob_sum'].mean() * 100
        
        print(f"{league:<10} {matches:>10} {top1_rate:>9.1f}% {top3_rate:>9.1f}% {avg_prob:>19.1f}%")
        
        league_stats.append({
            'league': league,
            'matches': matches,
            'top1_rate': top1_rate,
            'top3_rate': top3_rate,
            'avg_prob': avg_prob
        })

print("-" * 75)

# ============================================================================
# PART B — xG-BASED MATCH PROFILE SLICING
# ============================================================================

print("\n" + "="*80)
print("PART B — xG-BASED MATCH PROFILE SLICING")
print("="*80)

# Table B1: xG Total (Game Openness)
print("\nTable B1 — xG Total (Game Openness)")
print("-" * 70)
print(f"{'xG Total Bucket':<20} {'Matches':>12} {'Top-3 Coverage':>18} {'Avg Top-3 Prob':>18}")
print("-" * 70)

# Define buckets
results_df['xg_total_bucket'] = pd.cut(
    results_df['xg_total'],
    bins=[0, 2.2, 3.0, 10],
    labels=['Low (<2.2)', 'Medium (2.2-3.0)', 'High (>3.0)']
)

for bucket in ['Low (<2.2)', 'Medium (2.2-3.0)', 'High (>3.0)']:
    bucket_data = results_df[results_df['xg_total_bucket'] == bucket]
    if len(bucket_data) > 0:
        matches = len(bucket_data)
        coverage = bucket_data['top3_hit'].mean() * 100
        avg_prob = bucket_data['top3_prob_sum'].mean() * 100
        print(f"{bucket:<20} {matches:>12} {coverage:>17.1f}% {avg_prob:>17.1f}%")
    else:
        print(f"{bucket:<20} {0:>12} {'N/A':>18} {'N/A':>18}")

print("-" * 70)

# Table B2: xG Difference (Mismatch)
print("\nTable B2 — xG Difference (Match Balance)")
print("-" * 70)
print(f"{'xG Diff Bucket':<25} {'Matches':>12} {'Top-3 Coverage':>18} {'Avg Top-3 Prob':>13}")
print("-" * 70)

results_df['xg_diff_bucket'] = pd.cut(
    results_df['xg_diff'],
    bins=[0, 0.5, 1.2, 10],
    labels=['Balanced (<0.5)', 'Moderate (0.5-1.2)', 'Strong mismatch (>1.2)']
)

for bucket in ['Balanced (<0.5)', 'Moderate (0.5-1.2)', 'Strong mismatch (>1.2)']:
    bucket_data = results_df[results_df['xg_diff_bucket'] == bucket]
    if len(bucket_data) > 0:
        matches = len(bucket_data)
        coverage = bucket_data['top3_hit'].mean() * 100
        avg_prob = bucket_data['top3_prob_sum'].mean() * 100
        print(f"{bucket:<25} {matches:>12} {coverage:>17.1f}% {avg_prob:>12.1f}%")
    else:
        print(f"{bucket:<25} {0:>12} {'N/A':>18} {'N/A':>13}")

print("-" * 70)

# ============================================================================
# INTERPRETATION
# ============================================================================

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

# League analysis
if league_stats:
    league_df = pd.DataFrame(league_stats).sort_values('top3_rate', ascending=False)
    
    print("\n📊 **League Performance:**")
    
    # Best leagues
    high_perf = league_df[league_df['top3_rate'] >= 65]
    if len(high_perf) > 0:
        print(f"\n   ✅ Leagues exceeding 65% Top-3 coverage:")
        for _, row in high_perf.iterrows():
            print(f"      • {row['league']}: {row['top3_rate']:.1f}% ({row['matches']} matches)")
    
    # Underperforming
    low_perf = league_df[league_df['top3_rate'] < 60]
    if len(low_perf) > 0:
        print(f"\n   ⚠️ Leagues below 60% coverage:")
        for _, row in low_perf.iterrows():
            print(f"      • {row['league']}: {row['top3_rate']:.1f}% ({row['matches']} matches)")
    
    # Overall range
    print(f"\n   Range: {league_df['top3_rate'].min():.1f}% to {league_df['top3_rate'].max():.1f}%")
    print(f"   Variance: {league_df['top3_rate'].std():.1f}pp standard deviation")

# xG Total analysis
print("\n📈 **Game Openness Impact (xG Total):**")
for bucket in ['Low (<2.2)', 'Medium (2.2-3.0)', 'High (>3.0)']:
    bucket_data = results_df[results_df['xg_total_bucket'] == bucket]
    if len(bucket_data) > 0:
        cov = bucket_data['top3_hit'].mean() * 100
        print(f"   • {bucket}: {cov:.1f}% coverage")

low_xg = results_df[results_df['xg_total_bucket'] == 'Low (<2.2)']
high_xg = results_df[results_df['xg_total_bucket'] == 'High (>3.0)']
if len(low_xg) > 0 and len(high_xg) > 0:
    low_cov = low_xg['top3_hit'].mean() * 100
    high_cov = high_xg['top3_hit'].mean() * 100
    if low_cov > high_cov:
        print(f"\n   → Low-scoring games have BETTER coverage ({low_cov:.1f}% vs {high_cov:.1f}%)")
        print(f"      Defensive matches are more predictable")
    else:
        print(f"\n   → Open games have BETTER coverage ({high_cov:.1f}% vs {low_cov:.1f}%)")
        print(f"      High-scoring potential improves predictions")

# xG Diff analysis
print("\n⚖️ **Match Balance Impact (xG Difference):**")
for bucket in ['Balanced (<0.5)', 'Moderate (0.5-1.2)', 'Strong mismatch (>1.2)']:
    bucket_data = results_df[results_df['xg_diff_bucket'] == bucket]
    if len(bucket_data) > 0:
        cov = bucket_data['top3_hit'].mean() * 100
        print(f"   • {bucket}: {cov:.1f}% coverage")

balanced = results_df[results_df['xg_diff_bucket'] == 'Balanced (<0.5)']
mismatch = results_df[results_df['xg_diff_bucket'] == 'Strong mismatch (>1.2)']
if len(balanced) > 0 and len(mismatch) > 0:
    bal_cov = balanced['top3_hit'].mean() * 100
    mis_cov = mismatch['top3_hit'].mean() * 100
    if mis_cov > bal_cov:
        print(f"\n   → Imbalanced games have BETTER coverage ({mis_cov:.1f}% vs {bal_cov:.1f}%)")
        print(f"      Clear favorites make predictions more reliable")
    else:
        print(f"\n   → Balanced games have BETTER coverage ({bal_cov:.1f}% vs {mis_cov:.1f}%)")
        print(f"      Even matchups don't hurt prediction quality")

# Key factor
print("\n💡 **Top-3 Strength Depends On:**")
if league_stats:
    league_variance = pd.DataFrame(league_stats)['top3_rate'].std()
    
    # Calculate xG variance
    xg_total_variance = results_df.groupby('xg_total_bucket')['top3_hit'].mean().std() * 100
    xg_diff_variance = results_df.groupby('xg_diff_bucket')['top3_hit'].mean().std() * 100
    
    print(f"   • League variance: {league_variance:.1f}pp")
    print(f"   • xG Total variance: {xg_total_variance:.1f}pp")
    print(f"   • xG Diff variance: {xg_diff_variance:.1f}pp")
    
    if league_variance > max(xg_total_variance, xg_diff_variance):
        print(f"\n   → LEAGUE STRUCTURE is the dominant factor")
        print(f"      Different leagues have different predictability")
    else:
        print(f"\n   → MATCH PROFILE is the dominant factor")
        print(f"      xG characteristics matter more than league")

# Safety statement
print("\n" + "="*80)
print("SAFETY VERIFICATION")
print("="*80)
print("\nThis analysis is diagnostic only and does not modify the model or introduce overfitting.")
print()

# Save
results_df.to_csv('top3_league_profile_analysis.csv', index=False)
print("📁 Results saved to: top3_league_profile_analysis.csv\n")
