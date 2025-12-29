"""
First-Half Top-3 Coverage Evaluation (Extended)
===============================================

DIAGNOSTIC ONLY - extends Top-2 analysis to Top-3 predictions
Measures marginal gains and diminishing returns
"""

import pandas as pd
import numpy as np
from football_predictor import predict_match, Team

print("\n" + "="*80)
print("FIRST-HALF TOP-3 COVERAGE EVALUATION (Extended)")
print("="*80)
print("\n⚠️  DIAGNOSTIC MODE: Model remains unchanged\n")

# Load sample data file with half-time scores
sample_file = 'data/E0_2425.csv'

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
        
        # Get first-half probabilities (model provides these)
        ht_probs_list = pred.get('first_half_predictions', [])
        ht_probs = {score: prob for score, prob in ht_probs_list}
        
        if not ht_probs:
            continue
            
        sorted_probs = sorted(ht_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Top-3 (extended from Top-2)
        top1 = sorted_probs[0]
        top2 = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0)
        top3 = sorted_probs[2] if len(sorted_probs) > 2 else (None, 0)
        
        # Actual
        actual = (int(match['HTHG']), int(match['HTAG']))
        
        # Check hits
        top1_hit = (actual == top1[0])
        top2_hit = (actual in [top1[0], top2[0]])
        top3_hit = (actual in [top1[0], top2[0], top3[0]])
        
        results.append({
            'actual': actual,
            'top1': top1[0],
            'top1_prob': top1[1],
            'top2': top2[0],
            'top2_prob': top2[1],
            'top3': top3[0],
            'top3_prob': top3[1],
            'top1_hit': top1_hit,
            'top2_hit': top2_hit,
            'top3_hit': top3_hit,
            'confidence': pred.get('prediction_quality', 75.0)
        })
    
    # RESULTS
    results_df = pd.DataFrame(results)
    n = len(results_df)
    
    print(f"✅ Evaluated {n} matches\n")
    
    print("="*80)
    print("RESULTS")
    print("="*80)
    
    # Table 1: Global Coverage
    top1_rate = results_df['top1_hit'].mean() * 100
    top2_rate = results_df['top2_hit'].mean() * 100
    top3_rate = results_df['top3_hit'].mean() * 100
    avg_top2_prob = (results_df['top1_prob'] + results_df['top2_prob']).mean() * 100
    avg_top3_prob = (results_df['top1_prob'] + results_df['top2_prob'] + results_df['top3_prob']).mean() * 100
    
    print("\nTable 1 — Global Coverage")
    print("-" * 50)
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 50)
    print(f"{'Matches evaluated':<30} {n:>15}")
    print(f"{'Top-1 hit rate':<30} {top1_rate:>14.1f}%")
    print(f"{'Top-2 coverage':<30} {top2_rate:>14.1f}%")
    print(f"{'Top-3 coverage':<30} {top3_rate:>14.1f}%")
    print(f"{'Avg Top-3 probability sum':<30} {avg_top3_prob:>14.1f}%")
    print("-" * 50)
    
    # Table 2: Marginal Gain
    gain_1to2 = top2_rate - top1_rate
    gain_2to3 = top3_rate - top2_rate
    
    print("\nTable 2 — Marginal Gain")
    print("-" * 50)
    print(f"{'Transition':<30} {'Gain':>15}")
    print("-" * 50)
    print(f"{'Top-1 → Top-2':<30} {gain_1to2:>14.1f}%")
    print(f"{'Top-2 → Top-3':<30} {gain_2to3:>14.1f}%")
    print("-" * 50)
    
    # Table 3: Confidence Percentiles
    results_df['conf_rank'] = results_df['confidence'].rank(pct=True)
    
    print("\nTable 3 — Confidence Percentiles")
    print("-" * 60)
    print(f"{'Confidence Percentile':<25} {'Matches':>15} {'Top-3 Coverage':>15}")
    print("-" * 60)
    
    # Top 20%
    top20 = results_df[results_df['conf_rank'] >= 0.8]
    if len(top20) > 0:
        cov = top20['top3_hit'].mean() * 100
        print(f"{'Top 20%':<25} {len(top20):>15} {cov:>14.1f}%")
    else:
        print(f"{'Top 20%':<25} {0:>15} {'N/A':>15}")
    
    # Middle 60%
    mid60 = results_df[(results_df['conf_rank'] >= 0.2) & (results_df['conf_rank'] < 0.8)]
    if len(mid60) > 0:
        cov = mid60['top3_hit'].mean() * 100
        print(f"{'Middle 60%':<25} {len(mid60):>15} {cov:>14.1f}%")
    else:
        print(f"{'Middle 60%':<25} {0:>15} {'N/A':>15}")
    
    # Bottom 20%
    bot20 = results_df[results_df['conf_rank'] < 0.2]
    if len(bot20) > 0:
        cov = bot20['top3_hit'].mean() * 100
        print(f"{'Bottom 20%':<25} {len(bot20):>15} {cov:>14.1f}%")
    else:
        print(f"{'Bottom 20%':<25} {0:>15} {'N/A':>15}")
    
    print("-" * 60)
    
    # INTERPRETATION
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    print(f"\n📊 **Coverage Progression:**")
    print(f"   • Top-1:  {top1_rate:.1f}%")
    print(f"   • Top-2:  {top2_rate:.1f}% (+{gain_1to2:.1f}pp)")
    print(f"   • Top-3:  {top3_rate:.1f}% (+{gain_2to3:.1f}pp)")
    
    print(f"\n📉 **Diminishing Returns Analysis:**")
    if gain_2to3 < gain_1to2:
        ratio = (gain_2to3 / gain_1to2) * 100 if gain_1to2 > 0 else 0
        print(f"   • Top-2→Top-3 gain ({gain_2to3:.1f}pp) is {ratio:.0f}% of Top-1→Top-2 gain ({gain_1to2:.1f}pp)")
        print(f"   • ⚠️  DIMINISHING RETURNS CONFIRMED: Adding 3rd prediction is less valuable")
    else:
        print(f"   • Top-2→Top-3 gain ({gain_2to3:.1f}pp) >= Top-1→Top-2 gain ({gain_1to2:.1f}pp)")
        print(f"   • Unusual: 3rd prediction adds as much or more value")
    
    print(f"\n🎯 **Coverage Thresholds:**")
    if top3_rate >= 60:
        print(f"   • ✅ Top-3 crosses 60% threshold ({top3_rate:.1f}%)")
        print(f"   • This represents strong coverage for volatile first-half scores")
    else:
        print(f"   • Top-3 coverage is {top3_rate:.1f}% (below 60% threshold)")
        print(f"   • Gap to 60%: {60 - top3_rate:.1f}pp")
    
    print(f"\n📈 **Probability Mass:**")
    prob_mass_increase = avg_top3_prob - avg_top2_prob
    print(f"   • Top-2 avg probability sum: {avg_top2_prob:.1f}%")
    print(f"   • Top-3 avg probability sum: {avg_top3_prob:.1f}%")
    print(f"   • Additional mass captured: +{prob_mass_increase:.1f}pp")
    
    print(f"\n💡 **Key Findings:**")
    print(f"   1. Top-3 coverage reaches {top3_rate:.1f}%")
    print(f"   2. Marginal gain from 3rd prediction: {gain_2to3:.1f}pp")
    
    if gain_2to3 < 10:
        print(f"   3. ⚠️  Adding 3rd prediction provides minimal additional coverage (<10pp)")
    elif gain_2to3 < gain_1to2 * 0.5:
        print(f"   3. Diminishing returns evident (3rd adds <50% of 2nd's value)")
    else:
        print(f"   3. 3rd prediction still provides substantial value")
    
    # Note on percentiles
    print(f"\n📊 **Confidence Percentile Notes:**")
    if len(top20) == 0 and len(mid60) == 0 and len(bot20) == 0:
        print(f"   • All matches have identical confidence scores")
        print(f"   • Percentile breakdown not meaningful")
    else:
        print(f"   • Matches distributed across confidence levels")
        if len(top20) > 0 and len(bot20) > 0:
            top_cov = top20['top3_hit'].mean() * 100
            bot_cov = bot20['top3_hit'].mean() * 100
            if top_cov > bot_cov:
                print(f"   • High-confidence matches perform better ({top_cov:.1f}% vs {bot_cov:.1f}%)")
            else:
                print(f"   • Confidence doesn't strongly differentiate coverage")
    
    # SAFETY STATEMENT
    print("\n" + "="*80)
    print("SAFETY VERIFICATION")
    print("="*80)
    print("\nThis analysis is diagnostic only and does not modify the model or introduce overfitting.")
    print()
    
    # Save
    results_df.to_csv('first_half_top3_results.csv', index=False)
    print("📁 Results saved to: first_half_top3_results.csv\n")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
