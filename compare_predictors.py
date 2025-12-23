"""
Compare Base vs Enhanced Predictor Performance
===============================================
Tests both predictors on real match data to validate accuracy improvements.
"""

import pandas as pd
import glob
from football_predictor import Team as BaseTeam, predict_match as base_predict
from enhanced_predictor import Team, enhanced_predict_match
import numpy as np

def load_all_matches():
    """Load all match data."""
    csv_files = glob.glob("data/*.csv")
    all_matches = []
    
    for csv_file in csv_files:
        if 'learned_matches' in csv_file:
            continue
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            df.columns = df.columns.str.replace('\ufeff', '').str.strip()
            
            if all(col in df.columns for col in ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']):
                # Add league identifier from filename
                league_code = csv_file.split('\\')[-1].replace('.csv', '').replace('data/', '')
                df['League'] = league_code
                all_matches.append(df)
                print(f"Loaded {len(df)} matches from {league_code}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if all_matches:
        return pd.concat(all_matches, ignore_index=True)
    return pd.DataFrame()

def create_base_team(team_name, df, match_idx, is_home=True):
    """Create basic team for base predictor."""
    previous_matches = df.iloc[:match_idx]
    team_matches = previous_matches[
        (previous_matches['HomeTeam'] == team_name) | (previous_matches['AwayTeam'] == team_name)
    ].tail(5)
    
    goals_scored, goals_conceded, first_half = [], [], []
    
    for _, m in team_matches.iterrows():
        if m['HomeTeam'] == team_name:
            goals_scored.append(int(m['FTHG']))
            goals_conceded.append(int(m['FTAG']))
            first_half.append(int(m.get('HTHG', m['FTHG'] * 0.45)))
        else:
            goals_scored.append(int(m['FTAG']))
            goals_conceded.append(int(m['FTHG']))
            first_half.append(int(m.get('HTAG', m['FTAG'] * 0.45)))
    
    while len(goals_scored) < 5:
        goals_scored.insert(0, 1)
        goals_conceded.insert(0, 1)
        first_half.insert(0, 0)
    
    return BaseTeam(
        name=team_name,
        goals_scored=goals_scored[-5:],
        goals_conceded=goals_conceded[-5:],
        first_half_goals=first_half[-5:]
    )

def create_enhanced_team(team_name, df, match_idx, is_home=True, league='DEFAULT'):
    """Create enhanced team with SOT and home/away form."""
    previous_matches = df.iloc[:match_idx]
    team_matches = previous_matches[
        (previous_matches['HomeTeam'] == team_name) | (previous_matches['AwayTeam'] == team_name)
    ].tail(5)
    
    goals_scored, goals_conceded, first_half = [], [], []
    shots_on_target, sot_conceded = [], []
    home_goals, home_conceded, away_goals, away_conceded = [], [], [], []
    
    for _, m in team_matches.iterrows():
        if m['HomeTeam'] == team_name:
            # Home match for this team
            goals_scored.append(int(m['FTHG']))
            goals_conceded.append(int(m['FTAG']))
            first_half.append(int(m.get('HTHG', m['FTHG'] * 0.45)))
            
            if 'HST' in m and pd.notna(m['HST']):
                shots_on_target.append(int(m['HST']))
            if 'AST' in m and pd.notna(m['AST']):
                sot_conceded.append(int(m['AST']))
            
            home_goals.append(int(m['FTHG']))
            home_conceded.append(int(m['FTAG']))
        else:
            # Away match for this team
            goals_scored.append(int(m['FTAG']))
            goals_conceded.append(int(m['FTHG']))
            first_half.append(int(m.get('HTAG', m['FTAG'] * 0.45)))
            
            if 'AST' in m and pd.notna(m['AST']):
                shots_on_target.append(int(m['AST']))
            if 'HST' in m and pd.notna(m['HST']):
                sot_conceded.append(int(m['HST']))
            
            away_goals.append(int(m['FTAG']))
            away_conceded.append(int(m['FTHG']))
    
    # Pad if needed
    while len(goals_scored) < 5:
        goals_scored.insert(0, 1)
        goals_conceded.insert(0, 1)
        first_half.insert(0, 0)
    
    return Team(
        name=team_name,
        goals_scored=goals_scored[-5:],
        goals_conceded=goals_conceded[-5:],
        first_half_goals=first_half[-5:],
        shots_on_target=shots_on_target[-5:] if shots_on_target else None,
        shots_on_target_conceded=sot_conceded[-5:] if sot_conceded else None,
        home_goals_scored=home_goals[-3:] if len(home_goals) >= 2 else None,
        home_goals_conceded=home_conceded[-3:] if len(home_conceded) >= 2 else None,
        away_goals_scored=away_goals[-3:] if len(away_goals) >= 2 else None,
        away_goals_conceded=away_conceded[-3:] if len(away_conceded) >= 2 else None,
        league=league
    )

def compare_predictors():
    """Compare base vs enhanced predictor performance."""
    print("="*70)
    print("BASE vs ENHANCED PREDICTOR COMPARISON")
    print("="*70)
    
    df = load_all_matches()
    if df.empty:
        print("No data found!")
        return
    
    print(f"\nTotal matches: {len(df)}")
    
    # Stats trackers
    base_stats = {'total_error': [], 'off_by_1': 0, 'btts': 0, 'btts_total': 0, 'exact_top5': 0}
    enh_stats = {'total_error': [], 'off_by_1': 0, 'btts': 0, 'btts_total': 0, 'exact_top5': 0}
    total_tests = 0
    
    # Test on matches from each league
    leagues = df['League'].unique()
    test_indices = []
    
    print(f"Found leagues: {len(leagues)}")
    
    for league in leagues:
        league_indices = df[df['League'] == league].index
        # Get last 50 matches for this league
        if len(league_indices) > 60:
            last_50 = league_indices[-50:]
            test_indices.extend(last_50)
            # print(f"Added 50 matches from {league}")
        else:
            # If small dataset, take last few but leave some for history
            safe_count = max(0, len(league_indices) - 10)
            if safe_count > 0:
                subset = league_indices[-safe_count:]
                test_indices.extend(subset)
                # print(f"Added {len(subset)} matches from {league}")

    print(f"\nTotal matches to test: {len(test_indices)}\n")
    
    for idx in test_indices:
        try:
            match = df.iloc[idx]
            league = match.get('League', 'DEFAULT')
            
            # Create teams
            base_home = create_base_team(match['HomeTeam'], df, idx, True)
            base_away = create_base_team(match['AwayTeam'], df, idx, False)
            
            enh_home = create_enhanced_team(match['HomeTeam'], df, idx, True, league)
            enh_away = create_enhanced_team(match['AwayTeam'], df, idx, False, league)
            
            # Get predictions
            base_pred = base_predict(base_home, base_away)
            enh_pred = enhanced_predict_match(enh_home, enh_away)
            
            # Actual results
            actual_home = int(match['FTHG'])
            actual_away = int(match['FTAG'])
            actual_total = actual_home + actual_away
            actual_btts = (actual_home > 0) and (actual_away > 0)
            
            # Base predictor stats
            base_score = base_pred['full_match_predictions'][0][0]
            base_total = base_score[0] + base_score[1]
            base_error = abs(base_total - actual_total)
            base_stats['total_error'].append(base_error)
            if base_error == 1:
                base_stats['off_by_1'] += 1
            
            base_btts = base_pred['both_teams_score'] > 0.5
            base_stats['btts_total'] += 1
            if base_btts == actual_btts:
                base_stats['btts'] += 1
            
            base_top5 = [s for s, _ in base_pred['full_match_predictions'][:5]]
            if (actual_home, actual_away) in base_top5:
                base_stats['exact_top5'] += 1
            
            # Enhanced predictor stats
            enh_score = enh_pred['full_match_predictions'][0][0]
            enh_total = enh_score[0] + enh_score[1]
            enh_error = abs(enh_total - actual_total)
            enh_stats['total_error'].append(enh_error)
            if enh_error == 1:
                enh_stats['off_by_1'] += 1
            
            enh_btts = enh_pred['both_teams_score'] > 0.5
            enh_stats['btts_total'] += 1
            if enh_btts == actual_btts:
                enh_stats['btts'] += 1
            
            enh_top5 = [s for s, _ in enh_pred['full_match_predictions'][:5]]
            if (actual_home, actual_away) in enh_top5:
                enh_stats['exact_top5'] += 1
            
            total_tests += 1
            
        except Exception as e:
            continue
    
    
    
    # Calculate stats
    base_mae = np.mean(base_stats['total_error'])
    enh_mae = np.mean(enh_stats['total_error'])
    
    # Avoid division by zero
    total_valid = max(1, len(base_stats['total_error']))
    
    base_off1 = base_stats['off_by_1'] / total_valid * 100
    enh_off1 = enh_stats['off_by_1'] / total_valid * 100
    
    base_btts = base_stats['btts'] / max(1, base_stats['btts_total']) * 100
    enh_btts = enh_stats['btts'] / max(1, enh_stats['btts_total']) * 100
    
    base_exact = base_stats['exact_top5'] / max(1, total_tests) * 100
    enh_exact = enh_stats['exact_top5'] / max(1, total_tests) * 100

    report = []
    report.append("="*70)
    report.append("RESULTS")
    report.append("="*70)
    report.append(f"\nMatches Tested: {total_tests}\n")
    report.append("METRIC                    BASE        ENHANCED    IMPROVEMENT")
    report.append("-"*70)
    report.append(f"Total Goals MAE:          {base_mae:.2f}        {enh_mae:.2f}        {(base_mae - enh_mae):.2f}")
    report.append(f"Off-by-1 Rate:            {base_off1:.1f}%      {enh_off1:.1f}%      {base_off1 - enh_off1:+.1f}%")
    report.append(f"BTTS Accuracy:            {base_btts:.1f}%      {enh_btts:.1f}%      {enh_btts - base_btts:+.1f}%")
    report.append(f"Exact Score Top-5:        {base_exact:.1f}%      {enh_exact:.1f}%      {enh_exact - base_exact:+.1f}%")
    report.append("\n" + "="*70)
    report.append("VERDICT")
    report.append("="*70)
    
    improvements = 0
    if enh_mae < base_mae:
        improvements += 1
        report.append("[OK] Enhanced predictor has LOWER total goals error")
    if enh_off1 < base_off1:
        improvements += 1
        report.append("[OK] Enhanced predictor has LOWER off-by-1 rate")
    if enh_btts > base_btts:
        improvements += 1
        report.append("[OK] Enhanced predictor has BETTER BTTS accuracy")
    if enh_exact > base_exact:
        improvements += 1
        report.append("[OK] Enhanced predictor has BETTER exact score accuracy")
        
    report.append(f"\nEnhanced predictor is better in {improvements}/4 metrics")

    rec = ""
    if improvements >= 3:
        rec = "\nRECOMMENDATION: USE ENHANCED PREDICTOR - Significant improvement!"
        use_enhanced = True
    elif improvements >= 2:
        rec = "\nRECOMMENDATION: CONSIDER ENHANCED PREDICTOR - Moderate improvement"
        use_enhanced = True
    else:
        rec = "\nRECOMMENDATION: STICK WITH BASE PREDICTOR - Not enough improvement"
        use_enhanced = False
    report.append(rec)
    
    # Print to console
    for line in report:
        print(line)
        
    # Write to file
    with open('final_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
        
    return use_enhanced

if __name__ == "__main__":
    use_enhanced = compare_predictors()

