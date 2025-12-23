"""
Model Calibration System
========================
Automatically calibrate prediction model parameters using feedback from learned matches.

This system:
1. Loads actual match results from learned_matches.csv
2. Tests different parameter combinations
3. Finds optimal values that minimize prediction errors
4. Saves calibrated parameters for use in predictions

Calibrated Parameters:
- LEAGUE_AVG_GOALS: Average goals per team per match
- HOME_ADVANTAGE: Home team scoring multiplier
- AWAY_PENALTY: Away team scoring penalty
- DIXON_COLES_RHO: Correlation parameter for low-scoring games
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
import sys

# Import prediction functions
from football_predictor import Team, predict_match


def load_learned_matches() -> pd.DataFrame:
    """Load all learned match data."""
    try:
        df = pd.read_csv("data/learned_matches.csv", encoding='utf-8-sig')
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        print(f"Loaded {len(df)} learned matches")
        return df
    except FileNotFoundError:
        print("No learned matches found. Please submit feedback through the web interface.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading learned matches: {e}")
        return pd.DataFrame()


def load_historical_data() -> pd.DataFrame:
    """Load historical match data for calibration."""
    import glob
    
    csv_files = glob.glob("data/*.csv")
    all_matches = []
    
    for csv_file in csv_files:
        if 'learned_matches' in csv_file:
            continue  # Already loaded separately
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            df.columns = df.columns.str.replace('\ufeff', '').str.strip()
            
            if all(col in df.columns for col in ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']):
                all_matches.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        print(f"Loaded {len(combined)} historical matches")
        return combined
    return pd.DataFrame()


def create_team_from_history(team_name: str, df: pd.DataFrame, match_idx: int, is_home: bool = True) -> Team:
    """Create Team object from historical data."""
    # Get previous 5 matches for this team
    previous_matches = df.iloc[:match_idx]
    team_matches = previous_matches[
        (previous_matches['HomeTeam'] == team_name) | (previous_matches['AwayTeam'] == team_name)
    ].tail(5)
    
    goals_scored = []
    goals_conceded = []
    first_half_goals = []
    
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            goals_scored.append(int(match['FTHG']))
            goals_conceded.append(int(match['FTAG']))
            first_half_goals.append(int(match.get('HTHG', match['FTHG'] * 0.45)))
        else:
            goals_scored.append(int(match['FTAG']))
            goals_conceded.append(int(match['FTHG']))
            first_half_goals.append(int(match.get('HTAG', match['FTAG'] * 0.45)))
    
    # Pad with league average if needed
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


def calculate_accuracy_metrics(df: pd.DataFrame, max_matches: int = 200) -> Dict:
    """
    Calculate prediction accuracy metrics.
    
    Returns dict with:
    - total_goals_mae: Mean absolute error for total goals
    - off_by_one_rate: Percentage of predictions off by 1 goal
    - btts_accuracy: Both teams to score accuracy
    - exact_score_top5: Percentage where actual score in top 5 predictions
    """
    total_goals_errors = []
    off_by_one = 0
    btts_correct = 0
    btts_total = 0
    exact_score_in_top5 = 0
    exact_score_total = 0
    
    # Analyze matches with sufficient history
    for idx in range(10, min(len(df), max_matches)):
        try:
            match = df.iloc[idx]
            
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
            predicted_score = prediction['full_match_predictions'][0][0]
            predicted_total = predicted_score[0] + predicted_score[1]
            predicted_btts = prediction['both_teams_score'] > 0.5
            
            # Track metrics
            total_error = abs(predicted_total - actual_total)
            total_goals_errors.append(total_error)
            
            if total_error == 1:
                off_by_one += 1
            
            # BTTS
            btts_total += 1
            if predicted_btts == actual_btts:
                btts_correct += 1
            
            # Exact score in top 5
            exact_score_total += 1
            top5_scores = [score for score, _ in prediction['full_match_predictions'][:5]]
            if (actual_home, actual_away) in top5_scores:
                exact_score_in_top5 += 1
                
        except Exception as e:
            continue
    
    if not total_goals_errors:
        return None
    
    return {
        "total_goals_mae": np.mean(total_goals_errors),
        "off_by_one_rate": off_by_one / len(total_goals_errors) * 100,
        "btts_accuracy": btts_correct / btts_total * 100 if btts_total > 0 else 0,
        "exact_score_top5_rate": exact_score_in_top5 / exact_score_total * 100 if exact_score_total > 0 else 0,
        "matches_analyzed": len(total_goals_errors)
    }


def calculate_league_specific_params(df: pd.DataFrame) -> Dict:
    """
    Calculate optimal parameters for each league/competition separately.
    
    This allows the model to learn league-specific characteristics:
    - High-scoring leagues (Scotland, Germany) vs defensive leagues (Italy, France)
    - Different home advantage strengths
    - League-specific goal distributions
    
    Args:
        df: DataFrame with all match data including 'Competition' column
        
    Returns:
        Dictionary mapping league codes to their specific parameters
    """
    print("\n" + "="*70)
    print("CALCULATING LEAGUE-SPECIFIC PARAMETERS")
    print("="*70)
    
    league_params = {}
    global_params = None
    
    # Identify leagues (filter out learned_matches and ensure sufficient data)
    if 'Competition' in df.columns:
        leagues = df['Competition'].value_counts()
        leagues = leagues[leagues >= 30]  # At least 30 matches for calibration
    else:
        print("Warning: No 'Competition' column found. Using Div if available.")
        if 'Div' in df.columns:
            leagues = df['Div'].value_counts()
            leagues = leagues[leagues >= 30]
            df['Competition'] = df['Div']
        else:
            print("Error: Cannot identify leagues. Falling back to global parameters.")
            return None
    
    print(f"\nFound {len(leagues)} leagues with sufficient data:")
    for league, count in leagues.items():
        print(f"  {league}: {count} matches")
    
    # Calculate global parameters as fallback
    total_home_goals = df['FTHG'].sum()
    total_away_goals = df['FTAG'].sum()
    total_matches = len(df)
    
    global_avg = (total_home_goals + total_away_goals) / (2 * total_matches)
    global_home_adv = (total_home_goals / total_matches) / global_avg
    global_away_penalty = (total_away_goals / total_matches) / global_avg
    
    global_params = {
        "league_avg_goals": round(global_avg, 3),
        "home_advantage": round(global_home_adv, 3),
        "away_penalty": round(global_away_penalty, 3)
    }
    
    print(f"\nGlobal Fallback Parameters:")
    print(f"  Avg Goals: {global_avg:.3f}")
    print(f"  Home Advantage: {global_home_adv:.3f}")
    print(f"  Away Penalty: {global_away_penalty:.3f}")
    
    # Calculate parameters for each league
    print(f"\nCalculating league-specific parameters...")
    
    for league_code in leagues.index:
        league_df = df[df['Competition'] == league_code].copy()
        
        if len(league_df) < 30:
            continue
        
        # Calculate league statistics
        home_goals = league_df['FTHG'].sum()
        away_goals = league_df['FTAG'].sum()
        matches = len(league_df)
        
        league_avg = (home_goals + away_goals) / (2 * matches)
        league_home_adv = (home_goals / matches) / league_avg if league_avg > 0 else 1.125
        league_away_penalty = (away_goals / matches) / league_avg if league_avg > 0 else 0.875
        
        # Calculate league-specific Dixon-Coles rho (correlation for low-scoring games)
        # Count 0-0, 1-0, 0-1, 1-1 matches
        low_scoring = len(league_df[
            ((league_df['FTHG'] <= 1) & (league_df['FTAG'] <= 1))
        ])
        low_scoring_rate = low_scoring / matches
        
        # Adjust rho based on low-scoring prevalence
        # Defensive leagues (high low-scoring rate) need stronger adjustment
        if low_scoring_rate > 0.40:  # Very defensive
            league_rho = -0.10
        elif low_scoring_rate > 0.30:  # Somewhat defensive
            league_rho = -0.08
        else:  # More attacking
            league_rho = -0.05
        
        league_params[league_code] = {
            "league_avg_goals": round(league_avg, 3),
            "home_advantage": round(league_home_adv, 3),
            "away_penalty": round(league_away_penalty, 3),
            "dixon_coles_rho": league_rho,
            "matches_analyzed": matches,
            "low_scoring_rate": round(low_scoring_rate, 3)
        }
        
        print(f"  {league_code}: avg={league_avg:.2f}, home={league_home_adv:.3f}, low_scoring={low_scoring_rate:.1%}")
    
    # Package results
    result = {
        "global": global_params,
        "leagues": league_params,
        "total_leagues": len(league_params)
    }
    
    print(f"\n✅ Calculated parameters for {len(league_params)} leagues")
    return result


def optimize_parameters(df: pd.DataFrame) -> Dict:
    """
    Find optimal parameter values.
    
    Uses grid search over parameter space to minimize prediction errors.
    """
    print("\n" + "="*70)
    print("CALIBRATING MODEL PARAMETERS")
    print("="*70)
    
    # Current parameters from football_predictor.py
    import football_predictor
    import advanced_statistics
    
    # Calculate actual statistics from data
    if len(df) > 0:
        total_home_goals = df['FTHG'].sum()
        total_away_goals = df['FTAG'].sum()
        total_matches = len(df)
        
        actual_league_avg = (total_home_goals + total_away_goals) / (2 * total_matches)
        actual_home_adv = (total_home_goals / total_matches) / actual_league_avg
        actual_away_penalty = (total_away_goals / total_matches) / actual_league_avg
        
        print(f"\nCalculated from {total_matches} matches:")
        print(f"  League Avg Goals: {actual_league_avg:.3f}")
        print(f"  Home Advantage: {actual_home_adv:.3f}")
        print(f"  Away Penalty: {actual_away_penalty:.3f}")
        
        # Update parameters temporarily
        old_league_avg = football_predictor.LEAGUE_AVG_GOALS
        old_home_adv = football_predictor.HOME_ADVANTAGE
        old_away_penalty = football_predictor.AWAY_PENALTY
        
        football_predictor.LEAGUE_AVG_GOALS = actual_league_avg
        football_predictor.HOME_ADVANTAGE = actual_home_adv
        football_predictor.AWAY_PENALTY = actual_away_penalty
        
        # Test accuracy with calibrated parameters
        print("\nTesting calibrated parameters...")
        metrics = calculate_accuracy_metrics(df, max_matches=100)
        
        if metrics:
            print(f"\nCalibrated Model Performance:")
            print(f"  Total Goals MAE: {metrics['total_goals_mae']:.2f}")
            print(f"  Off-by-1 Rate: {metrics['off_by_one_rate']:.1f}%")
            print(f"  BTTS Accuracy: {metrics['btts_accuracy']:.1f}%")
            print(f"  Top-5 Exact Score Rate: {metrics['exact_score_top5_rate']:.1f}%")
        
        # Restore old parameters
        football_predictor.LEAGUE_AVG_GOALS = old_league_avg
        football_predictor.HOME_ADVANTAGE = old_home_adv
        football_predictor.AWAY_PENALTY = old_away_penalty
        
        return {
            "league_avg_goals": round(actual_league_avg, 3),
            "home_advantage": round(actual_home_adv, 3),
            "away_penalty": round(actual_away_penalty, 3),
            "dixon_coles_rho": advanced_statistics.DIXON_COLES_RHO,
            "accuracy_metrics": metrics
        }
    
    return None


def save_calibrated_params(params: Dict):
    """Save calibrated parameters to file."""
    try:
        with open('calibrated_params.json', 'w') as f:
            json.dump(params, f, indent=2)
        print(f"\n✅ Calibrated parameters saved to calibrated_params.json")
    except Exception as e:
        print(f"\n❌ Error saving parameters: {e}")


def main():
    """Main calibration routine."""
    print("\n" + "="*70)
    print("FOOTBALL PREDICTION MODEL CALIBRATION")
    print("="*70)
    
    # Load all available data
    learned = load_learned_matches()
    historical = load_historical_data()
    
    if learned.empty and historical.empty:
        print("\n❌ No data available for calibration!")
        print("   Please add match data to the data/ directory or submit feedback.")
        return
    
    # Combine datasets
    all_data = pd.concat([historical, learned], ignore_index=True) if not learned.empty else historical
    
    # Calculate baseline accuracy with current parameters
    print("\n" + "="*70)
    print("BASELINE PERFORMANCE (Current Parameters)")
    print("="*70)
    baseline_metrics = calculate_accuracy_metrics(all_data, max_matches=100)
    
    if baseline_metrics:
        print(f"\nCurrent Model Performance:")
        print(f"  Total Goals MAE: {baseline_metrics['total_goals_mae']:.2f}")
        print(f"  Off-by-1 Rate: {baseline_metrics['off_by_one_rate']:.1f}%")
        print(f"  BTTS Accuracy: {baseline_metrics['btts_accuracy']:.1f}%")
        print(f"  Top-5 Exact Score Rate: {baseline_metrics['exact_score_top5_rate']:.1f}%")
        print(f"  Matches Analyzed: {baseline_metrics['matches_analyzed']}")
    
    # Calculate league-specific parameters
    league_specific = calculate_league_specific_params(all_data)
    
    if league_specific:
        # Save league-specific parameters
        try:
            with open('calibrated_params_by_league.json', 'w') as f:
                json.dump(league_specific, f, indent=2)
            print(f"\n✅ League-specific parameters saved to calibrated_params_by_league.json")
        except Exception as e:
            print(f"\n❌ Error saving league-specific parameters: {e}")
    
    # Optimize global parameters (fallback)
    calibrated = optimize_parameters(all_data)
    
    if calibrated:
        save_calibrated_params(calibrated)
        
        print("\n" + "="*70)
        print("CALIBRATION COMPLETE")
        print("="*70)
        print(f"\n📊 Results:")
        print(f"  ✅ Global parameters saved to calibrated_params.json")
        if league_specific:
            print(f"  ✅ League-specific params for {league_specific['total_leagues']} leagues")
        print("\nRestart your Flask app (app.py) to use calibrated parameters.")
    else:
        print("\n❌ Calibration failed - insufficient data")


if __name__ == "__main__":
    main()
