"""
Enhanced Model Training Script
==============================
Uses combined dataset (domestic + international) for model training.

Usage:
    python train_model_enhanced.py
"""

import pandas as pd
from pathlib import Path
from football_predictor import Team, predict_match
from collections import defaultdict
import numpy as np
from data_processing import DataProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_combined_dataset():
    """Load the combined training dataset."""
    data_path = Path('data/combined_training_data.csv')
    
    if not data_path.exists():
        logger.warning("Combined dataset not found. Generating it now...")
        processor = DataProcessor()
        df = processor.merge_datasets(
            include_international=True,
            include_domestic=True,
            min_year=2010
        )
        processor.save_processed_data(df)
        return df
    
    logger.info(f"Loading combined dataset from {data_path}")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    logger.info(f"Loaded {len(df)} matches")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Sources: {df['Source'].value_counts().to_dict()}")
    
    return df


def extract_team_form(df, team_name, date, is_home=True, matches=5):
    """
    Extract last N matches for a team before a given date.
    
    Returns: Lists of goals_scored, goals_conceded, first_half_goals
    """
    # Get all matches for this team before the given date
    team_matches = df[
        ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) &
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(matches)
    
    if len(team_matches) == 0:
        return [1], [1], [0]  # Default values as lists
    
    goals_scored = []
    goals_conceded = []
    first_half_goals = []
    
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            goals_scored.append(int(match['FTHG']))
            goals_conceded.append(int(match['FTAG']))
            if 'HTHG' in match and pd.notna(match['HTHG']):
                first_half_goals.append(int(match['HTHG']))
            else:
                first_half_goals.append(int(match['FTHG'] / 2))  # Estimate
        else:
            goals_scored.append(int(match['FTAG']))
            goals_conceded.append(int(match['FTHG']))
            if 'HTAG' in match and pd.notna(match['HTAG']):
                first_half_goals.append(int(match['HTAG']))
            else:
                first_half_goals.append(int(match['FTAG'] / 2))  # Estimate
    
    # Ensure we return lists even if incomplete data
    if not goals_scored:
        goals_scored = [1]
    if not goals_conceded:
        goals_conceded = [1]
    if not first_half_goals:
        first_half_goals = [0]
    
    return goals_scored, goals_conceded, first_half_goals


def test_single_match(df, match_idx):
    """Test prediction for a single match."""
    match = df.iloc[match_idx]
    
    # Extract form for both teams
    home_form = extract_team_form(df, match['HomeTeam'], match['Date'], is_home=True)
    away_form = extract_team_form(df, match['AwayTeam'], match['Date'], is_home=False)
    
    # Create Team objects with lists
    home_team = Team(
        name=match['HomeTeam'],
        goals_scored=home_form[0],  # List
        goals_conceded=home_form[1],  # List
        first_half_goals=home_form[2]  # List
    )
    
    away_team = Team(
        name=match['AwayTeam'],
        goals_scored=away_form[0],  # List
        goals_conceded=away_form[1],  # List
        first_half_goals=away_form[2]  # List
    )
    
    # Make prediction
    prediction = predict_match(home_team, away_team)
    
    # Actual result
    actual_home = int(match['FTHG'])
    actual_away = int(match['FTAG'])
    actual_total = actual_home + actual_away
    actual_btts = (actual_home > 0 and actual_away > 0)
    
    return {
        'predicted': prediction,
        'actual_home': actual_home,
        'actual_away': actual_away,
        'actual_total': actual_total,
        'actual_btts': actual_btts,
        'competition': match.get('Competition', 'Unknown'),
        'source': match.get('Source', 'Unknown')
    }


def validate_model(max_matches=1000, test_split=0.2):
    """
    Validate model against combined dataset.
    
    Args:
        max_matches: Maximum matches to test (for speed)
        test_split: Fraction of data to use for testing
    """
    logger.info("Starting model validation...")
    
    # Load data
    df = load_combined_dataset()
    
    # Sort by date and take the most recent matches for testing
    df = df.sort_values('Date').reset_index(drop=True)
    test_size = min(int(len(df) * test_split), max_matches)
    start_idx = len(df) - test_size
    
    logger.info(f"Testing on {test_size} most recent matches")
    logger.info(f"Test period: {df.iloc[start_idx]['Date']} to {df.iloc[-1]['Date']}")
    
    results = []
    errors_by_source = defaultdict(list)
    errors_by_competition = defaultdict(list)
    
    for idx in range(start_idx, len(df)):
        if idx % 100 == 0:
            logger.info(f"Tested {idx - start_idx}/{test_size} matches...")
        
        try:
            result = test_single_match(df, idx)
            results.append(result)
            
            # Extract total goals prediction properly
            pred = result['predicted']
            predicted_total = pred['expected_goals']['home'] + pred['expected_goals']['away']
            error = abs(predicted_total - result['actual_total'])
            
            errors_by_source[result['source']].append(error)
            errors_by_competition[result['competition']].append(error)
            
        except Exception as e:
            logger.error(f"Error processing match {idx}: {str(e)}")
            continue
    
    if not results:
        logger.error("No results to analyze!")
        return
    
    # Calculate metrics
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS WITH ENHANCED DATASET")
    logger.info("="*60)
    
    # Overall metrics
    total_goals_errors = []
    btts_correct = 0
    exact_scores = 0
    
    for r in results:
        # Total goals error
        pred_total = r['predicted']['expected_goals']['home'] + r['predicted']['expected_goals']['away']
        total_goals_errors.append(abs(pred_total - r['actual_total']))
        
        # BTTS accuracy
        pred_btts = r['predicted']['both_teams_score'] > 0.5
        if pred_btts == r['actual_btts']:
            btts_correct += 1
        
        # Exact score in top 5
        pred_scores = [p[0] for p in r['predicted']['full_match_predictions']]
        actual_score = (r['actual_home'], r['actual_away'])
        if actual_score in pred_scores:
            exact_scores += 1
    
    logger.info(f"\nTotal matches tested: {len(results)}")
    logger.info(f"Date range: {df.iloc[start_idx]['Date']} to {df.iloc[-1]['Date']}")
    logger.info(f"\nMean Absolute Error (Total Goals): {np.mean(total_goals_errors):.3f}")
    logger.info(f"Median Absolute Error (Total Goals): {np.median(total_goals_errors):.3f}")
    logger.info(f"BTTS Accuracy: {btts_correct/len(results)*100:.1f}%")
    logger.info(f"Exact Score in Top 5: {exact_scores/len(results)*100:.1f}%")
    
    # By source
    logger.info("\n" + "-"*60)
    logger.info("PERFORMANCE BY SOURCE")
    logger.info("-"*60)
    for source in sorted(errors_by_source.keys()):
        errors = errors_by_source[source]
        logger.info(f"{source:15s}: MAE = {np.mean(errors):.3f} ({len(errors)} matches)")
    
    # By competition (top 10)
    logger.info("\n" + "-"*60)
    logger.info("PERFORMANCE BY COMPETITION (Top 10)")
    logger.info("-"*60)
    sorted_comps = sorted(errors_by_competition.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for comp, errors in sorted_comps:
        logger.info(f"{comp:30s}: MAE = {np.mean(errors):.3f} ({len(errors)} matches)")
    
    # Error distribution
    logger.info("\n" + "-"*60)
    logger.info("ERROR DISTRIBUTION")
    logger.info("-"*60)
    error_counts = defaultdict(int)
    for error in total_goals_errors:
        error_counts[int(error)] += 1
    
    for error in sorted(error_counts.keys()):
        pct = error_counts[error] / len(total_goals_errors) * 100
        logger.info(f"Error {error}: {error_counts[error]:4d} matches ({pct:5.1f}%)")
    
    # Save results to file
    with open('training_results_enhanced.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("ENHANCED MODEL VALIDATION RESULTS\n")
        f.write("Combined Dataset: International + Domestic Leagues\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total matches tested: {len(results)}\n")
        f.write(f"Test period: {df.iloc[start_idx]['Date']} to {df.iloc[-1]['Date']}\n\n")
        f.write(f"MAE (Total Goals): {np.mean(total_goals_errors):.3f}\n")
        f.write(f"Median AE (Total Goals): {np.median(total_goals_errors):.3f}\n")
        f.write(f"BTTS Accuracy: {btts_correct/len(results)*100:.1f}%\n")
        f.write(f"Exact Score in Top 5: {exact_scores/len(results)*100:.1f}%\n")
        f.write(f"\nBy Source:\n")
        for source in sorted(errors_by_source.keys()):
            errors = errors_by_source[source]
            f.write(f"  {source:15s}: MAE = {np.mean(errors):.3f} ({len(errors)} matches)\n")
        f.write(f"\nTop 10 Competitions:\n")
        sorted_comps = sorted(errors_by_competition.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        for comp, errors in sorted_comps:
            f.write(f"  {comp:30s}: MAE = {np.mean(errors):.3f} ({len(errors)} matches)\n")
    
    logger.info(f"\n✅ Results saved to training_results_enhanced.txt")
    
    return results


if __name__ == "__main__":
    results = validate_model(max_matches=1000, test_split=0.2)
    
    if results:
        logger.info(f"\n✅ Validation complete! Tested {len(results)} matches")
    else:
        logger.error("\n❌ Validation failed")
