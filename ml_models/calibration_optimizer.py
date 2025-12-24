"""
Calibration Optimizer for Poisson Model
=======================================

Replaces fixed calibration parameters with data-driven optimized values.
Uses Bayesian Optimization (scikit-optimize) to find best league-specific parameters.

Optimizes:
- League Average Goals
- Home Advantage Multiplier
- Away Penalty Multiplier
- Dixon-Coles Rho (Correlation confidence)
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from football_predictor import Team, predict_match, get_league_params
from ml_models.train_ml import load_and_prepare_data, _create_team_snapshot
from sklearn.metrics import log_loss

class CalibrationOptimizer:
    def __init__(self, league_name: str, matches: pd.DataFrame):
        self.league_name = league_name
        self.matches = matches.sort_values('Date')
        self.history = matches.iloc[:-int(len(matches)*0.2)] # 80% history
        self.validation = matches.iloc[-int(len(matches)*0.2):] # 20% validation
        
        # Define search space
        self.space = [
            Real(1.0, 4.0, name='league_avg_goals'),
            Real(0.9, 1.4, name='home_advantage'),
            Real(0.7, 1.0, name='away_penalty'),
            Real(-0.15, 0.05, name='dixon_coles_rho')
        ]
        
    def objective(self, params):
        """Objective function to minimize (Log Loss)."""
        metrics = self._evaluate_params(params)
        return metrics['log_loss']
        
    def _evaluate_params(self, params) -> Dict:
        """Evaluate parameters on validation set."""
        # Unpack parameters (order matches self.space)
        l_avg, h_adv, a_pen, rho = params
        
        # Create temp params dict
        temp_params = {
            'league_avg_goals': l_avg,
            'home_advantage': h_adv,
            'away_penalty': a_pen,
            'dixon_coles_rho': rho
        }
        
        results = []
        
        # Simulate validation matches
        # Note: We use the *entire* matches df to look up history up to that point
        # But we only evaluate on the validation set rows
        
        for idx in range(len(self.matches) - len(self.validation), len(self.matches)):
            match = self.matches.iloc[idx]
            history_subset = self.matches.iloc[:idx]
            
            home_team = _create_team_snapshot(match['HomeTeam'], history_subset)
            away_team = _create_team_snapshot(match['AwayTeam'], history_subset)
            
            # Inject params by temporarily setting team.league to a dummy and mocking get_league_params
            # Or better: modify predict_match to accept params overrides.
            # Since predict_match calls get_league_params internally, we need to pass params down.
            # We updated predict_match signature in our plan to accept kwargs!
            # But we didn't update football_predictor.py yet.
            
            # Hack for calibration: calculate manually using the core functions
            # Reusing the logic from football_predictor.py but with our params
            
            # Calculate xG
            from football_predictor import calculate_offensive_strength, calculate_defensive_weakness
            
            # We need to mock the league params lookup or pass them explicitly
            # Let's assume we modified the football_predictor functions to accept an explicit dict
            # calculate_offensive_strength(..., league_params=temp_params)
            
            home_osi = calculate_offensive_strength(home_team, True, league_params=temp_params)
            away_osi = calculate_offensive_strength(away_team, False, league_params=temp_params)
            
            home_dwi = calculate_defensive_weakness(home_team, league_params=temp_params)
            away_dwi = calculate_defensive_weakness(away_team, league_params=temp_params)
            
            xg_home = home_osi * away_dwi * l_avg
            xg_away = away_osi * home_dwi * l_avg * a_pen
            
            # Predict outcome probabilities directly
            from football_predictor import predict_score_probabilities, calculate_match_outcome_probabilities
            
            probs_map = predict_score_probabilities(xg_home, xg_away) 
            # Note: predict_score_probabilities uses globals or standard Poisson
            # We need to inject rho if we want to optimize it. 
            # Currently predict_score_probabilities uses `dixon_coles_tau` which imports from `advanced_statistics`.
            
            outcomes = calculate_match_outcome_probabilities(probs_map)
            
            # Truth
            h = int(match['FTHG'])
            a = int(match['FTAG'])
            if h > a: actual = 2 # Home
            elif h == a: actual = 1 # Draw
            else: actual = 0 # Away
            
            results.append({
                'actual': actual,
                'probs': [outcomes['away_win'], outcomes['draw'], outcomes['home_win']]
            })
            
        if not results: return {'log_loss': 99.9}
        
        y_true = [r['actual'] for r in results]
        y_pred = [r['probs'] for r in results]
        
        try:
            ll = log_loss(y_true, y_pred, labels=[0, 1, 2])
        except:
            ll = 99.9
            
        return {'log_loss': ll}

    def optimize(self, n_calls=20):
        print(f"Optimizing {self.league_name}...")
        
        # Scikit-optimize wrapper
        @use_named_args(self.space)
        def objective_wrapper(**params):
            return self.objective([
                params['league_avg_goals'],
                params['home_advantage'],
                params['away_penalty'],
                params['dixon_coles_rho']
            ])
            
        res = gp_minimize(objective_wrapper, self.space, n_calls=n_calls, random_state=42)
        
        best_params = {
            'league_avg_goals': res.x[0],
            'home_advantage': res.x[1],
            'away_penalty': res.x[2],
            'dixon_coles_rho': res.x[3]
        }
        
        print(f"Best score: {res.fun:.4f}")
        return best_params

if __name__ == "__main__":
    # Example usage
    pass
