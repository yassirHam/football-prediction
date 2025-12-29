"""
First-Half Betting Decision Logic
==================================

DIAGNOSTIC ONLY - operates on existing model outputs
Provides deterministic decision rules for first-half exact score betting

RESTRICTIONS:
- No model modifications
- No threshold tuning
- No learning from outcomes
- Deterministic logic only
"""


def make_decision(first_half_predictions, xg_home, xg_away):
    """
    Deterministic decision logic for first-half exact score betting.
    
    Args:
        first_half_predictions: List of (score, probability) tuples sorted by probability desc
        xg_home: Home team expected goals
        xg_away: Away team expected goals
    
    Returns:
        dict: {
            'decision': str,  # "BET_TOP_2", "BET_TOP_3", or "PASS"
            'xG_total': float,
            'xG_diff': float,
            'top2_prob_sum': float,
            'top3_prob_sum': float,
            'decision_reason': str
        }
    """
    # STEP 1: Compute Match Profile
    xg_total = xg_home + xg_away
    xg_diff = abs(xg_home - xg_away)
    
    # Extract probabilities (handle cases with fewer than 3 predictions)
    p1 = first_half_predictions[0][1] if len(first_half_predictions) > 0 else 0
    p2 = first_half_predictions[1][1] if len(first_half_predictions) > 1 else 0
    p3 = first_half_predictions[2][1] if len(first_half_predictions) > 2 else 0
    
    top2_prob_sum = p1 + p2
    top3_prob_sum = p1 + p2 + p3
    
    # STEP 2: Apply Decision Rules in Order
    
    # 🔵 RULE A — PASS (High Variance Zone)
    # Very open + balanced + weak probability concentration
    if xg_total > 3.2 and xg_diff < 0.5 and top3_prob_sum < 0.65:
        return {
            'decision': 'PASS',
            'xG_total': round(xg_total, 2),
            'xG_diff': round(xg_diff, 2),
            'top2_prob_sum': round(top2_prob_sum * 100, 1),
            'top3_prob_sum': round(top3_prob_sum * 100, 1),
            'decision_reason': 'High variance: Open balanced match with weak probability concentration'
        }
    
    # 🟢 RULE B — BET_TOP_2 (High Efficiency Zone)
    # Low variance match + Top-2 already captures most belief
    marginal_gain = (top3_prob_sum - top2_prob_sum) * 100  # Convert to percentage
    if xg_total <= 2.6 and top2_prob_sum >= 0.55 and marginal_gain < 15:
        return {
            'decision': 'BET_TOP_2',
            'xG_total': round(xg_total, 2),
            'xG_diff': round(xg_diff, 2),
            'top2_prob_sum': round(top2_prob_sum * 100, 1),
            'top3_prob_sum': round(top3_prob_sum * 100, 1),
            'decision_reason': 'High efficiency: Low variance with strong Top-2 concentration'
        }
    
    # 🟡 RULE C — BET_TOP_3 (High Coverage Zone)
    # Defensive or mismatched game + strong probability concentration
    if (xg_total < 2.2 or xg_diff > 1.2) and top3_prob_sum >= 0.68:
        return {
            'decision': 'BET_TOP_3',
            'xG_total': round(xg_total, 2),
            'xG_diff': round(xg_diff, 2),
            'top2_prob_sum': round(top2_prob_sum * 100, 1),
            'top3_prob_sum': round(top3_prob_sum * 100, 1),
            'decision_reason': 'High coverage: Defensive/imbalanced match with strong Top-3 concentration'
        }
    
    # 🔴 RULE D — DEFAULT PASS
    return {
        'decision': 'PASS',
        'xG_total': round(xg_total, 2),
        'xG_diff': round(xg_diff, 2),
        'top2_prob_sum': round(top2_prob_sum * 100, 1),
        'top3_prob_sum': round(top3_prob_sum * 100, 1),
        'decision_reason': 'No strong signal: Does not meet criteria for Top-2 or Top-3'
    }


def analyze_decision_distribution(decisions):
    """
    Analyze distribution of decisions across a batch of matches.
    
    Args:
        decisions: List of decision dicts from make_decision()
    
    Returns:
        dict: Summary statistics
    """
    total = len(decisions)
    if total == 0:
        return {
            'total_matches': 0,
            'bet_top2_pct': 0,
            'bet_top3_pct': 0,
            'pass_pct': 0
        }
    
    bet_top2_count = sum(1 for d in decisions if d['decision'] == 'BET_TOP_2')
    bet_top3_count = sum(1 for d in decisions if d['decision'] == 'BET_TOP_3')
    pass_count = sum(1 for d in decisions if d['decision'] == 'PASS')
    
    return {
        'total_matches': total,
        'bet_top2_count': bet_top2_count,
        'bet_top2_pct': round(bet_top2_count / total * 100, 1),
        'bet_top3_count': bet_top3_count,
        'bet_top3_pct': round(bet_top3_count / total * 100, 1),
        'pass_count': pass_count,
        'pass_pct': round(pass_count / total * 100, 1)
    }
