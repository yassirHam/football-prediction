"""
Quick demonstration of enhanced prediction features.
Shows how to use shots on target and home/away form data.
"""

from football_predictor import Team
from enhanced_predictor import enhanced_predict_match

print("="*70)
print("ENHANCED PREDICTION DEMONSTRATION")
print("="*70)

# Example 1: Basic team (backward compatible)
print("\n1. BASIC USAGE (Backward Compatible)")
print("-"*70)
basic_team1 = Team(
    name="Team A",
    goals_scored=[2, 1, 2, 1, 2],
    goals_conceded=[1, 1, 0, 1, 1],
    first_half_goals=[1, 0, 1, 1, 1],
    league="E0"  # English Premier League
)

basic_team2 = Team(
    name="Team B",
    goals_scored=[1, 2, 1, 2, 1],
    goals_conceded=[1, 0, 1, 1, 2],
    first_half_goals=[1, 1, 0, 1, 0],
    league="E0"
)

result = enhanced_predict_match(basic_team1, basic_team2)
print(f"Expected Goals: {result['expected_goals']['home']:.2f} - {result['expected_goals']['away']:.2f}")
print(f"BTTS Probability: {result['both_teams_score']:.1%}")
print(f"Most Likely Score: {result['full_match_predictions'][0]}")
print(f"Enhanced Features Used: {result['insights']['enhanced_features_used']}")

# Example 2: Enhanced with Shots on Target
print("\n\n2. WITH SHOTS ON TARGET DATA")
print("-"*70)
enhanced_team1 = Team(
    name="Attack Team",
    goals_scored=[3, 2, 3, 2, 3],
    goals_conceded=[1, 1, 2, 1, 1],
    first_half_goals=[2, 1, 2, 1, 2],
    shots_on_target=[8, 7, 9, 6, 8],  # High quality chances
    shots_on_target_conceded=[3, 4, 5, 3, 4],
    league="E0"
)

enhanced_team2 = Team(
    name="Defense Team",
    goals_scored=[1, 1, 0, 1, 1],
    goals_conceded=[1, 0, 1, 1, 0],
    first_half_goals=[0, 1, 0, 0, 1],
    shots_on_target=[4, 3, 2, 4, 3],  # Fewer chances
    shots_on_target_conceded=[6, 5, 4, 6, 5],
    league="E0"
)

result2 = enhanced_predict_match(enhanced_team1, enhanced_team2)
print(f"Expected Goals: {result2['expected_goals']['home']:.2f} - {result2['expected_goals']['away']:.2f}")
print(f"BTTS Probability: {result2['both_teams_score']:.1%}")
print(f"Most Likely Total: {result2['total_goals']['most_likely_total']}")
print(f"Enhanced Features Used: {result2['insights']['enhanced_features_used']}")

# Example 3: Home/Away Form Separation
print("\n\n3. WITH HOME/AWAY FORM SEPARATION")
print("-"*70)
form_team1 = Team(
    name="Home Strong Team",
    goals_scored=[2, 2, 2, 2, 2],  # Overall average
    goals_conceded=[1, 1, 1, 1, 1],
    first_half_goals=[1, 1, 1, 1, 1],
    home_goals_scored=[3, 3, 3],  # Strong at home!
    home_goals_conceded=[0, 1, 0],  # Solid defense at home
    away_goals_scored=[1, 1],  # Weak away
    away_goals_conceded=[2, 2],
    league="D1"  # German Bundesliga
)

form_team2 = Team(
    name="Away Strong Team",
    goals_scored=[2, 2, 2, 2, 2],
    goals_conceded=[1, 1, 1, 1, 1],
    first_half_goals=[1, 1, 1, 1, 1],
    home_goals_scored=[1, 1],  # Weak at home
    home_goals_conceded=[2, 1],
    away_goals_scored=[3, 3, 3],  # Strong away!
    away_goals_conceded=[0, 1, 1],  # Good defense away
    league="D1"
)

result3 = enhanced_predict_match(form_team1, form_team2)
print(f"Expected Goals: {result3['expected_goals']['home']:.2f} - {result3['expected_goals']['away']:.2f}")
print(f"Home Win: {result3['match_outcome']['home_win']:.1%}")
print(f"Away Win: {result3['match_outcome']['away_win']:.1%}")
print(f"Enhanced Features Used: {result3['insights']['enhanced_features_used']}")

print("\n" + "="*70)
print("✅ Enhanced predictor working! Use it in your app for better accuracy.")
print("="*70)
