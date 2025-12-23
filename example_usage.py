"""
Example Usage of Football Prediction System
============================================
This file demonstrates various ways to use the prediction system.
"""

from football_predictor import Team, predict_match, format_predictions
import json


def example_1_basic_usage():
    """Basic prediction example."""
    print("="*70)
    print("EXAMPLE 1: Basic Usage - Premier League Match")
    print("="*70)
    
    home_team = Team(
        name="Arsenal",
        goals_scored=[2, 3, 1, 2, 2],
        goals_conceded=[1, 0, 1, 2, 1],
        first_half_goals=[1, 2, 0, 1, 1]
    )
    
    away_team = Team(
        name="Chelsea",
        goals_scored=[1, 2, 2, 1, 3],
        goals_conceded=[1, 1, 2, 0, 2],
        first_half_goals=[0, 1, 1, 0, 2]
    )
    
    result = predict_match(home_team, away_team)
    print(format_predictions(result, "Arsenal", "Chelsea"))


def example_2_attacking_teams():
    """High-scoring match prediction."""
    print("\n" + "="*70)
    print("EXAMPLE 2: High-Scoring Match - Two Attacking Teams")
    print("="*70)
    
    home_team = Team(
        name="Bayern Munich",
        goals_scored=[4, 3, 5, 2, 3],
        goals_conceded=[2, 1, 2, 1, 2],
        first_half_goals=[2, 2, 3, 1, 2]
    )
    
    away_team = Team(
        name="Borussia Dortmund",
        goals_scored=[3, 4, 2, 3, 3],
        goals_conceded=[2, 2, 1, 2, 1],
        first_half_goals=[1, 2, 1, 2, 1]
    )
    
    result = predict_match(home_team, away_team)
    print(format_predictions(result, "Bayern Munich", "Borussia Dortmund"))


def example_3_defensive_teams():
    """Low-scoring match prediction."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Low-Scoring Match - Two Defensive Teams")
    print("="*70)
    
    home_team = Team(
        name="Inter Milan",
        goals_scored=[1, 0, 1, 1, 2],
        goals_conceded=[0, 0, 1, 0, 1],
        first_half_goals=[0, 0, 1, 0, 1]
    )
    
    away_team = Team(
        name="Juventus",
        goals_scored=[1, 1, 0, 2, 1],
        goals_conceded=[0, 1, 0, 1, 0],
        first_half_goals=[0, 1, 0, 1, 0]
    )
    
    result = predict_match(home_team, away_team)
    print(format_predictions(result, "Inter Milan", "Juventus"))


def example_4_json_output():
    """Example of JSON output for API usage."""
    print("\n" + "="*70)
    print("EXAMPLE 4: JSON Output - For API Integration")
    print("="*70)
    
    home_team = Team(
        name="Real Madrid",
        goals_scored=[3, 2, 1, 3, 2],
        goals_conceded=[1, 1, 0, 2, 1],
        first_half_goals=[2, 1, 0, 2, 1]
    )
    
    away_team = Team(
        name="Barcelona",
        goals_scored=[2, 3, 2, 1, 2],
        goals_conceded=[1, 2, 1, 1, 1],
        first_half_goals=[1, 1, 1, 0, 1]
    )
    
    result = predict_match(home_team, away_team)
    
    # Convert tuple keys to strings for JSON serialization
    json_result = {
        "match": f"{home_team.name} vs {away_team.name}",
        "first_half_predictions": [
            {"score": f"{h}-{a}", "probability": round(prob * 100, 1)}
            for (h, a), prob in result['first_half_predictions']
        ],
        "full_match_predictions": [
            {"score": f"{h}-{a}", "probability": round(prob * 100, 1)}
            for (h, a), prob in result['full_match_predictions']
        ],
        "total_goals": {
            key: round(val * 100, 1) for key, val in result['total_goals'].items()
        },
        "expected_goals": {
            "home": round(result['expected_goals']['home'], 2),
            "away": round(result['expected_goals']['away'], 2)
        },
        "insights": result['insights']
    }
    
    print(json.dumps(json_result, indent=2))


def example_5_form_comparison():
    """Compare teams with different form patterns."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Form Analysis - Improving vs Declining Team")
    print("="*70)
    
    # Team improving (older matches worse, recent better)
    home_team = Team(
        name="Newcastle United (Improving)",
        goals_scored=[3, 3, 2, 1, 0],  # Getting better
        goals_conceded=[0, 1, 1, 2, 3],  # Defense improving
        first_half_goals=[2, 1, 1, 0, 0]
    )
    
    # Team declining (older matches better, recent worse)
    away_team = Team(
        name="Tottenham (Declining)",
        goals_scored=[0, 1, 2, 3, 3],  # Getting worse
        goals_conceded=[3, 2, 1, 1, 0],  # Defense declining
        first_half_goals=[0, 0, 1, 1, 2]
    )
    
    result = predict_match(home_team, away_team)
    print(format_predictions(result, home_team.name, away_team.name))
    print("\nNote: Recent form is weighted more heavily (30%, 25%, 20%, 15%, 10%)")
    print("Newcastle's improving form gives them an advantage!")


def example_6_custom_analysis():
    """Extract specific insights from predictions."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Analysis - Extracting Specific Insights")
    print("="*70)
    
    home_team = Team(
        name="Manchester United",
        goals_scored=[2, 1, 3, 2, 1],
        goals_conceded=[1, 2, 1, 1, 2],
        first_half_goals=[1, 0, 2, 1, 0]
    )
    
    away_team = Team(
        name="West Ham",
        goals_scored=[1, 2, 1, 1, 2],
        goals_conceded=[2, 1, 2, 1, 1],
        first_half_goals=[0, 1, 0, 1, 1]
    )
    
    result = predict_match(home_team, away_team)
    
    print(f"\nMATCH: {home_team.name} vs {away_team.name}\n")
    
    # Home win probability (home goals > away goals)
    home_win_prob = sum(
        prob for (h, a), prob in 
        [(score, result['full_match_predictions'][i][1]) 
         for i, score in enumerate([s[0] for s in result['full_match_predictions']])]
        if h > a
    )
    
    # Most likely score
    most_likely_score, probability = result['full_match_predictions'][0]
    
    print(f"📊 ANALYSIS:")
    print(f"  • Most Likely Score: {most_likely_score[0]}-{most_likely_score[1]} ({probability*100:.1f}%)")
    print(f"  • Expected Goals: {home_team.name} {result['expected_goals']['home']:.2f}, "
          f"{away_team.name} {result['expected_goals']['away']:.2f}")
    print(f"  • Match Tempo: {result['insights']['tempo']}")
    print(f"  • Over 2.5 Goals: {result['total_goals']['over_2.5']*100:.1f}%")
    print(f"  • Both Teams to Score: HIGH (based on defensive weaknesses)" 
          if result['expected_goals']['home'] > 1.0 and result['expected_goals']['away'] > 1.0 
          else f"  • Both Teams to Score: MEDIUM/LOW")
    print(f"  • Confidence Level: {result['insights']['confidence']}")


if __name__ == "__main__":
    # Run all examples
    example_1_basic_usage()
    example_2_attacking_teams()
    example_3_defensive_teams()
    example_4_json_output()
    example_5_form_comparison()
    example_6_custom_analysis()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
