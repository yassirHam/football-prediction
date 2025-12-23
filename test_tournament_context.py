"""
Test Tournament Context Features
=================================
Demonstrates how the enhanced model handles:
- International cups with FIFA rankings
- Club cups with league positions  
- Tournament stage pressure (Group → Final)
"""

from football_predictor import Team, predict_match

print("="*70)
print("TOURNAMENT CONTEXT ENHANCEMENT - DEMO")
print("="*70)

# Test 1: World Cup Group Stage (International)
print("\n📍 TEST 1: World Cup Group Stage")
print("   Brazil (FIFA #1) vs Japan (FIFA #20)")
print("-" * 70)

brazil = Team(
    name="Brazil",
    goals_scored=[3, 2, 4, 3, 2],
    goals_conceded=[0, 1, 0, 1, 0],
    first_half_goals=[2, 1, 2, 2, 1],
    league_position=1,  # Not used for international
    fifa_ranking=1,  # FIFA ranking
    competition_type="INTERNATIONAL",
    tournament_stage="GROUP"  # Group stage = lower pressure
)

japan = Team(
    name="Japan",
    goals_scored=[1, 2, 1, 0, 2],
    goals_conceded=[1, 1, 2, 2, 1],
    first_half_goals=[0, 1, 0, 0, 1],
    league_position=10,  # Not used for international
    fifa_ranking=20,  # FIFA ranking
    competition_type="INTERNATIONAL",
    tournament_stage="GROUP"
)

result1 = predict_match(brazil, japan, neutral_venue=True)
print(f"\nExpected Goals:")
print(f"  Brazil: {result1['expected_goals']['home']:.2f}")
print(f"  Japan: {result1['expected_goals']['away']:.2f}")
print(f"\n💡 FIFA Ranking Gap: 19 positions")
print(f"   Stage Modifier: 0.8x (Group = more upsets possible)")

# Test 2: World Cup Final (same teams, different context)
print("\n📍 TEST 2: World Cup FINAL")
print("   Brazil (FIFA #1) vs Japan (FIFA #20)")
print("-" * 70)

brazil_final = Team(
    name="Brazil",
    goals_scored=[3, 2, 4, 3, 2],
    goals_conceded=[0, 1, 0, 1, 0],
    first_half_goals=[2, 1, 2, 2, 1],
    fifa_ranking=1,
    competition_type="INTERNATIONAL",
    tournament_stage="FINAL"  # Final = higher pressure
)

japan_final = Team(
    name="Japan",
    goals_scored=[1, 2, 1, 0, 2],
    goals_conceded=[1, 1, 2, 2, 1],
    first_half_goals=[0, 1, 0, 0, 1],
    fifa_ranking=20,
    competition_type="INTERNATIONAL",
    tournament_stage="FINAL"
)

result2 = predict_match(brazil_final, japan_final, neutral_venue=True)
print(f"\nExpected Goals:")
print(f"  Brazil: {result2['expected_goals']['home']:.2f}")
print(f"  Japan: {result2['expected_goals']['away']:.2f}")
print(f"\n💡 Same teams, FINAL stage:")
print(f"   Stage Modifier: 1.2x (Final = less upsets, favorites win more)")
print(f"   Brazil xG boost: {result2['expected_goals']['home'] - result1['expected_goals']['home']:.2f} vs Group stage")

# Test 3: Champions League Club Cup
print("\n📍 TEST 3: Champions League Quarter-Final")
print("   Real Madrid (1st in La Liga) vs PSG (3rd in Ligue 1)")
print("-" * 70)

real_madrid = Team(
    name="Real Madrid",
    goals_scored=[3, 2, 2, 4, 3],
    goals_conceded=[1, 0, 1, 1, 0],
    first_half_goals=[2, 1, 1, 2, 2],
    league_position=1,  # 1st in La Liga
    competition_type="CLUB_CUP",
    tournament_stage="QUARTER"  # Quarter-final
)

psg = Team(
    name="PSG",
    goals_scored=[2, 3, 1, 2, 2],
    goals_conceded=[1, 1, 0, 2, 1],
    first_half_goals=[1, 2, 0, 1, 1],
    league_position=3,  # 3rd in Ligue 1
    competition_type="CLUB_CUP",
    tournament_stage="QUARTER"
)

result3 = predict_match(real_madrid, psg, neutral_venue=True)
print(f"\nExpected Goals:")
print(f"  Real Madrid: {result3['expected_goals']['home']:.2f}")
print(f"  PSG: {result3['expected_goals']['away']:.2f}")
print(f"\n💡 Club Cup uses league positions")
print(f"   Ranking gap: 2 positions (closer than international)")
print(f"   Stage: Quarter-final (1.0x modifier = standard)")

# Test 4: Regular League Match (for comparison)
print("\n📍 TEST 4: Regular League Match")
print("   Liverpool (2nd) vs Everton (15th)")
print("-" * 70)

liverpool = Team(
    name="Liverpool",
    goals_scored=[3, 2, 2, 4, 2],
    goals_conceded=[0, 1, 0, 1, 1],
    first_half_goals=[2, 1, 1, 2, 1],
    league_position=2,
    competition_type="LEAGUE",  # Regular league match
    tournament_stage="GROUP"  # Not used for leagues
)

everton = Team(
    name="Everton",
    goals_scored=[1, 0, 1, 1, 2],
    goals_conceded=[2, 3, 2, 2, 1],
    first_half_goals=[0, 0, 1, 0, 1],
    league_position=15,
    competition_type="LEAGUE"
)

result4 = predict_match(liverpool, everton)
print(f"\nExpected Goals:")
print(f"  Liverpool: {result4['expected_goals']['home']:.2f}")
print(f"  Everton: {result4['expected_goals']['away']:.2f}")
print(f"\n💡 League match: Strong position gap effect")
print(f"   13 positions difference = significant boost for Liverpool")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("\n✅ International Tournaments:")
print("   - Use FIFA rankings (1-210 scale)")
print("   - Group stage: 0.8x modifier (more upsets)")
print("   - Final: 1.2x modifier (favorites dominate)")

print("\n✅ Club Cups:")
print("   - Use league positions (1-20 scale)")
print("   - Quarter-final: 1.0x (standard pressure)")
print("   - Stronger per-position effect than international")

print("\n✅ League Matches:")
print("   - Standard league position logic")
print("   - No tournament pressure modifiers")
print("   - Most predictable format")

print("\n🎯 Tournament progression matters!")
print("   Same teams, different xG based on stage!")
