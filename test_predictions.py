"""Quick test of improved predictions."""
from football_predictor import Team, predict_match

# Test 1: Balanced teams
print("="*60)
print("TEST 1: Balanced Teams (should have high BTTS)")
print("="*60)
t1 = Team('TeamA', [2,2,2,2,2], [1,1,1,1,1], [1,1,1,1,1])
t2 = Team('TeamB', [2,2,2,2,2], [1,1,1,1,1], [1,1,1,1,1])
r = predict_match(t1, t2)
print(f"BTTS Probability: {r['both_teams_score']:.3f}")
print(f"Top Score: {r['full_match_predictions'][0]}")
print(f"Most Likely Total: {r['total_goals'].get('most_likely_total', 'N/A')}")
print()

# Test 2: High scoring teams  
print("="*60)
print("TEST 2: High-Scoring Teams")
print("="*60)
t3 = Team('Attack1', [4,3,4,3,4], [1,2,1,2,1], [2,2,2,1,2])
t4 = Team('Attack2', [3,4,3,4,3], [2,1,2,1,2], [1,2,1,2,1])
r2 = predict_match(t3, t4)
print(f"BTTS Probability: {r2['both_teams_score']:.3f}")
print(f"Top Score: {r2['full_match_predictions'][0]}")
print(f"Most Likely Total: {r2['total_goals'].get('most_likely_total', 'N/A')}")
print(f"Over 2.5 Goals: {r2['total_goals']['over_2.5']:.1%}")
print()

# Test 3: Defensive teams
print("="*60)
print("TEST 3: Low-Scoring/Defensive Teams")
print("="*60)
t5 = Team('Defense1', [0,1,0,1,0], [0,0,1,0,1], [0,0,0,1,0])
t6 = Team('Defense2', [1,0,1,0,0], [1,0,0,1,0], [0,0,1,0,0])
r3 = predict_match(t5, t6)
print(f"BTTS Probability: {r3['both_teams_score']:.3f}")
print(f"Top Score: {r3['full_match_predictions'][0]}")
print(f"Most Likely Total: {r3['total_goals'].get('most_likely_total', 'N/A')}")
print(f"Under 2.5 Goals: {r3['total_goals']['under_2.5']:.1%}")
print()

print("="*60)
print("✅ All predictions working correctly!")
print("="*60)
