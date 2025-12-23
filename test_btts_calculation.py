"""
Unit Test for BTTS Calculation Fix
===================================
Tests that the Both Teams To Score (BTTS) calculation is correct.
"""

from football_predictor import Team, predict_match


def test_btts_high_scoring_teams():
    """Test BTTS with high-scoring teams - should have high probability."""
    print("Test 1: High-scoring teams (both likely to score)")
    
    # High-scoring teams
    team_a = Team("TeamA", [3, 2, 3, 2, 3], [1, 1, 2, 1, 1], [1, 1, 2, 1, 1])
    team_b = Team("TeamB", [2, 3, 2, 3, 2], [1, 2, 1, 1, 2], [1, 1, 1, 2, 1])
    
    result = predict_match(team_a, team_b)
    btts_prob = result['both_teams_score']
    
    print(f"  BTTS Probability: {btts_prob:.3f}")
    print(f"  Expected: >0.70 (both teams score frequently)")
    
    # Validate
    assert 0.0 <= btts_prob <= 1.0, f"❌ BTTS prob {btts_prob} outside valid range [0, 1]"
    assert btts_prob > 0.70, f"❌ BTTS prob {btts_prob} too low for attacking teams"
    
    print("  ✅ PASS\n")


def test_btts_low_scoring_teams():
    """Test BTTS with low-scoring teams - should have low probability."""
    print("Test 2: Low-scoring teams (both unlikely to score)")
    
    # Low-scoring defensive teams
    team_c = Team("TeamC", [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0])
    team_d = Team("TeamD", [0, 1, 0, 0, 1], [0, 0, 1, 0, 1], [0, 0, 0, 0, 1])
    
    result = predict_match(team_c, team_d)
    btts_prob = result['both_teams_score']
    
    print(f"  BTTS Probability: {btts_prob:.3f}")
    print(f"  Expected: <0.40 (both teams score rarely)")
    
    # Validate
    assert 0.0 <= btts_prob <= 1.0, f"❌ BTTS prob {btts_prob} outside valid range [0, 1]"
    assert btts_prob < 0.40, f"❌ BTTS prob {btts_prob} too high for defensive teams"
    
    print("  ✅ PASS\n")


def test_btts_mixed_teams():
    """Test BTTS with one strong offense, one strong defense."""
    print("Test 3: Mixed teams (one scores, one defends)")
    
    # Strong offense vs strong defense
    team_e = Team("Offense", [3, 3, 4, 2, 3], [0, 1, 0, 1, 0], [2, 1, 2, 1, 2])
    team_f = Team("Defense", [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0])
    
    result = predict_match(team_e, team_f)
    btts_prob = result['both_teams_score']
    
    print(f"  BTTS Probability: {btts_prob:.3f}")
    print(f"  Expected: 0.30-0.60 (mixed scenario)")
    
    # Validate
    assert 0.0 <= btts_prob <= 1.0, f"❌ BTTS prob {btts_prob} outside valid range [0, 1]"
    assert 0.20 < btts_prob < 0.70, f"❌ BTTS prob {btts_prob} outside expected range for mixed teams"
    
    print("  ✅ PASS\n")


def test_btts_formula_correctness():
    """Test that BTTS uses correct formula: P(both score) = P(A≥1) × P(B≥1)"""
    print("Test 4: Formula correctness check")
    
    from football_predictor import poisson_probability, calculate_expected_goals
    
    team_a = Team("TeamA", [2, 2, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
    team_b = Team("TeamB", [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1])
    
    # Get expected goals
    xg_home, xg_away = calculate_expected_goals(team_a, team_b)
    
    # Calculate BTTS manually
    prob_home_scores = 1 - poisson_probability(0, xg_home)
    prob_away_scores = 1 - poisson_probability(0, xg_away)
    expected_btts = prob_home_scores * prob_away_scores
    
    # Get prediction result
    result = predict_match(team_a, team_b)
    actual_btts = result['both_teams_score']
    
    print(f"  Manual calculation: {expected_btts:.3f}")
    print(f"  Prediction result:  {actual_btts:.3f}")
    print(f"  Difference: {abs(expected_btts - actual_btts):.6f}")
    
    # Should be identical (or very close due to floating point)
    assert abs(expected_btts - actual_btts) < 0.001, f"❌ BTTS calculation doesn't match expected formula"
    
    print("  ✅ PASS - Formula is correct!\n")


def run_all_tests():
    """Run all BTTS tests."""
    print("="*70)
    print("BTTS CALCULATION TEST SUITE")
    print("="*70)
    print()
    
    try:
        test_btts_high_scoring_teams()
        test_btts_low_scoring_teams()
        test_btts_mixed_teams()
        test_btts_formula_correctness()
        
        print("="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        return True
        
    except AssertionError as e:
        print("="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70)
        return False
    except Exception as e:
        print("="*70)
        print(f"❌ ERROR: {e}")
        print("="*70)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
