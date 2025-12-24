"""
Test Deterministic Confidence Gate
===================================

Verifies that the new confidence gate produces stable, deterministic outputs.
"""

import sys
import numpy as np
sys.path.append('.')

from hybrid.confidence_gate import (
    estimate_confidence_from_features,
    estimate_prediction_stability,
    evaluate_xgb_confidence
)

print("=" * 70)
print("DETERMINISTIC CONFIDENCE GATE - VERIFICATION TEST")
print("=" * 70)

# Test 1: Determinism check
print("\n1️⃣ DETERMINISM TEST")
print("-" * 70)

features = np.array([1.5, 2.0, 1.8, 0.0, 3.2, 1.1, 0.5, 2.5, 1.9, 0.8] * 4)
xg_home, xg_away = 1.8, 1.3

print(f"Features shape: {features.shape}")
print(f"xG: {xg_home:.2f} vs {xg_away:.2f}")

# Run 5 times to verify identical outputs
results = []
for run in range(5):
    conf = estimate_confidence_from_features(features, xg_home, xg_away)
    results.append(conf)
    print(f"  Run {run+1}: Confidence = {conf:.6f}")

# Check all identical
if len(set(results)) == 1:
    print("✅ PASS: All runs produced identical results")
else:
    print(f"❌ FAIL: Got {len(set(results))} different values!")

# Test 2: Low confidence with sparse features
print("\n2️⃣ SPARSE FEATURES TEST (expect LOW confidence)")
print("-" * 70)

sparse_features = np.zeros(40)
sparse_features[0] = 1.2
sparse_features[5] = 0.8
xg_home, xg_away = 2.0, 1.5

conf = estimate_confidence_from_features(sparse_features, xg_home, xg_away)
print(f"Feature sparsity: {np.sum(sparse_features == 0) / len(sparse_features) * 100:.1f}% zeros")
print(f"Confidence: {conf:.4f}")
print("✅ PASS: Low confidence for sparse data" if conf < 0.5 else "⚠️  Unexpected high confidence")

# Test 3: High confidence with complete features
print("\n3️⃣ COMPLETE FEATURES TEST (expect HIGH confidence)")
print("-" * 70)

complete_features = np.random.uniform(0.5, 3.0, 40)  # Dense, well-distributed
xg_home, xg_away = 1.4, 1.3  # Close to average

conf = estimate_confidence_from_features(complete_features, xg_home, xg_away)
print(f"Feature sparsity: {np.sum(complete_features == 0) / len(complete_features) * 100:.1f}% zeros")
print(f"xG total: {xg_home + xg_away:.2f} (league avg = 2.7)")
print(f"Confidence: {conf:.4f}")
print("✅ PASS: High confidence for complete data" if conf > 0.5 else "⚠️  Unexpected low confidence")

# Test 4: Extreme xG prediction
print("\n4️⃣ EXTREME xG TEST (expect LOWER confidence)")
print("-" * 70)

normal_features = np.random.uniform(0.5, 3.0, 40)
xg_home, xg_away = 4.5, 3.8  # Very high total

conf = estimate_confidence_from_features(normal_features, xg_home, xg_away)
print(f"xG total: {xg_home + xg_away:.2f} (deviation from 2.7 = {abs(xg_home + xg_away - 2.7):.2f})")
print(f"Confidence: {conf:.4f}")
print("✅ PASS: Lower confidence for extreme xG" if conf < 0.7 else "⚠️  Confidence not penalized")

# Test 5: Full pipeline test
print("\n5️⃣ FULL PIPELINE TEST (evaluate_xgb_confidence)")
print("-" * 70)

test_features = np.array([1.5, 2.0, 1.8, 1.2, 3.2, 1.1, 0.5, 2.5, 1.9, 0.8] * 4)
xg_home, xg_away = 1.8, 1.3

result = evaluate_xgb_confidence(
    xg_home=xg_home,
    xg_away=xg_away,
    features=test_features
)

print(f"Overall Confidence: {result['overall_confidence']:.1f}%")
print(f"  - Bounds Check: {result['bounds_check']:.1f}%")
print(f"  - Feature Quality: {result['feature_quality']:.1f}%")
print(f"  - Prediction Stability: {result['prediction_stability']:.1f}%")
print(f"Use XGBoost: {result['use_xgb']}")

# Run again to verify determinism
result2 = evaluate_xgb_confidence(
    xg_home=xg_home,
    xg_away=xg_away,
    features=test_features
)

if result['overall_confidence'] == result2['overall_confidence']:
    print("✅ PASS: Full pipeline is deterministic")
else:
    print("❌ FAIL: Pipeline produced different results!")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
