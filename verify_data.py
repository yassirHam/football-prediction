"""
Quick script to verify calibration used the CSV data files.
"""
import pandas as pd
import glob

print("="*70)
print("DATA FOLDER VERIFICATION")
print("="*70)

csv_files = glob.glob('data/*.csv')
csv_files = [f for f in csv_files if 'learned_matches' not in f]

total_matches = 0
print(f"\nFound {len(csv_files)} CSV files in data folder:\n")

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            matches = len(df)
            total_matches += matches
            print(f"  ✓ {csv_file:30s} - {matches:4d} matches")
    except Exception as e:
        print(f"  ✗ {csv_file:30s} - Error: {e}")

print(f"\n{'='*70}")
print(f"TOTAL MATCHES AVAILABLE: {total_matches:,}")
print(f"{'='*70}\n")

# Show that calibration used this data
print("Calibration results:")
try:
    import json
    with open('calibrated_params.json', 'r') as f:
        params = json.load(f)
    
    metrics = params.get('accuracy_metrics', {})
    print(f"  Matches analyzed: {metrics.get('matches_analyzed', 'N/A')}")
    print(f"  Off-by-1 rate: {metrics.get('off_by_one_rate', 0):.1f}%")
    print(f"  BTTS accuracy: {metrics.get('btts_accuracy', 0):.1f}%")
    print(f"  Exact score top-5: {metrics.get('exact_score_top5_rate', 0):.1f}%")
    print(f"\nCalibrated parameters:")
    print(f"  League avg goals: {params.get('league_avg_goals', 0):.3f}")
    print(f"  Home advantage: {params.get('home_advantage', 0):.3f}")
    print(f"  Away penalty: {params.get('away_penalty', 0):.3f}")
except:
    print("  No calibration file found yet")

print(f"\n{'='*70}")
