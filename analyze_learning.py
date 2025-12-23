"""
Learning Analytics - Analyze Model Performance from Feedback
=============================================================
Reads learned_matches.csv with enhanced tracking to show:
- Where model makes mistakes
- Error patterns by mistake range
- Continuous improvement over time
"""

import pandas as pd
import os

def analyze_learned_matches():
    """Analyze learned matches to show model improvement."""
    
    csv_path = "data/learned_matches.csv"
    
    if not os.path.exists(csv_path):
        print("❌ No learned matches found yet!")
        print("   Submit feedback through the web app to start tracking.")
        return
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print("="*70)
        print("📚 LEARNING ANALYTICS - Model Performance from Feedback")
        print("="*70)
        print(f"\n📊 Total Matches Tracked: {len(df)}")
        
        # Filter only matches with predictions (xG > 0)
        df_with_predictions = df[df['PredictedHomeXG'] > 0].copy()
        
        if len(df_with_predictions) == 0:
            print("\n⚠️  No matches with predictions yet!")
            print("   Make predictions before submitting feedback to track errors.")
            return
        
        print(f"📈 Matches with Predictions: {len(df_with_predictions)}")
        
        # Overall Error Metrics
        print("\n" + "="*70)
        print("🎯 OVERALL ACCURACY")
        print("="*70)
        
        if 'TotalGoalsError' in df_with_predictions.columns:
            avg_total_error = df_with_predictions['TotalGoalsError'].mean()
            print(f"  Average Total Goals Error: {avg_total_error:.2f} goals")
            
            # Error distribution
            perfect = len(df_with_predictions[df_with_predictions['TotalGoalsError'] == 0])
            within_1 = len(df_with_predictions[df_with_predictions['TotalGoalsError'] <= 1])
            within_2 = len(df_with_predictions[df_with_predictions['TotalGoalsError'] <= 2])
            
            print(f"\n  Error Distribution:")
            print(f"    Perfect (0 error): {perfect}/{len(df_with_predictions)} ({perfect/len(df_with_predictions)*100:.1f}%)")
            print(f"    Within ±1 goal: {within_1}/{len(df_with_predictions)} ({within_1/len(df_with_predictions)*100:.1f}%)")
            print(f"    Within ±2 goals: {within_2}/{len(df_with_predictions)} ({within_2/len(df_with_predictions)*100:.1f}%)")
        
        if 'XGErrorHome' in df_with_predictions.columns:
            avg_xg_error = (df_with_predictions['XGErrorHome'].mean() + df_with_predictions['XGErrorAway'].mean()) / 2
            print(f"\n  Average xG Error per Team: {avg_xg_error:.2f} goals")
        
        if 'ScoreExactMatch' in df_with_predictions.columns:
            exact_matches = df_with_predictions['ScoreExactMatch'].sum()
            print(f"\n  Exact Score Predictions: {exact_matches}/{len(df_with_predictions)} ({exact_matches/len(df_with_predictions)*100:.1f}%)")
        
        if 'OutcomeCorrect' in df_with_predictions.columns:
            outcome_correct = df_with_predictions['OutcomeCorrect'].sum()
            print(f"  Outcome (W/D/L) Accuracy: {outcome_correct}/{len(df_with_predictions)} ({outcome_correct/len(df_with_predictions)*100:.1f}%)")
        
        if 'BTTSCorrect' in df_with_predictions.columns:
            btts_correct = df_with_predictions['BTTSCorrect'].sum()
            print(f"  BTTS Accuracy: {btts_correct}/{len(df_with_predictions)} ({btts_correct/len(df_with_predictions)*100:.1f}%)")
        
        # Confidence Calibration
        print("\n" + "="*70)
        print("🎲 CONFIDENCE CALIBRATION")
        print("="*70)
        
        if 'Confidence' in df_with_predictions.columns and 'OutcomeCorrect' in df_with_predictions.columns:
            # Analyze high vs low confidence predictions
            high_conf = df_with_predictions[df_with_predictions['Confidence'] >= 70]
            low_conf = df_with_predictions[df_with_predictions['Confidence'] < 50]
            
            if len(high_conf) > 0:
                high_conf_acc = high_conf['OutcomeCorrect'].mean() * 100
                print(f"  High Confidence (≥70): {len(high_conf)} predictions, {high_conf_acc:.1f}% accurate")
            
            if len(low_conf) > 0:
                low_conf_acc = low_conf['OutcomeCorrect'].mean() * 100
                print(f"  Low Confidence (<50): {len(low_conf)} predictions, {low_conf_acc:.1f}% accurate")
        
        # Biggest Mistakes
        print("\n" + "="*70)
        print("⚠️  BIGGEST MISTAKES (Learn from these!)")
        print("="*70)
        
        if 'TotalGoalsError' in df_with_predictions.columns:
            worst_predictions = df_with_predictions.nlargest(5, 'TotalGoalsError')[
                ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 
                 'PredictedHomeXG', 'PredictedAwayXG', 'TotalGoalsError']
            ]
            
            print("\n  Top 5 Worst Predictions:")
            for idx, row in worst_predictions.iterrows():
                print(f"    {row['HomeTeam']} {row['FTHG']}-{row['FTAG']} {row['AwayTeam']}")
                print(f"      Predicted xG: {row['PredictedHomeXG']:.1f}-{row['PredictedAwayXG']:.1f} | Error: {row['TotalGoalsError']:.1f} goals")
        
        # Improvement Over Time
        if len(df_with_predictions) >= 10:
            print("\n" + "="*70)
            print("📈 IMPROVEMENT OVER TIME")
            print("="*70)
            
            # Split into first half and second half
            mid_point = len(df_with_predictions) // 2
            first_half = df_with_predictions.iloc[:mid_point]
            second_half = df_with_predictions.iloc[mid_point:]
            
            if 'TotalGoalsError' in df_with_predictions.columns:
                early_error = first_half['TotalGoalsError'].mean()
                recent_error = second_half['TotalGoalsError'].mean()
                improvement = early_error - recent_error
                
                print(f"  Early Predictions (first {len(first_half)}): {early_error:.2f} avg error")
                print(f"  Recent Predictions (last {len(second_half)}): {recent_error:.2f} avg error")
                
                if improvement > 0:
                    print(f"  ✅ Improvement: -{improvement:.2f} goals ({improvement/early_error*100:.1f}% better!)")
                else:
                    print(f"  ⚠️  Needs Work: +{abs(improvement):.2f} goals worse")
        
        print("\n" + "="*70)
        print("💡 RECOMMENDATIONS")
        print("="*70)
        print("\n  Based on your feedback data:")
        
        if 'TotalGoalsError' in df_with_predictions.columns:
            avg_error = df_with_predictions['TotalGoalsError'].mean()
            if avg_error > 2.0:
                print("  ⚠️  High error rate - consider recalibrating model")
            elif avg_error > 1.5:
                print("  📊 Moderate errors - model learning from feedback")
            else:
                print("  ✅ Good accuracy - model performing well!")
        
        if len(df_with_predictions) < 30:
            print(f"  📝 Need more data: {30 - len(df_with_predictions)} more feedback entries recommended")
        else:
            print("  ✅ Sufficient data for continuous learning")
        
    except Exception as e:
        print(f"❌ Error reading learned matches: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_learned_matches()
