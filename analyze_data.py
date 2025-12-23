import pandas as pd
import glob
import numpy as np

def analyze_dataset():
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("No CSV files found.")
        return

    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
            if 'FTHG' in df.columns and 'FTAG' in df.columns:
                all_dfs.append(df)
        except:
            pass
    
    if not all_dfs:
        print("No valid data found.")
        return

    full_data = pd.concat(all_dfs, ignore_index=True)
    
    total_matches = len(full_data)
    home_goals = full_data['FTHG'].sum()
    away_goals = full_data['FTAG'].sum()
    
    avg_home = home_goals / total_matches
    avg_away = away_goals / total_matches
    global_avg = (home_goals + away_goals) / (2 * total_matches)
    
    # Calculate Home Advantage (Ratio of Home/Away goals usually, or Home/GlobalAvg)
    # Standard definition: Home Strength / Away Strength. 
    # Simplified: Multiplier relative to "neutral".
    # If neutral team scores X. Home team scores X * Adv. Away team scores X * Penalty.
    # Avg Home = Global * HomeAdv
    # Avg Away = Global * AwayPenalty
    
    calc_home_adv = avg_home / global_avg
    calc_away_penalty = avg_away / global_avg
    
    print(f"Total Matches: {total_matches}")
    print(f"Total Goals: {home_goals + away_goals}")
    print(f"Avg Goals/Match (Total): {(home_goals + away_goals)/total_matches:.4f}")
    print(f"Global Avg per Team: {global_avg:.4f}")
    print("-" * 30)
    print(f"Avg Home Goals: {avg_home:.4f}")
    print(f"Avg Away Goals: {avg_away:.4f}")
    print("-" * 30)
    print(f"RECOMMENDED CONSTANTS:")
    print(f"LEAGUE_AVG_GOALS = {global_avg:.3f}")
    print(f"HOME_ADVANTAGE = {calc_home_adv:.3f}")
    print(f"AWAY_PENALTY = {calc_away_penalty:.3f}")

if __name__ == "__main__":
    analyze_dataset()
