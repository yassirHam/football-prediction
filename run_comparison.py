"""
Simple comparison test - Base vs Enhanced
Avoids Unicode issues and saves results to file
"""
import pandas as pd
import glob
from football_predictor import Team as BaseTeam, predict_match as base_predict
from enhanced_predictor import Team, enhanced_predict_match
import numpy as np
import sys

# Redirect output to file to avoid Unicode issues
output_file = open('test_results.txt', 'w', encoding='utf-8')
sys.stdout = output_file

def load_matches():
    csv_files = glob.glob("data/*.csv")
    all_matches = []
    for csv_file in csv_files:
        if 'learned' in csv_file:
            continue
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            df.columns = df.columns.str.replace('\ufeff', '').str.strip()
            if all(col in df.columns for col in ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']):
                league = csv_file.split('\\')[-1].replace('.csv', '')
                df['League'] = league
                all_matches.append(df)
        except:
            pass
    return pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

def create_base_team(team_name, df, idx):
    prev = df.iloc[:idx]
    matches = prev[(prev['HomeTeam'] == team_name) | (prev['AwayTeam'] == team_name)].tail(5)
    
    gs, gc, fh = [], [], []
    for _, m in matches.iterrows():
        if m['HomeTeam'] == team_name:
            gs.append(int(m['FTHG']))
            gc.append(int(m['FTAG']))
            fh.append(int(m.get('HTHG', m['FTHG'] * 0.45)))
        else:
            gs.append(int(m['FTAG']))
            gc.append(int(m['FTHG']))
            fh.append(int(m.get('HTAG', m['FTAG'] * 0.45)))
    
    while len(gs) < 5:
        gs.insert(0, 1)
        gc.insert(0, 1)
        fh.insert(0, 0)
    
    return BaseTeam(team_name, gs[-5:], gc[-5:], fh[-5:])

def create_enhanced_team(team_name, df, idx, league):
    prev = df.iloc[:idx]
    matches = prev[(prev['HomeTeam'] == team_name) | (prev['AwayTeam'] == team_name)].tail(5)
    
    gs, gc, fh, sot, sotc = [], [], [], [], []
    
    for _, m in matches.iterrows():
        if m['HomeTeam'] == team_name:
            gs.append(int(m['FTHG']))
            gc.append(int(m['FTAG']))
            fh.append(int(m.get('HTHG', m['FTHG'] * 0.45)))
            if 'HST' in m and pd.notna(m['HST']):
                sot.append(int(m['HST']))
            if 'AST' in m and pd.notna(m['AST']):
                sotc.append(int(m['AST']))
        else:
            gs.append(int(m['FTAG']))
            gc.append(int(m['FTHG']))
            fh.append(int(m.get('HTAG', m['FTAG'] * 0.45)))
            if 'AST' in m and pd.notna(m['AST']):
                sot.append(int(m['AST']))
            if 'HST' in m and pd.notna(m['HST']):
                sotc.append(int(m['HST']))
    
    while len(gs) < 5:
        gs.insert(0, 1)
        gc.insert(0, 1)
        fh.insert(0, 0)
    
    return Team(team_name, gs[-5:], gc[-5:], fh[-5:],
                shots_on_target=sot[-5:] if sot else None,
                shots_on_target_conceded=sotc[-5:] if sotc else None,
                league=league)

print("="*70)
print("BASE vs ENHANCED PREDICTOR - TEST RESULTS")
print("="*70)

df = load_matches()
print(f"\nTotal data: {len(df)} matches")

base_err, enh_err = [], []
base_off1, enh_off1 = 0, 0
base_btts, enh_btts = 0, 0
base_exact, enh_exact = 0, 0
total = 0

for idx in range(max(10, len(df)-310), min(len(df)-10, max(10, len(df)-310)+300)):
    try:
        m = df.iloc[idx]
        league = m.get('League', 'DEFAULT')
        
        bh = create_base_team(m['HomeTeam'], df, idx)
        ba = create_base_team(m['AwayTeam'], df, idx)
        eh = create_enhanced_team(m['HomeTeam'], df, idx, league)
        ea = create_enhanced_team(m['AwayTeam'], df, idx, league)
        
        bp = base_predict(bh, ba)
        ep = enhanced_predict_match(eh, ea)
        
        ah = int(m['FTHG'])
        aa = int(m['FTAG'])
        at = ah + aa
        abtts = ah > 0 and aa > 0
        
        # Base stats
        bs = bp['full_match_predictions'][0][0]
        bt = bs[0] + bs[1]
        be = abs(bt - at)
        base_err.append(be)
        if be == 1:
            base_off1 += 1
        if (bp['both_teams_score'] > 0.5) == abtts:
            base_btts += 1
        if (ah, aa) in [s for s, _ in bp['full_match_predictions'][:5]]:
            base_exact += 1
        
        # Enhanced stats  
        es = ep['full_match_predictions'][0][0]
        et = es[0] + es[1]
        ee = abs(et - at)
        enh_err.append(ee)
        if ee == 1:
            enh_off1 += 1
        if (ep['both_teams_score'] > 0.5) == abtts:
            enh_btts += 1
        if (ah, aa) in [s for s, _ in ep['full_match_predictions'][:5]]:
            enh_exact += 1
        
        total += 1
    except:
        continue

print(f"Tested: {total} matches\n")
print("="*70)
print("RESULTS")
print("="*70)
print("\nMETRIC                    BASE      ENHANCED  IMPROVEMENT")
print("-"*70)

bmae = np.mean(base_err)
emae = np.mean(enh_err)
print(f"Total Goals MAE:          {bmae:.2f}      {emae:.2f}      {bmae-emae:+.2f}")

boff = base_off1/total*100
eoff = enh_off1/total*100
print(f"Off-by-1 Rate:            {boff:.1f}%     {eoff:.1f}%     {boff-eoff:+.1f}%")

bbtts = base_btts/total*100
ebtts = enh_btts/total*100
print(f"BTTS Accuracy:            {bbtts:.1f}%     {ebtts:.1f}%     {ebtts-bbtts:+.1f}%")

bex = base_exact/total*100
eex = enh_exact/total*100
print(f"Exact Score Top-5:        {bex:.1f}%     {eex:.1f}%     {eex-bex:+.1f}%")

print("\n" + "="*70)
wins = sum([emae < bmae, eoff < boff, ebtts > bbtts, eex > bex])
print(f"Enhanced wins in {wins}/4 metrics")

if wins >= 3:
    print("\nRECOMMENDATION: USE ENHANCED - Clear improvement!")
elif wins >= 2:
    print("\nRECOMMENDATION: CONSIDER ENHANCED - Moderate improvement")
else:
    print("\nRECOMMENDATION: STICK WITH BASE - Not enough improvement")

print("="*70)

sys.stdout = sys.__stdout__
output_file.close()
print("Results saved to test_results.txt")
