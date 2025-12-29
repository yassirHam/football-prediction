# Daily Match Analyzer - User Guide

## 🎯 Overview

This automation system **eliminates all manual work** for daily match analysis. You only need to provide a match list from Flashscore - everything else is automated.

**What it does**:
- ✅ Automatically looks up historical data for each team
- ✅ Extracts last 5 matches with first-half statistics
- ✅ Runs the frozen prediction model
- ✅ Applies deterministic decision logic
- ✅ Outputs **only** BET_TOP_2 and BET_TOP_3 matches

**What it does NOT do**:
- ❌ No scraping
- ❌ No browser automation
- ❌ No manual H2H lookups
- ❌ No clicking through match pages

---

## 📋 Quick Start

### Step 1: Create Your Match List

Create a file called `daily_matches.txt` with your matches:

```text
Arsenal vs Tottenham
Manchester City vs Liverpool
Chelsea vs Manchester United
Burnley vs Everton
```

**Format**: `Home Team vs Away Team` (one per line)

**Supported separators**: 
- `vs` → Arsenal vs Tottenham
- `v` → Arsenal v Tottenham  
- `-` → Arsenal - Tottenham

---

### Step 2: Run the Analyzer

```bash
python daily_analyzer.py --input daily_matches.txt --output daily_betting_signals.json
```

**That's it!** The system will:
1. Load 21.9k historical matches
2. Find last 5 matches for each team
3. Extract first-half statistics
4. Run predictions
5. Apply decision logic
6. Save results to JSON

---

## 📊 Output Format

The system generates `daily_betting_signals.json`:

```json
{
  "date": "2025-12-26",
  "time_generated": "2025-12-26 23:17:09",
  "total_matches_analyzed": 5,
  "betting_opportunities": 3,
  "pass_count": 2,
  "summary": {
    "bet_top2_count": 1,
    "bet_top3_count": 2,
    "signal_rate": 60.0
  },
  "signals": [
    {
      "match": "Chelsea vs Manchester United",
      "decision": "BET_TOP_2",
      "top_predictions": ["0-0", "1-0", "2-0"],
      "probabilities": [61.8, 29.7, 7.2],
      "xG_total": 2.46,
      "xG_diff": 0.14,
      "top2_prob_sum": 91.6,
      "top3_prob_sum": 98.7,
      "reason": "High efficiency: Low variance with strong Top-2 concentration"
    }
  ]
}
```

---

## 🔧 Command-Line Options

### Basic Usage
```bash
python daily_analyzer.py
```
Uses defaults: `daily_matches.txt` → `daily_betting_signals.json`

### Custom Input/Output
```bash
python daily_analyzer.py --input my_matches.txt --output results.json
```

### Quiet Mode (No Console Output)
```bash
python daily_analyzer.py --quiet
```

### Custom Data Directory
```bash
python daily_analyzer.py --data-dir path/to/data
```

---

## 📝 Input Formats

### Format 1: Plain Text (Recommended)
```text
# Daily matches for 2025-12-27
Arsenal vs Tottenham
Manchester City vs Liverpool
Chelsea vs Manchester United
```

### Format 2: CSV with League Codes
```csv
HomeTeam,AwayTeam,League
Arsenal,Tottenham,E0
Lyon,Marseille,F1
Barcelona,Real Madrid,SP1
```

---

## 🎯 Understanding the Output

### Decision Types

| Decision | Meaning | Action |
|----------|---------|--------|
| `BET_TOP_2` | **High Efficiency** | Bet on top 2 scores |
| `BET_TOP_3` | **High Coverage** | Bet on top 3 scores |
| `PASS` | **No Signal** | Skip (not in output) |

### Key Metrics

- **xG_total**: Total expected goals (both teams)
- **xG_diff**: Difference in expected goals (imbalance)
- **top2_prob_sum**: Combined probability of top 2 scores
- **top3_prob_sum**: Combined probability of top 3 scores

### Decision Reasons

#### BET_TOP_2
> "High efficiency: Low variance with strong Top-2 concentration"

**Criteria**: Low-scoring, predictable match where top 2 scores capture most probability

**Example**: Chelsea vs Man Utd → 0-0 (62%) + 1-0 (30%) = 92%

---

#### BET_TOP_3
> "High coverage: Defensive/imbalanced match with strong Top-3 concentration"

**Criteria**: Defensive or mismatched game where top 3 scores needed for coverage

**Example**: Arsenal vs Spurs → 0-0 (44%) + 1-0 (27%) + 0-1 (12%) = 83%

---

## 📊 Example Workflow

### Monday Morning Routine

1. **Copy matches from Flashscore** (1 minute)
   ```text
   Arsenal vs Tottenham
   Liverpool vs Chelsea
   ...
   ```

2. **Run analyzer** (30 seconds)
   ```bash
   python daily_analyzer.py
   ```

3. **Review signals** (1 minute)
   - Open `daily_betting_signals.json`
   - Check "signals" array
   - Only 3 matches qualify → BET them

**Total time**: 2.5 minutes (vs 20+ minutes manual)

---

## 🔍 Troubleshooting

### "Team not found in historical data"

**Cause**: Team name doesn't match dataset

**Solutions**:
1. Try variations: "Man Utd" → "Manchester United"
2. Check fuzzy matching log for suggestions
3. Team may not have 5+ historical matches

### "Insufficient historical data"

**Cause**: Team has fewer than 3 matches in database

**Solution**: Skip this team or manually add data

### NaN/Missing Data Errors

**Already Fixed**: System converts NaN → 0 automatically

---

## 🚀 Advanced Usage

### One-Click Daily Analysis

Create a batch script `analyze_today.bat`:

```batch
@echo off
cd c:\path\to\football-prediction
python daily_analyzer.py
type daily_betting_signals.json
pause
```

### Auto-Email Results

Add to `daily_analyzer.py` (at the end):

```python
import smtplib
from email.message import EmailMessage

# Send results via email
msg = EmailMessage()
msg.set_content(json.dumps(output, indent=2))
msg['Subject'] = f"Betting Signals - {datetime.now().strftime('%Y-%m-%d')}"
msg['From'] = "your_email@gmail.com"
msg['To'] = "recipient@gmail.com"

# Configure SMTP and send
```

---

## 📚 How It Works (Technical)

### Pipeline Architecture

```
Manual Input (Flashscore)
         ↓
  daily_matches.txt
         ↓
  [daily_analyzer.py] ────┐
         ↓                │
  [team_data_lookup.py]←─┘
  • Loads combined_training_data.csv
  • Fuzzy matches team names
  • Extracts last 5 matches
  • Extracts FTHG, FTAG, HTHG, HTAG
         ↓
  Team Objects (Home + Away)
         ↓
  [football_predictor.py]
  • Poisson/XGBoost model
  • Calculates xG_home, xG_away
  • Generates first-half score probabilities
         ↓
  [decision_logic.py]
  • Applies deterministic rules
  • Returns BET_TOP_2 / BET_TOP_3 / PASS
         ↓
  Filter PASS decisions
         ↓
  daily_betting_signals.json
```

### Data Flow Example

**Input**: "Arsenal vs Tottenham"

1. **Lookup**: Find Arsenal's last 5 matches
   ```
   Arsenal 2-1 Man City (HT: 1-0)
   Arsenal 3-0 Brighton (HT: 2-0)
   Arsenal 1-1 Liverpool (HT: 0-1)
   Arsenal 2-0 Chelsea (HT: 1-0)
   Arsenal 4-2 Leeds (HT: 2-1)
   ```

2. **Extract**: 
   - goals_scored = [2, 3, 1, 2, 4]
   - goals_conceded = [1, 0, 1, 0, 2]
   - first_half_goals = [1, 2, 0, 1, 2]

3. **Predict**: 
   - xG_home = 1.78
   - xG_away = 1.03
   - FH Probs: 0-0 (44%), 1-0 (27%), 0-1 (12%)

4. **Decide**:
   - xG_total = 2.81
   - xG_diff = 1.24
   - top3_prob_sum = 82.3%
   - **Decision: BET_TOP_3** ✅

5. **Output**: Include in `signals` array

---

## 🛡️ Safety Statement

> **This automation relies exclusively on local historical data and does not scrape or automate third-party websites.**

All data comes from `combined_training_data.csv` - no external API calls or web scraping.

---

## 📞 Support

For issues:
1. Check this guide
2. Review `daily_analyzer.py` console output
3. Verify input file format
4. Ensure historical data exists

---

## ✅ Success Checklist

- [x] Manual match list from Flashscore (only external input)
- [x] Automated historical data lookup
- [x] Automated H2H statistics extraction
- [x] Automated model prediction
- [x] Automated decision logic
- [x] Filtered output (only BET signals)
- [x] Zero scraping/automation
- [x] Deterministic and repeatable

**You did it!** 🎉
