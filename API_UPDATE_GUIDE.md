# Football-Data.org API Update Guide

## 🎯 Purpose

This script keeps your local match database up-to-date by fetching recent matches from the **official Football-Data.org API**.

**What it does**:
- ✅ Fetches finished matches with half-time scores
- ✅ Appends only NEW matches to your CSV
- ✅ Preserves chronological order
- ✅ Uses official API (no scraping)

---

## 🔑 Setup (One-Time)

### Step 1: Get Your API Token

1. Go to https://www.football-data.org/
2. Register for a free account
3. Copy your API token

### Step 2: Set Environment Variable

**Windows**:
```cmd
set FOOTBALL_DATA_TOKEN=your_token_here
```

**Linux/Mac**:
```bash
export FOOTBALL_DATA_TOKEN=your_token_here
```

**Permanent (Windows)**:
```cmd
setx FOOTBALL_DATA_TOKEN "your_token_here"
```

---

## 🚀 Usage

### Basic Usage (Last 10 Days)
```bash
python update_match_data.py
```

### Custom Date Range
```bash
# Fetch last 30 days
python update_match_data.py --days 30
```

### Custom CSV Path
```bash
python update_match_data.py --csv path/to/custom.csv
```

### Specify Token Directly
```bash
python update_match_data.py --token your_token_here
```

---

## 📊 Example Output

```
============================================================
🔄 FOOTBALL-DATA.ORG CSV UPDATER
============================================================

📂 Loaded existing CSV: 21,823 matches
📡 Fetching matches from 2025-12-16 to 2025-12-26...
✅ Fetched 124 finished matches
💾 CSV updated successfully: 21,841 total matches

============================================================
📊 UPDATE SUMMARY
============================================================
✅ Matches fetched: 124
➕ New matches added: 18
⏭️  Skipped (duplicates): 96
⚠️  Skipped (missing HT data): 10
============================================================

✅ Update completed successfully!
```

---

## 📅 Recommended Workflow

### Daily Routine (Every Morning)

1. **Update match database** (5 seconds):
   ```bash
   python update_match_data.py
   ```

2. **Run daily analyzer** (5 seconds):
   ```bash
   python daily_analyzer.py
   ```

3. **Review betting signals** (1 minute):
   - Open `daily_betting_signals.json`

**Total time**: ~10 seconds automation + 1 minute review

---

## 🔧 How It Works

### Step 1: Load Existing CSV
- Reads `combined_training_data.csv`
- Creates unique keys: `Date|HomeTeam|AwayTeam`
- Tracks existing matches to prevent duplicates

### Step 2: Fetch Recent Matches
- Calls Football-Data.org API
- Parameters:
  - `dateFrom`: X days ago
  - `dateTo`: Today
  - `status`: FINISHED

### Step 3: Normalize Data
API format → CSV format:

| API Field | CSV Column |
|-----------|------------|
| `utcDate` | `Date` |
| `homeTeam.name` | `HomeTeam` |
| `awayTeam.name` | `AwayTeam` |
| `score.halfTime.home` | `HTHG` |
| `score.halfTime.away` | `HTAG` |
| `score.fullTime.home` | `FTHG` |
| `score.fullTime.away` | `FTAG` |
| `competition.name` | `Competition` |

### Step 4: Filter & Deduplicate
- Skip matches already in CSV
- Skip matches missing half-time scores
- Skip invalid/incomplete data

### Step 5: Append & Save
- Append new rows only
- Sort by date (chronological order)
- Overwrite CSV with updated data

---

## ⚠️ Important Notes

### Half-Time Scores Required
Matches **without half-time scores** are automatically skipped because:
- Your model requires `HTHG` and `HTAG`
- Incomplete data would break predictions

### API Rate Limits
Free tier limits:
- **10 requests per minute**
- **10 requests per day** (check current limits)

Solution: Run once per day, not continuously.

### Team Name Matching
API team names may differ from your CSV:
- API: "Manchester United FC"
- CSV: "Manchester United"

**Fuzzy matching in `team_data_lookup.py` handles this automatically**.

---

## 🔍 Troubleshooting

### "API token required"
**Solution**: Set `FOOTBALL_DATA_TOKEN` environment variable

### "No matches fetched"
**Possible causes**:
1. API rate limit exceeded
2. No finished matches in date range
3. Invalid token

**Solution**: Check API response, wait 24h if rate-limited

### "Matches skipped (missing HT data)"
**Normal**: Some leagues don't report half-time scores

**Action**: None needed - these matches are correctly skipped

---

## 🎯 Integration with Daily Analyzer

### Before (Without API Update)
```
Last 5 matches → May be outdated if CSV old
```

### After (With API Update)
```
Step 1: python update_match_data.py  ← Fetch latest matches
Step 2: python daily_analyzer.py     ← Use fresh data
```

**Result**: Daily analyzer now uses **real current form**

---

## 📚 API Documentation

Full API docs: https://www.football-data.org/documentation/api

**Endpoints used**:
- `GET /v4/matches` - Get matches by date range

**Authentication**: Header-based
```
X-Auth-Token: your_token_here
```

---

## ✅ Safety Verification

- [x] Uses official API only
- [x] No web scraping
- [x] No browser automation
- [x] Registered API token required
- [x] Respects rate limits
- [x] Deduplicates to prevent double-counting

---

## 🛡️ Safety Statement

> **This script updates local match data using the official Football-Data.org API and does not scrape or automate third-party websites.**
