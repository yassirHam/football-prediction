@echo off
REM ===========================================================================
REM DAILY BETTING AUTOMATION - ONE-CLICK WORKFLOW
REM ===========================================================================
REM
REM This batch file runs the complete daily workflow:
REM   1. Update match database from Football-Data.org API
REM   2. Analyze today's fixtures
REM   3. Display betting signals
REM
REM SETUP REQUIRED (ONE-TIME):
REM   set FOOTBALL_DATA_TOKEN=your_token_here
REM
REM ===========================================================================

echo.
echo ============================================================
echo           DAILY BETTING AUTOMATION WORKFLOW
echo ============================================================
echo.

REM Check if API token is set
if "%FOOTBALL_DATA_TOKEN%"=="" (
    echo ERROR: FOOTBALL_DATA_TOKEN environment variable not set
    echo.
    echo Please set your API token:
    echo   set FOOTBALL_DATA_TOKEN=your_token_here
    echo.
    echo Get your token at: https://www.football-data.org/
    echo.
    pause
    exit /b 1
)

REM Change to script directory
cd /d %~dp0

echo.
echo [STEP 1/3] Updating match database from Football-Data.org...
echo ------------------------------------------------------------
python update_match_data.py --days 10

if errorlevel 1 (
    echo.
    echo ERROR: Failed to update match database
    pause
    exit /b 1
)

echo.
echo.
echo [STEP 2/3] Analyzing daily matches...
echo ------------------------------------------------------------

REM Check if daily_matches.txt exists
if not exist daily_matches.txt (
    echo ERROR: daily_matches.txt not found
    echo.
    echo Please create daily_matches.txt with your fixtures:
    echo   Arsenal vs Tottenham
    echo   Chelsea vs Liverpool
    echo   ...
    echo.
    pause
    exit /b 1
)

python daily_analyzer.py --input daily_matches.txt --output daily_betting_signals.json

if errorlevel 1 (
    echo.
    echo ERROR: Failed to analyze matches
    pause
    exit /b 1
)

echo.
echo.
echo [STEP 3/3] Displaying betting signals...
echo ------------------------------------------------------------
echo.

REM Display the JSON output nicely
type daily_betting_signals.json

echo.
echo.
echo ============================================================
echo                  WORKFLOW COMPLETE!
echo ============================================================
echo.
echo Betting signals saved to: daily_betting_signals.json
echo.

pause
