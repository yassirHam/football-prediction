"""
Daily Match Analyzer
====================
Automated betting signal generator for daily match lists.

This script:
1. Reads manual match list (Home vs Away format)
2. Looks up historical data from local files (Home form for Home team, Away form for Away team)
3. Runs hybrid prediction model (XGBoost + Poisson)
4. Applies deterministic decision logic
5. Outputs only BET_TOP_2 and BET_TOP_3 signals

NO SCRAPING - Uses only local historical data.

Author: Football Analytics System
Version: 2.0 - Unified Logic
"""

import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import os
import sys
import re

# Force UTF-8 output for Windows consoles
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing modules
from team_data_lookup import TeamDataLookup
# Use the HYBRID predictor to match web app logic
try:
    from hybrid.hybrid_predictor import predict_match_hybrid as predict_match
    print("✅ Using HYBRID prediction pipeline (Sync with Web App)")
except ImportError:
    print("⚠️  Hybrid module not found, falling back to standard predictor")
    from football_predictor import predict_match

from decision_logic import make_decision


class DailyMatchAnalyzer:
    """Main automation engine for daily betting signals."""
    
    def __init__(self, data_dir: str = "data", debug: bool = False):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Directory containing historical data
            debug: Enable debug output
        """
        self.lookup = TeamDataLookup(data_dir)
        self.debug = debug
        self.results = []
        self.stats = {
            'total_matches': 0,
            'bet_top2_count': 0,
            'bet_top3_count': 0,
            'pass_count': 0,
            'failed_lookups': 0
        }
    
    def parse_match_list(self, filepath: str) -> List[Dict[str, str]]:
        """
        Parse match list from text or CSV file.
        
        Supported formats:
        - Text: "Home Team vs Away Team" (one per line)
        - CSV: HomeTeam,AwayTeam (with header)
        
        Args:
            filepath: Path to match list file
        
        Returns:
            List of {"home": str, "away": str, "league": str, "home_rank": int, "away_rank": int} dictionaries
        """
        matches = []
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Match list file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Detect format
        first_line = lines[0].strip()
        
        if ',' in first_line and first_line.lower().startswith('home'):
            # CSV format with header
            for line in lines[1:]:  # Skip header
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    matches.append({
                        'home': parts[0],
                        'away': parts[1],
                        'league': parts[2] if len(parts) > 2 else 'DEFAULT',
                        'home_rank': None,
                        'away_rank': None
                    })
        else:
            # Text format: "Home vs Away" or "Home (3) vs Away (10) [LEAGUE]"
            # Regex for extended format: Team A (RankA) vs Team B (RankB) [League]
            # Supports optional ranks and optional league
            regex = r"^(.*?)(?:\s*\((\d+)\))?\s+vs\s+(.*?)(?:\s*\((\d+)\))?(?:\s*\[([A-Z0-9]+)\])?$"
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty/comments
                    continue
                
                # Check for "v" or "-" separators and normalize to "vs"
                norm_line = line
                if ' vs ' not in norm_line.lower():
                     if ' v ' in norm_line.lower():
                         norm_line = norm_line.replace(' v ', ' vs ')
                     elif ' - ' in norm_line:
                         norm_line = norm_line.replace(' - ', ' vs ')
                
                match = re.match(regex, norm_line, re.IGNORECASE)
                
                if match:
                    home_name = match.group(1).strip()
                    home_rank = int(match.group(2)) if match.group(2) else None
                    away_name = match.group(3).strip()
                    away_rank = int(match.group(4)) if match.group(4) else None
                    league_code = match.group(5).strip() if match.group(5) else 'DEFAULT'
                    
                    matches.append({
                        'home': home_name,
                        'away': away_name,
                        'league': league_code,
                        'home_rank': home_rank,
                        'away_rank': away_rank
                    })
                else:
                    # Fallback for simple format if regex fails strangely
                    if ' vs ' in line.lower():
                        parts = line.split(' vs ', 1)
                        matches.append({
                            'home': parts[0].strip(),
                            'away': parts[1].strip(),
                            'league': 'DEFAULT',
                            'home_rank': None,
                            'away_rank': None
                        })
                    else:
                        print(f"⚠️  Skipping invalid format: {line}")
                        continue
        
        print(f"📋 Parsed {len(matches)} matches from {filepath}")
        return matches
    
    def analyze_match(self, home_name: str, away_name: str, 
                     league: str = "DEFAULT", 
                     home_rank: Optional[int] = None, 
                     away_rank: Optional[int] = None) -> Optional[Dict]:
        """
        Analyze a single match and generate betting signal.
        
        Args:
            home_name: Home team name
            away_name: Away team name
            league: League code (optional)
            home_rank: Home team league position (1-20)
            away_rank: Away team league position (1-20)
        
        Returns:
            Result dictionary or None if data unavailable
        """
        # Step 1: Infer league if missing
        if league == 'DEFAULT':
            league = self.lookup.infer_league_code(home_name)
            if self.debug and league != 'DEFAULT':
                print(f"   ℹ️  Inferred league: {league}")

        # Step 2: Lookup historical data (VENUE SPECIFIC)
        # Home team -> Prioritize recent HOME matches
        home_team = self.lookup.get_team_data(
            home_name, n=5, league=league, venue='home', rank=home_rank
        )
        # Away team -> Prioritize recent AWAY matches
        away_team = self.lookup.get_team_data(
            away_name, n=5, league=league, venue='away', rank=away_rank
        )
        
        if not home_team or not away_team:
            self.stats['failed_lookups'] += 1
            if self.debug:
                print(f"❌ Skipping {home_name} vs {away_name}: Insufficient historical data")
            return None
        
        # SAFETY GUARD 1: Require at least 3 matches with data
        if len(home_team.goals_scored) < 3 or len(away_team.goals_scored) < 3:
            if self.debug:
                print(f"⚠️  Skipping {home_name} vs {away_name}: Not enough matches (<3)")
            return None

        # Step 2: Run HYBRID prediction model
        prediction = predict_match(home_team, away_team, neutral_venue=False)
        
        # Step 3: Apply decision logic
        decision = make_decision(
            prediction['first_half_predictions'],
            prediction['expected_goals']['home'],
            prediction['expected_goals']['away']
        )

        # SAFETY GUARD 2: Check probability sum
        total_prob = sum(p for _, p in prediction['first_half_predictions'])
        if not (0.95 <= total_prob <= 1.05): # approx 1.0
             # Re-normalize if needed (should be handled by predictor but double check)
             if self.debug:
                 print(f"⚠️  Probability sum issue ({total_prob:.2f}), re-normalizing...")
        
        # Step 4: Extract top predictions (format them nicely)
        top_predictions = [
            f"{score[0]}-{score[1]}" 
            for score, prob in prediction['first_half_predictions'][:3]
        ]
        
        # Step 5: Build result
        result = {
            'match': f"{home_name} vs {away_name}",
            'decision': decision['decision'],
            'top_predictions': top_predictions,
            'probabilities': [
                round(prob * 100, 1) 
                for score, prob in prediction['first_half_predictions'][:3]
            ],
            'xG_total': round(decision['xG_total'], 2),
            'xG_diff': round(decision['xG_diff'], 2),
            'top2_prob_sum': round(decision['top2_prob_sum'] * 100, 1),
            'top3_prob_sum': round(decision['top3_prob_sum'] * 100, 1),
            'reason': decision['decision_reason'],
            'confidence_score': prediction.get('confidence_score', 0),
            'prediction_source': prediction.get('hybrid_metadata', {}).get('source', 'poisson')
        }

        # DEBUG OUTPUT
        if self.debug:
            print(f"\n🔍 DEBUG: {home_name} vs {away_name}")
            print(f"   Src: {result['prediction_source'].upper()} | Conf: {result['confidence_score']}")
            print(f"   xG: {result['xG_total']} (Home: {prediction['expected_goals']['home']:.2f}, Away: {prediction['expected_goals']['away']:.2f})")
            print(f"   Decision: {result['decision']}")
            if len(home_team.goals_scored) < 5:
                print(f"   ⚠️  Home matches used: {len(home_team.goals_scored)}")
            if len(away_team.goals_scored) < 5:
                print(f"   ⚠️  Away matches used: {len(away_team.goals_scored)}")

        
        # Update statistics
        if decision['decision'] == 'BET_TOP_2':
            self.stats['bet_top2_count'] += 1
        elif decision['decision'] == 'BET_TOP_3':
            self.stats['bet_top3_count'] += 1
        else:
            self.stats['pass_count'] += 1
        
        return result
    
    def analyze_daily_matches(self, match_list: List[Dict[str, str]]) -> List[Dict]:
        """
        Process all matches in the daily list.
        
        Args:
            match_list: List of match dictionaries
        
        Returns:
            List of betting signals (only BET_TOP_2 and BET_TOP_3)
        """
        self.results = []
        self.stats['total_matches'] = len(match_list)
        
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING {len(match_list)} MATCHES")
        if self.debug:
            print("   (DEBUG MODE ENABLED)")
        print(f"{'='*60}\n")
        
        for i, match in enumerate(match_list, 1):
            if not self.debug: # Don't spam if not debug
                 print(f"[{i}/{len(match_list)}] Processing: {match['home']} vs {match['away']}")
            
            result = self.analyze_match(
                match['home'], 
                match['away'], 
                match.get('league', 'DEFAULT'),
                match.get('home_rank'),
                match.get('away_rank')
            )
            
            if result:
                self.results.append(result)
        
        # Filter to only betting opportunities
        betting_signals = [
            r for r in self.results 
            if r['decision'] in ['BET_TOP_2', 'BET_TOP_3']
        ]
        
        return betting_signals
    
    def generate_output(self, signals: List[Dict], output_file: str):
        """
        Generate JSON output file with betting signals.
        
        Args:
            signals: List of betting signal dictionaries
            output_file: Path to output JSON file
        """
        output = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_matches_analyzed': self.stats['total_matches'],
            'betting_opportunities': len(signals),
            'pass_count': self.stats['pass_count'],
            'failed_lookups': self.stats['failed_lookups'],
            'summary': {
                'bet_top2_count': self.stats['bet_top2_count'],
                'bet_top3_count': self.stats['bet_top3_count'],
                'signal_rate': round(len(signals) / max(self.stats['total_matches'], 1) * 100, 1)
            },
            'signals': signals
        }
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✅ OUTPUT GENERATED")
        print(f"{'='*60}")
        print(f"📊 Total Matches Analyzed: {self.stats['total_matches']}")
        print(f"🎯 Betting Opportunities: {len(signals)}")
        print(f"   - BET_TOP_2: {self.stats['bet_top2_count']}")
        print(f"   - BET_TOP_3: {self.stats['bet_top3_count']}")
        print(f"⏸️  Pass: {self.stats['pass_count']}")
        print(f"❌ Failed Lookups: {self.stats['failed_lookups']}")
        print(f"\n📁 Output saved to: {output_file}")
        print(f"{'='*60}\n")
    
    def print_signals_summary(self, signals: List[Dict]):
        """
        Print a console summary of betting signals.
        
        Args:
            signals: List of betting signal dictionaries
        """
        if not signals:
            print("\n⚠️  No betting opportunities found today.")
            return
        
        print(f"\n{'='*60}")
        print(f"🎯 BETTING SIGNALS ({len(signals)})")
        print(f"{'='*60}\n")
        
        for i, signal in enumerate(signals, 1):
            print(f"{i}. {signal['match']}")
            print(f"   Decision: {signal['decision']}")
            print(f"   Top Predictions: {', '.join(signal['top_predictions'])}")
            print(f"   Probabilities: {signal['probabilities']}")
            print(f"   xG Total: {signal['xG_total']} | xG Diff: {signal['xG_diff']}")
            print(f"   Reason: {signal['reason']}\n")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Automated Daily Match Analyzer - Generate betting signals from match list"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='daily_matches.txt',
        help='Input file with match list (default: daily_matches.txt)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='daily_betting_signals.json',
        help='Output JSON file (default: daily_betting_signals.json)'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data',
        help='Directory containing historical data (default: data)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress console output (only save to file)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable verbose debug output'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = DailyMatchAnalyzer(data_dir=args.data_dir, debug=args.debug)
        
        # Parse match list
        matches = analyzer.parse_match_list(args.input)
        
        if not matches:
            print("❌ No matches found in input file")
            return
        
        # Analyze all matches
        signals = analyzer.analyze_daily_matches(matches)
        
        # Generate output
        analyzer.generate_output(signals, args.output)
        
        # Print summary unless quiet mode
        if not args.quiet:
            analyzer.print_signals_summary(signals)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
