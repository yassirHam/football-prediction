"""
Football-Data.org CSV Updater
==============================
Updates local match database using official Football-Data.org API.

This script:
- Fetches recent completed matches via official API
- Extracts half-time and full-time scores
- Appends only NEW matches to combined_training_data.csv
- Preserves chronological order

NO SCRAPING - Uses only official API with registered token.

Author: Football Analytics System
Version: 1.0
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple
import time


class FootballDataUpdater:
    """Handles CSV updates using Football-Data.org API."""
    
    def __init__(self, csv_path: str = "data/combined_training_data.csv", 
                 api_token: str = None):
        """
        Initialize the updater.
        
        Args:
            csv_path: Path to combined training data CSV
            api_token: Football-Data.org API token (or use env var)
        """
        self.csv_path = csv_path
        self.api_token = api_token or os.getenv("FOOTBALL_DATA_TOKEN")
        
        if not self.api_token:
            raise ValueError(
                "API token required. Set FOOTBALL_DATA_TOKEN environment variable "
                "or pass api_token parameter."
            )
        
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {"X-Auth-Token": self.api_token}
        
        # Statistics
        self.stats = {
            'fetched': 0,
            'new_added': 0,
            'duplicates_skipped': 0,
            'missing_ht_skipped': 0
        }
    
    def load_existing_csv(self) -> Tuple[pd.DataFrame, Set[str]]:
        """
        Load existing CSV and extract unique match keys.
        
        Returns:
            Tuple of (DataFrame, Set of match keys)
        """
        if not os.path.exists(self.csv_path):
            print(f"⚠️  CSV not found at {self.csv_path}, will create new file")
            return pd.DataFrame(), set()
        
        df = pd.read_csv(self.csv_path)
        
        # Create unique keys: Date + HomeTeam + AwayTeam
        existing_keys = set()
        for _, row in df.iterrows():
            key = self._create_match_key(
                row.get('Date', ''),
                row.get('HomeTeam', ''),
                row.get('AwayTeam', '')
            )
            existing_keys.add(key)
        
        print(f"📂 Loaded existing CSV: {len(df)} matches")
        return df, existing_keys
    
    def _create_match_key(self, date: str, home: str, away: str) -> str:
        """Create unique match identifier."""
        return f"{date}|{home}|{away}"
    
    def fetch_recent_matches(self, days_back: int = 10) -> List[Dict]:
        """
        Fetch recent matches from Football-Data.org API.
        
        Args:
            days_back: Number of days to look back
        
        Returns:
            List of match dictionaries
        """
        # Calculate date range
        date_to = datetime.now().strftime('%Y-%m-%d')
        date_from = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        print(f"📡 Fetching matches from {date_from} to {date_to}...")
        
        # Fetch matches
        url = f"{self.base_url}/matches"
        params = {
            'dateFrom': date_from,
            'dateTo': date_to,
            'status': 'FINISHED'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            matches = data.get('matches', [])
            self.stats['fetched'] = len(matches)
            print(f"✅ Fetched {len(matches)} finished matches")
            
            return matches
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error: {e}")
            return []
    
    def normalize_match(self, api_match: Dict) -> Dict:
        """
        Convert API match format to CSV row format.
        
        Args:
            api_match: Match data from API
        
        Returns:
            Dictionary with CSV columns or None if invalid
        """
        try:
            # Extract score data
            score = api_match.get('score', {})
            half_time = score.get('halfTime', {})
            full_time = score.get('fullTime', {})
            
            # Require valid half-time scores
            if half_time.get('home') is None or half_time.get('away') is None:
                return None
            
            # Require valid full-time scores
            if full_time.get('home') is None or full_time.get('away') is None:
                return None
            
            # Extract team names
            home_team = api_match.get('homeTeam', {}).get('name', '')
            away_team = api_match.get('awayTeam', {}).get('name', '')
            
            # Convert UTC date to YYYY-MM-DD
            utc_date = api_match.get('utcDate', '')
            match_date = datetime.fromisoformat(utc_date.replace('Z', '+00:00')).strftime('%Y-%m-%d')
            
            # Get competition info
            competition = api_match.get('competition', {}).get('name', 'Unknown')
            
            # Build CSV row
            return {
                'Date': match_date,
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'FTHG': int(full_time['home']),
                'FTAG': int(full_time['away']),
                'HTHG': int(half_time['home']),
                'HTAG': int(half_time['away']),
                'Competition': competition,
                'Source': 'FootballData.org'
            }
            
        except (KeyError, ValueError, TypeError) as e:
            return None
    
    def append_new_matches(self, existing_df: pd.DataFrame, 
                          existing_keys: Set[str],
                          api_matches: List[Dict]) -> pd.DataFrame:
        """
        Append only new matches to DataFrame.
        
        Args:
            existing_df: Current DataFrame
            existing_keys: Set of existing match keys
            api_matches: Matches from API
        
        Returns:
            Updated DataFrame
        """
        new_rows = []
        
        for api_match in api_matches:
            # Normalize to CSV format
            csv_row = self.normalize_match(api_match)
            
            if not csv_row:
                self.stats['missing_ht_skipped'] += 1
                continue
            
            # Check if already exists
            match_key = self._create_match_key(
                csv_row['Date'],
                csv_row['HomeTeam'],
                csv_row['AwayTeam']
            )
            
            if match_key in existing_keys:
                self.stats['duplicates_skipped'] += 1
                continue
            
            # New match - add it
            new_rows.append(csv_row)
            existing_keys.add(match_key)
            self.stats['new_added'] += 1
        
        # Append new rows
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = existing_df
        
        # Sort by date
        if 'Date' in combined_df.columns:
            combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
            combined_df = combined_df.sort_values('Date', ascending=True)
            combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
        
        return combined_df
    
    def save_csv(self, df: pd.DataFrame):
        """
        Save updated DataFrame to CSV.
        
        Args:
            df: DataFrame to save
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(self.csv_path, index=False)
        print(f"💾 CSV updated successfully: {len(df)} total matches")
    
    def print_summary(self):
        """Print update summary statistics."""
        print(f"\n{'='*60}")
        print(f"📊 UPDATE SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Matches fetched: {self.stats['fetched']}")
        print(f"➕ New matches added: {self.stats['new_added']}")
        print(f"⏭️  Skipped (duplicates): {self.stats['duplicates_skipped']}")
        print(f"⚠️  Skipped (missing HT data): {self.stats['missing_ht_skipped']}")
        print(f"{'='*60}\n")
    
    def update(self, days_back: int = 10):
        """
        Main update workflow.
        
        Args:
            days_back: Number of days to fetch
        """
        print(f"\n{'='*60}")
        print(f"🔄 FOOTBALL-DATA.ORG CSV UPDATER")
        print(f"{'='*60}\n")
        
        # Step 1: Load existing CSV
        existing_df, existing_keys = self.load_existing_csv()
        
        # Step 2: Fetch recent matches
        api_matches = self.fetch_recent_matches(days_back)
        
        if not api_matches:
            print("⚠️  No matches fetched from API")
            return
        
        # Step 3 & 4: Normalize and append new matches
        updated_df = self.append_new_matches(existing_df, existing_keys, api_matches)
        
        # Step 5: Save and report
        self.save_csv(updated_df)
        self.print_summary()


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Update match database using Football-Data.org API"
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='data/combined_training_data.csv',
        help='Path to CSV file (default: data/combined_training_data.csv)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=10,
        help='Number of days to look back (default: 10)'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='API token (or use FOOTBALL_DATA_TOKEN env var)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create updater
        updater = FootballDataUpdater(
            csv_path=args.csv,
            api_token=args.token
        )
        
        # Run update
        updater.update(days_back=args.days)
        
        print("✅ Update completed successfully!")
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nTo set API token:")
        print("  Windows: set FOOTBALL_DATA_TOKEN=your_token_here")
        print("  Linux/Mac: export FOOTBALL_DATA_TOKEN=your_token_here")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
