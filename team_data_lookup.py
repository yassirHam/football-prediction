"""
Team Data Lookup Module
=======================
Queries local historical match data to extract team statistics.

This module:
- Loads historical data from local CSV files (no scraping)
- Finds last N matches for any team
- Extracts statistics in Team dataclass format
- Handles team name variations and fuzzy matching

Author: Football Analytics System
Version: 1.0
"""

import pandas as pd
import os
from typing import List, Dict, Optional, Tuple
from football_predictor import Team
from datetime import datetime
from difflib import get_close_matches


class TeamDataLookup:
    """Handles all historical data lookups from local files."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the lookup engine.
        
        Args:
            data_dir: Directory containing historical CSV files
        """
        self.data_dir = data_dir
        self.historical_data = None
        self.team_name_variants = {}  # Cache for name variations
        
    def load_historical_data(self) -> pd.DataFrame:
        """
        Load all historical match data from local CSV files.
        
        Returns:
            DataFrame with all historical matches
        """
        if self.historical_data is not None:
            return self.historical_data
        
        # Load combined training data
        combined_file = os.path.join(self.data_dir, "combined_training_data.csv")
        
        if os.path.exists(combined_file):
            print(f"📂 Loading historical data from {combined_file}...")
            df = pd.read_csv(combined_file)
            
            # Convert Date to datetime for sorting
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Sort by date (most recent last for easier slicing)
            df = df.sort_values('Date', ascending=True)
            
            # Cache the data
            self.historical_data = df
            print(f"✅ Loaded {len(df):,} historical matches")
            return df
        else:
            raise FileNotFoundError(f"Could not find {combined_file}")
    
    def find_team_name(self, team_name: str, all_teams: List[str], threshold: float = 0.8) -> Optional[str]:
        """
        Find the best match for a team name using fuzzy matching.
        
        Args:
            team_name: Team name to search for
            all_teams: List of all team names in the dataset
            threshold: Similarity threshold (0-1)
        
        Returns:
            Best matching team name or None
        """
        # Check cache first
        if team_name in self.team_name_variants:
            return self.team_name_variants[team_name]
        
        # Exact match (case-insensitive)
        exact_matches = [t for t in all_teams if t.lower() == team_name.lower()]
        if exact_matches:
            self.team_name_variants[team_name] = exact_matches[0]
            return exact_matches[0]
        
        # Fuzzy match
        close_matches = get_close_matches(team_name, all_teams, n=1, cutoff=threshold)
        if close_matches:
            matched_name = close_matches[0]
            self.team_name_variants[team_name] = matched_name
            print(f"🔍 Matched '{team_name}' → '{matched_name}'")
            return matched_name
        
        return None
    
    def find_team_last_n_matches(self, team_name: str, n: int = 5, 
                                  before_date: Optional[str] = None,
                                  venue: Optional[str] = None) -> List[Dict]:
        """
        Find the last N matches for a team, optionally filtering by venue.
        
        Args:
            team_name: Name of the team
            n: Number of recent matches to retrieve
            before_date: Optional date limit (YYYY-MM-DD)
            venue: 'home' to get only home matches, 'away' for away matches, None for mixed
        
        Returns:
            List of match dictionaries with stats
        """
        df = self.load_historical_data()
        
        # Get all unique team names
        all_teams = set(df['HomeTeam'].dropna().unique()) | set(df['AwayTeam'].dropna().unique())
        
        # Find best matching team name
        matched_name = self.find_team_name(team_name, list(all_teams))
        
        if not matched_name:
            print(f"⚠️  Warning: Could not find team '{team_name}' in historical data")
            return []
        
        # Filter matches where team played
        if venue == 'home':
            team_matches = df[df['HomeTeam'] == matched_name].copy()
        elif venue == 'away':
            team_matches = df[df['AwayTeam'] == matched_name].copy()
        else:
            team_matches = df[
                (df['HomeTeam'] == matched_name) | (df['AwayTeam'] == matched_name)
            ].copy()
        
        # Apply date filter if provided
        if before_date:
            team_matches = team_matches[team_matches['Date'] < pd.to_datetime(before_date)]
        
        # Sort by date (most recent first) for iteration
        team_matches = team_matches.sort_values('Date', ascending=False)
        
        # Convert to list of dictionaries, ONLY including matches with valid HT data
        matches = []
        for _, row in team_matches.iterrows():
            # Skip matches without half-time data
            if pd.isna(row.get('HTHG')) or pd.isna(row.get('HTAG')):
                continue
            
            is_home = row['HomeTeam'] == matched_name
            
            matches.append({
                'date': row['Date'],
                'is_home': is_home,
                'opponent': row['AwayTeam'] if is_home else row['HomeTeam'],
                'goals_scored': row['FTHG'] if is_home else row['FTAG'],
                'goals_conceded': row['FTAG'] if is_home else row['FTHG'],
                'first_half_goals': row['HTHG'] if is_home else row['HTAG'],
                'competition': row.get('Competition', 'Unknown')
            })
            
            # Stop when we have enough matches with valid HT data
            if len(matches) >= n:
                break
        
        # Fallback logic: If explicitly filtering by venue but didn't find enough matches,
        # try to fill the rest with mixed venue data
        if venue and len(matches) < n:
            print(f"⚠️  Insufficient {venue} matches for {team_name} ({len(matches)}/{n}). looking for mixed venue as fallback.")
            
            # Get mixed matches excluding the ones we already found (by date)
            existing_dates = {m['date'] for m in matches}
            
            mixed_matches = df[
                (df['HomeTeam'] == matched_name) | (df['AwayTeam'] == matched_name)
            ].copy()
            
            if before_date:
                mixed_matches = mixed_matches[mixed_matches['Date'] < pd.to_datetime(before_date)]
            
            mixed_matches = mixed_matches.sort_values('Date', ascending=False)
            
            for _, row in mixed_matches.iterrows():
                # Skip if already added
                if row['Date'] in existing_dates:
                    continue
                
                # Skip invalid HT data
                if pd.isna(row.get('HTHG')) or pd.isna(row.get('HTAG')):
                    continue
                
                is_home = row['HomeTeam'] == matched_name
                
                matches.append({
                    'date': row['Date'],
                    'is_home': is_home,
                    'opponent': row['AwayTeam'] if is_home else row['HomeTeam'],
                    'goals_scored': row['FTHG'] if is_home else row['FTAG'],
                    'goals_conceded': row['FTAG'] if is_home else row['FTHG'],
                    'first_half_goals': row['HTHG'] if is_home else row['HTAG'],
                    'competition': row.get('Competition', 'Unknown')
                })
                
                if len(matches) >= n:
                    break
        
        if len(matches) < n:
            print(f"⚠️  Warning: Only found {len(matches)} matches with HT data for '{team_name}' (requested {n})")
        
        return matches
    
    def infer_league_code(self, team_name: str) -> str:
        """
        Infer league code from team's most recent history.
        
        Args:
            team_name: Name of team to lookup
            
        Returns:
            League code (e.g., 'E0') or 'DEFAULT'
        """
        matches = self.find_team_last_n_matches(team_name, n=1)
        if matches and matches[0].get('competition'):
            return matches[0]['competition']
        return 'DEFAULT'
    def extract_team_stats(self, matches: List[Dict], team_name: str, 
                           league: str = "DEFAULT", rank: Optional[int] = None) -> Optional[Team]:
        """
        Convert match history to Team dataclass format.
        
        Args:
            matches: List of match dictionaries from find_team_last_n_matches
            team_name: Name of the team
            league: League code for league-specific parameters
            rank: Optional league rank override
        
        Returns:
            Team object or None if insufficient data
        """
        if not matches or len(matches) < 3:
            print(f"⚠️  Insufficient data for {team_name}: only {len(matches)} matches found")
            return None
        
        # Extract statistics (already in chronological order, most recent first)
        # Handle NaN values by converting to 0
        goals_scored = []
        goals_conceded = []
        first_half_goals = []
        
        for m in matches:
            # Safe conversion with NaN handling
            try:
                gs = int(m['goals_scored']) if pd.notna(m['goals_scored']) else 0
                gc = int(m['goals_conceded']) if pd.notna(m['goals_conceded']) else 0
                fh = int(m['first_half_goals']) if pd.notna(m['first_half_goals']) else 0
            except (ValueError, TypeError):
                gs, gc, fh = 0, 0, 0
            
            goals_scored.append(gs)
            goals_conceded.append(gc)
            first_half_goals.append(fh)
        
        # Ensure we have at least 5 values (pad with zeros if needed)
        while len(goals_scored) < 5:
            goals_scored.append(0)
            goals_conceded.append(0)
            first_half_goals.append(0)
        
        # Create Team object
        return Team(
            name=team_name,
            goals_scored=goals_scored[:5],  # Take only last 5
            goals_conceded=goals_conceded[:5],
            first_half_goals=first_half_goals[:5],
            league=league,
            league_position=rank if rank is not None else 10  # Use provided rank or default mid-table
        )
    
    def get_team_data(self, team_name: str, n: int = 5, 
                     league: str = "DEFAULT", venue: Optional[str] = None,
                     rank: Optional[int] = None) -> Optional[Team]:
        """
        Convenience method to get Team object directly.
        
        Args:
            team_name: Name of the team
            n: Number of recent matches to use
            n: Number of recent matches to use
            league: League code
            venue: 'home', 'away', or None
            rank: Optional league rank override
        
        Returns:
            Team object or None
        """
        matches = self.find_team_last_n_matches(team_name, n, venue=venue)
        return self.extract_team_stats(matches, team_name, league, rank=rank)


# Singleton instance for module-level access
_lookup_instance = None

def get_lookup_instance(data_dir: str = "data") -> TeamDataLookup:
    """Get or create the singleton lookup instance."""
    global _lookup_instance
    if _lookup_instance is None:
        _lookup_instance = TeamDataLookup(data_dir)
    return _lookup_instance


# Convenience functions for direct access
def find_team_last_n_matches(team_name: str, n: int = 5) -> List[Dict]:
    """Find last N matches for a team."""
    lookup = get_lookup_instance()
    return lookup.find_team_last_n_matches(team_name, n)


def extract_team_stats(matches: List[Dict], team_name: str, league: str = "DEFAULT") -> Optional[Team]:
    """Extract team statistics from match history."""
    lookup = get_lookup_instance()
    return lookup.extract_team_stats(matches, team_name, league)


def get_team_data(team_name: str, n: int = 5, league: str = "DEFAULT") -> Optional[Team]:
    """Get Team object directly."""
    lookup = get_lookup_instance()
    return lookup.get_team_data(team_name, n, league)


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Team Data Lookup - Example Usage")
    print("="*60 + "\n")
    
    # Initialize lookup
    lookup = TeamDataLookup()
    
    # Example 1: Find last 5 matches for Arsenal
    print("Example 1: Arsenal's Last 5 Matches")
    print("-" * 40)
    matches = lookup.find_team_last_n_matches("Arsenal", n=5)
    for i, match in enumerate(matches, 1):
        loc = "Home" if match['is_home'] else "Away"
        print(f"{i}. {loc} vs {match['opponent']}: {match['goals_scored']}-{match['goals_conceded']} "
              f"(HT: {match['first_half_goals']})")
    
    # Example 2: Get Team object
    print("\n\nExample 2: Create Team Object")
    print("-" * 40)
    team = lookup.get_team_data("Arsenal", n=5, league="E0")
    if team:
        print(f"Team: {team.name}")
        print(f"Goals Scored: {team.goals_scored}")
        print(f"Goals Conceded: {team.goals_conceded}")
        print(f"First Half Goals: {team.first_half_goals}")
    
    print("\n✅ Team data lookup working correctly!")
