"""
Data processing utilities for football prediction model.
Handles data standardization, filtering, and merging from multiple sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and standardize football data from multiple sources."""
    
    # Standard column names for unified dataset
    STANDARD_COLUMNS = {
        'date': 'Date',
        'home_team': 'HomeTeam',
        'away_team': 'AwayTeam',
        'home_score': 'FTHG',
        'away_score': 'FTAG',
        'tournament': 'Competition',
        'city': 'City',
        'country': 'Country',
        'neutral': 'Neutral'
    }
    
    # Competition priorities for weighting
    COMPETITION_WEIGHTS = {
        'FIFA World Cup': 1.2,
        'UEFA Euro': 1.1,
        'Copa América': 1.1,
        'UEFA Nations League': 0.9,
        'Friendly': 0.6,
        'FIFA World Cup qualification': 0.8,
        'UEFA Euro qualification': 0.8,
        # Cup competitions
        'UEFA Champions League': 1.15,
        'EC': 1.15,  # Champions League code from football-data.co.uk
        'UEFA Europa League': 1.0,
        'FA Cup': 0.95,
        'Copa del Rey': 0.95,
        'DFB-Pokal': 0.95,
        'Coppa Italia': 0.95,
        # Domestic leagues
        'E0': 1.0,  # Premier League
        'SP1': 1.0,  # La Liga
        'D1': 1.0,  # Bundesliga
        'I1': 1.0,  # Serie A
        'F1': 1.0,  # Ligue 1
        'P1': 1.0,  # Portugal Primeira Liga
        'B1': 0.95,  # Belgium Pro League
        'T1': 0.95,  # Turkey Süper Lig
    }
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        
    def load_international_matches(self, min_year: int = 2000) -> pd.DataFrame:
        """
        Load and standardize international match data.
        
        Args:
            min_year: Minimum year to include (default: 2000)
            
        Returns:
            Standardized DataFrame
        """
        logger.info("Loading international matches...")
        
        file_path = self.data_dir / 'international' / 'results.csv'
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        # Load data
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} international matches")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by year
        df = df[df['date'].dt.year >= min_year].copy()
        logger.info(f"Filtered to {len(df)} matches since {min_year}")
        
        # Rename columns to match standard format
        df = df.rename(columns=self.STANDARD_COLUMNS)
        
        # Add competition weight
        df['Weight'] = df['Competition'].map(self.COMPETITION_WEIGHTS).fillna(0.7)
        
        # Add source identifier
        df['Source'] = 'International'
        
        # Calculate derived features
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        df['GoalDiff'] = df['FTHG'] - df['FTAG']
        df['Result'] = df.apply(
            lambda row: 'H' if row['FTHG'] > row['FTAG'] 
            else ('A' if row['FTHG'] < row['FTAG'] else 'D'), 
            axis=1
        )
        
        logger.info(f"Processed {len(df)} international matches")
        return df
    
    def load_domestic_league(self, league_code: str) -> pd.DataFrame:
        """
        Load domestic league data.
        
        Args:
            league_code: League identifier (e.g., 'E0', 'SP1')
            
        Returns:
            Standardized DataFrame
        """
        logger.info(f"Loading league: {league_code}")
        
        file_path = self.data_dir / f'{league_code}.csv'
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in {league_code}: {missing_cols}")
                return pd.DataFrame()
            
            # Convert date to datetime
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            
            # Remove rows with invalid dates
            df = df.dropna(subset=['Date'])
            
            # Add competition info
            df['Competition'] = league_code
            df['Weight'] = self.COMPETITION_WEIGHTS.get(league_code, 1.0)
            df['Source'] = 'Domestic'
            
            # Calculate derived features if not present
            if 'TotalGoals' not in df.columns:
                df['TotalGoals'] = df['FTHG'] + df['FTAG']
            
            if 'GoalDiff' not in df.columns:
                df['GoalDiff'] = df['FTHG'] - df['FTAG']
            
            if 'Result' not in df.columns and 'FTR' in df.columns:
                df['Result'] = df['FTR']
            elif 'Result' not in df.columns:
                df['Result'] = df.apply(
                    lambda row: 'H' if row['FTHG'] > row['FTAG'] 
                    else ('A' if row['FTHG'] < row['FTAG'] else 'D'), 
                    axis=1
                )
            
            logger.info(f"Loaded {len(df)} matches from {league_code}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {league_code}: {str(e)}")
            return pd.DataFrame()
    
    def load_all_domestic_leagues(self) -> pd.DataFrame:
        """
        Load all domestic league files from data directory.
        
        Returns:
            Combined DataFrame
        """
        logger.info("Loading all domestic leagues...")
        
        # Find all league CSV files (excluding subdirectories and 2425 season files for now)
        league_files = [
            f.stem for f in self.data_dir.glob('*.csv') 
            if f.stem not in ['learned_matches'] and '2425' not in f.stem
        ]
        
        dfs = []
        for league_code in league_files:
            df = self.load_domestic_league(league_code)
            if not df.empty:
                dfs.append(df)
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(combined)} total domestic matches from {len(dfs)} leagues")
            return combined
        else:
            logger.warning("No domestic league data loaded")
            return pd.DataFrame()
    
    def load_cup_competition(self, cup_name: str, file_pattern: str = None) -> pd.DataFrame:
        """
        Load cup competition data (Champions League, FA Cup, etc).
        
        Args:
            cup_name: Name of the cup competition
            file_pattern: Optional file pattern to match (e.g., 'champions_league_*.csv')
            
        Returns:
            Standardized DataFrame
        """
        logger.info(f"Loading cup competition: {cup_name}")
        
        cups_dir = self.data_dir / 'cups'
        if not cups_dir.exists():
            logger.warning(f"Cups directory not found: {cups_dir}")
            return pd.DataFrame()
        
        # Find cup files
        if file_pattern:
            cup_files = list(cups_dir.glob(file_pattern))
        else:
            cup_files = list(cups_dir.glob('*.csv'))
        
        if not cup_files:
            logger.warning(f"No cup files found for pattern: {file_pattern}")
            return pd.DataFrame()
        
        dfs = []
        for cup_file in cup_files:
            try:
                df = pd.read_csv(cup_file)
                
                # Handle different formats
                if 'Season' in df.columns and 'home' in df.columns:
                    # FA Cup format from jalapic/engsoccerdata
                    df = self._process_facup_format(df)
                else:
                    # football-data.co.uk format
                    df = self._process_standard_format(df, cup_name)
                
                if not df.empty:
                    dfs.append(df)
                    logger.info(f"Loaded {len(df)} matches from {cup_file.name}")
                    
            except Exception as e:
                logger.error(f"Error loading {cup_file}: {str(e)}")
                continue
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(combined)} total cup matches for {cup_name}")
            return combined
        else:
            return pd.DataFrame()
    
    def _process_facup_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process FA Cup data from jalapic/engsoccerdata format."""
        try:
            # Rename columns
            column_map = {
                'Season': 'Season',
                'Date': 'Date',
                'home': 'HomeTeam',
                'visitor': 'AwayTeam',
                'hgoal': 'FTHG',
                'vgoal': 'FTAG'
            }
            
            df = df.rename(columns=column_map)
            
            # Keep only required columns
            required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logger.error(f"Missing columns in FA Cup data: {missing}")
                return pd.DataFrame()
            
            df = df[required_cols].copy()
            
            # Convert date
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            
            # Add metadata
            df['Competition'] = 'FA Cup'
            df['Weight'] = self.COMPETITION_WEIGHTS.get('FA Cup', 0.95)
            df['Source'] = 'Cup'
            
            # Calculate derived features
            df['TotalGoals'] = df['FTHG'] + df['FTAG']
            df['GoalDiff'] = df['FTHG'] - df['FTAG']
            df['Result'] = df.apply(
                lambda row: 'H' if row['FTHG'] > row['FTAG'] 
                else ('A' if row['FTHG'] < row['FTAG'] else 'D'), 
                axis=1
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing FA Cup format: {str(e)}")
            return pd.DataFrame()
    
    def _process_standard_format(self, df: pd.DataFrame, comp_name: str) -> pd.DataFrame:
        """Process cup data in football-data.co.uk format."""
        try:
            # Check for required columns
            required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logger.error(f"Missing columns in {comp_name}: {missing}")
                return pd.DataFrame()
            
            # Convert date
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            df = df.dropna(subset=['Date'])
            
            # Add metadata (use 'Div' column if present, otherwise use comp_name)
            if 'Div' in df.columns:
                df['Competition'] = df['Div']
            else:
                df['Competition'] = comp_name
            
            df['Weight'] = df['Competition'].map(self.COMPETITION_WEIGHTS).fillna(1.0)
            df['Source'] = 'Cup'
            
            # Calculate derived features
            if 'TotalGoals' not in df.columns:
                df['TotalGoals'] = df['FTHG'] + df['FTAG']
            
            if 'GoalDiff' not in df.columns:
                df['GoalDiff'] = df['FTHG'] - df['FTAG']
            
            if 'Result' not in df.columns and 'FTR' in df.columns:
                df['Result'] = df['FTR']
            elif 'Result' not in df.columns:
                df['Result'] = df.apply(
                    lambda row: 'H' if row['FTHG'] > row['FTAG'] 
                    else ('A' if row['FTHG'] < row['FTAG'] else 'D'), 
                    axis=1
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing {comp_name} format: {str(e)}")
            return pd.DataFrame()
    
    def load_all_cup_competitions(self) -> pd.DataFrame:
        """
        Load all available cup competition data.
        
        Returns:
            Combined DataFrame
        """
        logger.info("Loading all cup competitions...")
        
        dfs = []
        
        # Load Champions League (multiple seasons)
        cl_df = self.load_cup_competition('UEFA Champions League', 'champions_league_*.csv')
        if not cl_df.empty:
            dfs.append(cl_df)
        
        # Load FA Cup
        fa_df = self.load_cup_competition('FA Cup', 'fa_cup.csv')
        if not fa_df.empty:
            dfs.append(fa_df)
        
        # Load Europa League if available
        el_df = self.load_cup_competition('UEFA Europa League', 'europa_league_*.csv')
        if not el_df.empty:
            dfs.append(el_df)
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(combined)} total cup competition matches")
            return combined
        else:
            logger.warning("No cup competition data loaded")
            return pd.DataFrame()
    
    def merge_datasets(self, 
                      include_international: bool = True,
                      include_domestic: bool = True,
                      include_cups: bool = True,
                      min_year: int = 2000) -> pd.DataFrame:
        """
        Merge all data sources into a unified dataset.
        
        Args:
            include_international: Include international matches
            include_domestic: Include domestic leagues
            min_year: Minimum year to include
            
        Returns:
            Unified DataFrame
        """
        logger.info("Merging datasets...")
        
        dfs = []
        
        if include_domestic:
            domestic_df = self.load_all_domestic_leagues()
            if not domestic_df.empty:
                # Filter by year
                domestic_df = domestic_df[domestic_df['Date'].dt.year >= min_year]
                dfs.append(domestic_df)
        
        if include_international:
            international_df = self.load_international_matches(min_year=min_year)
            if not international_df.empty:
                dfs.append(international_df)
        
        if include_cups:
            cups_df = self.load_all_cup_competitions()
            if not cups_df.empty:
                # Filter by year
                cups_df = cups_df[cups_df['Date'].dt.year >= min_year]
                dfs.append(cups_df)
        
        if not dfs:
            logger.error("No data loaded!")
            return pd.DataFrame()
        
        # Combine all datasets
        combined = pd.concat(dfs, ignore_index=True, sort=False)
        
        # Sort by date
        combined = combined.sort_values('Date').reset_index(drop=True)
        
        # Remove duplicates (same teams, same date, same score)
        before_dedup = len(combined)
        combined = combined.drop_duplicates(
            subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'],
            keep='first'
        )
        after_dedup = len(combined)
        
        if before_dedup > after_dedup:
            logger.info(f"Removed {before_dedup - after_dedup} duplicate matches")
        
        logger.info(f"Final dataset: {len(combined)} matches")
        logger.info(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")
        logger.info(f"Sources: {combined['Source'].value_counts().to_dict()}")
        
        return combined
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating data...")
        
        validation = {
            'total_rows': len(df),
            'date_range': (df['Date'].min(), df['Date'].max()),
            'missing_values': df.isnull().sum().to_dict(),
            'invalid_scores': len(df[(df['FTHG'] < 0) | (df['FTAG'] < 0)]),
            'sources': df['Source'].value_counts().to_dict(),
            'competitions': df['Competition'].nunique()
        }
        
        logger.info(f"Validation results: {validation}")
        return validation
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = 'combined_training_data.csv'):
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = self.data_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
    
    def filter_by_competitions(self, df: pd.DataFrame, competitions: List[str]) -> pd.DataFrame:
        """
        Filter dataset by specific competitions.
        
        Args:
            df: Input DataFrame
            competitions: List of competition names
            
        Returns:
            Filtered DataFrame
        """
        filtered = df[df['Competition'].isin(competitions)].copy()
        logger.info(f"Filtered to {len(filtered)} matches from {len(competitions)} competitions")
        return filtered
    
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive statistics about the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_matches': len(df),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
            'sources': df['Source'].value_counts().to_dict(),
            'avg_goals_per_match': df['TotalGoals'].mean(),
            'home_win_pct': (df['Result'] == 'H').mean() * 100,
            'draw_pct': (df['Result'] == 'D').mean() * 100,
            'away_win_pct': (df['Result'] == 'A').mean() * 100,
            'competitions': df['Competition'].nunique(),
            'top_competitions': df['Competition'].value_counts().head(10).to_dict()
        }
        
        return stats


if __name__ == '__main__':
    # Example usage
    processor = DataProcessor()
    
    # Load and merge all data
    combined_df = processor.merge_datasets(
        include_international=True,
        include_domestic=True,
        include_cups=True,  # Include cup competitions
        min_year=2010  # Last 15 years
    )
    
    # Validate
    validation = processor.validate_data(combined_df)
    
    # Get statistics
    stats = processor.get_dataset_stats(combined_df)
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save processed data
    processor.save_processed_data(combined_df)
    
    print("\n✓ Data processing complete!")
