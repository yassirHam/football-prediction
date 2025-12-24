"""
Lineup-Aware xG Adjustments
============================

Rule-based post-prediction adjustments for lineup changes.
DOES NOT modify models, only adjusts final xG values transparently.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class LineupAdjuster:
    """
    Applies rule-based xG adjustments for lineup changes.
    
    Rules:
    - Missing key attacker: -0.15 xG
    - Missing key defender: +0.12 conceded xG  
    - Multiple absences: multiplier effect
    - Cap: ±0.3 xG per team
    
    All adjustments are transparent and reversible.
    """
    
    def __init__(self, rules_file: str = 'adjustments/rules.json'):
        """
        Initialize lineup adjuster.
        
        Args:
            rules_file: Path to rules configuration
        """
        rules_path = Path(rules_file)
        if rules_path.exists():
            with open(rules_path, 'r') as f:
                config = json.load(f)
                self.rules = config.get('adjustments', {})
        else:
            # Default rules
            self.rules = {
                'key_attacker_missing': -0.15,
                'key_defender_missing': 0.12,
                'goalkeeper_missing': 0.20,
                'multiple_absences_multiplier': 1.5,
                'max_adjustment_per_team': 0.3
            }
    
    def adjust_for_absences(self,
                           xg_home: float,
                           xg_away: float,
                           home_absences: List[str] = None,
                           away_absences: List[str] = None) -> Tuple[float, float, Dict]:
        """
        Adjust xG values for player absences.
        
        Args:
            xg_home: Original home xG
            xg_away: Original away xG
            home_absences: List of absent player types for home team
                          (e.g., ['key_attacker', 'defender'])
            away_absences: List of absent player types for away team
            
        Returns:
            Tuple of (adjusted_xg_home, adjusted_xg_away, details_dict)
        """
        home_absences = home_absences or []
        away_absences = away_absences or []
        
        # Track adjustments for transparency
        adjustments = {
            'home': {'original': xg_home, 'changes': [], 'final': xg_home},
            'away': {'original': xg_away, 'changes': [], 'final': xg_away}
        }
        
        # Apply home team adjustments
        home_adj = 0
        away_conceded_adj = 0  # Affects away team's conceded goals
        
        for absence in home_absences:
            if absence == 'key_attacker':
                adj_value = self.rules['key_attacker_missing']
                home_adj += adj_value
                adjustments['home']['changes'].append(f"Key attacker missing: {adj_value:+.2f}")
            
            elif absence == 'key_defender':
                adj_value = self.rules['key_defender_missing']
                away_conceded_adj += adj_value  # Away team scores more
                adjustments['home']['changes'].append(f"Key defender missing: away +{adj_value:.2f}")
            
            elif absence == 'goalkeeper':
                adj_value = self.rules['goalkeeper_missing']
                away_conceded_adj += adj_value
                adjustments['home']['changes'].append(f"Goalkeeper missing: away +{adj_value:.2f}")
        
        # Multiple absences multiplier
        if len(home_absences) > 1:
            multiplier = self.rules['multiple_absences_multiplier']
            home_adj *= multiplier
            away_conceded_adj *= multiplier
            adjustments['home']['changes'].append(f"Multiple absences multiplier: x{multiplier}")
        
        # Apply to xG values
        xg_home_adjusted = xg_home + home_adj
        xg_away_adjusted = xg_away + away_conceded_adj
        
        # Apply away team adjustments (symmetric logic)
        away_adj = 0
        home_conceded_adj = 0
        
        for absence in away_absences:
            if absence == 'key_attacker':
                adj_value = self.rules['key_attacker_missing']
                away_adj += adj_value
                adjustments['away']['changes'].append(f"Key attacker missing: {adj_value:+.2f}")
            
            elif absence == 'key_defender':
                adj_value = self.rules['key_defender_missing']
                home_conceded_adj += adj_value
                adjustments['away']['changes'].append(f"Key defender missing: home +{adj_value:.2f}")
            
            elif absence == 'goalkeeper':
                adj_value = self.rules['goalkeeper_missing']
                home_conceded_adj += adj_value
                adjustments['away']['changes'].append(f"Goalkeeper missing: home +{adj_value:.2f}")
        
        if len(away_absences) > 1:
            multiplier = self.rules['multiple_absences_multiplier']
            away_adj *= multiplier
            home_conceded_adj *= multiplier
            adjustments['away']['changes'].append(f"Multiple absences multiplier: x{multiplier}")
        
        xg_home_adjusted += home_conceded_adj
        xg_away_adjusted += away_adj
        
        # Apply caps
        max_adj = self.rules['max_adjustment_per_team']
        
        home_total_change = xg_home_adjusted - xg_home
        if abs(home_total_change) > max_adj:
            xg_home_adjusted = xg_home + (max_adj if home_total_change > 0 else -max_adj)
            adjustments['home']['changes'].append(f"Capped at ±{max_adj}")
        
        away_total_change = xg_away_adjusted - xg_away
        if abs(away_total_change) > max_adj:
            xg_away_adjusted = xg_away + (max_adj if away_total_change > 0 else -max_adj)
            adjustments['away']['changes'].append(f"Capped at ±{max_adj}")
        
        adjustments['home']['final'] = xg_home_adjusted
        adjustments['away']['final'] = xg_away_adjusted
        
        return xg_home_adjusted, xg_away_adjusted, adjustments
    
    def print_adjustments(self, adjustments: Dict):
        """
        Print adjustments in human-readable format.
        
        Args:
            adjustments: Adjustments dictionary from adjust_for_absences()
        """
        print(f"\n{'='*60}")
        print("LINEUP ADJUSTMENTS")
        print(f"{'='*60}")
        
        for team in ['home', 'away']:
            team_adj = adjustments[team]
            print(f"\n{team.upper()} Team:")
            print(f"  Original xG: {team_adj['original']:.2f}")
            
            if team_adj['changes']:
                print(f"  Changes:")
                for change in team_adj['changes']:
                    print(f"    • {change}")
                print(f"  Final xG: {team_adj['final']:.2f} ({team_adj['final'] - team_adj['original']:+.2f})")
            else:
                print(f"  No adjustments")
        
        print(f"\n{'='*60}\n")


# Example usage
if __name__ == '__main__':
    adjuster = LineupAdjuster()
    
    # Test: Home team missing key attacker and defender
    xg_h, xg_a, details = adjuster.adjust_for_absences(
        xg_home=1.8,
        xg_away=1.2,
        home_absences=['key_attacker', 'key_defender'],
        away_absences=[]
    )
    
    print(f"Adjusted xG: {xg_h:.2f} - {xg_a:.2f}")
    adjuster.print_adjustments(details)
    
    # Test: Both teams missing players
    xg_h2, xg_a2, details2 = adjuster.adjust_for_absences(
        xg_home=1.5,
        xg_away=1.4,
        home_absences=['goalkeeper'],
        away_absences=['key_attacker', 'key_defender']
    )
    
    print(f"\nTest 2 Adjusted xG: {xg_h2:.2f} - {xg_a2:.2f}")
    adjuster.print_adjustments(details2)
    
    print("✅ Lineup adjuster test completed")
