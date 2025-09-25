"""
Feature Engineering Module
Handles calculation of team statistics, player features, and match features
"""

import numpy as np
from typing import Dict, Any, List


class FeatureEngineer:
    """Handles all feature engineering operations for match prediction"""
    
    def __init__(self, teams: Dict, players: Dict, historical_fixtures: List):
        self.teams = teams
        self.players = players
        self.historical_fixtures = historical_fixtures
    
    def calculate_team_strength_features(self, team_id: int, is_home: bool = True) -> Dict[str, float]:
        """Calculate team strength features from FPL API data"""
        if team_id not in self.teams:
            return self._default_team_features()
        
        team = self.teams[team_id]
        
        # Base strength features
        features = {
            'overall_strength': team['strength_overall_home'] if is_home else team['strength_overall_away'],
            'attack_strength': team['strength_attack_home'] if is_home else team['strength_attack_away'],
            'defence_strength': team['strength_defence_home'] if is_home else team['strength_defence_away'],
            'general_strength': team['strength']
        }
        
        return features
    
    def calculate_recent_form(self, team_id: int, current_gameweek: int, num_games: int = 5) -> Dict[str, float]:
        """Calculate recent form for a team based on last N matches"""
        recent_matches = []
        
        # Look through completed fixtures to find team's recent matches
        for fixture in self.historical_fixtures:
            if (fixture['finished'] and 
                fixture['team_h_score'] is not None and 
                fixture['event'] < current_gameweek):
                
                if fixture['team_h'] == team_id:
                    # Team played at home
                    home_score = fixture['team_h_score']
                    away_score = fixture['team_a_score']
                    goals_scored = home_score
                    goals_conceded = away_score
                    if home_score > away_score:
                        result = 3  # Win
                    elif home_score == away_score:
                        result = 1  # Draw
                    else:
                        result = 0  # Loss
                        
                elif fixture['team_a'] == team_id:
                    # Team played away
                    home_score = fixture['team_h_score']
                    away_score = fixture['team_a_score']
                    goals_scored = away_score
                    goals_conceded = home_score
                    if away_score > home_score:
                        result = 3  # Win
                    elif away_score == home_score:
                        result = 1  # Draw
                    else:
                        result = 0  # Loss
                else:
                    continue
                    
                recent_matches.append({
                    'gameweek': fixture['event'],
                    'goals_scored': goals_scored,
                    'goals_conceded': goals_conceded,
                    'points': result
                })
        
        # Sort by gameweek and take last N matches
        recent_matches.sort(key=lambda x: x['gameweek'], reverse=True)
        recent_matches = recent_matches[:num_games]
        
        if not recent_matches:
            return {
                'recent_points': 1.0,
                'recent_goals_scored': 1.0,
                'recent_goals_conceded': 1.0,
                'recent_goal_difference': 0.0,
                'recent_wins': 0.0,
                'recent_matches_count': 0
            }
        
        total_points = sum(match['points'] for match in recent_matches)
        total_goals_scored = sum(match['goals_scored'] for match in recent_matches)
        total_goals_conceded = sum(match['goals_conceded'] for match in recent_matches)
        wins = sum(1 for match in recent_matches if match['points'] == 3)
        
        return {
            'recent_points': total_points / len(recent_matches),
            'recent_goals_scored': total_goals_scored / len(recent_matches),
            'recent_goals_conceded': total_goals_conceded / len(recent_matches),
            'recent_goal_difference': (total_goals_scored - total_goals_conceded) / len(recent_matches),
            'recent_wins': wins / len(recent_matches),
            'recent_matches_count': len(recent_matches)
        }

    def calculate_team_player_features(self, team_id: int) -> Dict[str, float]:
        """Calculate team features based on player statistics"""
        if team_id not in self.players:
            return self._default_player_features()
        
        team_players = self.players[team_id]
        
        # Aggregate player statistics
        total_points = sum(p['total_points'] for p in team_players)
        avg_form = np.mean([p['form'] for p in team_players if p['form'] > 0]) if any(p['form'] > 0 for p in team_players) else 0
        total_goals = sum(p['goals_scored'] for p in team_players)
        total_assists = sum(p['assists'] for p in team_players)
        total_minutes = sum(p['minutes'] for p in team_players)
        avg_ict = np.mean([p['ict_index'] for p in team_players if p['ict_index'] > 0]) if any(p['ict_index'] > 0 for p in team_players) else 0
        
        # Defensive features (goalkeepers and defenders)
        defenders = [p for p in team_players if p['position'] in [1, 2]]  # GK and DEF
        total_clean_sheets = sum(p['clean_sheets'] for p in defenders)
        total_goals_conceded = sum(p['goals_conceded'] for p in defenders if p['position'] == 1)  # Only GK
        
        return {
            'team_total_points': total_points,
            'team_avg_form': avg_form,
            'team_total_goals': total_goals,
            'team_total_assists': total_assists,
            'team_avg_ict': avg_ict,
            'team_clean_sheets': total_clean_sheets,
            'team_goals_conceded': total_goals_conceded,
            'team_minutes_played': total_minutes
        }
    
    def create_match_features(self, home_team_id: int, away_team_id: int, current_gameweek: int = None) -> np.ndarray:
        """Create feature vector for a match"""
        # Get team strength features
        home_strength = self.calculate_team_strength_features(home_team_id, is_home=True)
        away_strength = self.calculate_team_strength_features(away_team_id, is_home=False)
        
        # Get player-based features
        home_player_stats = self.calculate_team_player_features(home_team_id)
        away_player_stats = self.calculate_team_player_features(away_team_id)
        
        # Get recent form features if current_gameweek is provided
        if current_gameweek:
            home_recent_form = self.calculate_recent_form(home_team_id, current_gameweek)
            away_recent_form = self.calculate_recent_form(away_team_id, current_gameweek)
        else:
            # Use default values for recent form
            home_recent_form = {'recent_points': 1.0, 'recent_goals_scored': 1.0, 'recent_goals_conceded': 1.0, 'recent_goal_difference': 0.0, 'recent_wins': 0.3}
            away_recent_form = {'recent_points': 1.0, 'recent_goals_scored': 1.0, 'recent_goals_conceded': 1.0, 'recent_goal_difference': 0.0, 'recent_wins': 0.3}
        
        # Combine all features
        features = []
        
        # Home team features (strength + player stats + recent form)
        features.extend([
            home_strength['overall_strength'],
            home_strength['attack_strength'],
            home_strength['defence_strength'],
            home_player_stats['team_avg_form'],
            home_player_stats['team_avg_ict'],
            home_recent_form['recent_points'],
            home_recent_form['recent_goals_scored'],
            home_recent_form['recent_goals_conceded'],
            home_recent_form['recent_goal_difference'],
            home_recent_form['recent_wins']
        ])
        
        # Away team features (strength + player stats + recent form)
        features.extend([
            away_strength['overall_strength'],
            away_strength['attack_strength'],
            away_strength['defence_strength'],
            away_player_stats['team_avg_form'],
            away_player_stats['team_avg_ict'],
            away_recent_form['recent_points'],
            away_recent_form['recent_goals_scored'],
            away_recent_form['recent_goals_conceded'],
            away_recent_form['recent_goal_difference'],
            away_recent_form['recent_wins']
        ])
        
        # Differential features (key comparisons)
        features.extend([
            home_strength['attack_strength'] - away_strength['defence_strength'],
            away_strength['attack_strength'] - home_strength['defence_strength'],
            home_player_stats['team_avg_form'] - away_player_stats['team_avg_form'],
            home_recent_form['recent_points'] - away_recent_form['recent_points'],
            home_recent_form['recent_goal_difference'] - away_recent_form['recent_goal_difference']
        ])
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get the names of all features in order"""
        return [
            'home_overall_strength', 'home_attack_strength', 'home_defence_strength',
            'home_avg_form', 'home_ict',
            'home_recent_points', 'home_recent_goals_scored', 'home_recent_goals_conceded',
            'home_recent_goal_difference', 'home_recent_wins',
            'away_overall_strength', 'away_attack_strength', 'away_defence_strength',
            'away_avg_form', 'away_ict',
            'away_recent_points', 'away_recent_goals_scored', 'away_recent_goals_conceded',
            'away_recent_goal_difference', 'away_recent_wins',
            'attack_vs_defence_home', 'attack_vs_defence_away', 'form_difference',
            'recent_points_difference', 'recent_goal_diff_difference'
        ]
    
    def _default_team_features(self) -> Dict[str, float]:
        """Default team strength features"""
        return {
            'overall_strength': 3,
            'attack_strength': 3,
            'defence_strength': 3,
            'general_strength': 3
        }
    
    def _default_player_features(self) -> Dict[str, float]:
        """Default player-based features"""
        return {
            'team_total_points': 50,
            'team_avg_form': 2.0,
            'team_total_goals': 5,
            'team_total_assists': 3,
            'team_avg_ict': 50,
            'team_clean_sheets': 1,
            'team_goals_conceded': 5,
            'team_minutes_played': 1000
        }
