"""
FPL Data API Module
Handles all interactions with the Fantasy Premier League API
"""

import requests
import pandas as pd
import numpy as np
from collections import defaultdict
import io
import csv


class FPLDataAPI:
    """Handles all FPL API data fetching operations"""
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.bootstrap_data = None
        self.teams = {}
        self.players = {}
        self.fixtures = {}
        self.historical_fixtures = []
        
    def fetch_bootstrap_data(self):
        """Fetch main FPL API data containing teams, players, and events"""
        print("Fetching FPL bootstrap data...")
        try:
            url = f"{self.base_url}bootstrap-static/"
            response = requests.get(url)
            response.raise_for_status()
            self.bootstrap_data = response.json()
            
            self._process_bootstrap_data()
            return True
            
        except Exception as e:
            print(f"Error fetching bootstrap data: {e}")
            return False
    
    def _process_bootstrap_data(self):
        """Process bootstrap data to extract teams and players"""

        # transform raw API data into usable dictionaries
        if not self.bootstrap_data:
            return
            
        # Process teams
        self.teams = {}
        for team in self.bootstrap_data['teams']:
            self.teams[team['id']] = {
                'name': team['name'],
                'short_name': team['short_name'],
                'strength': team['strength'],
                'strength_overall_home': team['strength_overall_home'],
                'strength_overall_away': team['strength_overall_away'],
                'strength_attack_home': team['strength_attack_home'],
                'strength_attack_away': team['strength_attack_away'],
                'strength_defence_home': team['strength_defence_home'],
                'strength_defence_away': team['strength_defence_away'],
                'pulse_id': team['pulse_id']
            }
        
        # Process players
        self.players = {}
        for player in self.bootstrap_data['elements']:
            team_id = player['team']
            if team_id not in self.players:
                self.players[team_id] = []
            
            self.players[team_id].append({
                'id': player['id'],
                'name': f"{player['first_name']} {player['second_name']}",
                'position': player['element_type'],
                'total_points': player['total_points'],
                'form': float(player['form']) if player['form'] else 0.0,
                'points_per_game': float(player['points_per_game']) if player['points_per_game'] else 0.0,
                'minutes': player['minutes'],
                'goals_scored': player['goals_scored'],
                'assists': player['assists'],
                'clean_sheets': player['clean_sheets'],
                'goals_conceded': player['goals_conceded'],
                'influence': float(player['influence']) if player['influence'] else 0.0,
                'creativity': float(player['creativity']) if player['creativity'] else 0.0,
                'threat': float(player['threat']) if player['threat'] else 0.0,
                'ict_index': float(player['ict_index']) if player['ict_index'] else 0.0
            })
        
        print(f"Successfully processed data for {len(self.teams)} teams and {len(self.bootstrap_data['elements'])} players")
    
    def fetch_fixtures(self, gameweek=None):
        """Fetch fixture data for all or specific gameweek"""
        print(f"Fetching fixtures{f' for gameweek {gameweek}' if gameweek else ''}...")
        try:
            url = f"{self.base_url}fixtures/"
            if gameweek:
                url += f"?event={gameweek}"
            
            response = requests.get(url)
            response.raise_for_status()
            fixtures_data = response.json()
            
            if gameweek:
                self.fixtures[gameweek] = fixtures_data
            else:
                # Group all fixtures by gameweek and store historical data
                self.historical_fixtures = fixtures_data
                for fixture in fixtures_data:
                    gw = fixture['event']
                    if gw not in self.fixtures:
                        self.fixtures[gw] = []
                    self.fixtures[gw].append(fixture)
            
            print(f"Successfully loaded {len(fixtures_data)} fixtures")
            return fixtures_data
            
        except Exception as e:
            print(f"Error fetching fixtures: {e}")
            return []
    
    def fetch_historical_season_data(self, seasons_back=2):
        """Fetch historical data from previous seasons"""
        print(f"Fetching historical data from last {seasons_back} seasons...")
        all_historical_fixtures = []
        
        # Try to fetch data from previous seasons
        # Note: This is a workaround since FPL API doesn't expose historical seasons
        # Use archived data or simulate with current season patterns
        
        # use the current season data and simulate previous seasons
        # by treating early gameweeks as "previous seasons"
        current_fixtures = self.fetch_fixtures()
        
        if len(current_fixtures) > 0:
            # Sort fixtures by gameweek
            current_fixtures_sorted = sorted(current_fixtures, key=lambda x: x['event'] if x['event'] else 0)
            
            # Create simulated historical data by treating different gameweek ranges as different seasons
            
            season_1_fixtures = []  # Gameweeks 1-19 (first half)
            season_2_fixtures = []  # Gameweeks 20-38 (second half)
            
            for fixture in current_fixtures_sorted:
                if fixture['finished'] and fixture['event']:
                    if fixture['event'] <= 19:
                        # Simulate as "last season" data
                        season_1_fixture = fixture.copy()
                        season_1_fixture['simulated_season'] = 'previous_1'
                        season_1_fixtures.append(season_1_fixture)
                    else:
                        # Simulate as "two seasons ago" data  
                        season_2_fixture = fixture.copy()
                        season_2_fixture['simulated_season'] = 'previous_2'
                        season_2_fixtures.append(season_2_fixture)
            
            # Add simulated historical fixtures
            all_historical_fixtures.extend(season_1_fixtures)
            all_historical_fixtures.extend(season_2_fixtures)
            all_historical_fixtures.extend([f for f in current_fixtures if f['finished']])
            
            print(f"Simulated historical data: {len(season_1_fixtures)} fixtures from 'season 1', "
                  f"{len(season_2_fixtures)} from 'season 2', "
                  f"{len([f for f in current_fixtures if f['finished']])} from current season")
            
        # Store all historical fixtures for training
        self.historical_fixtures = all_historical_fixtures
        
        return all_historical_fixtures
    
    def fetch_enhanced_training_data(self):
        """Fetch enhanced training data with more samples"""
        print("Fetching enhanced training data with historical seasons...")
        
        # Get current season data
        current_fixtures = self.fetch_fixtures()
        
        # Get historical data  
        historical_fixtures = self.fetch_historical_season_data(seasons_back=2)
        
        # Combine all data
        all_training_fixtures = historical_fixtures
        
        print(f"Total training fixtures available: {len(all_training_fixtures)}")
        print(f"Finished fixtures: {len([f for f in all_training_fixtures if f['finished']])}")
        
        return all_training_fixtures
    
    def fetch_real_historical_data(self):
        """
        Alternative method to get real historical Premier League data
        This method attempts to get actual historical data from multiple sources
        """
        print("Attempting to fetch real historical Premier League data...")
        
        all_historical_data = []
        
        # Method 1: Try FPL API historical endpoints (if they exist)
        historical_seasons = ['2023-24', '2022-23', '2021-22']
        
        for season in historical_seasons:
            print(f"Trying to fetch data for season {season}...")
            try:
                # trying my luck to see if the API endpoint has historical data
                url = f"https://fantasy.premierleague.com/api/bootstrap-static/{season}/"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"Found historical data for {season}")
                    # Process this data...
                else:
                    print(f"No API data available for {season}")
            except:
                print(f"Could not access {season} data via API")
        
        # Method 2: Use football-data.co.uk (free historical data)
        print("Attempting to fetch from football-data.co.uk...")
        try:
            # source of historical Premier League data
            historical_url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
            response = requests.get(historical_url, timeout=10)
            if response.status_code == 200:
                print("Successfully fetched historical data from football-data.co.uk")
                # Parse CSV data and convert to our format
                csv_data = io.StringIO(response.text)
                reader = csv.DictReader(csv_data)
                
                for row in reader:
                    if row.get('FTHG') and row.get('FTAG'):  # Full Time Home/Away Goals
                        # Convert to our fixture format
                        historical_fixture = {
                            'team_h': self.get_team_id_by_name(row.get('HomeTeam', '')),
                            'team_a': self.get_team_id_by_name(row.get('AwayTeam', '')),
                            'team_h_score': int(row['FTHG']),
                            'team_a_score': int(row['FTAG']),
                            'finished': True,
                            'event': 1,  # Assign a dummy gameweek
                            'source': 'football-data.co.uk',
                            'season': '2023-24'
                        }
                        all_historical_data.append(historical_fixture)
                
                print(f"Processed {len(all_historical_data)} historical matches from external source")
                        
        except Exception as e:
            print(f"Could not fetch from football-data.co.uk: {e}")
        
        if len(all_historical_data) > 0:
            # Combine with current season data
            current_fixtures = self.fetch_fixtures()
            current_finished = [f for f in current_fixtures if f['finished']]
            
            all_training_data = all_historical_data + current_finished
            self.historical_fixtures = all_training_data
            
            print(f"Total training data: {len(all_training_data)} matches")
            print(f"- Historical: {len(all_historical_data)} matches")
            print(f"- Current season: {len(current_finished)} matches")
            
            return all_training_data
        else:
            print("No historical data sources available. Using simulation method...")
            return self.fetch_enhanced_training_data()
    
    def get_team_id_by_name(self, team_name):
        """Helper method to get team ID by name for external data"""
        # Mapping from common team names to FPL team IDs
        team_mapping = {
            'Arsenal': 1, 'Aston Villa': 2, 'Brentford': 3, 'Brighton': 4,
            'Burnley': 5, 'Chelsea': 6, 'Crystal Palace': 7, 'Everton': 8,
            'Fulham': 9, 'Liverpool': 10, 'Leeds': 11, 'Man City': 12,
            'Man Utd': 13, 'Newcastle': 14, 'Nott\'m Forest': 15, 'Sheffield Utd': 16,
            'Tottenham': 17, 'West Ham': 18, 'Wolves': 19, 'Bournemouth': 20,
            # Alternative spellings
            'Manchester City': 12, 'Manchester United': 13, 'Tottenham Hotspur': 17,
            'Nottingham Forest': 15, 'Sheffield United': 16, 'West Ham United': 18,
            'Wolverhampton Wanderers': 19, 'Brighton & Hove Albion': 4
        }
        
        return team_mapping.get(team_name, 1)  # Default to Arsenal if not found
    
    def get_current_gameweek(self):
        """Get the current and next gameweek numbers"""
        if not self.bootstrap_data:
            self.fetch_bootstrap_data()
        
        current_gw = None
        next_gw = None
        
        for event in self.bootstrap_data['events']:
            if event['is_current']:
                current_gw = event['id']
            elif event['is_next']:
                next_gw = event['id']
        
        return current_gw, next_gw
