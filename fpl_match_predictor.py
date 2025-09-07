import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FPLMatchPredictor:
    """
    Premier League Match Outcome Predictor using Fantasy Premier League API
    Predicts W/L/D outcomes for upcoming gameweeks
    """
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.bootstrap_data = None
        self.teams = {}
        self.players = {}
        self.fixtures = {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def fetch_bootstrap_data(self):
        """Fetch main FPL API data containing teams, players, and events"""
        print("Fetching FPL bootstrap data...")
        try:
            url = f"{self.base_url}bootstrap-static/"
            response = requests.get(url)
            response.raise_for_status()
            self.bootstrap_data = response.json()
            
            # Process teams
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
            
            print(f"Successfully loaded data for {len(self.teams)} teams and {len(self.bootstrap_data['elements'])} players")
            return True
            
        except Exception as e:
            print(f"Error fetching bootstrap data: {e}")
            return False
    
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
                # Group all fixtures by gameweek
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
    
    def calculate_team_strength_features(self, team_id, is_home=True):
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
    
    def calculate_team_player_features(self, team_id):
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
    
    def _default_team_features(self):
        """Default team strength features"""
        return {
            'overall_strength': 3,
            'attack_strength': 3,
            'defence_strength': 3,
            'general_strength': 3
        }
    
    def _default_player_features(self):
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
    
    def create_match_features(self, home_team_id, away_team_id):
        """Create feature vector for a match"""
        # Get team strength features
        home_strength = self.calculate_team_strength_features(home_team_id, is_home=True)
        away_strength = self.calculate_team_strength_features(away_team_id, is_home=False)
        
        # Get player-based features
        home_player_stats = self.calculate_team_player_features(home_team_id)
        away_player_stats = self.calculate_team_player_features(away_team_id)
        
        # Combine all features
        features = []
        
        # Home team features
        features.extend([
            home_strength['overall_strength'],
            home_strength['attack_strength'],
            home_strength['defence_strength'],
            home_strength['general_strength'],
            home_player_stats['team_total_points'],
            home_player_stats['team_avg_form'],
            home_player_stats['team_total_goals'],
            home_player_stats['team_total_assists'],
            home_player_stats['team_avg_ict'],
            home_player_stats['team_clean_sheets'],
            home_player_stats['team_goals_conceded']
        ])
        
        # Away team features
        features.extend([
            away_strength['overall_strength'],
            away_strength['attack_strength'],
            away_strength['defence_strength'],
            away_strength['general_strength'],
            away_player_stats['team_total_points'],
            away_player_stats['team_avg_form'],
            away_player_stats['team_total_goals'],
            away_player_stats['team_total_assists'],
            away_player_stats['team_avg_ict'],
            away_player_stats['team_clean_sheets'],
            away_player_stats['team_goals_conceded']
        ])
        
        # Differential features
        features.extend([
            home_strength['attack_strength'] - away_strength['defence_strength'],
            away_strength['attack_strength'] - home_strength['defence_strength'],
            home_player_stats['team_avg_form'] - away_player_stats['team_avg_form'],
            home_player_stats['team_total_points'] - away_player_stats['team_total_points'],
            home_player_stats['team_avg_ict'] - away_player_stats['team_avg_ict']
        ])
        
        return np.array(features)
    
    def prepare_training_data_from_api(self):
        """Prepare training data using historical fixture results from API"""
        print("Preparing training data from API...")
        
        if not self.bootstrap_data:
            self.fetch_bootstrap_data()
        
        # Fetch all fixtures
        all_fixtures = self.fetch_fixtures()
        
        X = []
        y = []
        
        # Define feature names
        self.feature_names = [
            'home_overall_strength', 'home_attack_strength', 'home_defence_strength', 'home_general_strength',
            'home_total_points', 'home_avg_form', 'home_goals', 'home_assists', 'home_ict', 
            'home_clean_sheets', 'home_goals_conceded',
            'away_overall_strength', 'away_attack_strength', 'away_defence_strength', 'away_general_strength',
            'away_total_points', 'away_avg_form', 'away_goals', 'away_assists', 'away_ict',
            'away_clean_sheets', 'away_goals_conceded',
            'attack_vs_defence_home', 'attack_vs_defence_away', 'form_difference', 
            'points_difference', 'ict_difference'
        ]
        
        # Process finished fixtures
        for fixture in all_fixtures:
            if fixture['finished'] and fixture['team_h_score'] is not None:
                home_team_id = fixture['team_h']
                away_team_id = fixture['team_a']
                home_score = fixture['team_h_score']
                away_score = fixture['team_a_score']
                
                # Create features
                features = self.create_match_features(home_team_id, away_team_id)
                
                # Determine outcome
                if home_score > away_score:
                    outcome = 'H'  # Home win
                elif away_score > home_score:
                    outcome = 'A'  # Away win
                else:
                    outcome = 'D'  # Draw
                
                X.append(features)
                y.append(outcome)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training data prepared: {len(X)} samples with {X.shape[1] if len(X) > 0 else 0} features")
        if len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            print(f"Outcome distribution: {dict(zip(unique, counts))}")
        
        return X, y
    
    def train_model(self):
        """Train the prediction model using API data"""
        print("Training model with FPL API data...")
        
        X, y = self.prepare_training_data_from_api()
        
        if len(X) == 0:
            print("Warning: No training data available. Using dummy model.")
            return False
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Test Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return True
    
    def predict_match(self, home_team_id, away_team_id, home_team_name=None, away_team_name=None):
        """Predict outcome of a single match"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Get team names if not provided
        if home_team_name is None:
            home_team_name = self.teams[home_team_id]['name']
        if away_team_name is None:
            away_team_name = self.teams[away_team_id]['name']
        
        # Create features
        features = self.create_match_features(home_team_id, away_team_id)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction and probabilities
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Map prediction to readable format with team names
        outcome_map = {
            'H': f'{home_team_name} Win', 
            'D': 'Draw', 
            'A': f'{away_team_name} Win'
        }
        class_labels = self.model.classes_
        
        prob_dict = {}
        for i, label in enumerate(class_labels):
            prob_dict[outcome_map[label]] = probabilities[i]
        
        return {
            'prediction': outcome_map[prediction],
            'probabilities': prob_dict,
            'confidence': max(probabilities)
        }
    
    def predict_gameweek_fixtures(self, gameweek):
        """Predict all fixtures for a specific gameweek"""
        print(f"Predicting fixtures for Gameweek {gameweek}...")
        
        # Fetch fixtures for the gameweek
        fixtures = self.fetch_fixtures(gameweek)
        
        predictions = []
        
        for fixture in fixtures:
            if not fixture['finished']:  # Only predict unfinished matches
                home_team_id = fixture['team_h']
                away_team_id = fixture['team_a']
                home_team_name = self.teams[home_team_id]['name']
                away_team_name = self.teams[away_team_id]['name']
                
                try:
                    result = self.predict_match(home_team_id, away_team_id, home_team_name, away_team_name)
                    predictions.append({
                        'fixture_id': fixture['id'],
                        'home_team': home_team_name,
                        'away_team': away_team_name,
                        'kickoff_time': fixture['kickoff_time'],
                        'prediction': result['prediction'],
                        'probabilities': result['probabilities'],
                        'confidence': result['confidence']
                    })
                except Exception as e:
                    print(f"Error predicting {home_team_name} vs {away_team_name}: {e}")
                    predictions.append({
                        'fixture_id': fixture['id'],
                        'home_team': home_team_name,
                        'away_team': away_team_name,
                        'kickoff_time': fixture['kickoff_time'],
                        'prediction': 'Error',
                        'probabilities': {},
                        'confidence': 0.0
                    })
        
        return predictions
    
    def display_predictions(self, predictions):
        """Display predictions in a formatted way"""
        print("\n" + "="*80)
        print(f"PREMIER LEAGUE MATCH PREDICTIONS")
        print("="*80)
        
        for pred in predictions:
            print(f"\n{pred['home_team']} vs {pred['away_team']}")
            print(f"Kickoff: {pred['kickoff_time']}")
            print(f"Prediction: {pred['prediction']} (Confidence: {pred['confidence']:.1%})")
            
            if pred['probabilities']:
                print("Probabilities:")
                for outcome, prob in pred['probabilities'].items():
                    print(f"  {outcome}: {prob:.1%}")
            print("-" * 60)


def main():
    """Main function to predict upcoming Premier League matches"""
    print("FPL API Premier League Match Predictor")
    print("="*50)
    
    # Initialize predictor
    predictor = FPLMatchPredictor()
    
    # Fetch data
    if not predictor.fetch_bootstrap_data():
        print("Failed to fetch FPL data. Exiting.")
        return
    
    # Get current gameweek info
    current_gw, next_gw = predictor.get_current_gameweek()
    print(f"Current Gameweek: {current_gw}")
    print(f"Next Gameweek: {next_gw}")
    
    # Train model
    if predictor.train_model():
        # Predict next gameweek fixtures
        if next_gw:
            predictions = predictor.predict_gameweek_fixtures(next_gw)
            predictor.display_predictions(predictions)
        else:
            print("No upcoming gameweek found.")
    else:
        print("Failed to train model.")


if __name__ == "__main__":
    main()
