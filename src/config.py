"""
Configuration Module
Contains constants, settings, and team mappings
"""

# FPL API Configuration
FPL_BASE_URL = "https://fantasy.premierleague.com/api/"
FPL_BOOTSTRAP_ENDPOINT = "bootstrap-static/"
FPL_FIXTURES_ENDPOINT = "fixtures/"

# External Data Sources (Historical match data)
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"

# Team ID Mapping (FPL team names to IDs)
TEAM_NAME_TO_ID = {
    'Arsenal': 1, 'Aston Villa': 2, 'Brentford': 3, 'Brighton': 4,
    'Burnley': 5, 'Chelsea': 6, 'Crystal Palace': 7, 'Everton': 8,
    'Fulham': 9, 'Liverpool': 10, 'Luton': 11, 'Man City': 12,
    'Man Utd': 13, 'Newcastle': 14, 'Nott\'m Forest': 15, 'Sheffield Utd': 16,
    'Tottenham': 17, 'West Ham': 18, 'Wolves': 19, 'Bournemouth': 20,
    # Alternative spellings
    'Manchester City': 12, 'Manchester United': 13, 'Tottenham Hotspur': 17,
    'Nottingham Forest': 15, 'Sheffield United': 16, 'West Ham United': 18,
    'Wolverhampton Wanderers': 19, 'Brighton & Hove Albion': 4
}

# Model Configuration
MODEL_PARAMS = {
    'default_rf_params': {
        'random_state': 42,
        'n_jobs': -1
    },
    'param_grid': {
        'n_estimators': [100, 200],
        'max_depth': [5, 8, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    },
    'cv_splits': 3,
    'feature_selection_k': 15,
    'train_test_split': 0.8
}

# Feature Configuration
FEATURE_NAMES = [
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

# Default Values
DEFAULT_TEAM_FEATURES = {
    'overall_strength': 3,
    'attack_strength': 3,
    'defence_strength': 3,
    'general_strength': 3
}

DEFAULT_PLAYER_FEATURES = {
    'team_total_points': 50,
    'team_avg_form': 2.0,
    'team_total_goals': 5,
    'team_total_assists': 3,
    'team_avg_ict': 50,
    'team_clean_sheets': 1,
    'team_goals_conceded': 5,
    'team_minutes_played': 1000
}

DEFAULT_RECENT_FORM = {
    'recent_points': 1.0,
    'recent_goals_scored': 1.0,
    'recent_goals_conceded': 1.0,
    'recent_goal_difference': 0.0,
    'recent_wins': 0.3
}

# Player Position Mapping
PLAYER_POSITIONS = {
    1: 'GK',   # Goalkeeper
    2: 'DEF',  # Defender
    3: 'MID',  # Midfielder
    4: 'FWD'   # Forward
}

# Training Data Thresholds
TRAINING_DATA_THRESHOLDS = {
    'minimum_samples': 100,
    'recommended_samples': 300,
    'optimal_samples': 500
}

# API Request Configuration
API_CONFIG = {
    'timeout': 10,
    'max_retries': 3,
    'retry_delay': 1  # seconds
}

import os

# Cache Configuration
CACHE_CONFIG = {
    'cache_directory': os.getenv('CACHE_DIRECTORY', 'cache'),
    'model_cache_expiry_days': 7,
    'bootstrap_data_cache_expiry_hours': 24,
    'fixtures_cache_expiry_hours': 6,
    'training_data_cache_expiry_days': 7,
    'enable_persistent_cache': True
}

# Scheduler Configuration
SCHEDULER_CONFIG = {
    'enable_auto_retraining': os.getenv('ENABLE_SCHEDULER', 'true').lower() == 'true',
    'check_interval_hours': 1,
    'force_retrain_day': 'sunday',  # Day of week for forced retraining
    'force_retrain_time': '03:00',  # Time for forced retraining (24h format)
    'retraining_on_gameweek_change': True,
    'minimum_retraining_interval_hours': 6
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': os.path.join(CACHE_CONFIG['cache_directory'], 'app.log'),
    'max_file_size_mb': 10,
    'backup_count': 3
}