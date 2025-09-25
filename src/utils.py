"""
Utilities Module
Helper functions and display utilities
"""

from typing import List, Dict, Any
import numpy as np


def display_predictions(predictions: List[Dict[str, Any]]) -> None:
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


def validate_training_data_size(data_size: int) -> str:
    """Validate training data size and return status message"""
    if data_size < 100:
        return f"WARNING: Only {data_size} training samples available. Recommend at least 300+ for stable results."
    elif data_size < 300:
        return f"NOTICE: {data_size} training samples is better, but 500+ recommended for optimal performance."
    else:
        return f"GOOD: {data_size} training samples should provide stable model training."


def get_outcome_distribution(outcomes: np.ndarray) -> Dict[str, int]:
    """Get distribution of match outcomes"""
    unique, counts = np.unique(outcomes, return_counts=True)
    return dict(zip(unique, counts))


def format_team_name(team_name: str) -> str:
    """Format team name for consistent display"""
    return team_name.strip().title()


def calculate_win_percentage(recent_points: float, num_games: int = 5) -> float:
    """Calculate win percentage from recent points"""
    if num_games == 0:
        return 0.0
    max_points = num_games * 3  # 3 points for a win
    return (recent_points / max_points) * 100 if max_points > 0 else 0.0


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default value when denominator is zero"""
    return numerator / denominator if denominator != 0 else default


def get_match_result_code(home_score: int, away_score: int) -> str:
    """Get match result code (H/D/A) from scores"""
    if home_score > away_score:
        return 'H'  # Home win
    elif away_score > home_score:
        return 'A'  # Away win
    else:
        return 'D'  # Draw


def format_confidence_percentage(confidence: float) -> str:
    """Format confidence as percentage string"""
    return f"{confidence:.1%}"


def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate string to maximum length with ellipsis"""
    return text[:max_length] + "..." if len(text) > max_length else text


def is_valid_team_id(team_id: int) -> bool:
    """Check if team ID is valid (1-20 for Premier League)"""
    return 1 <= team_id <= 20


def is_valid_gameweek(gameweek: int) -> bool:
    """Check if gameweek is valid (1-38 for Premier League)"""
    return 1 <= gameweek <= 38
