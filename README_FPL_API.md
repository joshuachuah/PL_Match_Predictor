# FPL API Premier League Match Predictor

This program uses the Fantasy Premier League (FPL) API to predict upcoming Premier League match outcomes. It replaces the CSV-based approach with real-time data from the official FPL API.

## Features

- **Real-time Data**: Fetches live data from the FPL API instead of static CSV files
- **Current Season Focus**: Uses current season team strengths and player statistics
- **Gameweek Predictions**: Automatically predicts all matches in the next gameweek
- **Machine Learning**: Uses Random Forest classifier to predict W/L/D outcomes
- **Confidence Scores**: Provides probability distributions for each prediction

## How It Works

### Data Sources
The program uses several FPL API endpoints:

1. **Bootstrap Static** (`/api/bootstrap-static/`):
   - Team strength ratings (overall, attack, defense for home/away)
   - Player statistics (points, form, goals, assists, etc.)
   - Gameweek information

2. **Fixtures** (`/api/fixtures/`):
   - Historical match results for training
   - Upcoming fixtures for prediction

### Prediction Features

The model uses the following features for each team:

**Team Strength (from FPL)**:
- Overall strength (home/away specific)
- Attack strength (home/away specific)  
- Defense strength (home/away specific)
- General strength rating

**Player Statistics**:
- Total team points
- Average player form
- Total goals and assists
- Average ICT index (Influence, Creativity, Threat)
- Clean sheets and goals conceded

**Derived Features**:
- Attack vs defense differentials
- Form differences between teams
- Points and ICT differences

## Usage

### Basic Usage
```python
from fpl_match_predictor import FPLMatchPredictor

# Initialize predictor
predictor = FPLMatchPredictor()

# Fetch data and train model
predictor.fetch_bootstrap_data()
predictor.train_model()

# Get current gameweek info
current_gw, next_gw = predictor.get_current_gameweek()

# Predict all matches in next gameweek
predictions = predictor.predict_gameweek_fixtures(next_gw)
predictor.display_predictions(predictions)
```

### Run Complete Program
```bash
python fpl_match_predictor.py
```

## Sample Output

```
FPL API Premier League Match Predictor
==================================================
Fetching FPL bootstrap data...
Successfully loaded data for 20 teams and 720 players
Current Gameweek: 3
Next Gameweek: 4
Training model with FPL API data...
Preparing training data from API...
Successfully loaded 380 fixtures
Training data prepared: 30 samples with 27 features
Model trained successfully!
Test Accuracy: 0.667

================================================================================
PREMIER LEAGUE MATCH PREDICTIONS
================================================================================

Arsenal vs Nott'm Forest
Kickoff: 2025-09-13T11:30:00Z
Prediction: Home Win (Confidence: 45.5%)
Probabilities:
  Away Win: 18.8%
  Draw: 35.7%
  Home Win: 45.5%
------------------------------------------------------------

Bournemouth vs Brighton
Kickoff: 2025-09-13T14:00:00Z
Prediction: Home Win (Confidence: 72.8%)
Probabilities:
  Away Win: 8.2%
  Draw: 19.0%
  Home Win: 72.8%
------------------------------------------------------------
```

## Requirements

```python
requests
pandas
numpy
scikit-learn
```

## API Rate Limits

The FPL API is generally stable but:
- Avoid making too many requests in quick succession
- The API is sometimes slower during high-traffic periods (gameweek deadlines)
- Data is typically updated after matches finish

## Model Performance

The model achieves reasonable accuracy given the limited training data available early in the season. As more matches are played, the model will have more training data and should improve in accuracy.

Current features focus on team strength and current form rather than long-term historical patterns, making it well-suited for current season predictions.

