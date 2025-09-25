"""
Main Predictor Module
Orchestrates all components for Premier League match prediction
"""

from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .data_api import FPLDataAPI
from .feature_engineering import FeatureEngineer
from .model import MatchPredictor
from .utils import display_predictions, get_match_result_code
from .config import DEFAULT_RECENT_FORM, CACHE_CONFIG
from .cache_manager import ModelCacheManager
import logging

logger = logging.getLogger(__name__)


class FPLMatchPredictor:
    """
    Premier League Match Outcome Predictor using Fantasy Premier League API
    Predicts W/L/D outcomes for upcoming gameweeks
    """
    
    def __init__(self, enable_cache=True):
        # Initialize cache manager
        self.cache_manager = ModelCacheManager(CACHE_CONFIG['cache_directory']) if enable_cache else None
        
        # Initialize components with cache manager
        self.data_api = FPLDataAPI()
        self.feature_engineer = None
        self.model = MatchPredictor(self.cache_manager)
        self._model_trained = False
        
        # Try to load cached data on initialization
        if self.cache_manager:
            self._load_cached_data()
        
    def _load_cached_data(self) -> bool:
        """Load cached bootstrap data and check model status"""
        try:
            # Try to load cached bootstrap data
            cached_bootstrap = self.cache_manager.load_bootstrap_data()
            if cached_bootstrap:
                self.data_api.bootstrap_data = cached_bootstrap
                self.data_api._process_bootstrap_data()  # Process teams and players
                logger.info("Bootstrap data loaded from cache")
            
            # Check if model is already trained and loaded
            if self.model.is_trained():
                self._model_trained = True
                logger.info("Model loaded from cache and ready for predictions")
                
                # Initialize feature engineer for cached model
                if self.feature_engineer is None:
                    self._initialize_feature_engineer()
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            return False
    
    def fetch_bootstrap_data(self) -> bool:
        """Fetch main FPL API data containing teams, players, and events"""
        # First try to load from cache
        if self.cache_manager:
            cached_data = self.cache_manager.load_bootstrap_data()
            if cached_data:
                self.data_api.bootstrap_data = cached_data
                self.data_api._process_bootstrap_data()
                logger.info("Using cached bootstrap data")
                return True
        
        # If no cache or cache expired, fetch fresh data
        success = self.data_api.fetch_bootstrap_data()
        
        # Cache the fresh data if successful
        if success and self.cache_manager:
            self.cache_manager.save_bootstrap_data(self.data_api.bootstrap_data)
        
        return success
    
    def fetch_fixtures(self, gameweek: Optional[int] = None) -> List[Dict]:
        """Fetch fixture data for all or specific gameweek"""
        return self.data_api.fetch_fixtures(gameweek)
    
    def get_current_gameweek(self) -> Tuple[Optional[int], Optional[int]]:
        """Get the current and next gameweek numbers"""
        return self.data_api.get_current_gameweek()
    
    def prepare_training_data_from_api(self) -> Tuple[List, List]:
        """Prepare training data using enhanced historical fixture results from API"""
        print("Preparing enhanced training data from API...")
        
        # Try to load cached training data first
        if self.cache_manager:
            cached_training_data = self.cache_manager.load_training_data()
            if cached_training_data:
                logger.info("Using cached training data")
                data = cached_training_data['data']
                
                # Recreate feature engineer from cached data
                self.feature_engineer = FeatureEngineer(
                    teams=self.data_api.teams,
                    players=self.data_api.players,
                    historical_fixtures=data['historical_fixtures']
                )
                
                return data['X'], data['y']
        
        if not self.data_api.bootstrap_data:
            self.fetch_bootstrap_data()
        
        # Try to fetch real historical data first, fall back to enhanced simulation
        try:
            all_fixtures = self.data_api.fetch_real_historical_data()
        except Exception as e:
            print(f"Could not fetch real historical data: {e}")
            print("Falling back to simulated historical data...")
            all_fixtures = self.data_api.fetch_enhanced_training_data()
        
        # Initialize feature engineer with current data
        self.feature_engineer = FeatureEngineer(
            teams=self.data_api.teams,
            players=self.data_api.players,
            historical_fixtures=all_fixtures
        )
        
        # Prepare training data using feature engineer and model
        X, y = self.model.prepare_training_data(self.feature_engineer, all_fixtures)
        
        # Cache the training data
        if self.cache_manager:
            training_data = {
                'X': X,
                'y': y,
                'historical_fixtures': all_fixtures,
                'feature_names': self.feature_engineer.get_feature_names() if self.feature_engineer else []
            }
            self.cache_manager.save_training_data(training_data)
            logger.info("Training data cached successfully")
        
        return X, y
    
    def train_model(self, force_retrain=False) -> bool:
        """Train the prediction model using API data with optimizations"""
        # Check if model is already trained and loaded from cache
        if not force_retrain and self.model.is_trained() and self.model.is_model_cached():
            print("Model already trained and loaded from cache. Skipping training.")
            self._model_trained = True
            return True
        
        print("Training model with FPL API data...")
        
        # Check if retraining is needed based on cache manager
        if self.cache_manager and not force_retrain:
            current_gw, _ = self.get_current_gameweek()
            if current_gw and not self.cache_manager.is_retraining_needed(current_gw):
                # Try to load existing model instead of retraining
                if self.model.load_model_from_cache():
                    print("Loaded existing model from cache instead of retraining")
                    self._model_trained = True
                    return True
        
        X, y = self.prepare_training_data_from_api()
        
        if len(X) == 0:
            print("Warning: No training data available.")
            return False
        
        # Add current gameweek to training metadata
        current_gw, _ = self.get_current_gameweek()
        
        success = self.model.train_model(X, y, current_gw)
        if success:
            self._model_trained = True
            
            # Update training metadata with current gameweek info
            if self.cache_manager and hasattr(self.model, 'cache_manager'):
                # This will be saved automatically by the model's train_model method
                pass
        
        return success
    
    def _initialize_feature_engineer(self):
        """Initialize feature engineer when needed (e.g., after loading model from cache)"""
        logger.info("Initializing feature engineer for cached model...")
        
        # Get some historical fixtures for feature engineering
        try:
            # Try to fetch recent fixtures for feature engineering context
            all_fixtures = self.data_api.fetch_fixtures()  # Get all fixtures
            
            # Initialize feature engineer with current data
            self.feature_engineer = FeatureEngineer(
                teams=self.data_api.teams,
                players=self.data_api.players,
                historical_fixtures=all_fixtures
            )
            
            logger.info("Feature engineer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing feature engineer: {e}")
            # Fallback: initialize with minimal data
            self.feature_engineer = FeatureEngineer(
                teams=self.data_api.teams,
                players=self.data_api.players,
                historical_fixtures=[]
            )
            logger.warning("Feature engineer initialized with minimal data")
    
    def predict_match(self, home_team_id: int, away_team_id: int, 
                     home_team_name: Optional[str] = None, 
                     away_team_name: Optional[str] = None, 
                     current_gameweek: Optional[int] = None) -> Dict[str, Any]:
        """Predict outcome of a single match"""
        if not self._model_trained:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Ensure feature engineer is initialized
        if self.feature_engineer is None:
            self._initialize_feature_engineer()
        
        # Get team names if not provided
        if home_team_name is None:
            home_team_name = self.data_api.teams[home_team_id]['name']
        if away_team_name is None:
            away_team_name = self.data_api.teams[away_team_id]['name']
        
        # Create features with current gameweek for recent form
        features = self.feature_engineer.create_match_features(
            home_team_id, away_team_id, current_gameweek
        )
        
        # Get prediction from model
        return self.model.predict_match(features, home_team_name, away_team_name)
    
    def predict_gameweek_fixtures(self, gameweek: int) -> List[Dict[str, Any]]:
        """Predict all fixtures for a specific gameweek"""
        print(f"Predicting fixtures for Gameweek {gameweek}...")
        
        # Fetch fixtures for the gameweek
        fixtures = self.fetch_fixtures(gameweek)
        
        predictions = []
        
        for fixture in fixtures:
            if not fixture['finished']:  # Only predict unfinished matches
                home_team_id = fixture['team_h']
                away_team_id = fixture['team_a']
                home_team_name = self.data_api.teams[home_team_id]['name']
                away_team_name = self.data_api.teams[away_team_id]['name']
                
                try:
                    result = self.predict_match(
                        home_team_id, away_team_id, 
                        home_team_name, away_team_name, gameweek
                    )
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
    
    def display_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        """Display predictions in a formatted way"""
        display_predictions(predictions)
    
    @property
    def teams(self) -> Dict:
        """Get teams data"""
        return self.data_api.teams
    
    @property
    def model_trained(self) -> bool:
        """Check if model is trained"""
        return self._model_trained


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
