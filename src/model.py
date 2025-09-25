"""
Model Training and Prediction Module
Handles machine learning model training, hyperparameter tuning, and predictions
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MatchPredictor:
    """Handles model training and prediction operations"""
    
    def __init__(self, cache_manager=None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_names = []
        self.cache_manager = cache_manager
        self._model_loaded_from_cache = False
        
        # Try to load existing model from cache if available
        if self.cache_manager:
            self.load_model_from_cache()
        
    def prepare_training_data(self, feature_engineer, historical_fixtures: List) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data using historical fixture results"""
        print("Preparing training data from historical fixtures...")
        
        X = []
        y = []
        
        # Sort fixtures by gameweek to process chronologically
        all_fixtures_sorted = sorted(historical_fixtures, key=lambda x: x['event'] if x['event'] else 0)
        
        print(f"Processing {len(all_fixtures_sorted)} total fixtures for training...")
        
        # Process finished fixtures
        for fixture in all_fixtures_sorted:
            if fixture['finished'] and fixture['team_h_score'] is not None and fixture['event']:
                home_team_id = fixture['team_h']
                away_team_id = fixture['team_a']
                home_score = fixture['team_h_score']
                away_score = fixture['team_a_score']
                current_gw = fixture['event']
                
                # Create features with gameweek context for recent form
                features = feature_engineer.create_match_features(home_team_id, away_team_id, current_gw)
                
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
        
        # Store feature names
        self.feature_names = feature_engineer.get_feature_names()
        
        print(f"Training data prepared: {len(X)} samples with {X.shape[1] if len(X) > 0 else 0} features")
        if len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            print(f"Outcome distribution: {dict(zip(unique, counts))}")
        
        # Check if we have sufficient data
        if len(X) < 100:
            print(f"WARNING: Only {len(X)} training samples available. Recommend at least 300+ for stable results.")
        elif len(X) < 300:
            print(f"NOTICE: {len(X)} training samples is better, but 500+ recommended for optimal performance.")
        else:
            print(f"GOOD: {len(X)} training samples should provide stable model training.")
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray, current_gameweek: int = None) -> bool:
        """Train the prediction model with optimizations"""
        print("Training model with optimizations...")
        
        if len(X) == 0:
            print("Warning: No training data available. Using dummy model.")
            return False
        
        print(f"Initial dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection to reduce overfitting
        print("Performing feature selection...")
        k_best = min(15, X.shape[1])  # Select top 15 features or all if less
        self.feature_selector = SelectKBest(score_func=f_classif, k=k_best)
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        # Get selected feature names
        selected_features = []
        feature_scores = self.feature_selector.scores_
        selected_indices = self.feature_selector.get_support(indices=True)
        
        for idx in selected_indices:
            selected_features.append(self.feature_names[idx])
        
        print(f"Selected {len(selected_features)} features: {selected_features[:5]}{'...' if len(selected_features) > 5 else ''}")
        
        # Use time-series aware split (train on earlier matches, test on later)
        # Sort data chronologically and split 80/20
        split_point = int(0.8 * len(X_selected))
        X_train, X_test = X_selected[:split_point], X_selected[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Calculate class weights for balancing
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # Hyperparameter tuning with time series cross-validation
        print("Performing hyperparameter tuning...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 8, 10],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        # Use TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        rf_base = RandomForestClassifier(
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            rf_base, 
            param_grid, 
            cv=tscv, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Use best model
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        
        # Train final model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance analysis
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = list(zip(selected_features, self.model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            print(f"\nTop 5 most important features:")
            for feature, importance in feature_importance[:5]:
                print(f"  {feature}: {importance:.3f}")
        
        print(f"\nModel trained successfully!")
        print(f"Best CV Score: {grid_search.best_score_:.3f}")
        print(f"Test Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model to cache if cache manager is available
        if self.cache_manager:
            training_metadata = {
                'training_samples': len(X),
                'test_accuracy': accuracy,
                'best_cv_score': grid_search.best_score_,
                'best_params': best_params,
                'training_time': datetime.now().isoformat(),
                'current_gameweek': current_gameweek
            }
            self.save_model_to_cache(training_metadata)
        
        return True
    
    def load_model_from_cache(self) -> bool:
        """Load model components from cache if available"""
        if not self.cache_manager:
            return False
        
        try:
            model, scaler, feature_selector, feature_names, metadata = self.cache_manager.load_model()
            
            if model is not None:
                self.model = model
                self.scaler = scaler
                self.feature_selector = feature_selector
                self.feature_names = feature_names
                self._model_loaded_from_cache = True
                
                logger.info(f"Model loaded from cache (trained at: {metadata.get('trained_at')})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading model from cache: {e}")
            return False
    
    def save_model_to_cache(self, training_metadata: Dict[str, Any]) -> bool:
        """Save model components to cache"""
        if not self.cache_manager or not self.is_trained():
            return False
        
        try:
            success = self.cache_manager.save_model(
                self.model,
                self.scaler,
                self.feature_selector,
                self.feature_names,
                training_metadata
            )
            
            if success:
                logger.info("Model saved to cache successfully")
            else:
                logger.error("Failed to save model to cache")
                
            return success
            
        except Exception as e:
            logger.error(f"Error saving model to cache: {e}")
            return False
    
    def is_model_cached(self) -> bool:
        """Check if model was loaded from cache"""
        return self._model_loaded_from_cache
    
    def predict_match(self, features: np.ndarray, home_team_name: str, away_team_name: str) -> Dict[str, Any]:
        """Predict outcome of a single match"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Scale and transform features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Apply feature selection if trained
        if self.feature_selector is not None:
            features_selected = self.feature_selector.transform(features_scaled)
        else:
            features_selected = features_scaled
        
        # Get prediction and probabilities
        prediction = self.model.predict(features_selected)[0]
        probabilities = self.model.predict_proba(features_selected)[0]
        
        # Map prediction to readable format with team names
        outcome_map = {
            'H': f'{home_team_name}', 
            'D': 'Draw', 
            'A': f'{away_team_name}'
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
    
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self.model is not None
