"""
Model Training and Prediction Module
Handles machine learning model training, hyperparameter tuning, and predictions
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
import logging

# Try to import XGBoost, but make it optional
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available. Install with: pip install xgboost")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MatchPredictor:
    """Handles model training and prediction operations"""
    
    def __init__(self, cache_manager=None):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.feature_names = []
        self.cache_manager = cache_manager
        self._model_loaded_from_cache = False
        
        # Try to load existing model from cache if available
        if self.cache_manager:
            self.load_model_from_cache()
        
    def prepare_training_data(self, feature_engineer, historical_fixtures: List) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data using historical fixture results (excluding draws)"""
        print("Preparing training data from historical fixtures...")

        X = []
        y = []
        draws_excluded = 0

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

                # Skip draws - only include definitive wins
                if home_score == away_score:
                    draws_excluded += 1
                    continue

                # Create features with gameweek context for recent form
                features = feature_engineer.create_match_features(home_team_id, away_team_id, current_gw)

                # Binary classification: Home win vs Away win
                if home_score > away_score:
                    outcome = 'H'  # Home win
                else:
                    outcome = 'A'  # Away win

                X.append(features)
                y.append(outcome)

        X = np.array(X)
        y = np.array(y)

        # Store feature names
        self.feature_names = feature_engineer.get_feature_names()

        print(f"Training data prepared: {len(X)} samples with {X.shape[1] if len(X) > 0 else 0} features")
        print(f"Excluded {draws_excluded} draws - focusing on win/loss prediction only")
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
        """Train and compare multiple models, then optimize the best one"""
        print("="*70)
        print("TRAINING BINARY CLASSIFICATION MODEL (WIN/LOSS ONLY)")
        print("="*70)

        if len(X) == 0:
            print("Warning: No training data available. Using dummy model.")
            return False

        print(f"\nInitial dataset: {X.shape[0]} samples, {X.shape[1]} features")

        # Encode labels (convert 'H'/'A' to 0/1)
        y = self.label_encoder.fit_transform(y)
        print(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Feature selection to reduce overfitting
        print("\nPerforming feature selection...")
        k_best = min(15, X.shape[1])  # Select top 15 features or all if less
        self.feature_selector = SelectKBest(score_func=f_classif, k=k_best)
        X_selected = self.feature_selector.fit_transform(X_scaled, y)

        # Get selected feature names
        selected_features = []
        selected_indices = self.feature_selector.get_support(indices=True)

        for idx in selected_indices:
            selected_features.append(self.feature_names[idx])

        print(f"Selected {len(selected_features)} features: {selected_features[:5]}{'...' if len(selected_features) > 5 else ''}")

        # Use time-series aware split (train on earlier matches, test on later)
        split_point = int(0.8 * len(X_selected))
        X_train, X_test = X_selected[:split_point], X_selected[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

        print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        # Calculate class weights for balancing
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"Class weights: {class_weight_dict}")

        # Use TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=3)

        print("\n" + "="*70)
        print("STEP 1: COMPARING MULTIPLE ALGORITHMS")
        print("="*70)

        # Define models to test
        models_to_test = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, # num of trees
                max_depth=8, # max depth of decision tree, reduces overfitting
                class_weight=class_weight_dict, # 
                random_state=42, # reproducibility
                n_jobs=-1, # use all cpu cores
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5, # max depth of each weak leearner
                learning_rate=0.1, # shrink each tree's contribution, stable learning without excessive training time
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                class_weight=class_weight_dict, # Adjusts loss function to compensate for class imbalance, prevents model from predicting only majority class
                max_iter=1000, # Maximum number of optimization iterations, needs more iterations to converge ( further training no longer improves the model )
                random_state=42
            )
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            # Convert class weights for XGBoost
            scale_pos_weight = class_weight_dict.get('H', 1.0) / class_weight_dict.get('A', 1.0)
            models_to_test['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight, # Balances positive vs negative classes
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )

        # Test each model
        results = {}
        print("\nTesting algorithms:")
        for name, model in models_to_test.items():
            print(f"\n  Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': y_pred
            }
            print(f"    Accuracy: {accuracy:.3f} | F1-Score: {f1:.3f}")

        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        best_model_info = results[best_model_name]

        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"  Accuracy: {best_model_info['accuracy']:.3f}")
        print(f"  F1-Score: {best_model_info['f1_score']:.3f}")
        print(f"{'='*70}")

        # Now optimize the best model
        print(f"\nSTEP 2: OPTIMIZING {best_model_name.upper()}")
        print("="*70)

        # Define parameter grids for each model type
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300], # no of trees in forest
                'max_depth': [5, 8, 10, 15], # max depth of each tree
                'min_samples_split': [2, 5, 10], # min number of samples required to split a node
                'min_samples_leaf': [1, 2, 4] # min samples allowed in leaf node
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300], # no of boosting stages
                'max_depth': [3, 5, 7], # depth of each weak learner
                'learning_rate': [0.01, 0.1, 0.2], # how much each tree contributes (slower but safe learning, faster but risk overfitting)
                'subsample': [0.8, 0.9, 1.0] # fraction of data used per tree
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1.0, 10.0], # inverse of regularization strength 
                'penalty': ['l2'], # penalize large weights smoothly
                'solver': ['lbfgs', 'liblinear']
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }

        param_grid = param_grids.get(best_model_name, {})

        if param_grid:
            print("Performing hyperparameter tuning...")

            # Recreate base model with class weights
            if best_model_name == 'Random Forest':
                base_model = RandomForestClassifier(
                    class_weight=class_weight_dict,
                    random_state=42,
                    n_jobs=-1
                )
            elif best_model_name == 'Gradient Boosting':
                base_model = GradientBoostingClassifier(random_state=42)
            elif best_model_name == 'Logistic Regression':
                base_model = LogisticRegression(
                    class_weight=class_weight_dict,
                    max_iter=1000,
                    random_state=42
                )
            elif best_model_name == 'XGBoost':
                scale_pos_weight = class_weight_dict.get('H', 1.0) / class_weight_dict.get('A', 1.0)
                base_model = XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=tscv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_

            print(f"\nBest parameters: {best_params}")
            print(f"Best CV F1-Score: {best_cv_score:.3f}")
        else:
            self.model = best_model_info['model']
            best_params = "No hyperparameter tuning for this model"
            best_cv_score = best_model_info['f1_score']

        # Final evaluation
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Feature importance analysis
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = list(zip(selected_features, self.model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            print(f"\nTop 5 most important features:")
            for feature, importance in feature_importance[:5]:
                print(f"  {feature}: {importance:.3f}")

        print(f"\n{'='*70}")
        print(f"FINAL MODEL PERFORMANCE ({best_model_name})")
        print(f"{'='*70}")
        print(f"Test Accuracy: {accuracy:.3f}")
        print(f"Test F1-Score: {f1:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save model to cache if cache manager is available
        if self.cache_manager:
            training_metadata = {
                'model_type': best_model_name,
                'training_samples': len(X),
                'test_accuracy': accuracy,
                'test_f1_score': f1,
                'best_cv_score': best_cv_score,
                'best_params': str(best_params),
                'training_time': datetime.now().isoformat(),
                'current_gameweek': current_gameweek,
                'binary_classification': True
            }
            self.save_model_to_cache(training_metadata)

        return True
    
    def load_model_from_cache(self) -> bool:
        """Load model components from cache if available"""
        if not self.cache_manager:
            return False

        try:
            model, scaler, feature_selector, label_encoder, feature_names, metadata = self.cache_manager.load_model()

            if model is not None:
                self.model = model
                self.scaler = scaler
                self.feature_selector = feature_selector
                if label_encoder is not None:
                    self.label_encoder = label_encoder
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
                self.label_encoder,
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
        """Predict outcome of a single match (binary: home win vs away win)"""
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
        prediction_encoded = self.model.predict(features_selected)[0]
        probabilities = self.model.predict_proba(features_selected)[0]

        # Decode prediction back to original label ('H' or 'A')
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]

        # Map prediction to readable format with team names (binary classification)
        outcome_map = {
            'H': f'{home_team_name}',
            'A': f'{away_team_name}'
        }

        # Get original class labels for probability mapping
        class_labels_encoded = self.model.classes_
        class_labels = self.label_encoder.inverse_transform(class_labels_encoded)

        prob_dict = {}
        for i, label in enumerate(class_labels):
            prob_dict[outcome_map[label]] = probabilities[i]

        # Add draw probability as "uncertain" since we don't predict it
        # This is calculated as the overlap/uncertainty between the two classes
        max_prob = max(probabilities)
        # prob_dict['Draw (uncertain)'] = 1.0 - max_prob if max_prob < 0.8 else 0.0

        return {
            'prediction': outcome_map[prediction],
            'probabilities': prob_dict,
            'confidence': max(probabilities)
        }
    
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self.model is not None
