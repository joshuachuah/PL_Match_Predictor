"""
Model and Data Cache Management Module
Handles persistence, validation, and lifecycle of trained models and data
"""

import os
import pickle
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelCacheManager:
    """Manages caching and persistence of trained models and data"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # cache file paths
        self.model_file = self.cache_dir / "trained_model.pkl"
        self.scaler_file = self.cache_dir / "scaler.pkl"
        self.feature_selector_file = self.cache_dir / "feature_selector.pkl"
        self.label_encoder_file = self.cache_dir / "label_encoder.pkl"
        self.metadata_file = self.cache_dir / "model_metadata.json"
        self.bootstrap_data_file = self.cache_dir / "bootstrap_data.json"
        self.training_data_file = self.cache_dir / "training_data.pkl"
        self.fixtures_cache_file = self.cache_dir / "fixtures_cache.json"
        
    def save_model(self, model, scaler, feature_selector, label_encoder, feature_names: list,
                   training_metadata: Dict[str, Any]) -> bool:
        """Save trained model and associated components"""
        try:
            # save model components
            with open(self.model_file, 'wb') as f:
                pickle.dump(model, f)

            with open(self.scaler_file, 'wb') as f:
                pickle.dump(scaler, f)

            if feature_selector is not None:
                with open(self.feature_selector_file, 'wb') as f:
                    pickle.dump(feature_selector, f)

            if label_encoder is not None:
                with open(self.label_encoder_file, 'wb') as f:
                    pickle.dump(label_encoder, f)

            # save metadata
            metadata = {
                'trained_at': datetime.now().isoformat(),
                'feature_names': feature_names,
                'training_metadata': training_metadata,
                'model_version': '1.0',
                'has_feature_selector': feature_selector is not None,
                'has_label_encoder': label_encoder is not None
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Model saved successfully to {self.cache_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self) -> Tuple[Any, Any, Any, Any, list, Dict[str, Any]]:
        """Load trained model and associated components"""
        try:
            # check if all required files exist
            if not all([
                self.model_file.exists(),
                self.scaler_file.exists(),
                self.metadata_file.exists()
            ]):
                logger.warning("Model cache files not found")
                return None, None, None, None, [], {}

            # load metadata first
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

            # load model components
            with open(self.model_file, 'rb') as f:
                model = pickle.load(f)

            with open(self.scaler_file, 'rb') as f:
                scaler = pickle.load(f)

            feature_selector = None
            if metadata.get('has_feature_selector', False) and self.feature_selector_file.exists():
                with open(self.feature_selector_file, 'rb') as f:
                    feature_selector = pickle.load(f)

            label_encoder = None
            if metadata.get('has_label_encoder', False) and self.label_encoder_file.exists():
                with open(self.label_encoder_file, 'rb') as f:
                    label_encoder = pickle.load(f)

            feature_names = metadata.get('feature_names', [])

            logger.info(f"Model loaded successfully from {self.cache_dir}")
            return model, scaler, feature_selector, label_encoder, feature_names, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None, None, None, [], {}
    
    def save_bootstrap_data(self, bootstrap_data: Dict[str, Any]) -> bool:
        """Save FPL bootstrap data"""
        try:
            # add cache timestamp
            cache_data = {
                'data': bootstrap_data,
                'cached_at': datetime.now().isoformat(),
                'data_hash': self._hash_data(bootstrap_data)
            }
            
            with open(self.bootstrap_data_file, 'w') as f:
                json.dump(cache_data, f)
                
            logger.info("Bootstrap data cached successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving bootstrap data: {e}")
            return False
    
    def load_bootstrap_data(self) -> Optional[Dict[str, Any]]:
        """Load cached FPL bootstrap data"""
        try:
            if not self.bootstrap_data_file.exists():
                return None
                
            with open(self.bootstrap_data_file, 'r') as f:
                cache_data = json.load(f)
            
            # check if data is still fresh (within 24 hours)
            cached_at = datetime.fromisoformat(cache_data['cached_at'])
            if datetime.now() - cached_at > timedelta(hours=24):
                logger.info("Bootstrap data cache expired")
                return None
            
            logger.info("Bootstrap data loaded from cache")
            return cache_data['data']
            
        except Exception as e:
            logger.error(f"Error loading bootstrap data: {e}")
            return None
    
    def save_training_data(self, training_data: Dict[str, Any]) -> bool:
        """Save processed training data"""
        try:
            cache_data = {
                'data': training_data,
                'cached_at': datetime.now().isoformat(),
                'data_hash': self._hash_data(training_data)
            }
            
            with open(self.training_data_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.info("Training data cached successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            return False
    
    def load_training_data(self) -> Optional[Dict[str, Any]]:
        """Load cached training data"""
        try:
            if not self.training_data_file.exists():
                return None
                
            with open(self.training_data_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # check if data is still fresh (within 7 days)
            cached_at = datetime.fromisoformat(cache_data['cached_at'])
            if datetime.now() - cached_at > timedelta(days=7):
                logger.info("Training data cache expired")
                return None
            
            logger.info("Training data loaded from cache")
            return cache_data['data']
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None
    
    def save_fixtures_cache(self, fixtures_data: Dict[str, Any]) -> bool:
        """Save fixtures data with gameweek information"""
        try:
            cache_data = {
                'fixtures': fixtures_data,
                'cached_at': datetime.now().isoformat(),
                'current_gameweek': fixtures_data.get('current_gameweek'),
                'next_gameweek': fixtures_data.get('next_gameweek')
            }
            
            with open(self.fixtures_cache_file, 'w') as f:
                json.dump(cache_data, f)
                
            logger.info("Fixtures data cached successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving fixtures data: {e}")
            return False
    
    def load_fixtures_cache(self) -> Optional[Dict[str, Any]]:
        """Load cached fixtures data"""
        try:
            if not self.fixtures_cache_file.exists():
                return None
                
            with open(self.fixtures_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # check if data is still fresh (within 6 hours)
            cached_at = datetime.fromisoformat(cache_data['cached_at'])
            if datetime.now() - cached_at > timedelta(hours=6):
                logger.info("Fixtures cache expired")
                return None
            
            logger.info("Fixtures data loaded from cache")
            return cache_data
            
        except Exception as e:
            logger.error(f"Error loading fixtures data: {e}")
            return None
    
    def is_retraining_needed(self, current_gameweek: int) -> bool:
        """Check if model retraining is needed based on gameweek progression"""
        try:
            if not self.metadata_file.exists():
                logger.info("No existing model found - retraining needed")
                return True
            
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # check when model was last trained
            trained_at = datetime.fromisoformat(metadata['trained_at'])
            
            # check if it's been more than 7 days since last training
            if datetime.now() - trained_at > timedelta(days=7):
                logger.info("Model is older than 7 days - retraining needed")
                return True
            
            # check if gameweek has progressed significantly
            last_gameweek = metadata.get('training_metadata', {}).get('current_gameweek')
            if last_gameweek and current_gameweek > last_gameweek:
                logger.info(f"Gameweek progressed from {last_gameweek} to {current_gameweek} - retraining needed")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retraining status: {e}")
            return True  # Default to retraining if unsure
    
    def clear_cache(self) -> bool:
        """Clear all cached data and models"""
        try:
            cache_files = [
                self.model_file, self.scaler_file, self.feature_selector_file,
                self.metadata_file, self.bootstrap_data_file, 
                self.training_data_file, self.fixtures_cache_file
            ]
            
            for file_path in cache_files:
                if file_path.exists():
                    file_path.unlink()
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data"""
        info = {
            'cache_directory': str(self.cache_dir),
            'model_cached': self.model_file.exists(),
            'bootstrap_data_cached': self.bootstrap_data_file.exists(),
            'training_data_cached': self.training_data_file.exists(),
            'fixtures_cached': self.fixtures_cache_file.exists()
        }
        
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    info['model_trained_at'] = metadata.get('trained_at')
                    info['model_version'] = metadata.get('model_version')
            except:
                pass
        
        return info
    
    def _hash_data(self, data: Any) -> str:
        """Generate hash for data integrity checking"""
        return hashlib.md5(str(data).encode()).hexdigest()
