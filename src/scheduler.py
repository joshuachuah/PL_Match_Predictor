"""
Training Scheduler Module
Handles automatic retraining based on gameweek progression and time intervals
"""

import threading
import time
import schedule
from datetime import datetime, timedelta
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """Manages automatic model retraining schedule"""
    
    def __init__(self, predictor_instance, cache_manager):
        self.predictor = predictor_instance
        self.cache_manager = cache_manager
        self.scheduler_thread = None
        self.running = False
        self.last_check_time = None
        self.last_known_gameweek = None
        
    def start_scheduler(self):
        """Start the background scheduler"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        
        # Schedule checks every hour
        schedule.every().hour.do(self._check_retraining_needed)
        
        # Schedule weekly retraining as fallback (every Sunday at 3 AM)
        schedule.every().sunday.at("03:00").do(self._force_retrain)
        
        # Start scheduler in background thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Training scheduler started")
    
    def stop_scheduler(self):
        """Stop the background scheduler"""
        self.running = False
        schedule.clear()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Training scheduler stopped")
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _check_retraining_needed(self):
        """Check if retraining is needed based on gameweek progression"""
        try:
            logger.info("Checking if retraining is needed...")
            
            # Get current gameweek from predictor
            if not hasattr(self.predictor, 'data_api') or not self.predictor.data_api.bootstrap_data:
                # Try to fetch fresh bootstrap data
                if not self.predictor.fetch_bootstrap_data():
                    logger.warning("Could not fetch bootstrap data for retraining check")
                    return
            
            current_gw, next_gw = self.predictor.get_current_gameweek()
            
            if current_gw is None:
                logger.warning("Could not determine current gameweek")
                return
            
            # Check if gameweek has changed
            gameweek_changed = (
                self.last_known_gameweek is not None and 
                current_gw != self.last_known_gameweek
            )
            
            # Check if enough time has passed since last check
            time_for_recheck = (
                self.last_check_time is None or 
                datetime.now() - self.last_check_time > timedelta(hours=6)
            )
            
            # Use cache manager to determine if retraining is needed
            needs_retraining = self.cache_manager.is_retraining_needed(current_gw)
            
            if needs_retraining or gameweek_changed:
                logger.info(f"Retraining triggered - GW: {current_gw}, Changed: {gameweek_changed}, Needed: {needs_retraining}")
                self._trigger_retraining(current_gw)
            elif time_for_recheck:
                logger.info(f"Retraining check complete - no retraining needed (GW: {current_gw})")
            
            self.last_known_gameweek = current_gw
            self.last_check_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error checking retraining status: {e}")
    
    def _force_retrain(self):
        """Force retraining (scheduled weekly fallback)"""
        logger.info("Weekly forced retraining triggered")
        
        try:
            if not self.predictor.fetch_bootstrap_data():
                logger.error("Failed to fetch bootstrap data for forced retraining")
                return
            
            current_gw, _ = self.predictor.get_current_gameweek()
            self._trigger_retraining(current_gw or 1)
            
        except Exception as e:
            logger.error(f"Error in forced retraining: {e}")
    
    def _trigger_retraining(self, current_gameweek: int):
        """Trigger model retraining in background"""
        def retrain():
            try:
                logger.info(f"Starting background retraining for gameweek {current_gameweek}")
                
                # Clear old cache to ensure fresh training
                self.cache_manager.clear_cache()
                
                # Perform fresh training
                if self.predictor.train_model():
                    logger.info("Background retraining completed successfully")
                else:
                    logger.error("Background retraining failed")
                    
            except Exception as e:
                logger.error(f"Error during background retraining: {e}")
        
        # Run retraining in separate daemon thread to avoid blocking
        retrain_thread = threading.Thread(target=retrain, daemon=True)
        retrain_thread.start()
    
    def trigger_manual_retrain(self) -> bool:
        """Manually trigger retraining (for API endpoints)"""
        try:
            logger.info("Manual retraining triggered")
            
            current_gw, _ = self.predictor.get_current_gameweek()
            self._trigger_retraining(current_gw or 1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in manual retraining: {e}")
            return False
    
    def get_scheduler_status(self) -> dict:
        """Get current scheduler status"""
        return {
            'running': self.running,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'last_known_gameweek': self.last_known_gameweek,
            'next_scheduled_jobs': [str(job) for job in schedule.jobs],
            'cache_info': self.cache_manager.get_cache_info()
        }


class GameweekMonitor:
    """Monitor gameweek progression for smarter retraining triggers"""
    
    def __init__(self, data_api):
        self.data_api = data_api
        self.gameweek_history = []
        self.last_update = None
    
    def check_gameweek_progression(self) -> tuple[bool, int, int]:
        """
        Check if gameweek has progressed
        Returns: (gameweek_changed, current_gw, previous_gw)
        """
        try:
            current_gw, next_gw = self.data_api.get_current_gameweek()
            
            if not current_gw:
                return False, 0, 0
            
            # Check if this is a new gameweek
            if len(self.gameweek_history) == 0:
                self.gameweek_history.append({
                    'gameweek': current_gw,
                    'detected_at': datetime.now().isoformat()
                })
                return False, current_gw, 0
            
            last_recorded_gw = self.gameweek_history[-1]['gameweek']
            
            if current_gw > last_recorded_gw:
                # Gameweek has progressed!
                self.gameweek_history.append({
                    'gameweek': current_gw,
                    'detected_at': datetime.now().isoformat()
                })
                
                # Keep only last 5 gameweeks in history
                if len(self.gameweek_history) > 5:
                    self.gameweek_history = self.gameweek_history[-5:]
                
                logger.info(f"Gameweek progression detected: {last_recorded_gw} -> {current_gw}")
                return True, current_gw, last_recorded_gw
            
            return False, current_gw, last_recorded_gw
            
        except Exception as e:
            logger.error(f"Error checking gameweek progression: {e}")
            return False, 0, 0
    
    def is_gameweek_ending_soon(self) -> bool:
        """Check if current gameweek is ending soon (within 24 hours)"""
        try:
            # This would require more detailed fixture timing analysis
            # For now, we'll implement a simple check
            fixtures = self.data_api.fetch_fixtures()
            
            current_gw, next_gw = self.data_api.get_current_gameweek()
            if not current_gw:
                return False
            
            # Get fixtures for current gameweek
            current_gw_fixtures = [f for f in fixtures if f.get('event') == current_gw]
            
            # Check if any unfinished fixtures remain
            unfinished_fixtures = [f for f in current_gw_fixtures if not f.get('finished', False)]
            
            # If no unfinished fixtures, gameweek is likely complete
            if len(unfinished_fixtures) == 0:
                return True
            
            # Additional logic could check kickoff times of remaining fixtures
            return False
            
        except Exception as e:
            logger.error(f"Error checking gameweek ending: {e}")
            return False
