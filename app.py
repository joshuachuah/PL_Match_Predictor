from flask import Flask, render_template, jsonify, request
import json
import logging
import os
from datetime import datetime
from src.predictor import FPLMatchPredictor
from src.scheduler import TrainingScheduler
from src.config import LOGGING_CONFIG, SCHEDULER_CONFIG
import threading
import time

# Configure logging
os.makedirs('cache', exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['file_path']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global predictor instance and scheduler
predictor = None
scheduler = None
model_trained = False
training_status = {"status": "not_started", "message": ""}
app_initialized = False

def initialize_app():
    """Initialize the app with caching and scheduling"""
    global predictor, scheduler, model_trained, training_status, app_initialized
    
    try:
        if app_initialized:
            logger.info("App already initialized")
            return True
            
        logger.info("Initializing FPL Match Predictor with caching...")
        training_status["status"] = "initializing"
        training_status["message"] = "Initializing with cache support..."
        
        # Initialize predictor with caching enabled
        predictor = FPLMatchPredictor(enable_cache=True)
        
        # Check if model is already trained and cached
        if predictor.model_trained:
            model_trained = True
            training_status["status"] = "completed"
            training_status["message"] = "Model loaded from cache and ready!"
            logger.info("Model loaded from cache - skipping training")
        else:
            # Need to fetch data and train
            training_status["message"] = "Fetching FPL data..."
            
            if not predictor.fetch_bootstrap_data():
                training_status["status"] = "error"
                training_status["message"] = "Failed to fetch FPL data"
                logger.error("Failed to fetch bootstrap data")
                return False
            
            training_status["message"] = "Training prediction model..."
            logger.info("Training new model...")
            
            # Train model (will use cache if available)
            if predictor.train_model():
                model_trained = True
                training_status["status"] = "completed"
                training_status["message"] = "Model trained successfully!"
                logger.info("Model training completed")
            else:
                training_status["status"] = "error"
                training_status["message"] = "Failed to train model"
                logger.error("Model training failed")
                return False
        
        # Initialize scheduler if enabled
        if SCHEDULER_CONFIG['enable_auto_retraining']:
            scheduler = TrainingScheduler(predictor, predictor.cache_manager)
            scheduler.start_scheduler()
            logger.info("Automatic retraining scheduler started")
        
        app_initialized = True
        logger.info("App initialization completed successfully")
        return True
        
    except Exception as e:
        error_msg = f"Error during initialization: {str(e)}"
        training_status["status"] = "error"
        training_status["message"] = error_msg
        logger.error(error_msg, exc_info=True)
        return False

def initialize_predictor():
    """Legacy function - calls new initialize_app function"""
    return initialize_app()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/robots.txt')
def robots_txt():
    """Serve robots.txt for SEO"""
    return app.send_static_file('robots.txt')

@app.route('/sitemap.xml')
def sitemap_xml():
    """Serve sitemap.xml for SEO"""
    return app.send_static_file('sitemap.xml')

@app.route('/api/status')
def get_status():
    """Get current training status with cache and scheduler info"""
    global predictor, scheduler, model_trained, training_status, app_initialized
    
    status = {
        "training_status": training_status,
        "model_ready": model_trained,
        "app_initialized": app_initialized,
        "current_gameweek": None,
        "next_gameweek": None,
        "cache_info": {},
        "scheduler_info": {},
        "model_from_cache": False
    }
    
    if predictor:
        try:
            current_gw, next_gw = predictor.get_current_gameweek()
            status["current_gameweek"] = current_gw
            status["next_gameweek"] = next_gw
            
            # Add cache information
            if predictor.cache_manager:
                status["cache_info"] = predictor.cache_manager.get_cache_info()
                status["model_from_cache"] = predictor.model.is_model_cached() if predictor.model else False
            
            # Add scheduler information
            if scheduler:
                status["scheduler_info"] = scheduler.get_scheduler_status()
                
        except Exception as e:
            logger.error(f"Error getting status: {e}")
    
    return jsonify(status)

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the predictor"""
    global training_status, app_initialized
    
    if training_status["status"] in ["initializing", "training"]:
        return jsonify({"message": "Already initializing..."})
    
    if app_initialized:
        return jsonify({"message": "App already initialized", "status": "success"})
    
    # Start initialization in background thread
    thread = threading.Thread(target=initialize_app)
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": "Initialization started"})

@app.route('/api/predict')
def predict():
    """Get predictions for next gameweek"""
    global predictor, model_trained
    
    if not model_trained or not predictor:
        return jsonify({"error": "Model not ready. Please initialize first."}), 400
    
    try:
        current_gw, next_gw = predictor.get_current_gameweek()
        
        if not next_gw:
            return jsonify({"error": "No upcoming gameweek found"}), 404
        
        predictions = predictor.predict_gameweek_fixtures(next_gw)
        
        return jsonify({
            "current_gameweek": current_gw,
            "next_gameweek": next_gw,
            "predictions": predictions
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_gameweek/<int:gameweek>')
def predict_gameweek(gameweek):
    """Get predictions for a specific gameweek"""
    global predictor, model_trained
    
    if not model_trained or not predictor:
        return jsonify({"error": "Model not ready. Please initialize first."}), 400
    
    try:
        predictions = predictor.predict_gameweek_fixtures(gameweek)
        
        return jsonify({
            "gameweek": gameweek,
            "predictions": predictions
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add new endpoints for cache management and retraining
@app.route('/api/retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger model retraining"""
    global predictor, scheduler
    
    if not predictor:
        return jsonify({"error": "Predictor not initialized"}), 400
    
    try:
        if scheduler:
            success = scheduler.trigger_manual_retrain()
            if success:
                return jsonify({"message": "Manual retraining triggered", "status": "success"})
            else:
                return jsonify({"error": "Failed to trigger retraining"}), 500
        else:
            # Trigger retraining directly
            thread = threading.Thread(
                target=lambda: predictor.train_model(force_retrain=True),
                daemon=True
            )
            thread.start()
            return jsonify({"message": "Manual retraining started", "status": "success"})
            
    except Exception as e:
        logger.error(f"Error triggering retrain: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all cached data"""
    global predictor
    
    if not predictor or not predictor.cache_manager:
        return jsonify({"error": "Cache manager not available"}), 400
    
    try:
        success = predictor.cache_manager.clear_cache()
        if success:
            return jsonify({"message": "Cache cleared successfully", "status": "success"})
        else:
            return jsonify({"error": "Failed to clear cache"}), 500
            
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache/info')
def get_cache_info():
    """Get detailed cache information"""
    global predictor
    
    if not predictor or not predictor.cache_manager:
        return jsonify({"error": "Cache manager not available"}), 400
    
    try:
        cache_info = predictor.cache_manager.get_cache_info()
        return jsonify(cache_info)
        
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        return jsonify({"error": str(e)}), 500

# Auto-initialize for WSGI servers
_initialization_started = False

@app.before_request
def ensure_initialized():
    """Ensure app is initialized before handling requests"""
    global app_initialized, _initialization_started
    
    if not app_initialized and not _initialization_started:
        logger.info("Auto-initializing app on first request")
        _initialization_started = True
        thread = threading.Thread(target=initialize_app, daemon=True)
        thread.start()

if __name__ == '__main__':
    # Initialize app on startup when running directly
    logger.info("Starting FPL Match Predictor app...")
    
    # Start initialization in background
    init_thread = threading.Thread(target=initialize_app, daemon=True)
    init_thread.start()
    
    # Get port from environment variable (for production) or default to 5000
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    
    app.run(debug=debug, host='0.0.0.0', port=port)
