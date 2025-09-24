from flask import Flask, render_template, jsonify, request
import json
from fpl_match_predictor import FPLMatchPredictor
import threading
import time

app = Flask(__name__)

# Global predictor instance
predictor = None
model_trained = False
training_status = {"status": "not_started", "message": ""}

def initialize_predictor():
    """Initialize and train the predictor in a separate thread"""
    global predictor, model_trained, training_status
    
    try:
        training_status["status"] = "initializing"
        training_status["message"] = "Fetching FPL data..."
        
        predictor = FPLMatchPredictor()
        
        # Fetch bootstrap data
        if not predictor.fetch_bootstrap_data():
            training_status["status"] = "error"
            training_status["message"] = "Failed to fetch FPL data"
            return
        
        training_status["message"] = "Training prediction model..."
        
        # Train model
        if predictor.train_model():
            model_trained = True
            training_status["status"] = "completed"
            training_status["message"] = "Model trained successfully!"
        else:
            training_status["status"] = "error"
            training_status["message"] = "Failed to train model"
            
    except Exception as e:
        training_status["status"] = "error"
        training_status["message"] = f"Error: {str(e)}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current training status"""
    global predictor, model_trained, training_status
    
    status = {
        "training_status": training_status,
        "model_ready": model_trained,
        "current_gameweek": None,
        "next_gameweek": None
    }
    
    if predictor and model_trained:
        try:
            current_gw, next_gw = predictor.get_current_gameweek()
            status["current_gameweek"] = current_gw
            status["next_gameweek"] = next_gw
        except:
            pass
    
    return jsonify(status)

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the predictor"""
    global training_status
    
    if training_status["status"] in ["initializing", "training"]:
        return jsonify({"message": "Already initializing..."})
    
    # Start initialization in background thread
    thread = threading.Thread(target=initialize_predictor)
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
