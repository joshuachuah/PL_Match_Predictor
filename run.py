#!/usr/bin/env python3
"""
Simple launcher script for the Premier League Match Predictor web app
"""

if __name__ == '__main__':
    try:
        from app import app
        print("Premier League Match Predictor")
        print("="*50)
        print("Starting web server...")
        print("📍 Open your browser and go to: http://localhost:5000")
        print("🔄 Press Ctrl+C to stop the server")
        print("="*50)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        print("📦 Please install requirements first:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
