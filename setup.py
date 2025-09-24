#!/usr/bin/env python3
"""
Setup script for Premier League Match Predictor
Creates virtual environment and installs dependencies
"""

import os
import sys
import subprocess
import platform

def run_command(command, shell=False):
    """Run a command and return success status"""
    try:
        if isinstance(command, str) and not shell:
            command = command.split()
        
        result = subprocess.run(command, shell=shell, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    print("Premier League Match Predictor Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Determine OS
    is_windows = platform.system() == "Windows"
    python_cmd = "python" if is_windows else "python3"
    
    # Create virtual environment
    print("\n📦 Creating virtual environment...")
    success, output = run_command(f"{python_cmd} -m venv venv")
    
    if not success:
        print(f"❌ Failed to create virtual environment: {output}")
        sys.exit(1)
    
    print("✅ Virtual environment created")
    
    # Determine activation command and pip path
    if is_windows:
        pip_cmd = os.path.join("venv", "Scripts", "pip")
        activate_cmd = "venv\\Scripts\\activate"
    else:
        pip_cmd = os.path.join("venv", "bin", "pip")
        activate_cmd = "source venv/bin/activate"
    
    # Install requirements
    print("\n📥 Installing dependencies...")
    success, output = run_command(f"{pip_cmd} install -r requirements.txt")
    
    if not success:
        print(f"❌ Failed to install dependencies: {output}")
        print("💡 Try running manually:")
        print(f"   {activate_cmd}")
        print(f"   pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ Dependencies installed successfully")
    
    # Success message
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To run the application:")
    print(f"   {activate_cmd}")
    print("   python run.py")
    print("\n🌐 Then open: http://localhost:5000")
    print("\n💡 To deactivate virtual environment when done:")
    print("   deactivate")

if __name__ == "__main__":
    main()
