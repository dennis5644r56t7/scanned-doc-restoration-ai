"""
Run script for the Document Restoration AI Web App
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required = ['streamlit', 'opencv-python', 'numpy', 'pandas']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    return missing

def install_dependencies(missing_packages):
    """Install missing packages"""
    print(f"Installing missing packages: {missing_packages}")
    for package in missing_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    """Main function to run the app"""
    print("=" * 60)
    print("Document Restoration AI - Web Application")
    print("=" * 60)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {missing}")
        try:
            install_dependencies(missing)
        except Exception as e:
            print(f"Failed to install dependencies: {e}")
            print("Please install manually: pip install -r requirements.txt")
            return
    
    # Check if model exists, offer to train
    model_path = Path("models") / "deskew_model_best.h5"
    if not model_path.exists():
        print("\n⚠️  AI model not trained yet.")
        train = input("Do you want to train the model now? (y/n): ")
        if train.lower() == 'y':
            print("Training model...")
            try:
                from scripts.train_model import train_model
                if train_model():
                    print("✅ Model trained successfully!")
                else:
                    print("❌ Model training failed.")
            except Exception as e:
                print(f"❌ Training error: {e}")
        else:
            print("You can train the model later from the web app sidebar.")
    
    # Run the Streamlit app
    print("\n🚀 Starting web application...")
    print("Open your browser and go to: http://localhost:8501")
    print("Press Ctrl+C to stop the application\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main()