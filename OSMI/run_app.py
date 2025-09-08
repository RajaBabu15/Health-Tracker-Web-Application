#!/usr/bin/env python3
"""
Launch script for the OSMI Mental Health Risk Predictor Streamlit app
"""

import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'sklearn', 
        'joblib', 'plotly', 'lightgbm', 'xgboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'sklearn':
                try:
                    __import__('scikit-learn')
                except ImportError:
                    missing_packages.append('scikit-learn')
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        
        print("\n🔧 Install them using:")
        print("pip install -r requirements_app.txt")
        print("\nOr install individually:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_model_files():
    """Check if trained model files exist"""
    required_files = [
        "models/trained/best_calibrated_model.joblib",
        "models/trained/best_calibrated_model_metadata.json",
        "models/trained/best_calibrated_model_results.json"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required model files:")
        for file in missing_files:
            print(f"  - {file}")
        
        print("\n🔧 Please run the training script first:")
        print("python src/modeling/train_models.py")
        return False
    
    return True

def main():
    """Main function to launch the app"""
    print("🧠 OSMI Mental Health Risk Predictor")
    print("=" * 40)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    # Check model files
    print("🔍 Checking model files...")
    if not check_model_files():
        sys.exit(1)
    
    print("✅ All checks passed!")
    print("🚀 Launching Streamlit app...")
    print("=" * 40)
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.serverAddress", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

if __name__ == "__main__":
    main()
