"""
BioVanguard - Installation Verification Script
Run this to check if all dependencies are installed correctly
"""

import sys

def check_import(module_name, package_name=None):
    """Try to import a module and report status"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"[OK] {package_name:20s} - Installed")
        return True
    except ImportError as e:
        print(f"[MISSING] {package_name:20s} - Not found")
        return False

def main():
    print("\n" + "="*60)
    print("  BioVanguard - Installation Verification")
    print("="*60 + "\n")
    
    required_packages = [
        ("flask", "Flask"),
        ("flask_cors", "Flask-CORS"),
        ("werkzeug", "Werkzeug"),
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("rdkit", "RDKit"),
        ("Bio", "BioPython"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "XGBoost"),
        ("joblib", "Joblib"),
        ("matplotlib", "Matplotlib"),
        ("reportlab", "ReportLab"),
        ("PIL", "Pillow"),
    ]
    
    print("Checking required packages:\n")
    
    results = []
    for module, name in required_packages:
        results.append(check_import(module, name))
    
    print("\n" + "="*60)
    
    if all(results):
        print("[SUCCESS] All packages installed successfully!")
        print("\nYou can now run the server:")
        print("  python server.py --host 127.0.0.1 --port 8000")
    else:
        print("[WARNING] Some packages are missing. Please install them:")
        print("  pip install -r requirements.txt")
    
    print("="*60 + "\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check CUDA availability
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
    except:
        pass

if __name__ == "__main__":
    main()

