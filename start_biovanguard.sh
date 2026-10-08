#!/bin/bash
# BioVanguard Startup Script for Linux/Mac
# This script activates the virtual environment and starts the server

echo ""
echo "============================================================"
echo "  BioVanguard - AI-Powered Drug Discovery Platform"
echo "  Target: c-Myc Protein"
echo "============================================================"
echo ""

# Check if virtual environment exists
if [ ! -f ".venv310/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run the following commands first:"
    echo ""
    echo "  python3.10 -m venv .venv310"
    echo "  source .venv310/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

# Activate virtual environment
echo "[1/3] Activating virtual environment..."
source .venv310/bin/activate

# Check if models exist
if [ ! -f "PredictionPipeline/gnn_model_final.pt" ]; then
    echo ""
    echo "[WARNING] Model files not found in PredictionPipeline/"
    echo "Please ensure all required files are present:"
    echo "  - gnn_model_final.pt"
    echo "  - xgb_model_final.joblib"
    echo "  - xgb_scaler.joblib"
    echo "  - receptor.pdb"
    echo "  - receptor.pdbqt"
    echo ""
    read -p "Press Enter to continue anyway..."
fi

# Create necessary directories
echo "[2/3] Creating directories..."
mkdir -p uploads
mkdir -p web_runs

# Start server
echo "[3/3] Starting BioVanguard server..."
echo ""
echo "Server will be available at: http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the server"
echo ""

python server.py --host 127.0.0.1 --port 8000

# If server stops
echo ""
echo "Server stopped."
