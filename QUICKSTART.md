# BioVanguard Quick Start

This guide starts BioVanguard locally with the included Flask dashboard.

## 1. Create The Environment

Windows:

```powershell
py -3.10 -m venv .venv310
.\.venv310\Scripts\activate
```

Linux or macOS:

```bash
python3.10 -m venv .venv310
source .venv310/bin/activate
```

## 2. Install Dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 3. Check Required Assets

For full functionality, these files and folders should exist:

```text
PredictionPipeline/gnn_model_final.pt
PredictionPipeline/xgb_model_final.joblib
PredictionPipeline/xgb_scaler.joblib
PredictionPipeline/receptor.pdb
PredictionPipeline/receptor.pdbqt
graphs_cache-001.pkl
102kOutFiles/
```

The application can still start without the pose library, but 3D pose retrieval will be limited.

## 4. Start The Server

Windows:

```powershell
.\start_biovanguard.bat
```

Manual start:

```powershell
python server.py --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## 5. Run A Prediction

1. Select an example ligand or upload an SDF file.
2. Click `Predict Affinity`.
3. Review the ensemble score, model diagnostics, 3D pose, binding residues, and drug-likeness scorecard.
4. Download the PDF or JSON report if needed.

## Common Issues

### Port 8000 Is Already In Use

Use another port:

```powershell
python server.py --host 127.0.0.1 --port 8080
```

### RDKit Installation Fails

Conda is often the easiest fallback:

```bash
conda install -c conda-forge rdkit
```

### First Prediction Is Slow

The first pose-retrieval run may build an embedding index. Later predictions are faster during the same server session.

### GitHub Upload Notes

Use `.gitignore` for generated outputs and virtual environments. Use Git LFS or an external artifact link for large model and pose-library files. The graph cache is larger than 2 GB, so an external artifact or split archive may be safer than a direct repository upload.
