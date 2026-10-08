# BioVanguard

Source code for the article **"BioVanguard: Graph-Attention and Ensemble Learning Platform for Interpretable Virtual Screening of MYC-MAX Interface Inhibitors"**.

BioVanguard predicts ligand binding affinity against the MYC bHLH-LZ domain using a weighted
ensemble of a graph-attention neural network (GNN) and XGBoost, retrieves a structural binding
pose by embedding similarity, and reports the result with 3D pose review, residue-level contact
analysis, a multi-filter drug-likeness scorecard and downloadable PDF/JSON reports.

A deployed instance is available at <https://myc-affinity.com>.

## Contents of this repository

```text
server.py                     Flask backend and prediction API
pose_retrieval.py             GNN-embedding based pose retrieval
web/                          Frontend dashboard (index.html, styles.css, app.js)
PredictionPipeline/
  Prediction.ipynb            Model training and evaluation notebook
  gnn_model_final.pt          Trained graph-attention network
  xgb_model_final.joblib      Trained XGBoost model
  xgb_scaler.joblib           Feature scaler for the XGBoost branch
  receptor.pdb / .pdbqt       MYC bHLH-LZ receptor structure
  ZINC*.sdf                   Four example ligands, including ZINC000015675944
requirements.txt              Python dependencies
QUICKSTART.md                 Short local setup guide
```

## Data archived on Zenodo

The large assets behind the platform are not stored in this repository because of file-size
limits. They are openly archived at Zenodo:

**DOI: [10.5281/zenodo.23239244](https://doi.org/10.5281/zenodo.23239244)**

| Asset | Description |
|---|---|
| `docking_outputs_101615_ligands.tar.gz` | AutoDock Vina output poses and binding energies for the full screening library; these are the docking-derived training labels |
| `graphs_cache-001.pkl` | Precomputed protein-ligand graphs used to train the GNN |
| `embedding_index.pt` | Graph embedding index used for cosine-similarity pose retrieval |
| `results3_final.zip` | Training curves, ROC/PR curves, confusion matrices and enrichment plots |

Affinity prediction runs without these files. Pose retrieval requires `embedding_index.pt` and
the pose library; without them the platform falls back to the input ligand coordinates.

## Installation

Python 3.10 is recommended.

**macOS / Linux**

```bash
python3.10 -m venv .venv310
source .venv310/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -3.10 -m venv .venv310
.\.venv310\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python verify_installation.py
```

## Running the platform

```bash
python server.py --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>, upload an SDF ligand or pick one of the bundled examples,
and run the prediction.

## Reproducing the reported results

1. Download `graphs_cache-001.pkl` from the Zenodo record above into the repository root.
2. Open `PredictionPipeline/Prediction.ipynb` and run it end to end. The notebook loads the
   precomputed graphs, trains the GNN and the XGBoost branch, and writes the evaluation
   figures and metrics reported in the article.
3. To regenerate the docking labels from scratch instead, use the AutoDock Vina outputs in
   `docking_outputs_101615_ligands.tar.gz`.

## Model stack

- **GNN** — graph-attention network over unified protein-ligand graphs with 18-dimensional
  atom-level features and spatial contact edges
- **XGBoost** — concatenated 2,048-bit Morgan fingerprints and nine RDKit physicochemical
  descriptors
- **Ensemble** — `0.6 x GNN + 0.4 x XGBoost`
- **Active threshold** — -7.0 kcal/mol

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/predict` | Affinity prediction from an uploaded SDF or an example ligand |
| `POST /api/download/pdf` | Full PDF report for a prediction payload |
| `POST /api/download/json` | Compact JSON export |
| `GET /api/health` | Server status, device and model-load state |

`POST /api/predict` returns `ensemble_score`, `gnn_pred`, `xgb_pred`, `properties`,
`drug_likeness`, `model_diagnostics`, `pose_meta`, `binding_residues`, `receptor_pdb` and
`ligand_pdb`.

## Citation

If you use this software, please cite the article:

> Öncü E, Arı Yuka S (2026) BioVanguard: Graph-Attention and Ensemble Learning Platform for Interpretable Virtual Screening of MYC-MAX Interface Inhibitors. *Molecular Diversity* (in press).

A machine-readable citation is provided in `CITATION.cff`.

## License

Released under the MIT License. See `LICENSE`.

Predictions are computational estimates intended as pre-laboratory decision support, and
should be validated experimentally before biological conclusions are drawn.

## Acknowledgements

The numerical calculations reported in the associated article were fully/partially performed
at TÜBİTAK ULAKBİM, High Performance and Grid Computing Center (TRUBA resources).

Built with RDKit, PyTorch, PyTorch Geometric, XGBoost, BioPython and AutoDock Vina.
