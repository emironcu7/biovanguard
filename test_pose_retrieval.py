"""
Test script to verify pose retrieval system is working
"""

import sys
from pathlib import Path

# Check if required files exist
BASE_DIR = Path(__file__).parent
GRAPH_CACHE = BASE_DIR / 'graphs_cache-001.pkl'
LIGAND_DIR = BASE_DIR / '102kOutFiles'
PIPELINE_DIR = BASE_DIR / 'PredictionPipeline'

print("="*60)
print("BioVanguard Pose Retrieval System - Verification")
print("="*60)

print("\n1. Checking required files...")

if GRAPH_CACHE.exists():
    size_mb = GRAPH_CACHE.stat().st_size / (1024*1024)
    print(f"   [OK] Graph cache found: {GRAPH_CACHE}")
    print(f"     Size: {size_mb:.1f} MB")
else:
    print(f"   [MISSING] Graph cache NOT found: {GRAPH_CACHE}")
    print("     WARNING: Pose retrieval will not work!")

if LIGAND_DIR.exists():
    pdbqt_files = list(LIGAND_DIR.glob("*.pdbqt"))
    print(f"   [OK] Ligand directory found: {LIGAND_DIR}")
    print(f"     Contains {len(pdbqt_files)} PDBQT files")
else:
    print(f"   [MISSING] Ligand directory NOT found: {LIGAND_DIR}")
    print("     WARNING: Pose retrieval will not work!")

print("\n2. Checking model files...")
gnn_model = PIPELINE_DIR / 'gnn_model_final.pt'
xgb_model = PIPELINE_DIR / 'xgb_model_final.joblib'
receptor = PIPELINE_DIR / 'receptor.pdb'

for file in [gnn_model, xgb_model, receptor]:
    if file.exists():
        print(f"   [OK] {file.name}")
    else:
        print(f"   [MISSING] {file.name} NOT found")

print("\n3. Testing pose retrieval import...")
try:
    from pose_retrieval import PoseRetrievalEngine
    print("   [OK] PoseRetrievalEngine imported successfully")
except Exception as e:
    print(f"   [FAIL] Failed to import: {e}")
    sys.exit(1)

print("\n4. Testing server integration...")
try:
    import torch
    from server import ProteinLigandGAT, DEVICE
    
    print(f"   [OK] PyTorch device: {DEVICE}")
    
    # Try to load GNN model
    gnn = ProteinLigandGAT().to(DEVICE)
    gnn.load_state_dict(torch.load(gnn_model, map_location=DEVICE))
    print("   [OK] GNN model loaded")
    
    # Try to initialize pose engine
    if GRAPH_CACHE.exists() and LIGAND_DIR.exists():
        print("\n   Initializing PoseRetrievalEngine...")
        print("   (This will take ~84 seconds to load training graphs)")
        
        import time
        t0 = time.time()
        
        pose_engine = PoseRetrievalEngine(
            gnn_model=gnn,
            graph_cache_path=GRAPH_CACHE,
            ligand_dir=LIGAND_DIR,
            device=DEVICE,
            top_k=5
        )
        
        elapsed = time.time() - t0
        print(f"\n   [OK] PoseRetrievalEngine initialized in {elapsed:.1f}s")
        print(f"   [OK] Loaded {len(pose_engine.train_graphs):,} training graphs")
        print(f"   [OK] Training filenames: {len(pose_engine.train_filenames)}")
        
        print("\n" + "="*60)
        print("SUCCESS: Pose retrieval system is fully functional!")
        print("="*60)
        print("\nWhen you make a prediction:")
        print("  - First prediction: 30-60 seconds (builds embedding index)")
        print("  - Subsequent predictions: 5-15 seconds (uses cached embeddings)")
        print("  - 3D viewer will show aligned ligand pose")
        print("  - Console will show similarity scores and RMSD values")
        
    else:
        print("\n   [WARNING] Required files missing")
        print("   Pose retrieval will be disabled")
        print("   Predictions will use original ligand coordinates")
        
except Exception as e:
    print(f"   [FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)

