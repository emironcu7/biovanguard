"""
Test script for Drug-Likeness Scorecard feature
Tests the backend calculation functions
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from server import compute_molecular_properties, compute_drug_likeness_scorecard

def test_drug_likeness():
    """Test drug-likeness calculation with a sample molecule"""
    
    print("="*60)
    print("Drug-Likeness Scorecard Test")
    print("="*60)
    
    # Test with aspirin (simple drug-like molecule)
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(aspirin_smiles)
    
    if mol is None:
        print("[FAIL] Failed to create molecule from SMILES")
        return False
    
    print(f"\n[OK] Test Molecule: Aspirin")
    print(f"  SMILES: {aspirin_smiles}")
    
    # Compute properties
    print("\n1. Computing molecular properties...")
    try:
        properties = compute_molecular_properties(mol)
        print(f"   [OK] Properties calculated")
        print(f"     - MW: {properties['mw']:.2f}")
        print(f"     - LogP: {properties['logp']:.2f}")
        print(f"     - HBD: {properties['hbd']}")
        print(f"     - HBA: {properties['hba']}")
        print(f"     - TPSA: {properties['tpsa']:.2f}")
        print(f"     - QED: {properties['qed']:.3f}")
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        return False
    
    # Compute drug-likeness scorecard
    print("\n2. Computing drug-likeness scorecard...")
    try:
        drug_likeness = compute_drug_likeness_scorecard(properties)
        print(f"   [OK] Scorecard calculated")
        print(f"\n   Overall Assessment: {drug_likeness['assessment']}")
        print(f"   Overall Score: {drug_likeness['overall_score']:.3f}")
        print(f"   Assessment Class: {drug_likeness['assessment_class']}")
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Display metrics
    print("\n3. Individual Metrics:")
    metrics = drug_likeness['metrics']
    for key, metric in metrics.items():
        status = "[PASS]" if metric['passed'] else "[FAIL]"
        print(f"   {status} {metric['name']}: {metric['details']}")
    
    # Display radar data
    print("\n4. Radar Chart Data:")
    radar = drug_likeness['radar_data']
    print(f"   Labels: {radar['labels']}")
    print(f"   Values: {[f'{v:.1f}%' for v in radar['values']]}")
    
    # Test with a non-drug-like molecule (too large)
    print("\n" + "="*60)
    print("Testing with non-drug-like molecule...")
    print("="*60)
    
    # Large peptide-like molecule
    large_smiles = "CC(C)C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCNC(N)=N)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc2ccccc2)N)C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](Cc3c[nH]c4ccccc34)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)NCC(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CCCNC(N)=N)C(=O)N[C@@H](Cc5ccccc5)C(=O)N[C@@H](Cc6ccccc6)C(=O)N[C@@H](Cc7c[nH]c8ccccc78)C(=O)N[C@@H]([C@@H](C)O)C(=O)N9CCC[C@H]9C(=O)N[C@@H](CCSC)C(O)=O"
    mol2 = Chem.MolFromSmiles(large_smiles)
    
    if mol2:
        props2 = compute_molecular_properties(mol2)
        dl2 = compute_drug_likeness_scorecard(props2)
        print(f"\n[OK] Large Molecule Test")
        print(f"  MW: {props2['mw']:.2f}")
        print(f"  Assessment: {dl2['assessment']}")
        print(f"  Overall Score: {dl2['overall_score']:.3f}")
    
    print("\n" + "="*60)
    print("[OK] All tests completed successfully!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_drug_likeness()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

