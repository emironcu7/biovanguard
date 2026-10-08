"""
BioVanguard - Flask Backend Server
Handles ligand upload, prediction, and report generation
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from Bio.PDB import PDBParser

# Import pose retrieval engine
from pose_retrieval import PoseRetrievalEngine

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
OUTPUT_FOLDER = BASE_DIR / 'web_runs'
PIPELINE_DIR = BASE_DIR / 'PredictionPipeline'
LIGAND_DIR = BASE_DIR / '102kOutFiles'
GRAPH_CACHE = BASE_DIR / 'graphs_cache-001.pkl'

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'sdf'}
ACTIVE_THRESHOLD = -7.0
GNN_WEIGHT = 0.6
XGB_WEIGHT = 1.0 - GNN_WEIGHT
BINDING_RESIDUE_CUTOFF = 4.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_STACK = {
    'gnn_model': 'gnn_model_final.pt',
    'xgb_model': 'xgb_model_final.joblib',
    'xgb_scaler': 'xgb_scaler.joblib',
    'ensemble': 'weighted_best_models',
    'ensemble_weights': {
        'gnn': GNN_WEIGHT,
        'xgboost': XGB_WEIGHT,
    },
    'active_threshold': ACTIVE_THRESHOLD,
}

# Flask app
app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ============================================================================
# GNN Model Architecture
# ============================================================================

class ProteinLigandGAT(nn.Module):
    def __init__(self, node_dim=18, edge_dim=2, hidden=128, heads=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden)
        self.conv1 = GATConv(hidden, hidden//heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.conv2 = GATConv(hidden, hidden//heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.conv3 = GATConv(hidden, hidden//heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, data):
        x, ei, ea, b = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.input_proj(x))
        x = self.bn1(F.elu(self.conv1(x, ei, ea)) + x); x = self.dropout(x)
        x = self.bn2(F.elu(self.conv2(x, ei, ea)) + x); x = self.dropout(x)
        x = self.bn3(F.elu(self.conv3(x, ei, ea)) + x); x = self.dropout(x)
        x = global_mean_pool(x, b)
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        return self.fc3(x).squeeze(-1)

    def get_embedding(self, data):
        x, ei, ea, b = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.input_proj(x))
        x = self.bn1(F.elu(self.conv1(x, ei, ea)) + x); x = self.dropout(x)
        x = self.bn2(F.elu(self.conv2(x, ei, ea)) + x); x = self.dropout(x)
        x = self.bn3(F.elu(self.conv3(x, ei, ea)) + x); x = self.dropout(x)
        return global_mean_pool(x, b)

# ============================================================================
# Graph Building Functions
# ============================================================================

ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53]

def get_coords(mol):
    c = mol.GetConformer()
    return np.array([list(c.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

def atom_features(atom, is_protein=False):
    oh = [int(atom.GetAtomicNum() == t) for t in ATOM_TYPES]
    oh.append(int(atom.GetAtomicNum() not in ATOM_TYPES))
    return oh + [
        atom.GetTotalValence()/8., atom.GetFormalCharge()/2.,
        float(atom.GetIsAromatic()), float(atom.IsInRing()),
        float(atom.GetTotalNumHs()>0),
        float(atom.GetAtomicNum() in [7,8]),
        float(is_protein), 0.
    ]

def extract_pocket_for_graph(lig_coords, receptor_pdb, cutoff=10.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", receptor_pdb)
    pc, pf = [], []
    for model in structure:
        for chain in model:
            for res in chain:
                rc, rf = [], []
                for atom in res:
                    c = np.array(atom.get_vector().get_array())
                    f = [0]*len(ATOM_TYPES)+[0,0.,0.,0.,0.,0.,0.,1.,0.]
                    rc.append(c); rf.append(f)
                if rc:
                    md = min(np.linalg.norm(lig_coords-c,axis=1).min() for c in rc)
                    if md <= cutoff:
                        pc.extend(rc); pf.extend(rf)
    return np.array(pc), np.array(pf)

def build_graph_from_mol(mol, receptor_pdb):
    lc = get_coords(mol)
    pc, pf = extract_pocket_for_graph(lc, receptor_pdb)
    lf = [atom_features(a, False) for a in mol.GetAtoms()]
    nl = len(lf)
    x = torch.tensor(lf+list(pf), dtype=torch.float)
    es, ed, ea = [], [], []
    
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d = float(np.linalg.norm(lc[i]-lc[j]))
        for s, t in [(i,j),(j,i)]:
            es.append(s); ed.append(t); ea.append([d,0.])
    
    for li in range(nl):
        for pi in range(len(pc)):
            d = float(np.linalg.norm(lc[li]-pc[pi]))
            if d <= 5.:
                pg = pi+nl
                for s, t in [(li,pg),(pg,li)]:
                    es.append(s); ed.append(t); ea.append([d,1.])
    
    if not es:
        return None
    
    return Data(
        x=x,
        edge_index=torch.tensor([es,ed], dtype=torch.long),
        edge_attr=torch.tensor(ea, dtype=torch.float)
    )

# ============================================================================
# Molecular Property Calculation
# ============================================================================

def compute_molecular_properties(mol):
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    arom = Descriptors.NumAromaticRings(mol)
    heavy = Descriptors.HeavyAtomCount(mol)
    csp3 = Descriptors.FractionCSP3(mol)
    qed_v = QED.qed(mol)
    
    # Lipinski
    lip_passes = sum([
        mw <= 500,
        logp <= 5,
        hbd <= 5,
        hba <= 10
    ])
    
    # PAINS and Brenk
    params_pains = FilterCatalogParams()
    params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params_brenk = FilterCatalogParams()
    params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    
    return {
        'mw': mw,
        'logp': logp,
        'hbd': hbd,
        'hba': hba,
        'tpsa': tpsa,
        'rotb': rotb,
        'arom_rings': arom,
        'heavy_atoms': heavy,
        'frac_csp3': csp3,
        'qed': qed_v,
        'lip_passes': lip_passes,
        'veber': (tpsa <= 140) and (rotb <= 10),
        'ghose': (160 <= mw <= 480) and (-0.4 <= logp <= 5.6),
        'egan': (tpsa <= 131.6) and (logp <= 5.88),
        'muegge': (200 <= mw <= 600) and (-2 <= logp <= 5) and (tpsa <= 150),
        'pains_clean': not FilterCatalog(params_pains).HasMatch(mol),
        'brenk_clean': not FilterCatalog(params_brenk).HasMatch(mol),
    }


def compute_drug_likeness_scorecard(properties):
    """
    Generate a drug-likeness scorecard with radar chart data.
    Returns assessment scores and interpretation for various drug-likeness rules.
    """
    # Extract properties
    mw = properties.get('mw', 0)
    logp = properties.get('logp', 0)
    hbd = properties.get('hbd', 0)
    hba = properties.get('hba', 0)
    tpsa = properties.get('tpsa', 0)
    rotb = properties.get('rotb', 0)
    qed = properties.get('qed', 0)
    
    # Lipinski's Rule of Five (normalized 0-1)
    lipinski_score = 0
    lipinski_violations = 0
    if mw <= 500: lipinski_score += 0.25
    else: lipinski_violations += 1
    if logp <= 5: lipinski_score += 0.25
    else: lipinski_violations += 1
    if hbd <= 5: lipinski_score += 0.25
    else: lipinski_violations += 1
    if hba <= 10: lipinski_score += 0.25
    else: lipinski_violations += 1
    
    # Veber's Rule (oral bioavailability)
    veber_score = 0
    if tpsa <= 140: veber_score += 0.5
    if rotb <= 10: veber_score += 0.5
    
    # Ghose filter
    ghose_score = 0
    if 160 <= mw <= 480: ghose_score += 0.5
    if -0.4 <= logp <= 5.6: ghose_score += 0.5
    
    # PAINS and Brenk (from properties)
    pains_score = 1.0 if properties.get('pains_clean', True) else 0.0
    brenk_score = 1.0 if properties.get('brenk_clean', True) else 0.0
    
    # QED is already 0-1
    qed_score = qed
    
    # Overall assessment
    scores = {
        'lipinski': lipinski_score,
        'veber': veber_score,
        'ghose': ghose_score,
        'pains': pains_score,
        'brenk': brenk_score,
        'qed': qed_score
    }
    
    # Calculate average score
    avg_score = sum(scores.values()) / len(scores)
    
    # Determine overall assessment
    if avg_score >= 0.85:
        assessment = "Drug-like"
        assessment_class = "excellent"
    elif avg_score >= 0.70:
        assessment = "Promising"
        assessment_class = "good"
    elif avg_score >= 0.50:
        assessment = "Needs Review"
        assessment_class = "moderate"
    else:
        assessment = "Poor Drug-likeness"
        assessment_class = "poor"
    
    # Detailed metrics for scorecard
    metrics = {
        'lipinski': {
            'name': 'Lipinski (Ro5)',
            'score': lipinski_score,
            'violations': lipinski_violations,
            'passed': lipinski_violations <= 1,
            'details': f"{4 - lipinski_violations}/4 criteria met"
        },
        'veber': {
            'name': 'Veber',
            'score': veber_score,
            'passed': veber_score == 1.0,
            'details': f"TPSA: {tpsa:.1f} A^2, RotB: {rotb}"
        },
        'ghose': {
            'name': 'Ghose',
            'score': ghose_score,
            'passed': ghose_score == 1.0,
            'details': f"MW: {mw:.1f}, LogP: {logp:.2f}"
        },
        'pains': {
            'name': 'PAINS',
            'score': pains_score,
            'passed': pains_score == 1.0,
            'details': 'Clean' if pains_score == 1.0 else 'Contains PAINS'
        },
        'brenk': {
            'name': 'Brenk',
            'score': brenk_score,
            'passed': brenk_score == 1.0,
            'details': 'Clean' if brenk_score == 1.0 else 'Contains unwanted groups'
        },
        'qed': {
            'name': 'QED',
            'score': qed_score,
            'passed': qed_score >= 0.5,
            'details': f"Score: {qed_score:.3f}"
        }
    }
    
    # Radar chart data (normalized 0-100 for visualization)
    radar_data = {
        'labels': ['Lipinski', 'Veber', 'Ghose', 'PAINS', 'Brenk', 'QED'],
        'values': [
            lipinski_score * 100,
            veber_score * 100,
            ghose_score * 100,
            pains_score * 100,
            brenk_score * 100,
            qed_score * 100
        ]
    }
    
    return {
        'overall_score': round(avg_score, 3),
        'assessment': assessment,
        'assessment_class': assessment_class,
        'metrics': metrics,
        'radar_data': radar_data
    }


def compute_ligand_metadata(mol, filename=None):
    """Return compact ligand identifiers for reports and exports."""
    metadata = {
        'filename': filename or 'Ligand',
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'canonical_smiles': Chem.MolToSmiles(mol, canonical=True),
        'atom_count': mol.GetNumAtoms(),
        'bond_count': mol.GetNumBonds(),
        'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
        'ring_count': Descriptors.RingCount(mol),
    }
    if mol.GetNumConformers():
        coords = get_coords(mol)
        metadata['centroid'] = [float(v) for v in coords.mean(axis=0)]
    return metadata


def generate_ligand_2d_png(mol, size=(520, 360)):
    """Generate a base64 PNG 2D depiction for reports."""
    try:
        import base64
        from io import BytesIO
        from rdkit.Chem import Draw

        mol_2d = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_2d)
        drawer_options = Draw.MolDrawOptions()
        drawer_options.addAtomIndices = False
        drawer_options.bondLineWidth = 2
        drawer_options.fixedBondLength = 30
        drawer_options.padding = 0.08
        image = Draw.MolToImage(mol_2d, size=size, options=drawer_options)
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('ascii')
    except Exception as e:
        print(f"2D ligand depiction error: {e}")
        return None

# ============================================================================
# Protein-Ligand Interaction Analysis
# ============================================================================

def analyze_protein_ligand_interactions(mol, receptor_pdb, cutoff=4.5):
    """
    Analyze protein-ligand interactions and classify them by type.
    
    Returns a list of residues with interaction types:
    - Hydrogen bonds (H-bond donor/acceptor)
    - Hydrophobic interactions
    - Pi-stacking
    - Salt bridges
    - Van der Waals contacts
    """
    from Bio.PDB import PDBParser, NeighborSearch, Selection
    from rdkit.Chem import Lipinski
    
    # Get ligand coordinates and properties
    conf = mol.GetConformer()
    lig_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    
    # Parse receptor
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", receptor_pdb)
    
    # Get all protein atoms
    protein_atoms = list(Selection.unfold_entities(structure, 'A'))
    
    # Build neighbor search
    ns = NeighborSearch(protein_atoms)
    
    # Store residue interactions
    residue_interactions = {}
    
    # Analyze each ligand atom
    for lig_idx in range(mol.GetNumAtoms()):
        lig_atom = mol.GetAtomWithIdx(lig_idx)
        lig_pos = lig_coords[lig_idx]
        
        # Find nearby protein atoms
        nearby_atoms = ns.search(lig_pos, cutoff, level='A')
        
        for prot_atom in nearby_atoms:
            residue = prot_atom.get_parent()
            res_id = f"{residue.get_resname()}{residue.get_id()[1]}:{residue.get_parent().get_id()}"
            
            if res_id not in residue_interactions:
                residue_interactions[res_id] = {
                    'label': res_id,
                    'resname': residue.get_resname(),
                    'resnum': residue.get_id()[1],
                    'chain': residue.get_parent().get_id(),
                    'min_distance': float('inf'),
                    'interactions': set(),
                    'atom_contacts': []
                }
            
            # Calculate distance
            prot_pos = np.array(prot_atom.get_vector().get_array())
            distance = float(np.linalg.norm(lig_pos - prot_pos))
            
            if distance < residue_interactions[res_id]['min_distance']:
                residue_interactions[res_id]['min_distance'] = distance
            
            # Store contact
            residue_interactions[res_id]['atom_contacts'].append({
                'lig_atom': lig_atom.GetSymbol(),
                'prot_atom': prot_atom.element,
                'distance': distance
            })
            
            # Classify interaction type
            lig_element = lig_atom.GetSymbol()
            prot_element = prot_atom.element
            
            # Hydrogen bonds (distance < 3.5 Å)
            if distance < 3.5:
                # Donor-acceptor pairs
                h_bond_donors = {'N', 'O'}
                h_bond_acceptors = {'N', 'O', 'S'}
                
                # Check if ligand is donor and protein is acceptor
                if lig_element in h_bond_donors and prot_element in h_bond_acceptors:
                    if Lipinski.NumHDonors(Chem.MolFromSmarts(f'[{lig_element}]')) > 0:
                        residue_interactions[res_id]['interactions'].add('H-Bond')
                
                # Check if protein is donor and ligand is acceptor
                elif prot_element in h_bond_donors and lig_element in h_bond_acceptors:
                    if Lipinski.NumHAcceptors(Chem.MolFromSmarts(f'[{lig_element}]')) > 0:
                        residue_interactions[res_id]['interactions'].add('H-Bond')
            
            # Hydrophobic interactions (C-C contacts, distance < 4.0 Å)
            if distance < 4.0:
                if lig_element == 'C' and prot_element == 'C':
                    # Check if carbons are non-polar
                    if not lig_atom.GetIsAromatic() and prot_atom.name.startswith('C'):
                        residue_interactions[res_id]['interactions'].add('Hydrophobic')
            
            # Pi-stacking (aromatic-aromatic, distance < 4.5 Å)
            if distance < 4.5:
                if lig_atom.GetIsAromatic():
                    aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}
                    if residue.get_resname() in aromatic_residues:
                        residue_interactions[res_id]['interactions'].add('Pi-Stacking')
            
            # Salt bridges (charged interactions, distance < 4.0 Å)
            if distance < 4.0:
                charged_residues = {'ARG', 'LYS', 'ASP', 'GLU'}
                if residue.get_resname() in charged_residues:
                    lig_charge = lig_atom.GetFormalCharge()
                    if lig_charge != 0:
                        residue_interactions[res_id]['interactions'].add('Salt Bridge')
            
            # Van der Waals (all other close contacts)
            if distance < cutoff:
                if not residue_interactions[res_id]['interactions']:
                    residue_interactions[res_id]['interactions'].add('Van der Waals')
    
    # Convert to list and sort by distance
    result = []
    for res_id, data in residue_interactions.items():
        result.append({
            'label': data['label'],
            'resname': data['resname'],
            'resnum': data['resnum'],
            'chain': data['chain'],
            'min_distance': round(data['min_distance'], 2),
            'interactions': sorted(list(data['interactions'])),
            'contact_count': len(data['atom_contacts'])
        })
    
    result.sort(key=lambda x: x['min_distance'])
    return result


# ============================================================================
# 3D Visualization Generation
# ============================================================================

def generate_3d_viewer(mol, ligand_path):
    """Generate 3Dmol.js viewer HTML for protein-ligand complex"""
    try:
        # Read receptor PDB
        receptor_pdb = str(PIPELINE_DIR / 'receptor.pdb')
        with open(receptor_pdb, 'r') as f:
            receptor_pdb_data = f.read()
        
        # Convert molecule to PDB format for visualization
        ligand_pdb = Chem.MolToPDBBlock(mol)
        
        # Escape backticks and special characters for JavaScript
        receptor_pdb_data = receptor_pdb_data.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
        ligand_pdb = ligand_pdb.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
        
        # Generate 3Dmol.js viewer HTML
        viewer_html = f"""
<div id="container-3dmol" style="height: 500px; width: 100%; position: relative; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"></div>
<script>
setTimeout(function() {{
    try {{
        var element = document.getElementById('container-3dmol');
        if (!element) {{
            console.error('3D viewer container not found');
            return;
        }}
        
        var config = {{ backgroundColor: '#1a1a2e' }};
        var viewer = $3Dmol.createViewer(element, config);
        
        // Add receptor (protein)
        var receptorData = `{receptor_pdb_data}`;
        viewer.addModel(receptorData, "pdb");
        viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
        
        // Add ligand
        var ligandData = `{ligand_pdb}`;
        viewer.addModel(ligandData, "pdb");
        viewer.setStyle({{model: -1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.2}}}});
        
        // Center and zoom
        viewer.zoomTo();
        viewer.render();
        viewer.zoom(0.8, 1000);
        
        console.log('3D viewer initialized successfully');
    }} catch(e) {{
        console.error('3D viewer error:', e);
    }}
}}, 100);
</script>
"""
        
        return viewer_html
        
    except Exception as e:
        print(f"3D viewer generation error: {e}")
        import traceback
        traceback.print_exc()
        return f"""
<div style="text-align: center; padding: 2rem; color: var(--text-muted);">
    <p>3D visualization could not be generated</p>
    <p style="font-size: 0.9rem;">Error: {str(e)}</p>
</div>
"""

# ============================================================================
# Structured Visualization and Diagnostics
# ============================================================================

def analyze_protein_ligand_interactions(mol, receptor_pdb, cutoff=4.5):
    """Analyze receptor contacts around a posed ligand and classify interaction types."""
    from Bio.PDB import NeighborSearch, Selection

    donor_residue_atoms = {
        'ARG': {'NE', 'NH1', 'NH2'},
        'ASN': {'ND2'},
        'GLN': {'NE2'},
        'HIS': {'ND1', 'NE2'},
        'LYS': {'NZ'},
        'SER': {'OG'},
        'THR': {'OG1'},
        'TRP': {'NE1'},
        'TYR': {'OH'},
        'CYS': {'SG'},
    }
    acceptor_residue_atoms = {
        'ASP': {'OD1', 'OD2'},
        'ASN': {'OD1'},
        'CYS': {'SG'},
        'GLU': {'OE1', 'OE2'},
        'GLN': {'OE1'},
        'HIS': {'ND1', 'NE2'},
        'MET': {'SD'},
        'SER': {'OG'},
        'THR': {'OG1'},
        'TYR': {'OH'},
    }
    aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}
    basic_residue_atoms = {
        'ARG': {'NE', 'NH1', 'NH2'},
        'HIS': {'ND1', 'NE2'},
        'LYS': {'NZ'},
    }
    acidic_residue_atoms = {
        'ASP': {'OD1', 'OD2'},
        'GLU': {'OE1', 'OE2'},
    }
    hydrophobic_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'TYR', 'PRO', 'CYS'}
    backbone_atoms = {'N', 'CA', 'C', 'O', 'OXT'}
    halogens = {'F', 'CL', 'BR', 'I'}

    def safe_element(atom):
        element = (getattr(atom, 'element', '') or '').strip().upper()
        if element:
            return element
        return ''.join(ch for ch in atom.get_name().strip().upper() if ch.isalpha())[:2]

    def ligand_has_hydrogen(atom):
        try:
            return atom.GetTotalNumHs() > 0
        except Exception:
            return atom.GetNumExplicitHs() > 0

    def ligand_is_donor(atom):
        return atom.GetSymbol().upper() in {'N', 'O', 'S'} and ligand_has_hydrogen(atom)

    def ligand_is_acceptor(atom):
        element = atom.GetSymbol().upper()
        if element not in {'N', 'O', 'S'}:
            return False
        if atom.GetFormalCharge() > 0:
            return False
        if element == 'N' and atom.GetIsAromatic() and ligand_has_hydrogen(atom):
            return False
        return True

    def protein_is_donor(resname, atom_name, element):
        if atom_name == 'N':
            return True
        return element in {'N', 'O', 'S'} and atom_name in donor_residue_atoms.get(resname, set())

    def protein_is_acceptor(resname, atom_name, element):
        if atom_name in {'O', 'OXT'}:
            return True
        return element in {'N', 'O', 'S'} and atom_name in acceptor_residue_atoms.get(resname, set())

    def protein_is_hydrophobic(resname, atom_name, element):
        if resname not in hydrophobic_residues:
            return False
        if atom_name in backbone_atoms:
            return False
        return element in {'C', 'S'}

    def ligand_is_hydrophobic(atom):
        element = atom.GetSymbol().upper()
        if element in halogens:
            return True
        if element != 'C':
            return False
        return atom.GetFormalCharge() == 0

    def ordered_interactions(interactions):
        priority = {
            'Salt Bridge': 0,
            'H-Bond': 1,
            'Halogen Bond': 2,
            'Pi-Stacking': 3,
            'Cation-Pi': 4,
            'Hydrophobic': 5,
            'Van Der Waals': 6,
        }
        return sorted(interactions, key=lambda name: (priority.get(name, 99), name))

    conf = mol.GetConformer()
    lig_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", receptor_pdb)
    protein_atoms = [
        atom for atom in Selection.unfold_entities(structure, 'A')
        if safe_element(atom) != 'H' and atom.get_parent().get_resname().strip() not in {'HOH', 'WAT'}
    ]
    ns = NeighborSearch(protein_atoms)
    residue_interactions = {}

    for lig_idx in range(mol.GetNumAtoms()):
        lig_atom = mol.GetAtomWithIdx(lig_idx)
        lig_pos = lig_coords[lig_idx]

        for prot_atom in ns.search(lig_pos, cutoff, level='A'):
            residue = prot_atom.get_parent()
            resname = residue.get_resname().strip()
            if resname in {'HOH', 'WAT'}:
                continue

            chain_id = residue.get_parent().get_id()
            resnum = residue.get_id()[1]
            insertion_code = residue.get_id()[2].strip()
            res_label = f"{resname}{resnum}{insertion_code}:{chain_id}"

            if res_label not in residue_interactions:
                residue_interactions[res_label] = {
                    'label': res_label,
                    'resname': resname,
                    'res_name': resname,
                    'resnum': resnum,
                    'res_id': resnum,
                    'chain': chain_id,
                    'insertion_code': insertion_code,
                    'min_distance': float('inf'),
                    'interactions': set(),
                    'atom_contacts': [],
                }

            prot_pos = np.array(prot_atom.get_vector().get_array())
            distance = float(np.linalg.norm(lig_pos - prot_pos))
            if distance < residue_interactions[res_label]['min_distance']:
                residue_interactions[res_label]['min_distance'] = distance

            lig_element = lig_atom.GetSymbol().upper()
            prot_element = safe_element(prot_atom)
            atom_name = prot_atom.get_name().strip().upper()
            contact_types = set()

            if distance <= 3.6 and (
                (ligand_is_donor(lig_atom) and protein_is_acceptor(resname, atom_name, prot_element))
                or (ligand_is_acceptor(lig_atom) and protein_is_donor(resname, atom_name, prot_element))
            ):
                contact_types.add('H-Bond')

            if distance <= 4.0:
                if lig_atom.GetFormalCharge() > 0 and atom_name in acidic_residue_atoms.get(resname, set()):
                    contact_types.add('Salt Bridge')
                if lig_atom.GetFormalCharge() < 0 and atom_name in basic_residue_atoms.get(resname, set()):
                    contact_types.add('Salt Bridge')

            if distance <= 4.2 and ligand_is_hydrophobic(lig_atom) and protein_is_hydrophobic(resname, atom_name, prot_element):
                contact_types.add('Hydrophobic')

            if distance <= 5.0 and lig_atom.GetIsAromatic() and resname in aromatic_residues:
                contact_types.add('Pi-Stacking')

            if distance <= 5.0 and lig_atom.GetIsAromatic() and atom_name in basic_residue_atoms.get(resname, set()):
                contact_types.add('Cation-Pi')

            if distance <= 3.8 and lig_element in halogens and protein_is_acceptor(resname, atom_name, prot_element):
                contact_types.add('Halogen Bond')

            if distance <= cutoff and not contact_types:
                contact_types.add('Van Der Waals')

            residue_interactions[res_label]['interactions'].update(contact_types)
            residue_interactions[res_label]['atom_contacts'].append({
                'lig_atom': lig_atom.GetSymbol(),
                'lig_atom_index': lig_idx,
                'prot_atom': atom_name,
                'prot_element': prot_element,
                'distance': round(distance, 2),
                'types': ordered_interactions(contact_types),
            })

    result = []
    for data in residue_interactions.values():
        closest_contacts = sorted(data['atom_contacts'], key=lambda x: x['distance'])[:6]
        result.append({
            'label': data['label'],
            'resname': data['resname'],
            'res_name': data['res_name'],
            'resnum': data['resnum'],
            'res_id': data['res_id'],
            'chain': data['chain'],
            'insertion_code': data['insertion_code'],
            'min_distance': round(data['min_distance'], 2),
            'interactions': ordered_interactions(data['interactions']),
            'contact_count': len(data['atom_contacts']),
            'near_atom_pairs': len(data['atom_contacts']),
            'contacts': closest_contacts,
        })

    result.sort(key=lambda x: x['min_distance'])
    return result


def extract_binding_residues(mol, receptor_pdb, cutoff=BINDING_RESIDUE_CUTOFF, limit=24):
    """
    Find receptor residues within cutoff Angstroms of the posed ligand.
    Now includes interaction type analysis (H-bonds, hydrophobic, pi-stacking, etc.)
    """
    try:
        if mol is None or not mol.GetNumConformers():
            return []

        # Use the new interaction analysis function
        interactions = analyze_protein_ligand_interactions(mol, receptor_pdb, cutoff)
        
        # Add center coordinates for each residue
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("rec", receptor_pdb)
        
        for interaction in interactions:
            # Find the residue in structure to get center
            for model in structure:
                for chain in model:
                    if chain.id != interaction['chain']:
                        continue
                    for residue in chain:
                        if residue.id[1] == interaction['resnum']:
                            atom_coords = []
                            for atom in residue:
                                element = (getattr(atom, "element", "") or "").upper()
                                if element != "H":
                                    atom_coords.append(np.array(atom.get_vector().get_array(), dtype=float))
                            if atom_coords:
                                center = np.vstack(atom_coords).mean(axis=0)
                                interaction['center'] = [float(v) for v in center]
                            break
        
        # Limit results
        return interactions[:limit]
        
    except Exception as e:
        print(f"Binding residue extraction error: {e}")
        import traceback
        traceback.print_exc()
        return []


def compute_model_diagnostics(ensemble_score, gnn_pred, xgb_pred):
    """Expose ensemble agreement and residual-style diagnostics for the UI."""
    if gnn_pred is None:
        return {
            "ensemble_formula": "100% XGBoost fallback",
            "model_spread": 0.0,
            "agreement_score": 1.0,
            "uncertainty_band": 0.0,
            "residuals": {
                "gnn_to_ensemble": None,
                "xgboost_to_ensemble": 0.0,
                "gnn_minus_xgboost": None,
            },
            "interpretation": "GNN graph was unavailable; XGBoost prediction used as fallback.",
        }

    spread = abs(gnn_pred - xgb_pred)
    agreement_score = max(0.0, 1.0 - min(spread / 2.0, 1.0))
    uncertainty_band = spread / 2.0

    if spread < 0.35:
        interpretation = "High model agreement"
    elif spread < 0.75:
        interpretation = "Moderate model agreement"
    else:
        interpretation = "Low model agreement"

    return {
        "ensemble_formula": f"{GNN_WEIGHT:.0%} GNN + {XGB_WEIGHT:.0%} XGBoost",
        "model_spread": float(spread),
        "agreement_score": float(agreement_score),
        "uncertainty_band": float(uncertainty_band),
        "residuals": {
            "gnn_to_ensemble": float(gnn_pred - ensemble_score),
            "xgboost_to_ensemble": float(xgb_pred - ensemble_score),
            "gnn_minus_xgboost": float(gnn_pred - xgb_pred),
        },
        "interpretation": interpretation,
    }


def build_visualization_payload(posed_mol):
    """Return raw model blocks plus binding-site residue annotations."""
    receptor_pdb = str(PIPELINE_DIR / 'receptor.pdb')
    with open(receptor_pdb, 'r') as f:
        receptor_pdb_data = f.read()

    return {
        "receptor_pdb": receptor_pdb_data,
        "ligand_pdb": Chem.MolToPDBBlock(posed_mol),
        "binding_residues": extract_binding_residues(posed_mol, receptor_pdb),
        "binding_cutoff_angstrom": BINDING_RESIDUE_CUTOFF,
    }


def build_json_export(data, timestamp):
    """Build a clean, analysis-friendly JSON export from the prediction payload."""
    generated_at = datetime.now().isoformat(timespec='seconds')

    def safe_number(value, digits=None):
        try:
            if value is None:
                return None
            number = float(value)
            if not np.isfinite(number):
                return None
            return round(number, digits) if digits is not None else number
        except (TypeError, ValueError):
            return None

    def json_safe(value):
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        return value

    def compact_contact(contact):
        return {
            'ligand_atom_index': contact.get('lig_atom_index'),
            'ligand_atom': contact.get('lig_atom'),
            'protein_atom': contact.get('prot_atom'),
            'protein_element': contact.get('prot_element'),
            'distance_angstrom': safe_number(contact.get('distance'), 3),
            'interaction_types': contact.get('types', []),
        }

    residues = data.get('binding_residues') or []
    clean_residues = []
    interaction_summary = {}
    all_contacts = []

    for residue in residues:
        contacts = [compact_contact(contact) for contact in residue.get('contacts', [])]
        contacts = sorted(
            contacts,
            key=lambda item: item['distance_angstrom'] if item['distance_angstrom'] is not None else 999.0,
        )
        nearest_distance = contacts[0]['distance_angstrom'] if contacts else safe_number(residue.get('distance'), 3)
        interactions = sorted(set(residue.get('interactions', [])))

        for interaction in interactions:
            interaction_summary[interaction] = interaction_summary.get(interaction, 0) + 1

        residue_record = {
            'label': residue.get('label'),
            'residue_name': residue.get('residue'),
            'chain': residue.get('chain'),
            'residue_id': residue.get('res_id'),
            'nearest_distance_angstrom': nearest_distance,
            'contact_count': residue.get('contact_count', len(contacts)),
            'interaction_types': interactions,
            'center': json_safe(residue.get('center')),
            'contacts': contacts,
        }
        clean_residues.append(residue_record)

        for contact in contacts:
            all_contacts.append({
                'residue_label': residue.get('label'),
                **contact,
            })

    clean_residues = sorted(
        clean_residues,
        key=lambda item: (
            item['nearest_distance_angstrom'] if item['nearest_distance_angstrom'] is not None else 999.0,
            item['label'] or '',
        ),
    )
    all_contacts = sorted(
        all_contacts,
        key=lambda item: item['distance_angstrom'] if item['distance_angstrom'] is not None else 999.0,
    )

    diagnostics = data.get('model_diagnostics') or {}
    model_stack = data.get('model_stack') or {}
    pose = data.get('pose_meta') or {}
    ligand_pdb = data.get('ligand_pdb') or ''
    receptor_pdb = data.get('receptor_pdb') or ''
    viewer_html = data.get('viewer_html') or ''
    ligand_2d_png = data.get('ligand_2d_png') or ''
    active_threshold = safe_number(model_stack.get('active_threshold'), 3)
    if active_threshold is None:
        active_threshold = ACTIVE_THRESHOLD

    return json_safe({
        'schema_version': 'biovanguard.prediction.v2',
        'generated_at': generated_at,
        'source_prediction_timestamp': data.get('timestamp'),
        'target': {
            'name': 'MYC Receptor Affinity',
            'binding_site': 'C-Myc Binding Site',
            'receptor_file': str(PIPELINE_DIR / 'receptor.pdb'),
        },
        'ligand_identity': data.get('ligand_metadata') or {},
        'prediction': {
            'ensemble_score_kcal_per_mol': safe_number(data.get('ensemble_score'), 3),
            'gnn_prediction_kcal_per_mol': safe_number(data.get('gnn_pred'), 3),
            'xgboost_prediction_kcal_per_mol': safe_number(data.get('xgb_pred'), 3),
            'classification': 'Active' if data.get('is_active') else 'Inactive',
            'active_threshold_kcal_per_mol': active_threshold,
        },
        'model_stack': {
            'ensemble': model_stack.get('ensemble'),
            'formula': diagnostics.get('ensemble_formula', f'{GNN_WEIGHT:.0%} GNN + {XGB_WEIGHT:.0%} XGBoost'),
            'weights': model_stack.get('ensemble_weights', {'gnn': GNN_WEIGHT, 'xgboost': XGB_WEIGHT}),
            'gnn_model': model_stack.get('gnn_model'),
            'xgboost_model': model_stack.get('xgb_model'),
            'xgboost_scaler': model_stack.get('xgb_scaler'),
        },
        'model_diagnostics': diagnostics,
        'molecular_properties': data.get('properties') or {},
        'drug_likeness': data.get('drug_likeness') or {},
        'pose': {
            'source_file': pose.get('source_file'),
            'rank': pose.get('rank'),
            'reference_label_kcal_per_mol': safe_number(pose.get('ref_label'), 3),
            'similarity': safe_number(pose.get('similarity'), 4),
            'alignment_rmsd_angstrom': safe_number(pose.get('align_rmsd'), 3),
        },
        'binding_analysis': {
            'cutoff_angstrom': safe_number(data.get('binding_cutoff_angstrom'), 2),
            'residue_count': len(clean_residues),
            'contact_count': len(all_contacts),
            'interaction_summary': dict(sorted(interaction_summary.items())),
            'residues': clean_residues,
            'top_contacts': all_contacts[:75],
        },
        'structure_data': {
            'ligand_pose_pdb': ligand_pdb,
            'receptor_pdb_included': False,
            'receptor_pdb_characters_omitted': len(receptor_pdb),
            'viewer_html_included': False,
            'viewer_html_characters_omitted': len(viewer_html),
            'ligand_2d_png_included': False,
            'ligand_2d_png_characters_omitted': len(ligand_2d_png),
            'omission_reason': 'Large visualization assets are omitted from the JSON export to keep the data file compact and analysis-friendly.',
        },
    })


def build_pdf_report(data, timestamp):
    """Build a professional, data-rich PDF report from a prediction payload."""
    import base64
    import html
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.platypus import (
        HRFlowable,
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    output_path = OUTPUT_FOLDER / f'biovanguard_report_{timestamp}.pdf'
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    primary = colors.HexColor('#0C447C')
    accent = colors.HexColor('#1D9E75')
    muted = colors.HexColor('#5F5E5A')
    light_blue = colors.HexColor('#EAF3FB')
    light_green = colors.HexColor('#E8F6F1')
    light_gray = colors.HexColor('#F5F7FA')
    border = colors.HexColor('#D3D1C7')

    def esc(value):
        return html.escape('' if value is None else str(value))

    def num(value, digits=3, fallback='N/A'):
        try:
            if value is None:
                return fallback
            return f'{float(value):.{digits}f}'
        except (TypeError, ValueError):
            return fallback

    def status_text(flag):
        return 'Pass' if flag else 'Review'

    def pass_fail(condition):
        return 'Pass' if condition else 'Review'

    class ReportCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            page_count = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_page_decorations(page_count)
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)

        def draw_page_decorations(self, page_count):
            self.saveState()
            width, height = letter
            self.setStrokeColor(primary)
            self.setLineWidth(0.8)
            self.line(0.65 * inch, height - 0.55 * inch, width - 0.65 * inch, height - 0.55 * inch)
            self.setFont('Helvetica-Bold', 9)
            self.setFillColor(primary)
            self.drawString(0.65 * inch, height - 0.42 * inch, 'BioVanguard')
            self.setFont('Helvetica', 8)
            self.setFillColor(muted)
            self.drawRightString(width - 0.65 * inch, height - 0.42 * inch, 'MYC Receptor Affinity Report')
            self.setStrokeColor(border)
            self.line(0.65 * inch, 0.58 * inch, width - 0.65 * inch, 0.58 * inch)
            self.setFont('Helvetica', 7.5)
            self.drawString(0.65 * inch, 0.38 * inch, f'Generated: {generated_at}')
            self.drawRightString(width - 0.65 * inch, 0.38 * inch, f'Page {self._pageNumber} of {page_count}')
            self.restoreState()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title='BioVanguard Prediction Report',
        author='BioVanguard',
    )
    base_styles = getSampleStyleSheet()
    styles = {
        'title': ParagraphStyle(
            'ReportTitle',
            parent=base_styles['Heading1'],
            alignment=TA_CENTER,
            fontSize=22,
            leading=26,
            textColor=primary,
            spaceAfter=8,
        ),
        'subtitle': ParagraphStyle(
            'ReportSubtitle',
            parent=base_styles['Normal'],
            alignment=TA_CENTER,
            fontSize=11,
            leading=14,
            textColor=muted,
            spaceAfter=16,
        ),
        'section': ParagraphStyle(
            'Section',
            parent=base_styles['Heading2'],
            fontSize=13,
            leading=16,
            textColor=primary,
            spaceBefore=14,
            spaceAfter=8,
        ),
        'subsection': ParagraphStyle(
            'Subsection',
            parent=base_styles['Heading3'],
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor('#2C2C2A'),
            spaceBefore=10,
            spaceAfter=6,
        ),
        'body': ParagraphStyle(
            'Body',
            parent=base_styles['BodyText'],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#2C2C2A'),
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        ),
        'small': ParagraphStyle(
            'Small',
            parent=base_styles['BodyText'],
            fontSize=7.6,
            leading=9.2,
            textColor=muted,
        ),
        'cell': ParagraphStyle(
            'Cell',
            parent=base_styles['BodyText'],
            fontSize=8,
            leading=9.5,
            textColor=colors.HexColor('#2C2C2A'),
        ),
        'cell_bold': ParagraphStyle(
            'CellBold',
            parent=base_styles['BodyText'],
            fontSize=8,
            leading=9.5,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#2C2C2A'),
        ),
    }

    def p(value, style='cell'):
        return Paragraph(esc(value), styles[style])

    def table(rows, widths, header_color=primary, body_color=light_gray, repeat_rows=1):
        tbl = Table(rows, colWidths=widths, repeatRows=repeat_rows)
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), header_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8.5),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 1), (-1, -1), body_color),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [body_color, colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.35, border),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        return tbl

    def ligand_image_flowable():
        image_b64 = data.get('ligand_2d_png')
        if not image_b64:
            return Paragraph('2D Structure Not Available', styles['cell'])
        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image(BytesIO(image_bytes), width=2.15 * inch, height=1.5 * inch)
            image.hAlign = 'CENTER'
            return image
        except Exception:
            return Paragraph('2D Structure Not Available', styles['cell'])

    props = data.get('properties', {})
    ligand = data.get('ligand_metadata', {})
    diagnostics = data.get('model_diagnostics', {})
    residuals = diagnostics.get('residuals', {})
    pose = data.get('pose_meta', {})
    model_stack = data.get('model_stack', {})
    drug_likeness = data.get('drug_likeness') or {}
    residues = data.get('binding_residues', [])
    is_active = bool(data.get('is_active'))
    activity = 'Active' if is_active else 'Inactive'
    score = float(data.get('ensemble_score', 0.0))
    threshold = float(model_stack.get('active_threshold', ACTIVE_THRESHOLD))
    if not drug_likeness and props:
        drug_likeness = compute_drug_likeness_scorecard(props)

    interaction_summary = {}
    for residue in residues:
        for interaction in residue.get('interactions', []):
            interaction_summary[interaction] = interaction_summary.get(interaction, 0) + 1

    story = []
    story.append(Paragraph('BioVanguard Prediction Report', styles['title']))
    story.append(Paragraph('MYC Receptor Affinity And Binding Interaction Analysis', styles['subtitle']))
    story.append(HRFlowable(width='100%', thickness=1.2, color=primary, spaceAfter=12))

    summary_rows = [
        [p('Metric', 'cell_bold'), p('Value', 'cell_bold'), p('Interpretation', 'cell_bold')],
        [p('Ensemble Score'), p(f'{num(score)} kcal/mol'), p(f'{activity}; threshold {threshold:.2f} kcal/mol')],
        [p('Model Stack'), p(diagnostics.get('ensemble_formula', '60% GNN + 40% XGBoost')), p('Weighted best-model ensemble')],
        [p('Interaction Residues'), p(str(len(residues))), p(f'{data.get("binding_cutoff_angstrom", BINDING_RESIDUE_CUTOFF)} Angstrom contact cutoff')],
        [p('Drug-Likeness'), p(drug_likeness.get('assessment', 'N/A')), p(f"Overall score {num(drug_likeness.get('overall_score'), 3)}")],
        [p('Pose Source'), p(pose.get('source_file', 'Original Coordinates')), p(f"Similarity {num(pose.get('similarity'), 3)}; RMSD {num(pose.get('align_rmsd'), 2)} Angstrom")],
    ]
    story.append(table(summary_rows, [1.6 * inch, 2.1 * inch, 2.5 * inch], header_color=accent, body_color=light_green))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        'Executive Summary',
        styles['section'],
    ))
    margin = score - threshold
    summary_text = (
        f'The submitted ligand is classified as <b>{activity}</b> against the MYC receptor affinity '
        f'threshold. The ensemble score is <b>{score:.3f} kcal/mol</b>, which is '
        f'{abs(margin):.3f} kcal/mol {"below" if margin <= 0 else "above"} the active cutoff. '
        f'The model agreement assessment is <b>{esc(diagnostics.get("interpretation", "Not Available"))}</b>. '
        f'The binding-site analysis identified <b>{len(residues)}</b> proximal residues and '
        f'<b>{sum(interaction_summary.values())}</b> classified interaction labels. '
        f'The developability screen classifies the ligand as <b>{esc(drug_likeness.get("assessment", "Not Available"))}</b>.'
    )
    story.append(Paragraph(summary_text, styles['body']))

    story.append(Paragraph('Ligand Identity', styles['section']))
    ligand_rows = [
        [p('Field', 'cell_bold'), p('Value', 'cell_bold')],
        [p('Input File'), p(ligand.get('filename', 'Ligand'))],
        [p('Molecular Formula'), p(ligand.get('formula', 'N/A'))],
        [p('Canonical SMILES'), p(ligand.get('canonical_smiles', 'N/A'))],
        [p('Atom Count'), p(ligand.get('atom_count', 'N/A'))],
        [p('Bond Count'), p(ligand.get('bond_count', 'N/A'))],
        [p('Ring Count'), p(ligand.get('ring_count', 'N/A'))],
    ]
    ligand_info_table = table(ligand_rows, [1.3 * inch, 2.75 * inch])
    ligand_image_table = Table(
        [[p('2D Ligand Structure', 'cell_bold')], [ligand_image_flowable()]],
        colWidths=[2.0 * inch],
    )
    ligand_image_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.35, border),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    identity_layout = Table(
        [[ligand_image_table, ligand_info_table]],
        colWidths=[2.1 * inch, 4.1 * inch],
    )
    identity_layout.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(identity_layout)

    story.append(PageBreak())
    story.append(Paragraph('Prediction And Model Diagnostics', styles['section']))
    weights = model_stack.get('ensemble_weights', {})
    gnn_weight = float(weights.get('gnn', GNN_WEIGHT))
    xgb_weight = float(weights.get('xgboost', XGB_WEIGHT))
    model_rows = [
        [p('Model', 'cell_bold'), p('Prediction', 'cell_bold'), p('Weight', 'cell_bold'), p('Contribution', 'cell_bold')],
        [p('Graph Neural Network'), p(f'{num(data.get("gnn_pred"))} kcal/mol'), p(f'{gnn_weight:.0%}'), p(f'{num((data.get("gnn_pred") or 0) * gnn_weight)}')],
        [p('XGBoost'), p(f'{num(data.get("xgb_pred"))} kcal/mol'), p(f'{xgb_weight:.0%}'), p(f'{num((data.get("xgb_pred") or 0) * xgb_weight)}')],
        [p('Ensemble', 'cell_bold'), p(f'{num(score)} kcal/mol', 'cell_bold'), p('100%', 'cell_bold'), p(activity, 'cell_bold')],
    ]
    story.append(table(model_rows, [1.8 * inch, 1.55 * inch, 1.05 * inch, 1.35 * inch], header_color=primary, body_color=light_blue))
    story.append(Spacer(1, 8))
    diag_rows = [
        [p('Diagnostic', 'cell_bold'), p('Value', 'cell_bold')],
        [p('Agreement Score'), p(f'{num(diagnostics.get("agreement_score"), 3)} ({num((diagnostics.get("agreement_score") or 0) * 100, 1)}%)')],
        [p('Model Spread'), p(f'{num(diagnostics.get("model_spread"))} kcal/mol')],
        [p('Uncertainty Band'), p(f'+/- {num(diagnostics.get("uncertainty_band"))} kcal/mol')],
        [p('GNN Residual To Ensemble'), p(f'{num(residuals.get("gnn_to_ensemble"))} kcal/mol')],
        [p('XGBoost Residual To Ensemble'), p(f'{num(residuals.get("xgboost_to_ensemble"))} kcal/mol')],
        [p('Interpretation'), p(diagnostics.get('interpretation', 'N/A'))],
    ]
    story.append(table(diag_rows, [2.15 * inch, 4.05 * inch], header_color=accent, body_color=light_green))

    story.append(Paragraph('Molecular Properties And Developability', styles['section']))
    property_rows = [
        [p('Property', 'cell_bold'), p('Value', 'cell_bold'), p('Reference Range', 'cell_bold'), p('Status', 'cell_bold')],
        [p('Molecular Weight'), p(f'{num(props.get("mw"), 2)} Da'), p('< 500 Da'), p(pass_fail((props.get('mw') or 9999) < 500))],
        [p('LogP'), p(num(props.get('logp'), 2)), p('-0.4 to 5.6'), p(pass_fail(-0.4 <= (props.get('logp') or 999) <= 5.6))],
        [p('H-Bond Donors'), p(props.get('hbd', 'N/A')), p('<= 5'), p(pass_fail((props.get('hbd') or 99) <= 5))],
        [p('H-Bond Acceptors'), p(props.get('hba', 'N/A')), p('<= 10'), p(pass_fail((props.get('hba') or 99) <= 10))],
        [p('TPSA'), p(f'{num(props.get("tpsa"), 2)} A^2'), p('<= 140 A^2'), p(pass_fail((props.get('tpsa') or 999) <= 140))],
        [p('Rotatable Bonds'), p(props.get('rotb', 'N/A')), p('<= 10'), p(pass_fail((props.get('rotb') or 99) <= 10))],
        [p('Aromatic Rings'), p(props.get('arom_rings', 'N/A')), p('1 to 3 typical'), p(pass_fail(1 <= (props.get('arom_rings') or 0) <= 3))],
        [p('Fraction Csp3'), p(num(props.get('frac_csp3'), 3)), p('Higher may improve 3D character'), p('Informational')],
        [p('QED'), p(num(props.get('qed'), 3)), p('> 0.5'), p(pass_fail((props.get('qed') or 0) > 0.5))],
    ]
    story.append(table(property_rows, [1.55 * inch, 1.05 * inch, 1.95 * inch, 1.15 * inch]))
    story.append(Spacer(1, 8))
    drug_rows = [
        [p('Rule Or Filter', 'cell_bold'), p('Assessment', 'cell_bold')],
        [p('Lipinski Rule Of Five'), p(f"{props.get('lip_passes', 'N/A')}/4 Criteria Passed")],
        [p('Veber'), p(status_text(props.get('veber')))],
        [p('Ghose'), p(status_text(props.get('ghose')))],
        [p('Egan'), p(status_text(props.get('egan')))],
        [p('Muegge'), p(status_text(props.get('muegge')))],
        [p('PAINS Filter'), p('Clean' if props.get('pains_clean') else 'Review')],
        [p('Brenk Filter'), p('Clean' if props.get('brenk_clean') else 'Review')],
    ]
    story.append(table(drug_rows, [2.1 * inch, 4.1 * inch], header_color=accent, body_color=light_green))

    if drug_likeness:
        story.append(Spacer(1, 8))
        story.append(Paragraph('Drug-Likeness Scorecard', styles['subsection']))
        overall_score = drug_likeness.get('overall_score')
        scorecard_intro = (
            f'Overall assessment: <b>{esc(drug_likeness.get("assessment", "N/A"))}</b>. '
            f'Composite score: <b>{num(overall_score, 3)}</b>. '
            'The scorecard combines rule-based developability filters with the QED score used in the dashboard radar view.'
        )
        story.append(Paragraph(scorecard_intro, styles['body']))

        metric_rows = [[p('Metric', 'cell_bold'), p('Score', 'cell_bold'), p('Status', 'cell_bold'), p('Details', 'cell_bold')]]
        metrics = drug_likeness.get('metrics') or {}
        for key in ['lipinski', 'veber', 'ghose', 'pains', 'brenk', 'qed']:
            metric = metrics.get(key)
            if not metric:
                continue
            metric_rows.append([
                p(metric.get('name', key.title())),
                p(f"{num(metric.get('score'), 2)} ({num((metric.get('score') or 0) * 100, 0)}%)"),
                p('Pass' if metric.get('passed') else 'Review'),
                p(metric.get('details', 'N/A')),
            ])
        if len(metric_rows) > 1:
            story.append(table(metric_rows, [1.25 * inch, 1.05 * inch, 0.85 * inch, 3.05 * inch], header_color=primary, body_color=light_blue))

        radar = drug_likeness.get('radar_data') or {}
        labels = radar.get('labels') or []
        values = radar.get('values') or []
        if labels and values:
            radar_rows = [[p('Radar Axis', 'cell_bold'), p('Normalized Value', 'cell_bold')]]
            for label, value in zip(labels, values):
                radar_rows.append([p(label), p(f'{num(value, 0)}%')])
            story.append(Spacer(1, 8))
            story.append(table(radar_rows, [2.1 * inch, 1.5 * inch], header_color=accent, body_color=light_green))

    story.append(PageBreak())
    story.append(Paragraph('Protein-Ligand Binding Interaction Analysis', styles['section']))
    story.append(Paragraph(
        'Residues are reported when at least one receptor atom is within the configured contact cutoff '
        'of the posed ligand. Interaction labels are assigned from distance-based donor/acceptor, aromatic, '
        'charged, and hydrophobic contact rules.',
        styles['body'],
    ))
    if interaction_summary:
        summary_rows = [[p('Interaction Type', 'cell_bold'), p('Residue Count', 'cell_bold')]]
        for interaction, count in sorted(interaction_summary.items(), key=lambda item: (-item[1], item[0])):
            summary_rows.append([p(interaction), p(count)])
        story.append(table(summary_rows, [3.2 * inch, 1.6 * inch], header_color=primary, body_color=light_blue))
        story.append(Spacer(1, 8))

    if residues:
        residue_rows = [[p('Residue', 'cell_bold'), p('Distance', 'cell_bold'), p('Interactions', 'cell_bold'), p('Contacts', 'cell_bold')]]
        for residue in residues[:18]:
            residue_rows.append([
                p(residue.get('label', 'Unknown')),
                p(f'{num(residue.get("min_distance"), 2)} A'),
                p(', '.join(residue.get('interactions', [])) or 'Unclassified'),
                p(residue.get('contact_count', residue.get('near_atom_pairs', 0))),
            ])
        story.append(table(residue_rows, [1.25 * inch, 0.85 * inch, 3.1 * inch, 0.65 * inch]))
        story.append(Spacer(1, 8))

        detail_rows = [[p('Residue', 'cell_bold'), p('Nearest Contact', 'cell_bold'), p('Contact Count', 'cell_bold'), p('Primary Interaction Detail', 'cell_bold')]]
        for residue in residues[:14]:
            contacts = residue.get('contacts', [])
            nearest = contacts[0] if contacts else {}
            atom_text = 'N/A'
            if nearest:
                atom_text = f"{nearest.get('lig_atom', '?')}{nearest.get('lig_atom_index', '')} to {nearest.get('prot_atom', '?')}"
            detail_rows.append([
                p(residue.get('label', 'Unknown')),
                p(f'{atom_text}; {num(nearest.get("distance"), 2)} A' if nearest else 'N/A'),
                p(residue.get('contact_count', residue.get('near_atom_pairs', len(contacts)))),
                p(', '.join(nearest.get('types', [])) or ', '.join(residue.get('interactions', [])) or 'Contact'),
            ])
        story.append(Paragraph('Residue Interaction Detail', styles['subsection']))
        story.append(table(detail_rows, [1.15 * inch, 1.8 * inch, 0.9 * inch, 2.25 * inch], header_color=primary, body_color=light_blue))
        story.append(Spacer(1, 8))

        contact_rows = [[p('Residue', 'cell_bold'), p('Ligand Atom', 'cell_bold'), p('Receptor Atom', 'cell_bold'), p('Distance', 'cell_bold'), p('Type', 'cell_bold')]]
        for residue in residues[:12]:
            for contact in residue.get('contacts', [])[:3]:
                contact_rows.append([
                    p(residue.get('label', 'Unknown')),
                    p(f"{contact.get('lig_atom', '?')}{contact.get('lig_atom_index', '')}"),
                    p(contact.get('prot_atom', '?')),
                    p(f'{num(contact.get("distance"), 2)} A'),
                    p(', '.join(contact.get('types', [])) or 'Contact'),
                ])
        if len(contact_rows) > 1:
            story.append(Paragraph('Closest Atom-Level Contacts', styles['subsection']))
            story.append(table(contact_rows, [1.15 * inch, 1.0 * inch, 1.05 * inch, 0.85 * inch, 2.0 * inch], header_color=accent, body_color=light_green))
    else:
        story.append(Paragraph('No binding residues were detected within the configured cutoff.', styles['body']))

    story.append(PageBreak())
    story.append(Paragraph('Pose Retrieval And Structural Context', styles['section']))
    pose_rows = [
        [p('Field', 'cell_bold'), p('Value', 'cell_bold')],
        [p('Retrieval Method'), p('GNN Embedding Similarity Search With Pose Alignment')],
        [p('Template Source'), p(pose.get('source_file', 'Original Coordinates'))],
        [p('Template Rank'), p(pose.get('rank', 'N/A'))],
        [p('Template Similarity'), p(num(pose.get('similarity'), 4))],
        [p('Alignment RMSD'), p(f'{num(pose.get("align_rmsd"), 2)} A')],
        [p('Reference Label'), p(f"{pose.get('ref_label', 'N/A')} kcal/mol" if pose.get('ref_label') is not None else 'N/A')],
    ]
    story.append(table(pose_rows, [2.0 * inch, 4.2 * inch], header_color=primary, body_color=light_blue))
    story.append(Spacer(1, 8))
    story.append(Paragraph('Interactive 3D Review Presets', styles['section']))
    preset_rows = [
        [p('Viewer Preset', 'cell_bold'), p('Purpose In The Application', 'cell_bold'), p('Report Interpretation', 'cell_bold')],
        [p('Ligand Focus'), p('Centers the 3D scene on the posed ligand model.'), p('Used to inspect the final docked pose and ligand conformation.')],
        [p('Binding Pocket'), p('Shows the ligand with the same binding-site residue set reported in this PDF.'), p(f'{len(residues)} residues are considered in the binding-site view.')],
        [p('Interaction Map'), p('Color-codes residues by dominant interaction class.'), p('Matches the interaction summary and residue-detail tables above.')],
        [p('Pocket Surface'), p('Adds a restrained surface around the binding-site residue selection.'), p('Helps evaluate steric fit without hiding the ligand.')],
    ]
    story.append(table(preset_rows, [1.35 * inch, 2.45 * inch, 2.35 * inch], header_color=primary, body_color=light_blue))
    story.append(Spacer(1, 8))
    story.append(Paragraph('Methodology', styles['section']))
    method_rows = [
        [p('Component', 'cell_bold'), p('Description', 'cell_bold')],
        [p('GNN Model'), p('Graph Attention Network over ligand and receptor-pocket graph features.')],
        [p('XGBoost Model'), p('Morgan fingerprints plus molecular descriptors scaled through the trained pipeline.')],
        [p('Ensemble Strategy'), p(diagnostics.get('ensemble_formula', 'Weighted GNN and XGBoost ensemble.'))],
        [p('Interaction Analysis'), p('Distance-based classification of H-Bond, hydrophobic, aromatic, charged, halogen, and van der Waals contacts.')],
        [p('Activity Threshold'), p(f'{threshold:.2f} kcal/mol; lower scores indicate stronger predicted affinity.')],
    ]
    story.append(table(method_rows, [1.7 * inch, 4.5 * inch], header_color=accent, body_color=light_green))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width='100%', thickness=0.6, color=border, spaceAfter=8))
    story.append(Paragraph(
        '<b>Research Use Notice:</b> This report is computational and should be interpreted as a decision-support '
        'artifact. Experimental validation is required before biological or development conclusions are made.',
        styles['small'],
    ))

    doc.build(story, canvasmaker=ReportCanvas)
    return output_path

# ============================================================================
# Predictor Class
# ============================================================================

class BioVanguardPredictor:
    def __init__(self):
        print("Loading BioVanguard models...")
        self.device = DEVICE
        self.receptor_pdb = str(PIPELINE_DIR / 'receptor.pdb')
        
        # Load GNN
        self.gnn = ProteinLigandGAT().to(self.device)
        self.gnn.load_state_dict(
            torch.load(PIPELINE_DIR / 'gnn_model_final.pt', map_location=self.device)
        )
        self.gnn.eval()
        print("  [OK] GNN loaded")
        
        # Load XGBoost
        self.xgb = joblib.load(PIPELINE_DIR / 'xgb_model_final.joblib')
        self.scaler = joblib.load(PIPELINE_DIR / 'xgb_scaler.joblib')
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        print("  [OK] XGBoost loaded")
        
        # Initialize pose retrieval engine
        self.pose_engine = None
        if GRAPH_CACHE.exists() and LIGAND_DIR.exists():
            try:
                print("\nInitializing Pose Retrieval Engine...")
                self.pose_engine = PoseRetrievalEngine(
                    gnn_model=self.gnn,
                    graph_cache_path=GRAPH_CACHE,
                    ligand_dir=LIGAND_DIR,
                    device=self.device,
                    top_k=5
                )
                print("  [OK] Pose Retrieval Engine initialized")
            except Exception as e:
                print(f"  [WARNING] Pose Retrieval Engine initialization failed: {e}")
                print("  [WARNING] Will use fallback pose generation")
                self.pose_engine = None
        else:
            print("  [WARNING] Graph cache or ligand directory not found")
            print("  [WARNING] Pose retrieval disabled - using fallback")
        
        print(f"Device: {self.device}")
    
    def predict(self, mol):
        # XGBoost prediction
        fp = np.array(self.mfpgen.GetFingerprint(mol))
        desc = np.array([
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.HeavyAtomCount(mol),
            Descriptors.FractionCSP3(mol),
        ], dtype=float)
        xf = np.concatenate([fp, desc]).reshape(1, -1)
        xf[:, 2048:] = self.scaler.transform(xf[:, 2048:])
        xgb_pred = float(self.xgb.predict(xf)[0])
        
        # GNN prediction
        graph = build_graph_from_mol(mol, self.receptor_pdb)
        if graph is None:
            return xgb_pred, None, xgb_pred, None, {}
        
        batch = Batch.from_data_list([graph]).to(self.device)
        with torch.no_grad():
            gnn_pred = float(self.gnn(batch).cpu().numpy()[0])
        
        # Ensemble
        ensemble = GNN_WEIGHT * gnn_pred + XGB_WEIGHT * xgb_pred
        
        return ensemble, gnn_pred, xgb_pred, graph, {}
    
    def predict_with_pose(self, mol):
        """
        Complete prediction pipeline with pose retrieval.
        Returns: ensemble_score, gnn_pred, xgb_pred, posed_mol, pose_meta
        """
        # Step 1: ML predictions
        ensemble, gnn_pred, xgb_pred, graph, _ = self.predict(mol)
        
        # Step 2: Pose retrieval (if available)
        posed_mol = mol
        pose_meta = {"source_file": "original", "similarity": 0.0, "align_rmsd": 0.0}
        
        if self.pose_engine is not None and graph is not None:
            try:
                print("\n[Prediction] Running pose retrieval...")
                posed_mol, pose_meta = self.pose_engine.retrieve_pose(mol, graph)
                print("[Prediction] Pose retrieval complete")
            except Exception as e:
                print(f"[Prediction] Pose retrieval failed: {e}")
                print("[Prediction] Using original molecule coordinates")
                import traceback
                traceback.print_exc()
        else:
            print("[Prediction] Pose retrieval not available, using original coordinates")
        
        return ensemble, gnn_pred, xgb_pred, posed_mol, pose_meta

# Global predictor instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = BioVanguardPredictor()
    return predictor

# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

@app.route('/PredictionPipeline/<path:filename>')
def serve_pipeline_file(filename):
    return send_from_directory(PIPELINE_DIR, filename)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get ligand file
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Please upload SDF file'}), 400
            
            filename = secure_filename(file.filename)
            filepath = UPLOAD_FOLDER / filename
            file.save(filepath)
            
        elif 'example' in request.form:
            example_name = request.form['example']
            filepath = PIPELINE_DIR / example_name
            if not filepath.exists():
                return jsonify({'error': 'Example file not found'}), 404
        else:
            return jsonify({'error': 'No file or example provided'}), 400
        
        # Load molecule
        mol = Chem.MolFromMolFile(str(filepath), removeHs=True, sanitize=True)
        if mol is None:
            return jsonify({'error': 'Failed to parse SDF file'}), 400
        
        if not mol.GetNumConformers():
            AllChem.EmbedMolecule(mol, randomSeed=42)
        
        # Run prediction WITH pose retrieval
        pred = get_predictor()
        print("\n" + "="*60)
        print("Starting BioVanguard Prediction Pipeline")
        print("="*60)
        
        ensemble_score, gnn_pred, xgb_pred, posed_mol, pose_meta = pred.predict_with_pose(mol)
        
        print(f"\nPrediction Results:")
        print(f"  Ensemble: {ensemble_score:.3f} kcal/mol")
        print(f"  GNN: {gnn_pred:.3f} kcal/mol" if gnn_pred else "  GNN: N/A")
        print(f"  XGBoost: {xgb_pred:.3f} kcal/mol")
        print(f"  Activity: {'ACTIVE' if ensemble_score <= ACTIVE_THRESHOLD else 'INACTIVE'}")
        print("="*60 + "\n")
        
        # Compute properties and visualization data
        properties = compute_molecular_properties(mol)
        drug_likeness = compute_drug_likeness_scorecard(properties)
        ligand_meta = compute_ligand_metadata(mol, filepath.name)
        ligand_2d_png = generate_ligand_2d_png(mol)
        diagnostics = compute_model_diagnostics(ensemble_score, gnn_pred, xgb_pred)
        visualization = build_visualization_payload(posed_mol)

        # Keep the legacy HTML payload for old clients; the current UI uses the
        # structured PDB/residue data below.
        viewer_html = generate_3d_viewer(posed_mol, str(filepath))
        
        # Prepare response
        result = {
            'ensemble_score': ensemble_score,
            'gnn_pred': gnn_pred,
            'xgb_pred': xgb_pred,
            'is_active': ensemble_score <= ACTIVE_THRESHOLD,
            'properties': properties,
            'drug_likeness': drug_likeness,
            'ligand_metadata': ligand_meta,
            'ligand_2d_png': ligand_2d_png,
            'model_stack': MODEL_STACK,
            'model_diagnostics': diagnostics,
            'pose_meta': pose_meta,
            'receptor_pdb': visualization['receptor_pdb'],
            'ligand_pdb': visualization['ligand_pdb'],
            'binding_residues': visualization['binding_residues'],
            'binding_cutoff_angstrom': visualization['binding_cutoff_angstrom'],
            'timestamp': datetime.now().isoformat(),
            'viewer_html': viewer_html
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<format>', methods=['POST'])
def download(format):
    try:
        data = request.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            output_path = OUTPUT_FOLDER / f'biovanguard_result_{timestamp}.json'
            export_data = build_json_export(data, timestamp)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, allow_nan=False)
            return send_file(output_path, as_attachment=True, download_name=f'biovanguard_result_{timestamp}.json')
        
        elif format == 'pdf':
            output_path = build_pdf_report(data, timestamp)
            return send_file(
                output_path,
                as_attachment=True,
                download_name=f'biovanguard_report_{timestamp}.pdf'
            )

            # Generate comprehensive PDF report
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
            from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                           Paragraph, Spacer, PageBreak, Image,
                                           KeepTogether, HRFlowable)
            from reportlab.pdfgen import canvas
            
            output_path = OUTPUT_FOLDER / f'biovanguard_report_{timestamp}.pdf'
            
            # Custom page template with header/footer
            class NumberedCanvas(canvas.Canvas):
                def __init__(self, *args, **kwargs):
                    canvas.Canvas.__init__(self, *args, **kwargs)
                    self._saved_page_states = []

                def showPage(self):
                    self._saved_page_states.append(dict(self.__dict__))
                    self._startPage()

                def save(self):
                    num_pages = len(self._saved_page_states)
                    for state in self._saved_page_states:
                        self.__dict__.update(state)
                        self.draw_page_decorations(num_pages)
                        canvas.Canvas.showPage(self)
                    canvas.Canvas.save(self)

                def draw_page_decorations(self, page_count):
                    self.saveState()
                    # Header
                    self.setFont('Helvetica-Bold', 10)
                    self.setFillColor(colors.HexColor('#667eea'))
                    self.drawString(inch, letter[1] - 0.5*inch, "BIOVANGUARD")
                    self.setFont('Helvetica', 8)
                    self.setFillColor(colors.grey)
                    self.drawRightString(letter[0] - inch, letter[1] - 0.5*inch,
                                        f"c-Myc Receptor Affinity Prediction")
                    
                    # Footer
                    self.setFont('Helvetica', 8)
                    self.drawString(inch, 0.5*inch,
                                   f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    self.drawRightString(letter[0] - inch, 0.5*inch,
                                        f"Page {self._pageNumber} of {page_count}")
                    self.restoreState()

            doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                                   topMargin=0.75*inch, bottomMargin=0.75*inch)
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#667eea'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#764ba2'),
                spaceBefore=20,
                spaceAfter=12,
                fontName='Helvetica-Bold'
            )
            
            subheading_style = ParagraphStyle(
                'CustomSubHeading',
                parent=styles['Heading3'],
                fontSize=11,
                textColor=colors.HexColor('#555'),
                spaceBefore=10,
                spaceAfter=8,
                fontName='Helvetica-Bold'
            )
            
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['BodyText'],
                fontSize=10,
                alignment=TA_JUSTIFY,
                spaceAfter=12
            )
            
            # ========== TITLE PAGE ==========
            story.append(Spacer(1, 1.5*inch))
            story.append(Paragraph("BIOVANGUARD", title_style))
            story.append(Paragraph("AI-Powered Drug Discovery Platform",
                                  ParagraphStyle('subtitle', parent=styles['Normal'],
                                               fontSize=14, alignment=TA_CENTER,
                                               textColor=colors.grey)))
            story.append(Spacer(1, 0.3*inch))
            story.append(HRFlowable(width="80%", thickness=2,
                                   color=colors.HexColor('#667eea'),
                                   spaceAfter=0.3*inch, spaceBefore=0.3*inch))
            
            # Executive Summary Box
            summary_data = [[
                Paragraph("<b>PREDICTION SUMMARY</b>",
                         ParagraphStyle('box_title', parent=styles['Normal'],
                                      fontSize=12, textColor=colors.white,
                                      alignment=TA_CENTER))
            ]]
            
            activity_status = 'ACTIVE' if data['is_active'] else 'INACTIVE'
            activity_color = colors.green if data['is_active'] else colors.red
            
            summary_data.append([
                Paragraph(f"<b>Ensemble Score:</b> {data['ensemble_score']:.3f} kcal/mol<br/>"
                         f"<b>Activity Status:</b> <font color='{activity_color.hexval()}'>{activity_status}</font><br/>"
                         f"<b>Target:</b> c-Myc Protein Receptor<br/>"
                         f"<b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y')}",
                         body_style)
            ])
            
            summary_table = Table(summary_data, colWidths=[5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 12),
                ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#667eea')),
            ]))
            story.append(summary_table)
            story.append(PageBreak())
            
            # ========== PREDICTION RESULTS ==========
            story.append(Paragraph("1. PREDICTION RESULTS", heading_style))
            story.append(Paragraph(
                "The ensemble model combines Graph Neural Network (GNN) and XGBoost predictions "
                "to provide robust binding affinity estimates. Lower (more negative) scores indicate "
                "stronger binding affinity.",
                body_style
            ))
            
            pred_data = [
                ['Model', 'Prediction (kcal/mol)', 'Weight', 'Contribution'],
                ['GNN', f"{data.get('gnn_pred', 0):.3f}" if data.get('gnn_pred') else 'N/A',
                 '60%', f"{data.get('gnn_pred', 0) * 0.6:.3f}" if data.get('gnn_pred') else 'N/A'],
                ['XGBoost', f"{data.get('xgb_pred', 0):.3f}" if data.get('xgb_pred') else 'N/A',
                 '40%', f"{data.get('xgb_pred', 0) * 0.4:.3f}" if data.get('xgb_pred') else 'N/A'],
                ['', '', '', ''],
                ['ENSEMBLE', f"{data['ensemble_score']:.3f}", '100%',
                 f"<b>{data['ensemble_score']:.3f}</b>"],
            ]
            
            pred_table = Table(pred_data, colWidths=[1.5*inch, 1.5*inch, 1*inch, 1.5*inch])
            pred_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, 3), colors.beige),
                ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#e8eaf6')),
                ('FONTNAME', (0, 4), (-1, 4), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('LINEABOVE', (0, 4), (-1, 4), 2, colors.HexColor('#764ba2')),
            ]))
            story.append(pred_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Model Diagnostics
            if 'model_diagnostics' in data:
                diag = data['model_diagnostics']
                story.append(Paragraph("Model Diagnostics", subheading_style))
                diag_text = (
                    f"<b>Agreement:</b> {diag.get('agreement', 'N/A')}<br/>"
                    f"<b>Model Spread:</b> {diag.get('spread', 'N/A'):.3f} kcal/mol<br/>"
                    f"<b>Interpretation:</b> {diag.get('interpretation', 'N/A')}"
                )
                story.append(Paragraph(diag_text, body_style))
            
            story.append(Spacer(1, 0.3*inch))
            
            # ========== MOLECULAR PROPERTIES ==========
            story.append(Paragraph("2. MOLECULAR PROPERTIES", heading_style))
            story.append(Paragraph(
                "Comprehensive analysis of physicochemical properties and drug-likeness criteria.",
                body_style
            ))
            
            props = data['properties']
            
            # Basic Properties
            story.append(Paragraph("2.1 Physicochemical Properties", subheading_style))
            basic_props = [
                ['Property', 'Value', 'Optimal Range', 'Status'],
                ['Molecular Weight', f"{props['mw']:.2f} Da", '< 500 Da',
                 'PASS' if props['mw'] < 500 else 'FAIL'],
                ['LogP', f"{props['logp']:.2f}", '-0.4 to 5.6',
                 'PASS' if -0.4 <= props['logp'] <= 5.6 else 'FAIL'],
                ['H-Bond Donors', str(props['hbd']), '<= 5',
                 'PASS' if props['hbd'] <= 5 else 'FAIL'],
                ['H-Bond Acceptors', str(props['hba']), '<= 10',
                 'PASS' if props['hba'] <= 10 else 'FAIL'],
                ['TPSA', f"{props['tpsa']:.2f} A^2", '<= 140 A^2',
                 'PASS' if props['tpsa'] <= 140 else 'FAIL'],
                ['Rotatable Bonds', str(props['rotb']), '<= 10',
                 'PASS' if props['rotb'] <= 10 else 'FAIL'],
                ['Aromatic Rings', str(props['arom_rings']), '1-3',
                 'PASS' if 1 <= props['arom_rings'] <= 3 else 'FAIL'],
                ['QED Score', f"{props['qed']:.3f}", '> 0.5',
                 'PASS' if props['qed'] > 0.5 else 'FAIL'],
            ]
            
            props_table = Table(basic_props, colWidths=[1.8*inch, 1.2*inch, 1.5*inch, 0.8*inch])
            props_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ]))
            story.append(props_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Drug-Likeness Rules
            story.append(Paragraph("2.2 Drug-Likeness Assessment", subheading_style))
            lipinski_status = (
                "PASS"
                if props['lip_passes'] == 4
                else f"{props['lip_passes']}/4"
            )
            druglike_data = [
                ['Rule', 'Criteria', 'Status'],
                ['Lipinski (Ro5)', '4/4 criteria passed',
                 lipinski_status],
                ['Veber', 'TPSA <= 140 & RotB <= 10',
                 'PASS' if props['veber'] else 'FAIL'],
                ['Ghose', 'MW 160-480 & LogP -0.4-5.6',
                 'PASS' if props['ghose'] else 'FAIL'],
                ['Egan', 'TPSA <= 131.6 & LogP <= 5.88',
                 'PASS' if props['egan'] else 'FAIL'],
                ['Muegge', 'MW 200-600 & LogP -2-5 & TPSA <= 150',
                 'PASS' if props['muegge'] else 'FAIL'],
                ['PAINS Filter', 'No pan-assay interference',
                 'CLEAN' if props['pains_clean'] else 'WARNING'],
                ['Brenk Filter', 'No toxic/reactive groups',
                 'CLEAN' if props['brenk_clean'] else 'WARNING'],
            ]
            
            druglike_table = Table(druglike_data, colWidths=[1.5*inch, 3*inch, 1*inch])
            druglike_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (2, 0), (2, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ]))
            story.append(druglike_table)
            
            story.append(PageBreak())
            
            # ========== BINDING INTERACTIONS ==========
            story.append(Paragraph("3. PROTEIN-LIGAND INTERACTIONS", heading_style))
            story.append(Paragraph(
                "Detailed analysis of molecular interactions between the ligand and c-Myc receptor binding site. "
                "Interactions are classified by type and distance.",
                body_style
            ))
            
            residues = data.get('binding_residues', [])
            if residues:
                story.append(Paragraph(f"3.1 Binding Site Residues ({len(residues)} contacts)", subheading_style))
                
                # Group residues by interaction type
                interaction_summary = {}
                for res in residues:
                    for interaction in res.get('interactions', []):
                        if interaction not in interaction_summary:
                            interaction_summary[interaction] = 0
                        interaction_summary[interaction] += 1
                
                if interaction_summary:
                    summary_text = "<b>Interaction Summary:</b><br/>"
                    for int_type, count in sorted(interaction_summary.items(), key=lambda x: x[1], reverse=True):
                        summary_text += f"- {int_type}: {count} residues<br/>"
                    story.append(Paragraph(summary_text, body_style))
                    story.append(Spacer(1, 0.1*inch))
                
                # Detailed residue table
                res_data = [['Residue', 'Distance (A)', 'Interactions', 'Contacts']]
                for res in residues[:15]:  # Top 15 residues
                    interactions_str = ', '.join(res.get('interactions', ['Van der Waals']))
                    res_data.append([
                        res.get('label', res.get('res_name', 'Unknown')),
                        f"{res['min_distance']:.2f}",
                        interactions_str,
                        str(res.get('contact_count', res.get('near_atom_pairs', 0)))
                    ])
                
                res_table = Table(res_data, colWidths=[1.2*inch, 1*inch, 2.5*inch, 0.8*inch])
                res_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                    ('ALIGN', (3, 0), (3, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
                ]))
                story.append(res_table)
            else:
                story.append(Paragraph("No binding residues detected within cutoff distance.", body_style))
            
            story.append(Spacer(1, 0.3*inch))
            
            # ========== POSE INFORMATION ==========
            if 'pose_meta' in data and data['pose_meta'].get('source_file') != 'original':
                story.append(Paragraph("4. POSE RETRIEVAL INFORMATION", heading_style))
                pose_meta = data['pose_meta']
                pose_text = (
                    f"<b>Retrieval Method:</b> GNN Embedding Similarity Search<br/>"
                    f"<b>Template Source:</b> {pose_meta.get('source_file', 'N/A')}<br/>"
                    f"<b>Similarity Score:</b> {pose_meta.get('similarity', 0):.4f}<br/>"
                    f"<b>Alignment RMSD:</b> {pose_meta.get('align_rmsd', 0):.2f} A<br/>"
                    f"<b>Reference dG:</b> {pose_meta.get('ref_label', 'N/A')} kcal/mol"
                )
                story.append(Paragraph(pose_text, body_style))
                story.append(Spacer(1, 0.2*inch))
                story.append(Paragraph(
                    "The ligand pose was determined using a similarity-based retrieval system that identifies "
                    "the most similar training ligands based on GNN embeddings, then aligns the query ligand "
                    "to the retrieved docked pose using SVD-based rotation.",
                    body_style
                ))
            
            story.append(PageBreak())
            
            # ========== METHODOLOGY ==========
            story.append(Paragraph("5. METHODOLOGY", heading_style))
            
            story.append(Paragraph("5.1 Ensemble Model Architecture", subheading_style))
            story.append(Paragraph(
                "<b>Graph Neural Network (GNN):</b> A Graph Attention Network with 3 GAT layers, "
                "128-dimensional hidden representations, and multi-head attention mechanisms. "
                "The model processes both ligand chemistry and protein environment as a unified graph structure.<br/><br/>"
                "<b>XGBoost:</b> Gradient boosting model trained on Morgan fingerprints (radius=2, 2048 bits) "
                "and molecular descriptors. Provides complementary predictions based on chemical similarity.<br/><br/>"
                "<b>Ensemble Strategy:</b> Weighted combination (60% GNN + 40% XGBoost) to leverage both "
                "structural and chemical information.",
                body_style
            ))
            
            story.append(Paragraph("5.2 Training Data", subheading_style))
            story.append(Paragraph(
                "Models trained on 101,612 ligand-protein complexes with experimental binding affinity data. "
                "Training set includes diverse chemical scaffolds and binding modes to ensure robust generalization.",
                body_style
            ))
            
            story.append(Paragraph("5.3 Validation Metrics", subheading_style))
            story.append(Paragraph(
                "Model performance evaluated using cross-validation with metrics including RMSE, MAE, "
                "Pearson correlation, and Spearman rank correlation on held-out test sets.",
                body_style
            ))
            
            # ========== DISCLAIMER ==========
            story.append(Spacer(1, 0.5*inch))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
            story.append(Spacer(1, 0.2*inch))
            
            disclaimer_style = ParagraphStyle(
                'Disclaimer',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.grey,
                alignment=TA_JUSTIFY
            )
            
            story.append(Paragraph("<b>DISCLAIMER</b>", disclaimer_style))
            story.append(Paragraph(
                "This report was prepared by BioVanguard, an AI-powered computational tool for predicting "
                "ligand-protein binding affinity. Predictions are based on machine learning models and should "
                "be validated through experimental methods. Results are for research purposes only and should "
                "not be used as the sole basis for drug development decisions. The accuracy of predictions "
                "depends on the similarity of the query ligand to the training data.",
                disclaimer_style
            ))
            
            # Build PDF
            doc.build(story, canvasmaker=NumberedCanvas)
            return send_file(output_path, as_attachment=True,
                           download_name=f'biovanguard_report_{timestamp}.pdf')
        
        elif format == 'sdf':
            # Return the original SDF file or create a simple one
            output_path = OUTPUT_FOLDER / f'biovanguard_ligand_{timestamp}.sdf'
            
            # Create a simple SDF with the data
            with open(output_path, 'w') as f:
                f.write(f"BioVanguard Ligand Export\n")
                f.write(f"  Generated: {timestamp}\n")
                f.write(f"\n")
                f.write(f"  0  0  0  0  0  0  0  0  0  0999 V2000\n")
                f.write(f"M  END\n")
                f.write(f"> <Ensemble_Score>\n")
                f.write(f"{data['ensemble_score']:.3f}\n\n")
                f.write(f"> <Activity>\n")
                f.write(f"{'ACTIVE' if data['is_active'] else 'INACTIVE'}\n\n")
                f.write(f"$$$$\n")
            
            return send_file(output_path, as_attachment=True, download_name=f'biovanguard_ligand_{timestamp}.sdf')
        
        else:
            return jsonify({'error': 'Invalid format'}), 400
            
    except Exception as e:
        print(f"Download error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'models_loaded': predictor is not None,
        'model_stack': MODEL_STACK,
    })

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BioVanguard Web Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  BioVanguard - AI-Powered Drug Discovery Platform")
    print("  Target: c-Myc Protein")
    print("="*60)
    print(f"\n  Server starting on http://{args.host}:{args.port}")
    print(f"  Device: {DEVICE}")
    print("\n  Press Ctrl+C to stop the server\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

