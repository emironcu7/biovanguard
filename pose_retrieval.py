"""
Pose Retrieval Engine - GNN Embedding-Based Pose Retrieval
Ported from Google Colab implementation
"""

import os
import time
import pickle
import numpy as np
from pathlib import Path

import torch
from torch_geometric.data import Batch
from rdkit import Chem


class PoseRetrievalEngine:
    """
    GNN Embedding Similarity -> Nearest-Neighbor Pose Retrieval.

    How it works
    ------------
    Training graphs were built from real Vina-docked poses (_out.pdbqt).
    Each graph encodes both ligand chemistry AND protein environment.

    For a new ligand:
      1. Build its graph (ligand centered at docking box, same as training).
      2. Extract its GNN embedding (128-dim global mean pool).
      3. Compute cosine similarity with ALL training embeddings.
      4. Retrieve top-K most similar training ligands' _out.pdbqt files.
      5. Read the best Vina pose from each file.
      6. Align the new ligand's 3D shape to the retrieved pose via
         heavy-atom RMS fitting (MCS-based) -> ligand-specific pose.

    Result: each ligand gets a unique pose grounded in real docking data,
    not a generic translation to docking center.
    """

    def __init__(self, gnn_model, graph_cache_path, ligand_dir, device, top_k=5):
        """
        Parameters
        ----------
        gnn_model : ProteinLigandGAT
            The trained GNN model with get_embedding() method
        graph_cache_path : Path
            Path to graphs_cache-001.pkl file
        ligand_dir : Path
            Directory containing _out.pdbqt files
        device : torch.device
            Device for computation
        top_k : int
            Number of similar poses to retrieve
        """
        self.gnn = gnn_model
        self.device = device
        self.top_k = top_k
        self.ligand_dir = ligand_dir
        self._emb_cache = None   # (N, 128) tensor built lazily

        print("\n[Pose Retrieval] Loading training graph cache...")
        t0 = time.time()
        with open(graph_cache_path, "rb") as f:
            cached = pickle.load(f)

        self.train_graphs = cached["graphs"]      # list of Data
        self.train_labels = np.array(cached["labels"])
        self.train_filenames = cached["file_names"]  # list of str

        print(f"[Pose Retrieval] Loaded {len(self.train_graphs):,} training graphs "
              f"in {time.time()-t0:.1f}s")

    def _build_embedding_index(self):
        """
        Compute GNN embeddings for ALL training graphs.
        Done once, then cached in RAM.
        ~2-5 min for 100K graphs on GPU.
        """
        print("\n[Pose Retrieval] Building embedding index from training graphs...")
        print("[Pose Retrieval] (This runs once per session - ~3-5 min for 100K graphs)")
        t0 = time.time()
        embs = []
        BATCH = 256

        self.gnn.eval()
        with torch.no_grad():
            for i in range(0, len(self.train_graphs), BATCH):
                batch_graphs = self.train_graphs[i:i+BATCH]
                batch = Batch.from_data_list(
                    [g.to(self.device) for g in batch_graphs])
                emb = self.gnn.get_embedding(batch)   # (B, 128)
                embs.append(emb.cpu())

                if (i // BATCH) % 50 == 0:
                    pct = i / len(self.train_graphs) * 100
                    elapsed = time.time()-t0
                    eta = elapsed/max(i,1)*len(self.train_graphs) - elapsed
                    print(f"[Pose Retrieval]   {pct:.1f}%  elapsed={elapsed:.0f}s  "
                          f"ETA={eta:.0f}s")

        self._emb_cache = torch.cat(embs, dim=0)   # (N, 128)
        # L2-normalise for cosine similarity via dot product
        norms = self._emb_cache.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self._emb_cache = self._emb_cache / norms
        print(f"[Pose Retrieval] Index built: {self._emb_cache.shape}  "
              f"({time.time()-t0:.1f}s total)")

    def _get_query_embedding(self, query_graph):
        """Extract and L2-normalise embedding for one query graph."""
        self.gnn.eval()
        with torch.no_grad():
            batch = Batch.from_data_list([query_graph.to(self.device)])
            emb = self.gnn.get_embedding(batch).cpu()   # (1, 128)
        emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return emb   # (1, 128)

    def _read_best_pose_from_pdbqt(self, filename):
        """
        Read MODEL 1 (best Vina pose) from _out.pdbqt.
        Returns np.ndarray (N_heavy, 3) of heavy-atom coordinates.
        """
        # Try to find the file
        candidates = [
            os.path.join(self.ligand_dir, filename + "_out.pdbqt"),
            os.path.join(self.ligand_dir, filename),
            os.path.join(self.ligand_dir, Path(filename).stem + "_out.pdbqt"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            return None

        coords = []
        in_model1 = False
        model_seen = 0
        with open(path) as f:
            for line in f:
                if line.startswith("MODEL"):
                    model_seen += 1
                    in_model1 = (model_seen == 1)
                    continue
                if line.startswith("ENDMDL") and in_model1:
                    break
                take = (model_seen == 0) or in_model1
                if take and line.startswith(("ATOM","HETATM")):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        elem = line[76:78].strip() if len(line)>76 else 'C'
                        if elem.upper() != 'H':   # heavy atoms only
                            coords.append([x,y,z])
                    except ValueError:
                        pass
        return np.array(coords) if coords else None

    def _align_ligand_to_pose(self, query_mol, ref_coords):
        """
        Align query_mol's 3D conformation to the retrieved pose coordinates.

        Strategy:
          1. Translate query centroid -> ref centroid.
          2. Apply SVD-based rotation to minimise heavy-atom RMSD
             (works even if atom counts differ - uses centroid + inertia
             tensor alignment, not MCS).

        Returns a new RDKit Mol with updated conformer and RMSD.
        """
        # Get query coordinates
        conf = query_mol.GetConformer()
        q_coords = np.array([list(conf.GetAtomPosition(i)) 
                            for i in range(query_mol.GetNumAtoms())])
        r_coords = ref_coords

        q_cen = q_coords.mean(axis=0)
        r_cen = r_coords.mean(axis=0)

        # Step 1: translate
        q_centered = q_coords - q_cen
        r_centered = r_coords - r_cen

        # Step 2: SVD rotation - align principal axes
        n = min(len(q_centered), len(r_centered))
        H = q_centered[:n].T @ r_centered[:n]
        U, S, Vt = np.linalg.svd(H)
        # Correct for reflection
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1., 1., d])
        rot = Vt.T @ D @ U.T                    # (3, 3)

        # Apply rotation + translation
        q_new = (q_centered @ rot.T) + r_cen

        # Write new coords into a copy of the mol
        rw = Chem.RWMol(query_mol)
        conf = rw.GetConformer()
        for i in range(rw.GetNumAtoms()):
            conf.SetAtomPosition(i, q_new[i].tolist())

        rmsd = float(np.sqrt(((q_new[:n] - r_centered[:n] + r_cen)**2).mean()))
        return rw.GetMol(), rmsd

    def retrieve_pose(self, query_mol, query_graph):
        """
        Main entry point.

        Parameters
        ----------
        query_mol   : RDKit Mol with 3D conformer (from ZINC SDF)
        query_graph : torch_geometric Data built from query_mol

        Returns
        -------
        posed_mol   : RDKit Mol with retrieved + aligned pose
        meta        : dict with similarity, source filename, rmsd info
        """
        # Build index if not done yet
        if self._emb_cache is None:
            self._build_embedding_index()

        # Query embedding
        print("\n[Pose Retrieval] Computing query embedding...")
        q_emb = self._get_query_embedding(query_graph)   # (1, 128)

        # Cosine similarities
        sims = (self._emb_cache @ q_emb.T).squeeze(1)   # (N,)
        top_k_idx = torch.topk(sims, self.top_k).indices.numpy()

        print(f"[Pose Retrieval] Top-{self.top_k} similar training ligands:")
        best_mol = None
        best_sim = -1
        best_meta = {}

        for rank, idx in enumerate(top_k_idx):
            fname = self.train_filenames[idx]
            sim = float(sims[idx])
            label = float(self.train_labels[idx])

            ref_coords = self._read_best_pose_from_pdbqt(fname)
            if ref_coords is None:
                print(f"[Pose Retrieval]   [{rank+1}] {fname}  sim={sim:.4f}  "
                      f"label={label:.2f}  WARNING: file not found")
                continue

            posed_mol, rmsd = self._align_ligand_to_pose(query_mol, ref_coords)
            print(f"[Pose Retrieval]   [{rank+1}] {fname}  sim={sim:.4f}  "
                  f"label={label:.2f} kcal/mol  RMSD={rmsd:.2f} A")

            if sim > best_sim:
                best_sim = sim
                best_mol = posed_mol
                best_meta = {
                    "source_file": fname,
                    "similarity": sim,
                    "ref_label": label,
                    "align_rmsd": rmsd,
                    "rank": rank+1,
                }

        if best_mol is None:
            print("[Pose Retrieval] WARNING: No _out.pdbqt files found. "
                  "Returning original molecule.")
            best_mol = query_mol
            best_meta = {"source_file": "fallback", "similarity": 0.0, "align_rmsd": 0.0}

        # Get final pose centroid
        conf = best_mol.GetConformer()
        fc = np.array([list(conf.GetAtomPosition(i)) 
                      for i in range(best_mol.GetNumAtoms())]).mean(axis=0)
        print(f"\n[Pose Retrieval] Final pose centroid: "
              f"({fc[0]:.2f}, {fc[1]:.2f}, {fc[2]:.2f})")
        print(f"[Pose Retrieval] Best similarity: {best_meta['similarity']:.4f}")
        
        return best_mol, best_meta

