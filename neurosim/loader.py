"""
neurosim.loader
===============
BIDS-compatible data ingestion for NeuroSim pipelines.
Gracefully degrades if nibabel/nilearn/pybids are not installed.
"""
from __future__ import annotations
import os, warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from numpy.typing import NDArray

SCHAEFER_N_ROIS = [100, 200, 400, 600, 800, 1000]
ATLAS_NAMES = {
    "schaefer100":("schaefer",100),"schaefer200":("schaefer",200),
    "schaefer400":("schaefer",400),"schaefer600":("schaefer",600),
    "schaefer1000":("schaefer",1000),"glasser360":("glasser",360),
    "destrieux":("destrieux",148),"desikan":("desikan",84),
}

def _require_nibabel():
    try: import nibabel; return nibabel
    except ImportError: raise ImportError("NiBabel required: pip install nibabel")

def _require_nilearn():
    try:
        import nilearn
        import nilearn.datasets   # submodule must be explicitly imported
        return nilearn
    except ImportError:
        raise ImportError("Nilearn required: pip install nilearn")

def _require_pybids():
    try: import bids; return bids
    except ImportError: raise ImportError("PyBIDS required: pip install pybids")

def load_atlas(atlas, n_rois=None, resolution_mm=2):
    # Validate name FIRST — before importing heavy optional deps
    atlas_str = str(atlas)
    if not Path(atlas_str).exists():
        key = atlas_str.lower().replace("-","").replace("_","")
        if key not in ATLAS_NAMES:
            raise ValueError(f"Unknown atlas '{atlas_str}'. Choose from: {list(ATLAS_NAMES.keys())}")

    nib = _require_nibabel()
    nilearn_datasets = _require_nilearn().datasets

    if Path(atlas_str).exists():
        atlas_img = nib.load(atlas_str)
        n = int(atlas_img.get_fdata().max())
        return atlas_img, np.array([f"Region_{i}" for i in range(1, n+1)])
    key = atlas_str.lower().replace("-","").replace("_","")
    family, default_n = ATLAS_NAMES[key]
    n_rois = n_rois or default_n
    if family == "schaefer":
        if n_rois not in SCHAEFER_N_ROIS:
            raise ValueError(f"Schaefer supports n_rois in {SCHAEFER_N_ROIS}. Got {n_rois}.")
        dataset = nilearn_datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=resolution_mm)
        atlas_img = nib.load(dataset.maps)
        return atlas_img, np.array(dataset.labels)
    raise ValueError(f"Atlas family '{family}' not yet auto-downloadable. Provide a file path.")

def load_connectome(connectome_path, normalise=False, threshold_pct=None):
    connectome_path = Path(connectome_path)
    if not connectome_path.exists():
        raise FileNotFoundError(f"Connectome file not found: {connectome_path}")
    suffix = connectome_path.suffix.lower()
    if suffix == ".npy":
        SC = np.load(str(connectome_path))
    elif suffix == ".npz":
        data = np.load(str(connectome_path))
        SC = data["SC"] if "SC" in data else data[list(data.keys())[0]]
    elif suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix == ".tsv" else ","
        try:
            import pandas as pd
            SC = pd.read_csv(connectome_path, sep=sep, header=None).values.astype(float)
        except ImportError:
            SC = np.loadtxt(str(connectome_path), delimiter=sep)
    else:
        try: SC = np.loadtxt(str(connectome_path))
        except: raise ValueError(f"Cannot read connectome file with extension '{suffix}'.")
    if SC.ndim != 2 or SC.shape[0] != SC.shape[1]:
        raise ValueError(f"Connectome must be a square matrix. Got shape {SC.shape}.")
    SC = (SC + SC.T) / 2.0
    np.fill_diagonal(SC, 0.0)
    if threshold_pct is not None:
        nonzero = SC[SC > 0]
        if len(nonzero) > 0:
            SC[SC < np.percentile(nonzero, threshold_pct)] = 0.0
    if normalise:
        mx = SC.max()
        if mx > 0: SC = SC / mx
    return SC.astype(float)

def from_arrays(X, SC=None, subject_id="unknown", validate=True):
    X = np.asarray(X, dtype=float)
    if validate:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N_regions, T_timepoints). Got shape {X.shape}.")
        N, T = X.shape
        if T < N:
            warnings.warn(f"X has shape ({N}, {T}) — T < N. Check orientation.", UserWarning, stacklevel=2)
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values.")
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values.")
    N, T = X.shape
    if SC is not None:
        SC = np.asarray(SC, dtype=float)
        if validate:
            if SC.shape != (N, N):
                raise ValueError(f"SC shape {SC.shape} does not match X's N_regions={N}.")
            if not np.allclose(SC, SC.T, atol=1e-6):
                warnings.warn("SC is not symmetric. Enforcing symmetry.", UserWarning, stacklevel=2)
            SC = (SC + SC.T) / 2.0
            np.fill_diagonal(SC, 0.0)
    return {"X": X, "SC": SC, "subject_id": subject_id, "N": N, "T": T}

class BIDSLoader:
    def __init__(self, bids_dir, atlas="schaefer400", derivatives_dir=None, n_rois=None):
        self.bids_dir = Path(bids_dir)
        self.atlas = atlas
        self.n_rois = n_rois
        if derivatives_dir is None:
            self.derivatives_dir = self.bids_dir / "derivatives" / "fmriprep"
        else:
            self.derivatives_dir = Path(derivatives_dir)
        self._layout = None
        try:
            bids = _require_pybids()
            self._layout = bids.BIDSLayout(str(self.bids_dir),
                derivatives=str(self.derivatives_dir) if self.derivatives_dir.exists() else False)
        except ImportError:
            warnings.warn("PyBIDS not installed. Using path-pattern fallback.", UserWarning, stacklevel=2)

    @property
    def subjects(self):
        if self._layout is not None:
            return self._layout.get_subjects()
        return sorted([d.name for d in self.bids_dir.iterdir()
                       if d.is_dir() and d.name.startswith("sub-")])
