"""
NeuroSim
========
A physics-constrained Python toolkit for finite-horizon Network Control
Theory in macro-scale neuroimaging.

Install
-------
Core only::

    pip install neurosim

With neuroimaging data loading (NiBabel, Nilearn, PyBIDS)::

    pip install neurosim[neuroimaging]

With visualisation (UMAP)::

    pip install neurosim[viz]

Everything::

    pip install neurosim[all]

Quickstart
----------
>>> from neurosim.physics import normalise_matrix, compute_gramian_doubling, minimum_energy
>>> from neurosim.connectivity import graphnet_effective_connectivity
>>> from neurosim.harmonize import BlindHarmonizer
>>> from neurosim.simulation import WilsonCowanNetwork
>>> from neurosim.loader import BIDSLoader, from_arrays   # requires [neuroimaging]
"""

__version__ = "0.1.0-dev"
__author__  = "Ritam Kanti Roy"
__email__   = "rhitam2001@gmail.com"
__license__ = "Apache-2.0"

# ── Physics engine ────────────────────────────────────────────────────────────
from neurosim.physics import (
    normalise_matrix,
    compute_gramian_doubling,
    minimum_energy,
    average_controllability,
    modal_controllability,
)

# ── Connectivity estimation ───────────────────────────────────────────────────
from neurosim.connectivity import (
    functional_connectivity,
    ridge_effective_connectivity,
    graphnet_effective_connectivity,
    graph_laplacian,
    simulate_feedforward_network,
)

# ── Harmonisation ─────────────────────────────────────────────────────────────
from neurosim.harmonize import (
    BlindHarmonizer,
    detect_site_effects,
)

# ── Neural mass simulation ────────────────────────────────────────────────────
from neurosim.simulation import (
    WilsonCowanNode,
    WilsonCowanNetwork,
)

# ── Data loading (optional — requires pip install neurosim[neuroimaging]) ─────
try:
    from neurosim.loader import (
        BIDSLoader,
        load_atlas,
        load_bold,
        load_connectome,
        from_arrays,
    )
    _loader_available = True
except ImportError:
    _loader_available = False

__all__ = [
    "normalise_matrix", "compute_gramian_doubling", "minimum_energy",
    "average_controllability", "modal_controllability",
    "functional_connectivity", "ridge_effective_connectivity",
    "graphnet_effective_connectivity", "graph_laplacian",
    "simulate_feedforward_network",
    "BlindHarmonizer", "detect_site_effects",
    "WilsonCowanNode", "WilsonCowanNetwork",
    "BIDSLoader", "load_atlas", "load_bold", "load_connectome", "from_arrays",
]
