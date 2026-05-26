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
>>> from neurosim.loader import BIDSLoader, from_arrays      # requires [neuroimaging]
>>> from neurosim.plot import plot_neurosim_summary           # requires matplotlib
>>> from neurosim.clinical import AUDPipeline, ADNIPipeline  # core only
"""

__version__ = "0.1.0-dev"
__author__  = "Ritam Kanti Roy"
__email__   = "rhitam2001@gmail.com"
__license__ = "Apache-2.0"

# ── Physics engine ─────────────────────────────────────────────────────────────
from neurosim.physics import (
    normalise_matrix,
    compute_gramian_doubling,
    minimum_energy,
    average_controllability,
    modal_controllability,
)

# ── Connectivity estimation ────────────────────────────────────────────────────
from neurosim.connectivity import (
    functional_connectivity,
    ridge_effective_connectivity,
    graphnet_effective_connectivity,
    graph_laplacian,
    simulate_feedforward_network,
)

# ── Harmonisation ──────────────────────────────────────────────────────────────
from neurosim.harmonize import (
    BlindHarmonizer,
    detect_site_effects,
)

# ── Neural mass simulation ─────────────────────────────────────────────────────
from neurosim.simulation import (
    WilsonCowanNode,
    WilsonCowanNetwork,
)

# ── Clinical pipelines (core — no optional deps required) ─────────────────────
from neurosim.clinical import (
    AUDPipeline,
    ADNIPipeline,
    EpilepsyPipeline,
    SubjectResult,
    CohortResult,
)

# ── Data loading (optional — requires pip install neurosim[neuroimaging]) ──────
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

# ── Visualisation (optional — requires matplotlib) ────────────────────────────
try:
    from neurosim.plot import (
        set_style,
        plot_state_space,
        plot_state_trajectory,
        plot_energy_matrix,
        plot_energy_landscape_1d,
        plot_controllability,
        plot_stimulation_targets,
        plot_cohort_energy,
        plot_neurosim_summary,
        save_figure,
        PALETTE,
    )
    _plot_available = True
except ImportError:
    _plot_available = False

# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    # Physics
    "normalise_matrix", "compute_gramian_doubling", "minimum_energy",
    "average_controllability", "modal_controllability",
    # Connectivity
    "functional_connectivity", "ridge_effective_connectivity",
    "graphnet_effective_connectivity", "graph_laplacian",
    "simulate_feedforward_network",
    # Harmonisation
    "BlindHarmonizer", "detect_site_effects",
    # Simulation
    "WilsonCowanNode", "WilsonCowanNetwork",
    # Clinical
    "AUDPipeline", "ADNIPipeline", "EpilepsyPipeline",
    "SubjectResult", "CohortResult",
    # Loader (conditional)
    "BIDSLoader", "load_atlas", "load_bold", "load_connectome", "from_arrays",
    # Plot (conditional)
    "set_style", "plot_state_space", "plot_state_trajectory",
    "plot_energy_matrix", "plot_energy_landscape_1d",
    "plot_controllability", "plot_stimulation_targets",
    "plot_cohort_energy", "plot_neurosim_summary",
    "save_figure", "PALETTE",
]
