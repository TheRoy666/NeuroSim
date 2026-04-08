"""
NeuroSim
========
A physics-constrained Python toolkit for finite-horizon Network Control Theory
in macro-scale neuroimaging.

Quickstart
----------
>>> from neurosim.physics import normalise_matrix, compute_gramian_doubling, minimum_energy
>>> from neurosim.connectivity import graphnet_effective_connectivity
>>> from neurosim.harmonize import BlindHarmonizer
>>> from neurosim.simulation import WilsonCowanNetwork
"""

__version__ = "0.1.0-dev"
__author__ = "Ritam Kanti Roy"
__email__ = "rhitam2001@gmail.com"

from neurosim.physics import (
    normalise_matrix,
    compute_gramian_doubling,
    minimum_energy,
    average_controllability,
    modal_controllability,
)
from neurosim.connectivity import (
    functional_connectivity,
    ridge_effective_connectivity,
    graphnet_effective_connectivity,
    graph_laplacian,
    simulate_feedforward_network,
)
from neurosim.harmonize import (
    BlindHarmonizer,
    detect_site_effects,
)
from neurosim.simulation import (
    WilsonCowanNode,
    WilsonCowanNetwork,
)

__all__ = [
    # Physics
    "normalise_matrix",
    "compute_gramian_doubling",
    "minimum_energy",
    "average_controllability",
    "modal_controllability",
    # Connectivity
    "functional_connectivity",
    "ridge_effective_connectivity",
    "graphnet_effective_connectivity",
    "graph_laplacian",
    "simulate_feedforward_network",
    # Harmonise
    "BlindHarmonizer",
    "detect_site_effects",
    # Simulation
    "WilsonCowanNode",
    "WilsonCowanNetwork",
]
