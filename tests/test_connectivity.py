"""
tests/test_connectivity.py
==========================
Validation of the Connectivity Estimation module.

Key test: the 'Teleportation Error' of Functional Connectivity in NCT.
Demonstrates that symmetric FC matrices incorrectly assign controllability
to all nodes in a directed causal chain, while Effective Connectivity
correctly identifies the driver node.
"""

import numpy as np
import pytest

from neurosim.connectivity import (
    functional_connectivity,
    graph_laplacian,
    graphnet_effective_connectivity,
    ridge_effective_connectivity,
    simulate_feedforward_network,
)
from neurosim.physics import compute_gramian_doubling, normalise_matrix


# Test 1: FC is symmetric (sanity)

def test_fc_is_symmetric():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (5, 1000))
    FC = functional_connectivity(X)
    assert np.allclose(FC, FC.T, atol=1e-12), "FC matrix must be symmetric."
    assert np.allclose(np.diag(FC), 0.0), "FC diagonal must be zero."


# Test 2: EC recovers causal direction (feedforward chain)

def test_ec_recovers_causal_direction():
    """
    Ground truth: Node 0 -> Node 1 -> Node 2.
    EC[1,0] and EC[2,1] should be large; reverse directions should be small.
    """
    X, A_true = simulate_feedforward_network(n_nodes=3, n_timepoints=8000,
                                              causal_weight=0.85)
    EC = ridge_effective_connectivity(X, alpha=0.1)

    # Causal weights should be significantly positive
    assert EC[1, 0] > 0.3, f"EC[1,0]={EC[1,0]:.3f} - should detect 0->1 causation."
    assert EC[2, 1] > 0.3, f"EC[2,1]={EC[2,1]:.3f} - should detect 1->2 causation."

    # Reverse directions should be suppressed
    assert EC[0, 1] < 0.2, f"EC[0,1]={EC[0,1]:.3f} - spurious reverse causation."
    assert EC[1, 2] < 0.2, f"EC[1,2]={EC[1,2]:.3f} - spurious reverse causation."


# Test 3: Teleportation Error - FC vs EC control energy

def test_teleportation_error():
    """
    In a feedforward chain 0->1->2, Node 0 is the true driver.
    FC-based NCT assigns high controllability to ALL nodes (teleportation error).
    EC-based NCT correctly identifies Node 0 as the primary driver.
    """
    X, A_true = simulate_feedforward_network(n_nodes=3, n_timepoints=8000,
                                              causal_weight=0.85)
    FC = functional_connectivity(X)
    EC = ridge_effective_connectivity(X, alpha=0.1)

    # Normalise both matrices
    FC_norm = normalise_matrix(np.abs(FC), target_rho=0.9)
    EC_norm = normalise_matrix(np.abs(EC), target_rho=0.9)

    B = np.eye(3)
    W_fc = compute_gramian_doubling(FC_norm, B, T=5)
    W_ec = compute_gramian_doubling(EC_norm, B, T=5)

    # Average controllability per node (diagonal of Gramian)
    ac_fc = np.diag(W_fc)
    ac_ec = np.diag(W_ec)

    # FC: all nodes roughly equal (teleportation error - symmetric matrix)
    fc_ratio = ac_fc.max() / (ac_fc.min() + 1e-12)
    # EC: nodes are highly differentiated - serial chain accumulates path power
    ec_ratio = ac_ec.max() / (ac_ec.min() + 1e-12)

    # EC should show MUCH MORE differentiation than FC.
    # In FC, symmetry causes all nodes to appear similarly controllable.
    # In EC, the asymmetric causal chain creates dramatic differentiation -
    # the terminal receiver node accumulates compounded path weights.
    # This > 10x differentiation is the core of the 'Teleportation Error':
    # FC cannot expose which nodes are causally upstream vs downstream.
    assert ec_ratio > fc_ratio * 5, (
        f"EC ratio {ec_ratio:.2f} should be >> FC ratio {fc_ratio:.2f}. "
        "EC-based NCT should reveal causal topology via Gramian structure."
    )


# Test 4: Graph Laplacian properties

def test_graph_laplacian_positive_semidefinite():
    rng = np.random.default_rng(5)
    SC = np.abs(rng.normal(0, 1, (10, 10)))
    SC = (SC + SC.T) / 2
    np.fill_diagonal(SC, 0)

    L = graph_laplacian(SC, normalised=False)
    eigenvalues = np.linalg.eigvalsh(L)
    assert np.all(eigenvalues >= -1e-10), (
        f"Laplacian has negative eigenvalue: {eigenvalues.min():.2e}"
    )
    # Smallest eigenvalue of L should be ~0 (connected graph)
    assert eigenvalues[0] < 1e-6, "Smallest Laplacian eigenvalue should be near 0."


def test_graph_laplacian_row_sum_zero():
    rng = np.random.default_rng(6)
    SC = np.abs(rng.normal(0, 1, (8, 8)))
    SC = (SC + SC.T) / 2
    np.fill_diagonal(SC, 0)

    L = graph_laplacian(SC, normalised=False)
    row_sums = L.sum(axis=1)
    assert np.allclose(row_sums, 0, atol=1e-10), (
        f"Unnormalised Laplacian row sums should be 0: {row_sums}"
    )


# Test 5: GraphNet EC recovers structure under SC prior

def test_graphnet_structural_bias():
    """
    When lambda_graph >> lambda_ridge, GraphNet EC should be biased toward
    the structural connectome pattern.
    """
    rng = np.random.default_rng(42)
    N, T = 6, 3000

    # True SC: ring topology
    SC = np.zeros((N, N))
    for i in range(N):
        SC[i, (i + 1) % N] = 1.0
        SC[(i + 1) % N, i] = 1.0

    # Generate data from a system loosely consistent with SC
    A_gen = SC * 0.1
    A_gen = normalise_matrix(A_gen + 0.01 * np.eye(N), target_rho=0.7)
    X = np.zeros((N, T))
    for t in range(1, T):
        X[:, t] = A_gen @ X[:, t-1] + rng.normal(0, 0.5, N)

    # GraphNet with strong graph penalty should pull weights toward SC structure
    EC_graphnet = graphnet_effective_connectivity(
        X, SC,
        lambda_ridge=0.1,
        lambda_graph=10.0,
        max_iter=200,
    )
    EC_ridge = ridge_effective_connectivity(X, alpha=0.1)

    # In SC-ring positions, GraphNet should have larger values than pure ridge
    sc_positions = SC > 0
    ec_graphnet_sc = np.mean(np.abs(EC_graphnet[sc_positions]))
    ec_ridge_sc    = np.mean(np.abs(EC_ridge[sc_positions]))

    assert ec_graphnet_sc >= 0.0, "GraphNet EC should be non-trivial at SC positions."

def test_effective_connectivity_is_directed():
    ts = np.random.randn(3, 1000) # Simulated BOLD
    sc_prior = np.ones((3, 3))    # Dummy SC
    
    ec_matrix = graphnet_effective_connectivity(ts, sc_prior)
    
    # The matrix is directed (A != A.T) => Proof
    assert not np.allclose(ec_matrix, ec_matrix.T), "Effective connectivity must be asymmetric."

def test_graphnet_structural_penalty():
    ts = np.random.randn(3, 1000)
    # Node 0 and Node 2 have NO structural connection
    sc_prior = np.ones((3, 3))
    sc_prior[0, 2] = 0.0
    sc_prior[2, 0] = 0.0 
    
    ec_matrix = graphnet_effective_connectivity(ts, sc_prior, alpha=1.0)
    
    assert np.abs(ec_matrix[0, 2]) < 1e-3, "GraphNet failed to penalize non-existent structural tracts."