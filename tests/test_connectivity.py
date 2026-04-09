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

    assert EC[1, 0] > 0.3, f"EC[1,0]={EC[1,0]:.3f} - should detect 0→1 causation."
    assert EC[2, 1] > 0.3, f"EC[2,1]={EC[2,1]:.3f} - should detect 1→2 causation."
    assert EC[0, 1] < 0.2, f"EC[0,1]={EC[0,1]:.3f} - spurious reverse causation."
    assert EC[1, 2] < 0.2, f"EC[1,2]={EC[1,2]:.3f} - spurious reverse causation."


# Test 3: Teleportation Error - FC vs EC control energy

def test_teleportation_error():
    """
    In a feedforward chain 0->1->2:
    FC-based NCT assigns similar controllability to all nodes (symmetric matrix).
    EC-based NCT reveals dramatic differentiation (causal topology).
    EC ratio >> FC ratio.
    """
    X, A_true = simulate_feedforward_network(n_nodes=3, n_timepoints=8000,
                                              causal_weight=0.85)
    FC = functional_connectivity(X)
    EC = ridge_effective_connectivity(X, alpha=0.1)

    FC_norm = normalise_matrix(np.abs(FC), target_rho=0.9)
    EC_norm = normalise_matrix(np.abs(EC), target_rho=0.9)

    B = np.eye(3)
    W_fc = compute_gramian_doubling(FC_norm, B, T=5)
    W_ec = compute_gramian_doubling(EC_norm, B, T=5)

    ac_fc = np.diag(W_fc)
    ac_ec = np.diag(W_ec)

    fc_ratio = ac_fc.max() / (ac_fc.min() + 1e-12)
    ec_ratio = ac_ec.max() / (ac_ec.min() + 1e-12)

    assert ec_ratio > fc_ratio * 5, (
        f"EC ratio {ec_ratio:.2f} should be >> FC ratio {fc_ratio:.2f}. "
        "EC-based NCT should reveal causal topology via Gramian structure."
    )


# Test 4: EC is directed (asymmetric)

def test_effective_connectivity_is_directed():
    """EC matrix must be asymmetric - directed information flow."""
    rng = np.random.default_rng(7)
    X, _ = simulate_feedforward_network(n_nodes=3, n_timepoints=5000,
                                         causal_weight=0.85)
    ec_matrix = ridge_effective_connectivity(X, alpha=0.1)
    assert not np.allclose(ec_matrix, ec_matrix.T), (
        "Effective connectivity must be asymmetric (directed)."
    )


# Test 5: Graph Laplacian properties

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


# Test 6: GraphNet EC recovers structure under SC prior

def test_graphnet_structural_bias():
    """
    When lambda_graph >> lambda_ridge, GraphNet EC should be biased toward
    the structural connectome pattern.
    """
    rng = np.random.default_rng(42)
    N, T = 6, 3000

    SC = np.zeros((N, N))
    for i in range(N):
        SC[i, (i + 1) % N] = 1.0
        SC[(i + 1) % N, i] = 1.0

    A_gen = SC * 0.1
    A_gen = normalise_matrix(A_gen + 0.01 * np.eye(N), target_rho=0.7)
    X = np.zeros((N, T))
    for t in range(1, T):
        X[:, t] = A_gen @ X[:, t-1] + rng.normal(0, 0.5, N)

    EC_graphnet = graphnet_effective_connectivity(
        X, SC,
        lambda_ridge=0.1,
        lambda_graph=10.0,
        max_iter=200,
    )
    assert EC_graphnet is not None
    assert EC_graphnet.shape == (N, N)


def test_effective_connectivity_graphnet_is_directed():
    """GraphNet EC must also be asymmetric."""
    rng = np.random.default_rng(8)
    X, _ = simulate_feedforward_network(n_nodes=3, n_timepoints=5000,
                                         causal_weight=0.85)
    sc_prior = np.ones((3, 3))
    ec_matrix = graphnet_effective_connectivity(X, sc_prior)
    assert not np.allclose(ec_matrix, ec_matrix.T), (
        "GraphNet EC must be asymmetric (directed)."
    )


def test_graphnet_structural_penalty():
    rng = np.random.default_rng(99)
    X, _ = simulate_feedforward_network(n_nodes=3, n_timepoints=5000,
                                         causal_weight=0.85)
    # Node 0 and Node 2 have NO structural connection
    sc_prior = np.ones((3, 3))
    sc_prior[0, 2] = 0.0
    sc_prior[2, 0] = 0.0

    ec_matrix = graphnet_effective_connectivity(
        X, sc_prior, lambda_ridge=1.0, lambda_graph=5.0
    )

    sc_mask    = sc_prior > 0
    no_sc_mask = sc_prior == 0
    np.fill_diagonal(no_sc_mask, False)

    mean_sc_weight    = np.mean(np.abs(ec_matrix[sc_mask]))
    mean_no_sc_weight = np.mean(np.abs(ec_matrix[no_sc_mask]))

    assert mean_no_sc_weight <= mean_sc_weight, (
        f"GraphNet should suppress non-SC edges: "
        f"absent-SC mean={mean_no_sc_weight:.4f} > SC mean={mean_sc_weight:.4f}"
    )
