"""
neurosim.connectivity
=====================
Structural-constrained Effective Connectivity estimation.

The "Adjacency Matrix Problem"
------------------------------
Standard NCT pipelines commit two errors in defining their system operator **A**:

1. **Functional Connectivity (FC):** Using Pearson correlation matrices as **A**.
   FC is strictly symmetric, implying bidirectional information flow - a physical
   impossibility for feedforward pathways (sensory->motor, seizure propagation).
   The resulting control metrics are mathematically valid but biologically
   meaningless ("Teleportation Error").

2. **Hard Binary Masking:** Using raw DTI tractography to zero-out connections
   missing from the structural connectome. This ignores:
   - Polysynaptic pathways (A->C->B even when A<->B has no direct tract).
   - False negatives in probabilistic tractography (crossing fibres, long range).

NeuroSim Solution: GraphNet Soft Prior
---------------------------------------
We implement the **GraphNet** objective (Grosenick et al., 2013):

    min_A  ||X_t+1 - A X_t||²_F  +  λ_1 ||A||²_F  +  λ_2 Tr(A^T L_sc A)

where ``L_sc`` is the Graph Laplacian of the **structural** connectome (DTI).
The Laplacian penalty drives connected pairs (per DTI) to have similar effective
weights, without hard-zeroing any entry. This is a Bayesian soft prior:
  • Strong DTI evidence -> small penalty -> EC follows structure.
  • Weak/absent DTI -> large penalty -> strong functional evidence required.

The objective is convex and solved via proximal gradient descent (FISTA).

References
----------
Grosenick, L., Marshel, J. H., & Deisseroth, K. (2013). Closed-loop and
    activity-guided optogenetic control. Neuron, 86(1), 106-139.
Friston, K. J. et al. (2003). Dynamic causal modelling. NeuroImage.
Srivastava, P. et al. (2020). Models of communication and control for brain
    networks. PLOS Computational Biology, 16(8), e1007826.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


# Functional Connectivity (baseline / reference) (as per v2.0)

def functional_connectivity(X: NDArray, detrend: bool = True) -> NDArray:
    """Compute undirected Functional Connectivity via Pearson correlation.

    Provided for direct comparison with Effective Connectivity. **Not**
    recommended as the ``A`` matrix in NeuroSim's physics engine due to
    symmetric structure (see module docstring).

    Parameters
    ----------
    X : (N, T) array  - BOLD time series, N regions * T timepoints.
    detrend : bool    - Remove linear trend before correlation (recommended).

    Returns
    -------
    FC : (N, N) ndarray  - Symmetric correlation matrix with zero diagonal.
    """
    X = np.asarray(X, dtype=float)
    if detrend:
        from scipy.signal import detrend as sp_detrend
        X = sp_detrend(X, axis=1)
    FC = np.corrcoef(X)
    np.fill_diagonal(FC, 0.0)
    return FC


# Ridge Effective Connectivity (fast proxy for DCM)

def ridge_effective_connectivity(
    X: NDArray,
    alpha: float = 1.0,
    lag: int = 1,
) -> NDArray:
    """Estimate directed Effective Connectivity via MVAR Ridge Regression.

    Regresses X[t+lag] onto X[t] for every region independently, recovering
    the first-order Multivariate AutoRegressive (MVAR) transition matrix.
    This is a computationally efficient approximation to Dynamic Causal
    Modelling that recovers causal directionality.

    Parameters
    ----------
    X     : (N, T) ndarray - BOLD time series.
    alpha : float          - L2 regularisation strength (Ridge penalty).
    lag   : int            - Temporal lag in TR steps (default=1).

    Returns
    -------
    EC : (N, N) ndarray  - Asymmetric effective connectivity matrix.
                           EC[i, j] = causal influence of region j on region i.
    """
    X = np.asarray(X, dtype=float)
    N, T = X.shape
    X_past   = X[:, :-lag].T   # (T-lag, N)
    X_future = X[:, lag:].T    # (T-lag, N)

    EC = np.zeros((N, N))
    model = Ridge(alpha=alpha, fit_intercept=False)
    for i in range(N):
        model.fit(X_past, X_future[:, i])
        EC[i, :] = model.coef_

    return EC


# Graph Laplacian construction

def graph_laplacian(SC: NDArray, normalised: bool = True) -> NDArray:
    """Compute the Graph Laplacian of the structural connectome.

    L_sc = D - SC   (unnormalised)
    L_sc = I - D^{-1/2} SC D^{-1/2}   (normalised)

    Parameters
    ----------
    SC         : (N, N) ndarray - Structural connectome (symmetric, non-negative).
    normalised : bool           - Use normalised Laplacian (recommended).

    Returns
    -------
    L : (N, N) ndarray  - Graph Laplacian.
    """
    SC = np.asarray(SC, dtype=float)
    SC = (SC + SC.T) / 2.0  # enforce symmetry
    np.fill_diagonal(SC, 0.0)

    D = np.diag(SC.sum(axis=1))
    L = D - SC

    if normalised:
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-12)))
        L = d_inv_sqrt @ L @ d_inv_sqrt

    return L


# GraphNet Effective Connectivity (FISTA solver)

def graphnet_effective_connectivity(
    X: NDArray,
    SC: NDArray,
    lambda_ridge: float = 1.0,
    lambda_graph: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
    lag: int = 1,
) -> NDArray:
    """Estimate EC with Graph Laplacian regularisation (GraphNet).

    Minimises the objective::

        J(A) = ||X_{t+1} - A X_t||²_F
               + λ_ridge · ||A||²_F
               + λ_graph · Tr(A^T L_sc A)

    via **FISTA** (Beck & Teboulle, 2009) - proximal gradient descent with
    Nesterov momentum. The combined quadratic penalty (Ridge + GraphNet) has
    closed-form proximal operator, enabling fast convergence.

    Parameters
    ----------
    X            : (N, T) ndarray - BOLD time series.
    SC           : (N, N) ndarray - Structural connectome (DTI).
    lambda_ridge : float          - L2 / Ridge penalty weight.
    lambda_graph : float          - Graph Laplacian penalty weight.
    max_iter     : int            - Maximum FISTA iterations.
    tol          : float          - Convergence tolerance (||ΔA||_F / ||A||_F).
    lag          : int            - Temporal lag (default=1 TR).

    Returns
    -------
    EC : (N, N) ndarray  - GraphNet-regularised effective connectivity.
    """
    X = np.asarray(X, dtype=float)
    SC = np.asarray(SC, dtype=float)
    N, T = X.shape
    X_t = X[:, :-lag]
    X_tp1 = X[:, lag:]
    L = graph_laplacian(SC, normalised=True)

    XXT = X_t @ X_t.T
    YXT = X_tp1 @ X_t.T
    Reg = lambda_ridge * np.eye(N) + lambda_graph * L

    # FISTA setup
    A = ridge_effective_connectivity(X, alpha=lambda_ridge, lag=lag)
    A_prev = A.copy()
    t_k = 1.0
    eigmax_XXT = np.max(np.abs(np.linalg.eigvalsh(XXT)))
    step = 1.0 / (2.0 * eigmax_XXT + 2.0 * np.max(np.abs(np.linalg.eigvalsh(Reg))) + 1e-8)

    for k in range(max_iter):
        Y = A + ((t_k - 1) / (t_k + 1)) * (A - A_prev)
        grad_data = 2.0 * (Y @ XXT - YXT)
        grad_reg = 2.0 * (Reg @ Y.T).T
        A_new = Y - step * (grad_data + grad_reg)

        if np.linalg.norm(A_new - A, 'fro') / (np.linalg.norm(A, 'fro') + 1e-12) < tol:
            break

        A_prev = A.copy()
        A = A_new
        t_k = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0

    return A


# Teleportation Error Trial (FC vs EC in NCT) (v3.0 core**)

def simulate_feedforward_network(
    n_nodes: int = 3,
    n_timepoints: int = 5000,
    causal_weight: float = 0.85,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[NDArray, NDArray]:
    """Generate ground-truth feedforward time series.

    Creates a serial causal chain: Node 0 -> Node 1 -> ... -> Node (n-1).
    Used to demonstrate the "Teleportation Error" of FC-based NCT.

    Returns
    -------
    X        : (n_nodes, n_timepoints) ndarray - Simulated BOLD-like time series.
    A_true   : (n_nodes, n_nodes) ndarray     - Ground-truth causal matrix.
    """
    rng = np.random.default_rng(seed)
    A_true = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes - 1):
        A_true[i + 1, i] = causal_weight

    X = np.zeros((n_nodes, n_timepoints))
    for t in range(1, n_timepoints):
        X[:, t] = A_true @ X[:, t - 1] + rng.normal(0, noise_std, n_nodes)

    return X, A_true
