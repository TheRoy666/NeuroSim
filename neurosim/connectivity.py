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
    max_iter: int = 3000,
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

    CORRECTED (see CHANGELOG): two fixes applied to the original version.
    (1) max_iter raised from 500 to 3000 -- 500 was insufficient for
    lower-T/higher-N problems (e.g. ADNI, N=400, T=197), confirmed via
    direct convergence-trajectory checks showing genuine, still-ongoing
    convergence at 500 iterations, not a stall. (2) The convergence check
    was reordered: A is now assigned from A_new BEFORE the tolerance
    check, not after. The original ordering meant that whenever the loop
    broke, it returned the value from BEFORE the update that actually
    satisfied convergence -- for problems that converged in very few
    iterations (confirmed on HCP and UNAM data), this meant the function
    silently returned the unmodified ridge-only starting point,
    regardless of lambda_graph, since the one update that would have
    incorporated the graph term was computed and then discarded on every
    call.

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

        A_prev = A.copy()
        A = A_new
        if np.linalg.norm(A - A_prev, 'fro') / (np.linalg.norm(A_prev, 'fro') + 1e-12) < tol:
            break
        t_k = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0

    return A


# Moving-block bootstrap for EC estimation uncertainty (Path B)

def block_bootstrap_ec(
    X: NDArray,
    SC: NDArray,
    ec_func: "callable" = None,
    n_boot: int = 200,
    block_length: int = 15,
    seed: int = 0,
    **ec_kwargs,
) -> NDArray:
    """Moving-block bootstrap over BOLD time series, re-estimating EC each draw.

    Per Phase 0 plan (Path B): naive timepoint bootstrap is invalid for BOLD
    because of temporal autocorrelation. This resamples contiguous blocks of
    ``block_length`` TRs (with replacement) to preserve short-range temporal
    structure, concatenates them back to length T, and re-estimates EC on
    each resampled series.

    Parameters
    ----------
    X            : (N, T) ndarray - original BOLD time series.
    SC           : (N, N) ndarray - structural connectome (passed through
                   to ``ec_func``, e.g. for the GraphNet Laplacian prior).
    ec_func      : callable(X, SC, **ec_kwargs) -> (N, N) ndarray. Defaults
                   to ``graphnet_effective_connectivity`` if not given.
    n_boot       : int   - number of bootstrap resamples (B). Phase 0
                   default: start at 200, check convergence of the
                   rank-stability metric before scaling up.
    block_length : int   - block length in TRs (Phase 0 default: 15,
                   ≈ sqrt(T) rule of thumb for T≈197-200).
    seed         : int   - RNG seed for reproducibility.
    **ec_kwargs  : passed through to ec_func.

    Returns
    -------
    EC_boot : (n_boot, N, N) ndarray - one EC estimate per resample.

    Notes
    -----
    Compute cost is n_boot × (cost of ec_func). Benchmark on 2-3 subjects
    before launching the full batch (Phase 0 compute-budget check).

    Uses graphnet_effective_connectivity's corrected version by default
    (see that function's docstring) -- this function required no changes
    itself, but every call it makes now benefits from the fix.
    """
    if ec_func is None:
        ec_func = graphnet_effective_connectivity

    X = np.asarray(X, dtype=float)
    N, T = X.shape
    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(T / block_length))
    max_start = T - block_length
    if max_start < 0:
        raise ValueError(
            f"block_length ({block_length}) exceeds series length ({T})."
        )

    EC_boot = np.zeros((n_boot, N, N))
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:T]
        X_resampled = X[:, idx]
        EC_boot[b] = ec_func(X_resampled, SC, **ec_kwargs)

    return EC_boot


def driver_node_rank_stability(
    EC_boot: NDArray,
    controllability_func: "callable",
    top_k: int = 5,
) -> dict:
    """Quantify how stable identified driver-node rankings are under
    EC-estimation uncertainty (Path B core metric).

    Parameters
    ----------
    EC_boot : (n_boot, N, N) ndarray - from ``block_bootstrap_ec``.
    controllability_func : callable(A) -> (N,) ndarray - e.g.
        ``physics.average_controllability`` or ``physics.modal_controllability``.
        Applied to each bootstrap EC (after normalisation is the caller's
        responsibility - pass an already-appropriate A, or wrap the
        normalisation into this callable).
    top_k : int - size of the top-k set for Jaccard overlap.

    Returns
    -------
    dict with:
        "kendall_tau_mean", "kendall_tau_std" : float - Kendall's τ between
            each bootstrap ranking and the original (first EC in EC_boot is
            NOT special-cased; caller should pass the point-estimate ranking
            separately if comparing bootstrap-to-original specifically).
        "jaccard_topk_mean", "jaccard_topk_std" : float - top-k overlap
            fraction across all pairs of bootstrap rankings.
        "rankings" : (n_boot, N) ndarray - raw node rankings per resample
            (argsort of controllability, descending), for further analysis.
    """
    from itertools import combinations
    from scipy.stats import kendalltau

    n_boot = EC_boot.shape[0]
    N = EC_boot.shape[1]

    scores = np.array([controllability_func(EC_boot[b]) for b in range(n_boot)])
    rankings = np.argsort(-scores, axis=1)  # descending, (n_boot, N)

    taus = []
    jaccards = []
    for i, j in combinations(range(n_boot), 2):
        tau, _ = kendalltau(scores[i], scores[j])
        taus.append(tau)
        top_i = set(rankings[i, :top_k])
        top_j = set(rankings[j, :top_k])
        jaccards.append(len(top_i & top_j) / len(top_i | top_j))

    return {
        "kendall_tau_mean": float(np.mean(taus)),
        "kendall_tau_std": float(np.std(taus)),
        "jaccard_topk_mean": float(np.mean(jaccards)),
        "jaccard_topk_std": float(np.std(jaccards)),
        "rankings": rankings,
    }


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
