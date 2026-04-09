"""
neurosim.physics
================
Discrete finite-horizon controllability engine.

The central methodological contribution of NeuroSim is the replacement of the
infinite-horizon algebraic Lyapunov solver with a **Discrete-Time Finite-Horizon
Gramian** computed via the Van Loan Doubling Algorithm (Van Loan, 1978).

Background
----------
Standard NCT pipelines solve the continuous algebraic Lyapunov equation::

    A W + W A^T + B B^T = 0   ->   W_∞

This assumes T -> ∞, producing a "vanishing cost" problem: in any stable system,
the energy required to reach any reachable state collapses to zero as time grows
unbounded. For clinical neuroscience - where we care about state transitions on
the order of cognitive task windows (2–10 s) - this is biologically indefensible.

NeuroSim enforces a finite horizon T. The brain is modelled as a Discrete-Time
Linear Time-Invariant (DT-LTI) system::

    x[k+1] = A x[k] + B u[k]

The Reachability Gramian over horizon T is the matrix sum::

    W_T = Σ_{k=0}^{T-1}  A^k B B^T (A^T)^k

Naive computation costs O(T . N³). The Doubling Algorithm reduces this to
O(log T . N³) by iterative squaring::

    Φ_0 = A,   Ψ_0 = B B^T
    Φ_{j+1} = Φ_j²
    Ψ_{j+1} = Ψ_j + Φ_j Ψ_j Φ_j^T

After ⌈log₂ T⌉ iterations, W_T = Ψ_{⌈log₂ T⌉}.

References
----------
Van Loan, C. F. (1978). Computing integrals involving the matrix exponential.
    IEEE Transactions on Automatic Control, 23(3), 395–404.
Gu, S. et al. (2015). Controllability of structural brain networks. Nature Communications.
Srivastava, P. et al. (2020). Models of communication and control for brain networks.
    PLOS Computational Biology.
"""
from __future__ import annotations

import math
import warnings
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigvals, solve_discrete_lyapunov


# Spectral normalisation

def normalise_matrix(
    A: NDArray,
    target_rho: float = 0.9,
) -> NDArray:
    """Normalise adjacency matrix so its spectral radius equals target_rho.

    A stable DT-LTI system requires ρ(A) < 1. This function rescales A so
    that ρ(A_norm) = target_rho exactly, preserving the directional
    asymmetry of the matrix (eigenvalue ratios are unchanged).

    Parameters
    ----------
    A : (N, N) array-like
        Raw connectivity matrix (structural or effective). May be unstable.
    target_rho : float
        Desired spectral radius after normalisation (default 0.9).

    Returns
    -------
    A_norm : (N, N) ndarray
        Rescaled matrix with ρ(A_norm) = target_rho.

    Raises
    ------
    ValueError
        If the matrix has a near-zero spectral radius (degenerate input).

    Notes
    -----
    The stability guard (ρ ≥ 1 check) belongs in ``compute_gramian_doubling``,
    not here. ``normalise_matrix`` is designed to *fix* unstable matrices.
    """
    A = np.asarray(A, dtype=float)
    rho = np.max(np.abs(eigvals(A)))
    if rho < 1e-12:
        raise ValueError(
            f"Matrix has near-zero spectral radius (ρ = {rho:.2e}). "
            "Check your input - the matrix may be all-zeros or degenerate."
        )
    return A * (target_rho / rho)


# Van Loan Doubling Algorithm

def compute_gramian_doubling(
    A: NDArray,
    B: NDArray,
    T: int,
) -> NDArray:
    """Compute the Discrete Finite-Horizon Reachability Gramian via Doubling.

    Implements the iterative squaring (Van Loan) approach to compute::

        W_T = Σ_{k=0}^{T-1} A^k B B^T (A^T)^k

    in O(log T . N³) rather than O(T . N³).

    Parameters
    ----------
    A : (N, N) ndarray
        Normalised effective connectivity matrix (ρ(A) < 1 required).
    B : (N, M) ndarray
        Input matrix. For full-rank control, use ``np.eye(N)``.
    T : int
        Finite time horizon (number of TR steps).

    Returns
    -------
    W_T : (N, N) ndarray
        Positive semi-definite Reachability Gramian.

    Raises
    ------
    ValueError
        If ρ(A) ≥ 1 (unstable system). Call ``normalise_matrix()`` first.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    N = A.shape[0]

    rho = np.max(np.abs(eigvals(A)))
    if rho >= 1.0:
        raise ValueError(
            f"Spectral radius ρ(A) = {rho:.4f} ≥ 1. "
            "System is unstable. Call normalise_matrix() first."
        )

    # Binary-decomposition doubling
    W       = np.zeros((N, N))
    Phi_acc = np.eye(N)      # A^{accumulated bits}
    Phi_j   = A.copy()       # A^{2^j}
    Q_j     = B @ B.T        # BBᵀ doubled at each step

    iters = math.ceil(math.log2(T)) + 1 if T > 1 else 1
    t = T

    for _ in range(iters):
        if t & 1:
            W      += Phi_acc @ Q_j @ Phi_acc.T
            Phi_acc = Phi_acc @ Phi_j
        Q_j   = Q_j + Phi_j @ Q_j @ Phi_j.T
        Phi_j = Phi_j @ Phi_j
        t >>= 1
        if t == 0:
            break

    return (W + W.T) / 2.0


# Minimum-Energy Control

def minimum_energy(
    A: NDArray,
    B: NDArray,
    x0: NDArray,
    xT: NDArray,
    T: int,
    rcond: float = 1e-10,
) -> Tuple[float, NDArray]:
    """Compute the minimum control energy for a state transition.

    The optimal (minimum-norm) control input that steers the DT-LTI system
    from state x0 to xT in exactly T steps has energy::

        E* = (xT - A^T x0)^T  W_T^{-1}  (xT - A^T x0)

    Parameters
    ----------
    A  : (N, N) ndarray - normalised connectivity matrix.
    B  : (N, M) ndarray - input matrix.
    x0 : (N,) ndarray  - initial brain state.
    xT : (N,) ndarray  - target brain state.
    T  : int           - finite horizon (TR steps).
    rcond : float      - pseudo-inverse cutoff for ill-conditioned Gramians.

    Returns
    -------
    energy : float       - Scalar minimum control energy E*.
    u_opt  : (N,) ndarray - Optimal first-step control input.
    """
    W_T   = compute_gramian_doubling(A, B, T)
    A_T   = np.linalg.matrix_power(A, T)
    delta = xT - A_T @ x0

    cond = np.linalg.cond(W_T)
    if cond > 1e12:
        warnings.warn(
            f"Gramian condition number {cond:.2e} is very large. "
            "Energy estimate may be unreliable.",
            RuntimeWarning,
        )

    W_inv  = np.linalg.pinv(W_T, rcond=rcond)
    energy = float(delta @ W_inv @ delta)
    u_opt  = B.T @ W_inv @ delta

    return energy, u_opt


# Average / modal controllability (Gu et al. 2015)

def average_controllability(A: NDArray) -> NDArray:
    """Trace of the infinite-horizon Gramian per node (Gu et al., 2015).

    Returns
    -------
    ac : (N,) ndarray - Average controllability for each node.
    """
    A = np.asarray(A, dtype=float)
    N = A.shape[0]
    W_inf = solve_discrete_lyapunov(A, np.eye(N))
    return np.diag(W_inf)


def modal_controllability(A: NDArray) -> NDArray:
    """Modal controllability per node (Gu et al., 2015).

    Returns
    -------
    mc : (N,) ndarray - Modal controllability for each node.
    """
    A = np.asarray(A, dtype=float)
    eigenvalues, V = np.linalg.eig(A)
    mc = np.array([
        np.sum((1 - np.abs(eigenvalues) ** 2) * (np.abs(V[i, :]) ** 2))
        for i in range(A.shape[0])
    ])
    return mc.real
