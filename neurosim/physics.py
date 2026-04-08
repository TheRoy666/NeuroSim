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

    A W + W A^T + B B^T = 0   →   W_∞

This assumes T → ∞, producing a "vanishing cost" problem: in any stable system,
the energy required to reach any reachable state collapses to zero as time grows
unbounded. For clinical neuroscience — where we care about state transitions on
the order of cognitive task windows (2–10 s) — this is biologically indefensible.

NeuroSim enforces a finite horizon T. The brain is modelled as a Discrete-Time
Linear Time-Invariant (DT-LTI) system::

    x[k+1] = A x[k] + B u[k]

The Reachability Gramian over horizon T is the matrix sum::

    W_T = Σ_{k=0}^{T-1}  A^k B B^T (A^T)^k

Naiive computation costs O(T . N³). The Doubling Algorithm reduces this to
O(log T · N³) by iterative squaring::

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


def normalise_matrix(
    A: NDArray,
    target_rho: float = 0.9,
) -> NDArray:
    """Normalise adjacency matrix so its spectral radius < 1.

    A stable DT-LTI system requires ρ(A) < 1. Two strategies:

    * ``"spectral"``  - divide by (ρ(A) + δ), where δ = 0.01.
      Preserves directional asymmetry. Preferred for Effective Connectivity.
    * ``"trace"``     - divide by Tr(A) + 1.
      Cruder, suitable when A is ill-conditioned.

    Parameters
    ----------
    A : (N, N) array-like
        Raw connectivity matrix (structural or effective).
    target_rho : float
        Desired spectral radius after normalisation (default 0.9).
    method : {"spectral", "trace"}
        Normalisation strategy.

    Returns
    -------
    A_norm : (N, N) ndarray
        Normalised connectivity matrix with ρ(A_norm) = target_rho.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    N = A.shape[0]

    # Guardrail: system must be stable
    rho = np.max(np.abs(eigvals(A)))
    if rho >= 1.0:
        raise ValueError(
            f"System is unstable (spectral radius ρ = {rho:.4f} ≥ 1). "
            "Call normalise_matrix() first."
        )

    # Initialise
    Phi = A.copy()           # Φ_j = A^(2^j)
    Psi = B @ B.T            # Current Gramian block
    W = np.zeros((N, N))     # Accumulated Gramian
    Phi_acc = np.eye(N)      # Accumulated state transition

    t = T
    while t > 0:
        if t % 2 == 1:       # Current bit is set
            W += Phi_acc @ Psi @ Phi_acc.T
            Phi_acc = Phi_acc @ Phi

        # Double the block
        Psi = Psi + Phi @ Psi @ Phi.T
        Phi = Phi @ Phi

        t //= 2

    # Enforce symmetry
    W = 0.5 * (W + W.T)
    return W

# Van Loan Doubling Algorithm

def compute_gramian_doubling(
    A: NDArray,
    B: NDArray,
    T: int,
) -> NDArray:
    """Compute the Discrete Finite-Horizon Reachability Gramian via Doubling.

    Implements the iterative squaring (Van Loan) approach to compute::

        W_T = Σ_{k=0}^{T-1} A^k B B^T (A^T)^k

    in O(log T · N³) rather than O(T · N³).

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
        If ρ(A) ≥ 1 (unstable system).

    Notes
    -----
    The doubling loop handles non-power-of-2 horizons by decomposing T in
    binary and accumulating partial sums, analogous to fast exponentiation.
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

    # Seed values
    Phi = A.copy()          # Φ_j = A^{2^j}
    Q0  = B @ B.T           # base BBᵀ
    Psi = Q0.copy()         # accumulated Gramian

    # Binary decomposition of T for O(log T) accumulation
    W = np.zeros((N, N))
    Phi_acc = np.eye(N)     # tracks A^{bits accumulated so far}
    t = T

    iters = math.ceil(math.log2(T)) + 1 if T > 1 else 1
    Phi_j = A.copy()
    Q_j   = Q0.copy()

    for j in range(iters):
        if t & 1:  # binary bit is set
            W += Phi_acc @ Q_j @ Phi_acc.T
            Phi_acc = Phi_acc @ Phi_j

        # Double
        Q_j   = Q_j + Phi_j @ Q_j @ Phi_j.T
        Phi_j = Phi_j @ Phi_j

        t >>= 1
        if t == 0:
            break

    # Symmetrise
    W = (W + W.T) / 2.0
    return W


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

    where W_T is the Finite-Horizon Reachability Gramian.

    Parameters
    ----------
    A  : (N, N) ndarray - normalised connectivity matrix.
    B  : (N, M) ndarray - input matrix.
    x0 : (N,) ndarray  - initial brain state (e.g., resting-state BOLD).
    xT : (N,) ndarray  - target brain state (e.g., task-evoked BOLD).
    T  : int           - finite horizon (TR steps).
    rcond : float      - pseudo-inverse cutoff for ill-conditioned Gramians.

    Returns
    -------
    energy : float
        Scalar minimum control energy E*.
    u_opt : (N,) ndarray
        Optimal control input at t=0 (first time step).
    """
    W_T = compute_gramian_doubling(A, B, T)

    # Free evolution term
    A_T = np.linalg.matrix_power(A, T)
    delta = xT - A_T @ x0

    # Condition monitoring
    cond = np.linalg.cond(W_T)
    if cond > 1e12:
        warnings.warn(
            f"Gramian condition number {cond:.2e} is very large. "
            "Energy estimate may be unreliable.",
            RuntimeWarning,
        )

    W_inv = np.linalg.pinv(W_T, rcond=rcond)
    energy = float(delta @ W_inv @ delta)
    u_opt = B.T @ W_inv @ delta   # optimal first-step control

    return energy, u_opt


# Average / modal controllability (Gu et al. 2015)

def average_controllability(A: NDArray) -> NDArray:
    """Trace of the infinite-horizon Gramian per node (Gu et al., 2015).

    While NeuroSim's primary engine uses finite-horizon physics, average
    controllability (via the infinite-horizon Gramian) remains useful as a
    *network topology* descriptor, capturing how easily a node can steer the
    system to a broad range of states.

    Returns
    -------
    ac : (N,) ndarray
        Average controllability for each node (diagonal of W_∞).
    """
    A = np.asarray(A, dtype=float)
    N = A.shape[0]
    W_inf = solve_discrete_lyapunov(A, np.eye(N))
    return np.diag(W_inf)


def modal_controllability(A: NDArray) -> NDArray:
    """Modal controllability per node (Gu et al., 2015).

    Quantifies the ability of each node to steer the system into difficult-
    to-reach (low-energy) modes. Nodes with high modal controllability can
    push the system into states that require large control energy.

    Returns
    -------
    mc : (N,) ndarray
        Modal controllability for each node.
    """
    A = np.asarray(A, dtype=float)
    eigenvalues, V = np.linalg.eig(A)
    mc = np.array([
        np.sum((1 - np.abs(eigenvalues)**2) * (np.abs(V[i, :])**2))
        for i in range(A.shape[0])
    ])
    return mc.real
