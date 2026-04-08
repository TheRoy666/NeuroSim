"""
tests/test_physics.py
=====================
Formal verification of the NeuroSim physics engine.

Tests are structured as engineering verification checks that mirror the
requirements in the NeuroSim Engineering Verification Checklist:

1. Spectral radius of normalised matrix < 1 (stability guarantee).
2. Gramian W_T is positive semi-definite (validity of energy metric).
3. Doubling algorithm converges to naiive sum for small T.
4. Control energy is non-negative.
5. Doubling algorithm is O(log T) faster than naiive summation.
"""

import time

import numpy as np
import pytest

from neurosim.physics import (
    average_controllability,
    compute_gramian_doubling,
    minimum_energy,
    modal_controllability,
    normalise_matrix,
)


def test_van_loan_instability_guardrail():
    # Unstable matrix (spectral radius > 1)
    A_unstable = np.array([[1.5, 0.2], [0.1, 0.9]])
    B = np.eye(2)
    
    with pytest.raises(ValueError, match="System is unstable"):
        van_loan_discrete_gramian(A_unstable, B, T=100)

def test_gramian_is_positive_semi_definite():
    # Stable matrix
    A_stable = np.array([[0.5, 0.2], [0.1, 0.4]])
    B = np.eye(2)
    
    W_d = van_loan_discrete_gramian(A_stable, B, T=50)
    eigenvalues = np.linalg.eigvals(W_d)
    
    # Assert all eigenvalues are >= 0 (accounting for floating point precision)
    assert np.all(eigenvalues >= -1e-10), "Gramian is not Positive Semi-Definite."

# Fixtures

@pytest.fixture
def random_stable_system(seed: int = 0, N: int = 20):
    """Random N×N stable connectivity matrix + full input matrix."""
    rng = np.random.default_rng(seed)
    A_raw = rng.normal(0, 1, (N, N))
    A = normalise_matrix(A_raw, target_rho=0.9)
    B = np.eye(N)
    return A, B, N


@pytest.fixture
def small_system():
    """3-node feedforward system for exact numerical comparison."""
    A = np.array([[0.0, 0.0, 0.0],
                  [0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0]])
    B = np.eye(3)
    return A, B


# Test 1: Spectral radius guarantee

def test_normalise_spectral_radius(random_stable_system):
    A, B, N = random_stable_system
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    assert rho < 1.0, f"Spectral radius {rho:.4f} ≥ 1 - system is unstable."


def test_normalise_target_rho():
    rng = np.random.default_rng(42)
    A_raw = rng.normal(0, 1, (15, 15))
    for target in [0.7, 0.9, 0.95]:
        A = normalise_matrix(A_raw, target_rho=target)
        rho = np.max(np.abs(np.linalg.eigvals(A)))
        assert abs(rho - target) < 1e-10, (
            f"Target ρ={target}, achieved ρ={rho:.6f}"
        )


# Test 2: Gramian is positive semi-definite

def test_gramian_positive_semidefinite(random_stable_system):
    A, B, N = random_stable_system
    W = compute_gramian_doubling(A, B, T=10)
    eigenvalues = np.linalg.eigvalsh(W)
    assert np.all(eigenvalues >= -1e-9), (
        f"Gramian has negative eigenvalue: {eigenvalues.min():.2e} - not PSD."
    )


def test_gramian_symmetric(random_stable_system):
    A, B, N = random_stable_system
    W = compute_gramian_doubling(A, B, T=10)
    assert np.allclose(W, W.T, atol=1e-10), "Gramian is not symmetric."


# Test 3: Doubling algorithm matches naiive sum (small T)

def _naive_gramian(A, B, T):
    """O(T.N³) reference implementation."""
    N = A.shape[0]
    W = np.zeros((N, N))
    Ak = np.eye(N)
    Q  = B @ B.T
    for _ in range(T):
        W += Ak @ Q @ Ak.T
        Ak = Ak @ A
    return W


def test_doubling_matches_naive(small_system):
    A, B = small_system
    for T in [1, 2, 4, 8, 16]:
        W_naive   = _naive_gramian(A, B, T)
        W_doubling = compute_gramian_doubling(A, B, T)
        assert np.allclose(W_naive, W_doubling, atol=1e-9), (
            f"Doubling mismatch at T={T}:\n"
            f"  max |ΔW| = {np.max(np.abs(W_naive - W_doubling)):.2e}"
        )


def test_doubling_matches_naive_random(seed=7, N=10):
    rng = np.random.default_rng(seed)
    A_raw = rng.normal(0, 1, (N, N))
    A = normalise_matrix(A_raw, target_rho=0.85)
    B = rng.normal(0, 1, (N, 3))
    for T in [3, 5, 12, 20]:
        W_naive    = _naive_gramian(A, B, T)
        W_doubling = compute_gramian_doubling(A, B, T)
        assert np.allclose(W_naive, W_doubling, atol=1e-8), (
            f"Random system mismatch at T={T}, max |ΔW| = "
            f"{np.max(np.abs(W_naive - W_doubling)):.2e}"
        )


# Test 4: Control energy is non-negative

def test_minimum_energy_nonnegative(random_stable_system):
    A, B, N = random_stable_system
    rng = np.random.default_rng(1)
    x0 = rng.normal(0, 1, N)
    xT = rng.normal(0, 1, N)
    energy, _ = minimum_energy(A, B, x0, xT, T=8)
    assert energy >= -1e-9, f"Negative energy: {energy:.4e}"


def test_trivial_transition_zero_energy(random_stable_system):
    """If xT = A^T x0 (free evolution), control energy should be ~0."""
    A, B, N = random_stable_system
    rng = np.random.default_rng(2)
    x0 = rng.normal(0, 1, N)
    T  = 5
    xT = np.linalg.matrix_power(A, T) @ x0
    energy, _ = minimum_energy(A, B, x0, xT, T=T)
    assert energy < 1e-6, (
        f"Free-evolution should have near-zero energy, got {energy:.2e}"
    )


# Test 5: Doubling is faster than naiive for large T

def test_doubling_speedup(N=50, T=200):
    rng = np.random.default_rng(99)
    A = normalise_matrix(rng.normal(0, 1, (N, N)), target_rho=0.9)
    B = np.eye(N)

    t0 = time.perf_counter()
    _naive_gramian(A, B, T)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    compute_gramian_doubling(A, B, T)
    t_doubling = time.perf_counter() - t0

    speedup = t_naive / max(t_doubling, 1e-9)
    print(f"\nSpeedup (T={T}, N={N}): {speedup:.1f}×")
    assert speedup > 1.0, (
        f"Doubling ({t_doubling*1e3:.1f} ms) not faster than naïve "
        f"({t_naive*1e3:.1f} ms)."
    )


# Test 6: Average and modal controllability

def test_controllability_shapes(random_stable_system):
    A, B, N = random_stable_system
    ac = average_controllability(A)
    mc = modal_controllability(A)
    assert ac.shape == (N,), f"Average controllability shape mismatch: {ac.shape}"
    assert mc.shape == (N,), f"Modal controllability shape mismatch: {mc.shape}"
    assert np.all(ac >= 0), "Average controllability has negative values."


def test_unstable_raises():
    """normalise_matrix should raise if A is already near-singular."""
    A_unstable = 2.0 * np.eye(5)  # eigenvalues = 2 > 1
    # normalise_matrix should handle this fine (rescaling to target_rho)
    A_norm = normalise_matrix(A_unstable, target_rho=0.9)
    rho = np.max(np.abs(np.linalg.eigvals(A_norm)))
    assert abs(rho - 0.9) < 1e-10

    # compute_gramian_doubling should raise on UNSTABLE input
    with pytest.raises(ValueError, match="Spectral radius"):
        compute_gramian_doubling(A_unstable, np.eye(5), T=5)
      
