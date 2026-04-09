"""
tests/test_physics.py
=====================
Formal verification of the NeuroSim physics engine.

Tests are structured as engineering verification checks that mirror the
requirements in the NeuroSim Engineering Verification Checklist:

1. Spectral radius of normalised matrix == target_rho (normalisation correctness).
2. Gramian W_T is positive semi-definite (validity of energy metric).
3. Doubling algorithm converges to naive sum for small T.
4. Control energy is non-negative.
5. compute_gramian_doubling raises on unstable input (stability guardrail).
6. Doubling algorithm is faster than naive summation for large T.
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


# Fixture

@pytest.fixture
def random_stable_system():
    """Random 20×20 stable connectivity matrix + full input matrix."""
    rng = np.random.default_rng(0)
    A_raw = rng.normal(0, 1, (20, 20))
    A = normalise_matrix(A_raw, target_rho=0.9)
    B = np.eye(20)
    return A, B, 20


# Test 1: normalise_matrix produces correct spectral radius

def test_normalise_target_rho():
    """normalise_matrix should rescale ANY matrix (stable or not) to target_rho."""
    rng = np.random.default_rng(42)
    A_raw = rng.normal(0, 1, (15, 15))
    for target in [0.7, 0.9, 0.95]:
        A = normalise_matrix(A_raw, target_rho=target)
        rho = np.max(np.abs(np.linalg.eigvals(A)))
        assert abs(rho - target) < 1e-10, (
            f"Target ρ={target}, achieved ρ={rho:.6f}"
        )


def test_normalise_works_on_unstable_matrix():
    """

    This test verifies that normalise_matrix correctly rescales a wildly
    unstable matrix (ρ >> 1) to the target_rho.
    """
    A_unstable = np.array([[1.5, 0.2], [0.1, 0.9]])
    rho_before = np.max(np.abs(np.linalg.eigvals(A_unstable)))
    assert rho_before > 1.0, "Test fixture should be unstable"

    A_norm = normalise_matrix(A_unstable, target_rho=0.9)
    rho_after = np.max(np.abs(np.linalg.eigvals(A_norm)))
    assert abs(rho_after - 0.9) < 1e-10, (
        f"After normalisation: ρ={rho_after:.6f}, expected 0.9"
    )


def test_normalise_raises_on_degenerate():
    """normalise_matrix should raise only for a near-zero (degenerate) matrix."""
    A_zero = np.zeros((5, 5))
    with pytest.raises(ValueError, match="near-zero spectral radius"):
        normalise_matrix(A_zero)


# Test 2: Gramian is positive semi-definite and symmetric

def test_gramian_positive_semidefinite(random_stable_system):
    A, B, N = random_stable_system
    W = compute_gramian_doubling(A, B, T=10)
    eigenvalues = np.linalg.eigvalsh(W)
    assert np.all(eigenvalues >= -1e-9), (
        f"Gramian has negative eigenvalue: {eigenvalues.min():.2e}"
    )


def test_gramian_symmetric(random_stable_system):
    A, B, N = random_stable_system
    W = compute_gramian_doubling(A, B, T=10)
    assert np.allclose(W, W.T, atol=1e-10), "Gramian is not symmetric."


def test_gramian_psd_small_system():
    """Small known-stable system."""
    A_stable = np.array([[0.5, 0.2], [0.1, 0.4]])
    B = np.eye(2)
    W = compute_gramian_doubling(A_stable, B, T=50)
    eigenvalues = np.linalg.eigvalsh(W)
    assert np.all(eigenvalues >= -1e-10), "Gramian is not PSD."


# Test 3: Doubling matches naive summation

def _naive_gramian(A, B, T):
    """Reference O(T·N³) implementation for numerical comparison."""
    N = A.shape[0]
    W = np.zeros((N, N))
    Ak = np.eye(N)
    Q = B @ B.T
    for _ in range(T):
        W += Ak @ Q @ Ak.T
        Ak = Ak @ A
    return W


def test_doubling_matches_naive(random_stable_system):
    A, B, N = random_stable_system
    for T in [1, 2, 4, 8, 16]:
        W_naive    = _naive_gramian(A, B, T)
        W_doubling = compute_gramian_doubling(A, B, T)
        assert np.allclose(W_naive, W_doubling, atol=1e-8), (
            f"Doubling mismatch at T={T}: max|ΔW|={np.max(np.abs(W_naive - W_doubling)):.2e}"
        )


# Test 4: Control energy properties

def test_minimum_energy_nonnegative(random_stable_system):
    A, B, N = random_stable_system
    rng = np.random.default_rng(1)
    x0 = rng.normal(0, 1, N)
    xT = rng.normal(0, 1, N)
    energy, _ = minimum_energy(A, B, x0, xT, T=8)
    assert energy >= -1e-9, f"Negative energy: {energy:.4e}"


def test_trivial_transition_zero_energy(random_stable_system):
    """Free evolution xT = A^T x0 requires zero control energy."""
    A, B, N = random_stable_system
    rng = np.random.default_rng(2)
    x0 = rng.normal(0, 1, N)
    T  = 5
    xT = np.linalg.matrix_power(A, T) @ x0
    energy, _ = minimum_energy(A, B, x0, xT, T=T)
    assert energy < 1e-6, (
        f"Free-evolution energy should be ~0, got {energy:.2e}"
    )


# Test 5: compute_gramian_doubling raises on unstable input

def test_gramian_raises_on_unstable():
    """
    The stability guardrail (ValueError) belongs in compute_gramian_doubling,
    not in normalise_matrix.
    """
    A_unstable = 2.0 * np.eye(5)   # ρ = 2.0 > 1
    with pytest.raises(ValueError, match="unstable"):
        compute_gramian_doubling(A_unstable, np.eye(5), T=5)


# Test 6: Controllability metrics

def test_controllability_shapes(random_stable_system):
    A, B, N = random_stable_system
    ac = average_controllability(A)
    mc = modal_controllability(A)
    assert ac.shape == (N,)
    assert mc.shape == (N,)
    assert np.all(ac >= 0), "Average controllability must be non-negative."


# Test 7: Speedup (optional - verifies O(log T) claim)

def test_doubling_speedup():
    N, T = 30, 100
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
    assert speedup > 1.0, (
        f"Doubling ({t_doubling*1e3:.1f}ms) not faster than naive ({t_naive*1e3:.1f}ms)."
    )
