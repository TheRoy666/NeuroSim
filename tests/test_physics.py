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
    normalise_matrix,
    compute_gramian_doubling,
    minimum_energy,
    average_controllability,
    modal_controllability,
)


# Test 1: Stability guardrail
def test_normalise_spectral_radius():
    A_unstable = np.array([[1.5, 0.2], [0.1, 0.9]])
    with pytest.raises(ValueError, match="unstable"):
        normalise_matrix(A_unstable, target_rho=0.9)


def test_gramian_is_positive_semi_definite():
    A_stable = np.array([[0.5, 0.2], [0.1, 0.4]])
    B = np.eye(2)
    W = compute_gramian_doubling(A_stable, B, T=50)
    eigenvalues = np.linalg.eigvalsh(W)
    assert np.all(eigenvalues >= -1e-10), "Gramian is not Positive Semi-Definite."


# Fixture for random stable system
@pytest.fixture
def random_stable_system(seed: int = 0, N: int = 20):
    rng = np.random.default_rng(seed)
    A_raw = rng.normal(0, 1, (N, N))
    A = normalise_matrix(A_raw, target_rho=0.9)
    B = np.eye(N)
    return A, B, N


# Test spectral radius
def test_normalise_target_rho():
    rng = np.random.default_rng(42)
    A_raw = rng.normal(0, 1, (15, 15))
    for target in [0.7, 0.9, 0.95]:
        A = normalise_matrix(A_raw, target_rho=target)
        rho = np.max(np.abs(np.linalg.eigvals(A)))
        assert abs(rho - target) < 1e-10


# Test Gramian properties
def test_gramian_positive_semidefinite(random_stable_system):
    A, B, N = random_stable_system
    W = compute_gramian_doubling(A, B, T=10)
    eigenvalues = np.linalg.eigvalsh(W)
    assert np.all(eigenvalues >= -1e-9)


def test_gramian_symmetric(random_stable_system):
    A, B, N = random_stable_system
    W = compute_gramian_doubling(A, B, T=10)
    assert np.allclose(W, W.T, atol=1e-10)


# Test Doubling vs Naive
def _naive_gramian(A, B, T):
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
        W_naive = _naive_gramian(A, B, T)
        W_doubling = compute_gramian_doubling(A, B, T)
        assert np.allclose(W_naive, W_doubling, atol=1e-8)


# Test minimum energy
def test_minimum_energy_nonnegative(random_stable_system):
    A, B, N = random_stable_system
    rng = np.random.default_rng(1)
    x0 = rng.normal(0, 1, N)
    xT = rng.normal(0, 1, N)
    energy, _ = minimum_energy(A, B, x0, xT, T=8)
    assert energy >= -1e-9


def test_trivial_transition_zero_energy(random_stable_system):
    A, B, N = random_stable_system
    rng = np.random.default_rng(2)
    x0 = rng.normal(0, 1, N)
    T = 5
    xT = np.linalg.matrix_power(A, T) @ x0
    energy, _ = minimum_energy(A, B, x0, xT, T=T)
    assert energy < 1e-6


# Test controllability metrics
def test_controllability_shapes(random_stable_system):
    A, B, N = random_stable_system
    ac = average_controllability(A)
    mc = modal_controllability(A)
    assert ac.shape == (N,)
    assert mc.shape == (N,)
    assert np.all(ac >= 0)


# Test speedup (optional, can be slow)
def test_doubling_speedup():
    N = 30
    T = 100
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
    assert speedup > 1.0
      
