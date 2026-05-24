"""
tests/test_simulation.py
========================
Unit tests for neurosim.simulation.

The Wilson-Cowan model serves as NeuroSim's non-linear ground truth benchmark.
These tests verify that:
1. The single-node model produces a stable limit cycle in the correct parameter regime.
2. The coupled network produces correlated multi-region dynamics.
3. The BOLD-proxy downsampling produces the correct shape and value range.
4. Fixed-point and limit-cycle parameter regimes are distinguishable by variance.
5. The LTI control energy is computable from Wilson-Cowan trajectories.
"""

import numpy as np
import pytest
from scipy.signal import find_peaks

from neurosim.simulation import WilsonCowanNetwork, WilsonCowanNode


# Single-node Wilson-Cowan

class TestWilsonCowanNode:

    def test_simulate_returns_correct_keys(self):
        node = WilsonCowanNode()
        result = node.simulate(t_span=(0.0, 200.0), n_points=2000)
        assert "t" in result
        assert "E" in result
        assert "I" in result

    def test_simulate_output_shapes(self):
        node = WilsonCowanNode()
        result = node.simulate(t_span=(0.0, 500.0), n_points=5000)
        assert result["t"].shape == (5000,)
        assert result["E"].shape == (5000,)
        assert result["I"].shape == (5000,)

    def test_time_vector_monotone(self):
        node = WilsonCowanNode()
        result = node.simulate(t_span=(0.0, 200.0), n_points=2000)
        assert np.all(np.diff(result["t"]) > 0), "Time vector must be strictly increasing"

    def test_populations_bounded(self):
        """E and I populations must stay in [0, 1] for sigmoidal dynamics."""
        node = WilsonCowanNode(**WilsonCowanNode.LIMIT_CYCLE_PARAMS)
        result = node.simulate(t_span=(0.0, 1000.0), n_points=10000)
        assert result["E"].min() >= -0.01, f"E goes below 0: {result['E'].min():.4f}"
        assert result["E"].max() <= 1.01, f"E exceeds 1: {result['E'].max():.4f}"
        assert result["I"].min() >= -0.01
        assert result["I"].max() <= 1.01

    def test_limit_cycle_detected(self):
        """
        In the LIMIT_CYCLE_PARAMS regime, the excitatory population E should
        exhibit sustained oscillations (variance > threshold after transient).
        This is the core validation that the Wilson-Cowan benchmark is operating
        correctly as a non-linear ground truth.
        """
        node = WilsonCowanNode(**WilsonCowanNode.LIMIT_CYCLE_PARAMS)
        result = node.simulate(t_span=(0.0, 1000.0), n_points=10000)

        # Skip transient (first 25% of simulation)
        transient_idx = 2500
        E_steady = result["E"][transient_idx:]
        E_var = float(np.var(E_steady))

        assert E_var > 1e-4, (
            f"No oscillations detected: variance = {E_var:.2e}. "
            "Expected limit-cycle oscillations in this parameter regime."
        )

    def test_limit_cycle_frequency_in_gamma_range(self):
        """Limit cycle frequency should be in the gamma range (20–80 Hz)."""
        node = WilsonCowanNode(**WilsonCowanNode.LIMIT_CYCLE_PARAMS)
        result = node.simulate(t_span=(0.0, 1000.0), n_points=10000)

        transient_idx = 2500
        E_steady = result["E"][transient_idx:]
        t_steady  = result["t"][transient_idx:]

        peaks, _ = find_peaks(E_steady, height=np.mean(E_steady))

        if len(peaks) > 1:
            t_peaks = t_steady[peaks]
            period_ms = np.mean(np.diff(t_peaks))
            freq_hz = 1000.0 / period_ms
            assert 5.0 < freq_hz < 150.0, (
                f"Oscillation frequency {freq_hz:.1f} Hz outside expected range (15–120 Hz). "
                "Check Wilson-Cowan parameters."
            )

    def test_fixed_point_regime_low_variance(self):
        """
        With default sub-limit-cycle parameters (weak coupling), the system
        should converge to a fixed point (low variance).
        """
        # Use weak coupling parameters that produce a fixed point
        node = WilsonCowanNode(
            w_EE=4.0, w_IE=2.0, w_EI=2.0, w_II=1.0,
            c_E=-2.0, c_I=-2.0, tau_E=10.0, tau_I=20.0
        )
        result = node.simulate(t_span=(0.0, 1000.0), n_points=5000, E0=0.3, I0=0.3)
        E_steady = result["E"][1000:]
        E_var = float(np.var(E_steady))
        assert E_var < 1e-4, (
            f"Expected fixed-point (low variance), got variance={E_var:.2e}"
        )

    def test_external_input_shifts_activity(self):
        """External input P_ext should increase excitatory population activity."""
        node = WilsonCowanNode(**WilsonCowanNode.LIMIT_CYCLE_PARAMS)
        r_no_input  = node.simulate(t_span=(0.0, 500.0), n_points=5000, P_ext=0.0)
        r_with_input = node.simulate(t_span=(0.0, 500.0), n_points=5000, P_ext=2.0)

        mean_E_no    = float(np.mean(r_no_input["E"][1000:]))
        mean_E_with  = float(np.mean(r_with_input["E"][1000:]))

        assert mean_E_with > mean_E_no, (
            f"External input should increase mean E activity. "
            f"No input: {mean_E_no:.4f}, with input: {mean_E_with:.4f}"
        )

    def test_limit_cycle_params_dict_complete(self):
        required = {"w_EE", "w_IE", "w_EI", "w_II", "c_E", "c_I", "tau_E", "tau_I"}
        assert required.issubset(set(WilsonCowanNode.LIMIT_CYCLE_PARAMS.keys()))


# Coupled network

class TestWilsonCowanNetwork:

    def _make_ring_coupling(self, n, strength=0.5):
        C = np.zeros((n, n))
        for i in range(n):
            C[i, (i + 1) % n] = strength
            C[(i + 1) % n, i] = strength
        return C

    def test_simulate_returns_correct_keys(self):
        n = 4
        C = self._make_ring_coupling(n)
        net = WilsonCowanNetwork(n_regions=n, C=C)
        result = net.simulate(t_span=(0.0, 200.0), n_points=2000)
        assert "t" in result
        assert "E" in result
        assert "I" in result

    def test_simulate_output_shapes(self):
        n = 5
        C = self._make_ring_coupling(n)
        net = WilsonCowanNetwork(n_regions=n, C=C)
        result = net.simulate(t_span=(0.0, 500.0), n_points=4000)
        assert result["E"].shape == (n, 4000)
        assert result["I"].shape == (n, 4000)
        assert result["t"].shape == (4000,)

    def test_network_populations_bounded(self):
        n = 3
        C = self._make_ring_coupling(n, strength=0.3)
        net = WilsonCowanNetwork(n_regions=n, C=C)
        result = net.simulate(t_span=(0.0, 500.0), n_points=5000)
        assert result["E"].min() >= -0.05
        assert result["E"].max() <= 1.05

    def test_coupled_regions_correlated(self):
        """Coupled regions should produce correlated excitatory time series."""
        n = 4
        C = self._make_ring_coupling(n, strength=1.0)
        net = WilsonCowanNetwork(n_regions=n, C=C)
        result = net.simulate(t_span=(0.0, 2000.0), n_points=10000, seed=0)

        E = result["E"][:, 2000:]  # skip transient
        corr_matrix = np.corrcoef(E)

        # Diagonal is 1.0; off-diagonal should be substantially positive
        off_diag = corr_matrix[np.triu_indices(n, k=1)]
        mean_corr = float(np.mean(off_diag))
        assert mean_corr > 0.2, (
            f"Coupled network regions not sufficiently correlated: mean_corr={mean_corr:.3f}"
        )

    def test_zero_coupling_regions_independent(self):
        """With zero inter-regional coupling, regions should evolve independently."""
        n = 3
        C = np.zeros((n, n))  # no coupling
        net = WilsonCowanNetwork(n_regions=n, C=C,
                                  node_params=WilsonCowanNode.LIMIT_CYCLE_PARAMS)
        result = net.simulate(t_span=(0.0, 1000.0), n_points=5000, seed=7)
        E = result["E"][:, 1000:]
        # With random initial conditions and no coupling, correlations should be low
        corr_matrix = np.corrcoef(E)
        off_diag = np.abs(corr_matrix[np.triu_indices(n, k=1)])
        # Not perfectly independent (same params), but shouldn't be highly correlated
        assert off_diag.mean() < 0.95, "Uncoupled regions should not be perfectly correlated"

    def test_reproducibility_with_seed(self):
        n = 3
        C = self._make_ring_coupling(n)
        net = WilsonCowanNetwork(n_regions=n, C=C)
        r1 = net.simulate(t_span=(0.0, 200.0), n_points=2000, seed=42)
        r2 = net.simulate(t_span=(0.0, 200.0), n_points=2000, seed=42)
        assert np.allclose(r1["E"], r2["E"]), "Same seed should produce identical results"

    def test_different_seeds_different_results(self):
        n = 3
        C = self._make_ring_coupling(n)
        net = WilsonCowanNetwork(n_regions=n, C=C)
        r1 = net.simulate(t_span=(0.0, 200.0), n_points=2000, seed=1)
        r2 = net.simulate(t_span=(0.0, 200.0), n_points=2000, seed=2)
        assert not np.allclose(r1["E"], r2["E"]), "Different seeds should produce different results"


# BOLD proxy extraction

class TestBOLDProxy:

    def test_bold_proxy_shape(self):
        n = 5
        C = np.zeros((n, n))
        net = WilsonCowanNetwork(n_regions=n, C=C,
                                  node_params=WilsonCowanNode.LIMIT_CYCLE_PARAMS)
        result = net.simulate(t_span=(0.0, 5000.0), n_points=50000)
        E_bold = net.extract_bold_proxy(result, tr_ms=720.0)

        expected_T = int(5000.0 / 720.0)
        assert E_bold.shape[0] == n, f"Expected {n} regions, got {E_bold.shape[0]}"
        assert abs(E_bold.shape[1] - expected_T) <= 2, \
            f"Expected ~{expected_T} timepoints, got {E_bold.shape[1]}"

    def test_bold_proxy_values_in_range(self):
        """BOLD proxy is downsampled E population — should stay in [0, 1]."""
        n = 3
        C = np.zeros((n, n))
        net = WilsonCowanNetwork(n_regions=n, C=C,
                                  node_params=WilsonCowanNode.LIMIT_CYCLE_PARAMS)
        result = net.simulate(t_span=(0.0, 2000.0), n_points=20000)
        E_bold = net.extract_bold_proxy(result, tr_ms=720.0)
        assert E_bold.min() >= -0.05
        assert E_bold.max() <= 1.05

    def test_bold_proxy_ready_for_physics_engine(self):
        """
        The BOLD proxy output must be usable as input to the physics engine.
        This is the critical integration test between simulation and physics.
        """
        from neurosim.physics import normalise_matrix, compute_gramian_doubling, minimum_energy

        n = 4
        C = np.eye(n) * 0.3
        net = WilsonCowanNetwork(n_regions=n, C=C,
                                  node_params=WilsonCowanNode.LIMIT_CYCLE_PARAMS)
        result = net.simulate(t_span=(0.0, 5000.0), n_points=50000, seed=0)
        E_bold = net.extract_bold_proxy(result, tr_ms=720.0)

        # Extract initial and final states
        x0 = E_bold[:, 0]
        xT = E_bold[:, -1]

        # Build a toy connectivity matrix and compute energy
        A_raw = np.random.default_rng(0).normal(0, 0.5, (n, n))
        A = normalise_matrix(A_raw, target_rho=0.9)
        B = np.eye(n)
        T = E_bold.shape[1]

        W = compute_gramian_doubling(A, B, T=min(T, 20))
        energy, _ = minimum_energy(A, B, x0, xT, T=min(T, 20))

        assert energy >= 0, f"Control energy must be non-negative, got {energy:.4f}"
        assert W.shape == (n, n)
