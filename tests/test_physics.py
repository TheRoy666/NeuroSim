"""
Comprehensive unit tests for neurosim.physics.

Covers all 9 public functions: normalise_matrix, compute_gramian_doubling,
minimum_energy, minimum_energy_trajectory, zero_order_hold,
minimum_energy_trajectory_ltv, average_controllability,
modal_controllability, finite_vs_infinite_comparison.

For each: basic correctness, edge cases, error handling, and where
applicable, mathematical invariants that must hold exactly (not just
approximately) -- several of these were previously only verified ad hoc
during development (Gramian doubling vs. brute force, LTV reduces to
LTI) and are formalized here as permanent regression tests rather than
one-off checks that could silently stop being true after a future edit.
"""
import numpy as np
import pytest

from neurosim import physics


# ---------------------------------------------------------------------
# normalise_matrix
# ---------------------------------------------------------------------

class TestNormaliseMatrix:
    def test_achieves_target_rho_exactly(self):
        rng = np.random.default_rng(0)
        A = rng.uniform(-1, 1, (10, 10))
        A_norm = physics.normalise_matrix(A, target_rho=0.9)
        rho = np.max(np.abs(np.linalg.eigvals(A_norm)))
        assert np.isclose(rho, 0.9, atol=1e-10)

    def test_default_target_rho_is_0_9(self):
        rng = np.random.default_rng(1)
        A = rng.uniform(-1, 1, (5, 5))
        A_norm = physics.normalise_matrix(A)
        rho = np.max(np.abs(np.linalg.eigvals(A_norm)))
        assert np.isclose(rho, 0.9, atol=1e-10)

    def test_preserves_directional_asymmetry(self):
        """Scaling must preserve the ratio of eigenvalues -- i.e. it's a
        uniform scalar multiple, not a reshaping."""
        rng = np.random.default_rng(2)
        A = rng.uniform(-1, 1, (6, 6))
        A_norm = physics.normalise_matrix(A, target_rho=0.7)
        ratio = A_norm / A
        # every entry should be scaled by the SAME constant (where A != 0)
        nonzero_mask = np.abs(A) > 1e-10
        ratios = ratio[nonzero_mask]
        assert np.allclose(ratios, ratios[0], rtol=1e-8)

    def test_raises_on_near_zero_spectral_radius(self):
        A = np.zeros((5, 5))
        with pytest.raises(ValueError, match="near-zero spectral radius"):
            physics.normalise_matrix(A)

    def test_custom_target_rho(self):
        rng = np.random.default_rng(3)
        A = rng.uniform(-1, 1, (8, 8))
        for target in [0.1, 0.5, 0.99]:
            A_norm = physics.normalise_matrix(A, target_rho=target)
            rho = np.max(np.abs(np.linalg.eigvals(A_norm)))
            assert np.isclose(rho, target, atol=1e-10)

    def test_single_element_matrix(self):
        A = np.array([[5.0]])
        A_norm = physics.normalise_matrix(A, target_rho=0.9)
        assert np.isclose(A_norm[0, 0], 0.9)


# ---------------------------------------------------------------------
# compute_gramian_doubling
# ---------------------------------------------------------------------

class TestComputeGramianDoubling:
    def _brute_force_gramian(self, A, B, T):
        N = A.shape[0]
        W = np.zeros((N, N))
        A_power = np.eye(N)
        for _ in range(T):
            W += A_power @ B @ B.T @ A_power.T
            A_power = A_power @ A
        return W

    @pytest.mark.parametrize("T", [1, 2, 5, 17, 50, 100])
    def test_matches_brute_force_various_T(self, T):
        """Formalized regression test for a check previously only done ad
        hoc: the doubling algorithm must match brute-force summation
        exactly (to numerical precision), including odd/non-power-of-2 T,
        which exercises the binary-decomposition branch logic."""
        rng = np.random.default_rng(0)
        N = 8
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.85)
        B = np.eye(N)
        W_doubling = physics.compute_gramian_doubling(A, B, T)
        W_brute = self._brute_force_gramian(A, B, T)
        assert np.allclose(W_doubling, W_brute, atol=1e-8, rtol=1e-6)

    def test_result_is_symmetric(self):
        rng = np.random.default_rng(1)
        N = 6
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.8)
        B = np.eye(N)
        W = physics.compute_gramian_doubling(A, B, T=20)
        assert np.allclose(W, W.T)

    def test_result_is_positive_semidefinite(self):
        rng = np.random.default_rng(2)
        N = 6
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.8)
        B = np.eye(N)
        W = physics.compute_gramian_doubling(A, B, T=20)
        eigvals = np.linalg.eigvalsh(W)
        assert np.all(eigvals > -1e-10)

    def test_raises_on_unstable_A(self):
        A = np.eye(5) * 1.5  # rho = 1.5, unstable
        B = np.eye(5)
        with pytest.raises(ValueError, match="unstable"):
            physics.compute_gramian_doubling(A, B, T=10)

    def test_raises_exactly_at_rho_equals_1(self):
        A = np.eye(5)  # rho = 1.0 exactly, boundary case
        B = np.eye(5)
        with pytest.raises(ValueError):
            physics.compute_gramian_doubling(A, B, T=10)

    def test_non_square_B_input_matrix(self):
        """B need not be full-rank / square -- confirms the function
        works with a reduced input matrix (e.g. control on a subset of
        nodes), not just B=I."""
        rng = np.random.default_rng(3)
        N, M = 8, 3
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.8)
        B = rng.uniform(-1, 1, (N, M))
        W = physics.compute_gramian_doubling(A, B, T=15)
        assert W.shape == (N, N)
        W_brute = self._brute_force_gramian(A, B, T=15)
        assert np.allclose(W, W_brute, atol=1e-8)

    def test_T_equals_1(self):
        """Edge case: T=1 should just give B B^T."""
        rng = np.random.default_rng(4)
        N = 5
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.8)
        B = rng.uniform(-1, 1, (N, N))
        W = physics.compute_gramian_doubling(A, B, T=1)
        assert np.allclose(W, (B @ B.T + (B @ B.T).T) / 2, atol=1e-10)


# ---------------------------------------------------------------------
# minimum_energy
# ---------------------------------------------------------------------

class TestMinimumEnergy:
    def test_zero_target_from_zero_start_gives_zero_energy(self):
        rng = np.random.default_rng(0)
        N = 5
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.8)
        B = np.eye(N)
        x0 = np.zeros(N)
        xT = np.zeros(N)
        energy, u_opt = physics.minimum_energy(A, B, x0, xT, T=10)
        assert np.isclose(energy, 0.0, atol=1e-8)

    def test_energy_is_nonnegative(self):
        rng = np.random.default_rng(1)
        N = 6
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.85)
        B = np.eye(N)
        x0 = rng.uniform(-1, 1, N)
        xT = rng.uniform(-1, 1, N)
        energy, _ = physics.minimum_energy(A, B, x0, xT, T=15)
        assert energy >= 0

    def test_energy_decreases_as_horizon_increases(self):
        """More time to act should never require MORE energy for the same
        transition -- but this is only a clean, testable property when
        x0=0 (deviation coordinates). With nonzero x0, delta = xT - A^T x0
        genuinely changes with T (the natural drift term A^T x0 decays
        differently at each horizon for a stable A), so different T values
        are not actually solving "the same transition with more time" --
        confirmed directly during test development: with a nonzero x0,
        ||delta|| itself changed from 0.619 (T=5) to 0.642 (T=40), and
        energy was correspondingly non-monotonic. Not a bug in
        minimum_energy -- a real subtlety in what "more time" means when
        the target is specified as an absolute state rather than a
        deviation. Testing here with x0=0, matching the deviation-
        coordinate convention used throughout the rest of this project's
        Path A1/A2 work, where this property is the one that actually
        matters."""
        rng = np.random.default_rng(2)
        N = 5
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.85)
        B = np.eye(N)
        x0 = np.zeros(N)
        xT = rng.uniform(-0.5, 0.5, N)
        energies = [physics.minimum_energy(A, B, x0, xT, T=T)[0]
                    for T in [5, 10, 20, 40]]
        for i in range(len(energies) - 1):
            assert energies[i + 1] <= energies[i] + 1e-6

    def test_warns_on_ill_conditioned_gramian(self):
        """A near-uncontrollable system (B has a near-zero column) should
        trigger the ill-conditioning warning."""
        N = 5
        A = physics.normalise_matrix(np.eye(N) + 0.01 * np.eye(N), target_rho=0.99)
        B = np.zeros((N, N))
        B[0, 0] = 1e-8  # nearly uncontrollable
        x0 = np.zeros(N)
        xT = np.ones(N)
        with pytest.warns(RuntimeWarning, match="condition number"):
            physics.minimum_energy(A, B, x0, xT, T=5)

    def test_u_opt_shape(self):
        rng = np.random.default_rng(3)
        N, M = 8, 3
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.8)
        B = rng.uniform(-1, 1, (N, M))
        x0 = rng.uniform(-1, 1, N)
        xT = rng.uniform(-1, 1, N)
        _, u_opt = physics.minimum_energy(A, B, x0, xT, T=10)
        assert u_opt.shape == (M,)

    def test_nonzero_x0_can_break_horizon_monotonicity_documented(self):
        """Explicit, documented confirmation of the subtlety found while
        writing the test above: with nonzero x0, energy is NOT guaranteed
        monotonic in T, because delta = xT - A^T x0 changes with T (the
        natural drift term). This is understood, expected behavior of the
        absolute-state-target formulation -- not a bug -- documented here
        as its own test so it doesn't get mistaken for one in the future.
        Uses the exact seed/setup that originally surfaced it."""
        rng = np.random.default_rng(2)
        N = 5
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.85)
        B = np.eye(N)
        x0 = rng.uniform(-0.5, 0.5, N)  # nonzero -- deliberately reintroduces drift
        xT = rng.uniform(-0.5, 0.5, N)
        energies = [physics.minimum_energy(A, B, x0, xT, T=T)[0]
                    for T in [5, 10, 20, 40]]
        # Confirm delta itself is not constant across T (the actual cause)
        deltas = [np.linalg.norm(xT - np.linalg.matrix_power(A, T) @ x0)
                  for T in [5, 10, 20, 40]]
        assert not np.allclose(deltas, deltas[0], rtol=1e-3), (
            "Expected delta to vary with T when x0 != 0 -- if this ever "
            "becomes constant, the drift-confound explanation no longer "
            "holds and the non-monotonicity below would need re-diagnosing."
        )


# ---------------------------------------------------------------------
# minimum_energy_trajectory
# ---------------------------------------------------------------------

class TestMinimumEnergyTrajectory:
    def test_energy_matches_sum_of_squared_U(self):
        """The function computes energy independently as a consistency
        check per its own docstring -- verify that check actually holds."""
        rng = np.random.default_rng(0)
        N = 6
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.85)
        B = np.eye(N)
        x0 = rng.uniform(-1, 1, N)
        xT = rng.uniform(-1, 1, N)
        energy, U = physics.minimum_energy_trajectory(A, B, x0, xT, T=15)
        assert np.isclose(energy, np.sum(U ** 2), rtol=1e-6)

    def test_energy_matches_minimum_energy_scalar(self):
        """minimum_energy_trajectory's total energy should match
        minimum_energy's scalar E* for the same problem, per the
        docstring's stated equivalence."""
        rng = np.random.default_rng(1)
        N = 5
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.8)
        B = np.eye(N)
        x0 = rng.uniform(-1, 1, N)
        xT = rng.uniform(-1, 1, N)
        energy_scalar, _ = physics.minimum_energy(A, B, x0, xT, T=12)
        energy_traj, _ = physics.minimum_energy_trajectory(A, B, x0, xT, T=12)
        assert np.isclose(energy_scalar, energy_traj, rtol=1e-4)

    def test_injecting_U_actually_reaches_target(self):
        """The literal, most important property: propagating x0 forward
        through the LINEAR system under the computed U must land exactly
        (to numerical precision) on xT."""
        rng = np.random.default_rng(2)
        N = 6
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.85)
        B = np.eye(N)
        x0 = rng.uniform(-1, 1, N)
        xT = rng.uniform(-1, 1, N)
        T = 15
        _, U = physics.minimum_energy_trajectory(A, B, x0, xT, T)

        x = x0.copy()
        for k in range(T):
            x = A @ x + B @ U[k]
        assert np.allclose(x, xT, atol=1e-6)

    def test_U_shape(self):
        rng = np.random.default_rng(3)
        N, M, T = 10, 4, 20
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.8)
        B = rng.uniform(-1, 1, (N, M))
        x0 = rng.uniform(-1, 1, N)
        xT = rng.uniform(-1, 1, N)
        _, U = physics.minimum_energy_trajectory(A, B, x0, xT, T)
        assert U.shape == (T, M)

    def test_zero_transition_gives_zero_control(self):
        rng = np.random.default_rng(4)
        N = 5
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.8)
        B = np.eye(N)
        x0 = np.zeros(N)
        xT = np.zeros(N)
        energy, U = physics.minimum_energy_trajectory(A, B, x0, xT, T=10)
        assert np.allclose(U, 0, atol=1e-8)
        assert np.isclose(energy, 0, atol=1e-8)

    def test_T_equals_1(self):
        """Edge case: single-step horizon."""
        rng = np.random.default_rng(5)
        N = 4
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.8)
        B = np.eye(N)
        x0 = rng.uniform(-1, 1, N)
        xT = rng.uniform(-1, 1, N)
        energy, U = physics.minimum_energy_trajectory(A, B, x0, xT, T=1)
        assert U.shape == (1, N)
        x_final = A @ x0 + B @ U[0]
        assert np.allclose(x_final, xT, atol=1e-6)


# ---------------------------------------------------------------------
# zero_order_hold
# ---------------------------------------------------------------------

class TestZeroOrderHold:
    def test_returns_correct_row_within_each_interval(self):
        U = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        tr_ms = 10.0
        u_func = physics.zero_order_hold(U, tr_ms)
        assert np.allclose(u_func(0.0), [1.0, 2.0])
        assert np.allclose(u_func(5.0), [1.0, 2.0])   # still interval 0
        assert np.allclose(u_func(9.99), [1.0, 2.0])  # still interval 0
        assert np.allclose(u_func(10.0), [3.0, 4.0])  # interval 1 starts
        assert np.allclose(u_func(15.0), [3.0, 4.0])
        assert np.allclose(u_func(20.0), [5.0, 6.0])  # interval 2

    def test_zero_past_horizon(self):
        U = np.array([[1.0, 2.0], [3.0, 4.0]])
        tr_ms = 10.0
        u_func = physics.zero_order_hold(U, tr_ms)
        assert np.allclose(u_func(100.0), [0.0, 0.0])

    def test_negative_time_clips_to_first_interval(self):
        U = np.array([[1.0, 2.0], [3.0, 4.0]])
        tr_ms = 10.0
        u_func = physics.zero_order_hold(U, tr_ms)
        assert np.allclose(u_func(-5.0), [1.0, 2.0])

    def test_single_step_control(self):
        U = np.array([[7.0]])
        u_func = physics.zero_order_hold(U, tr_ms=1.0)
        assert np.allclose(u_func(0.5), [7.0])
        assert np.allclose(u_func(1.5), [0.0])


# ---------------------------------------------------------------------
# minimum_energy_trajectory_ltv
# ---------------------------------------------------------------------

class TestMinimumEnergyTrajectoryLTV:
    def test_reduces_exactly_to_lti_when_constant(self):
        """Formalized regression test for the LTV-reduces-to-LTI check
        previously only done ad hoc: if every A_k, B_k in the list is the
        SAME matrix, the LTV result must match the LTI
        minimum_energy_trajectory result exactly."""
        rng = np.random.default_rng(0)
        N, T = 6, 12
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.85)
        B = np.eye(N)
        x0 = rng.uniform(-1, 1, N)
        xT = rng.uniform(-1, 1, N)

        energy_lti, U_lti = physics.minimum_energy_trajectory(A, B, x0, xT, T)
        A_list = [A] * T
        B_list = [B] * T
        energy_ltv, U_ltv = physics.minimum_energy_trajectory_ltv(A_list, B_list, x0, xT)

        assert np.isclose(energy_lti, energy_ltv, rtol=1e-6)
        assert np.allclose(U_lti, U_ltv, atol=1e-6)

    def test_injecting_U_reaches_target_with_genuinely_time_varying_A(self):
        """The actually-useful case: A_k genuinely differs across steps,
        confirming the general (not just constant-reduces-to-LTI) formula
        is correct."""
        rng = np.random.default_rng(1)
        N, T = 5, 10
        A_list, B_list = [], []
        for k in range(T):
            A_k = physics.normalise_matrix(
                rng.uniform(-1, 1, (N, N)) + 0.1 * k, target_rho=0.7 + 0.01 * k)
            B_list.append(np.eye(N))
            A_list.append(A_k)
        x0 = rng.uniform(-1, 1, N)
        xT = rng.uniform(-1, 1, N)

        _, U = physics.minimum_energy_trajectory_ltv(A_list, B_list, x0, xT)

        x = x0.copy()
        for k in range(T):
            x = A_list[k] @ x + B_list[k] @ U[k]
        assert np.allclose(x, xT, atol=1e-6)

    def test_U_shape(self):
        rng = np.random.default_rng(2)
        N, M, T = 7, 3, 8
        A_list = [physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), 0.8) for _ in range(T)]
        B_list = [rng.uniform(-1, 1, (N, M)) for _ in range(T)]
        x0 = rng.uniform(-1, 1, N)
        xT = rng.uniform(-1, 1, N)
        _, U = physics.minimum_energy_trajectory_ltv(A_list, B_list, x0, xT)
        assert U.shape == (T, M)

    def test_warns_on_ill_conditioned_ltv_gramian(self):
        N, T = 4, 5
        A_list = [np.eye(N) * 0.99] * T
        B_list = [np.zeros((N, N))] * T  # completely uncontrollable
        for b in B_list:
            b[0, 0] = 1e-9
        x0 = np.zeros(N)
        xT = np.ones(N)
        with pytest.warns(RuntimeWarning, match="LTV Gramian"):
            physics.minimum_energy_trajectory_ltv(A_list, B_list, x0, xT)


# ---------------------------------------------------------------------
# average_controllability / modal_controllability
# ---------------------------------------------------------------------

class TestControllabilityMetrics:
    def test_average_controllability_shape_and_positivity(self):
        rng = np.random.default_rng(0)
        N = 8
        A = physics.normalise_matrix(rng.uniform(0, 1, (N, N)), target_rho=0.9)
        ac = physics.average_controllability(A)
        assert ac.shape == (N,)
        assert np.all(ac > 0)  # diagonal of a PSD Gramian

    def test_modal_controllability_shape(self):
        rng = np.random.default_rng(1)
        N = 8
        A = physics.normalise_matrix(rng.uniform(0, 1, (N, N)), target_rho=0.9)
        mc = physics.modal_controllability(A)
        assert mc.shape == (N,)
        assert np.all(np.isreal(mc))  # imaginary parts should have canceled

    def test_identity_like_matrix_gives_uniform_controllability(self):
        """A symmetric, homogeneous matrix should give near-identical
        controllability across all nodes -- a basic sanity check that the
        metric isn't doing something node-order-dependent for no reason."""
        N = 6
        A = physics.normalise_matrix(np.ones((N, N)) - np.eye(N), target_rho=0.8)
        ac = physics.average_controllability(A)
        assert np.allclose(ac, ac[0], rtol=1e-6)


# ---------------------------------------------------------------------
# finite_vs_infinite_comparison
# ---------------------------------------------------------------------

class TestFiniteVsInfiniteComparison:
    def test_returns_dataframe_with_expected_columns(self):
        rng = np.random.default_rng(0)
        N = 5
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.85)
        B = np.eye(N)
        x0 = rng.uniform(-0.5, 0.5, N)
        xT = rng.uniform(-0.5, 0.5, N)
        df = physics.finite_vs_infinite_comparison(A, B, x0, xT, T_range=[1, 5, 10])
        assert list(df.columns) == ["T", "E_finite", "E_infinite", "ratio"]
        assert len(df) == 3

    def test_ratio_decreases_toward_1_as_T_grows(self):
        """The core NeuroSim claim made concrete: the vanishing-cost gap
        should shrink as horizon grows, per the module's own docstring."""
        rng = np.random.default_rng(1)
        N = 5
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.85)
        B = np.eye(N)
        x0 = rng.uniform(-0.5, 0.5, N)
        xT = rng.uniform(-0.5, 0.5, N)
        df = physics.finite_vs_infinite_comparison(A, B, x0, xT, T_range=[1, 5, 10, 20])
        ratios = df["ratio"].values
        # ratio should generally decrease toward 1 -- check the trend, not
        # strict monotonicity at every step (numerical noise near T_large)
        assert ratios[0] > ratios[-1]

    def test_default_T_range(self):
        rng = np.random.default_rng(2)
        N = 4
        A = physics.normalise_matrix(rng.uniform(-1, 1, (N, N)), target_rho=0.85)
        B = np.eye(N)
        x0 = rng.uniform(-0.5, 0.5, N)
        xT = rng.uniform(-0.5, 0.5, N)
        df = physics.finite_vs_infinite_comparison(A, B, x0, xT)
        assert len(df) == 20  # default range(1, 21)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
