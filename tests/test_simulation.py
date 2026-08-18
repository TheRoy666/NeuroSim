"""
Comprehensive unit tests for neurosim.simulation.

Covers _sigmoid, WilsonCowanNode, WilsonCowanNetwork (all 6 public
methods), input_jacobian_at, discretize_system, discretize_jacobian.

Includes several exact mathematical-consistency checks not previously
formalized anywhere in this project's development history:
- discretize_system's A block must EXACTLY match discretize_jacobian's
  output for the same J (a real, provable property of the augmented-
  matrix-exponential construction, not just a plausible-sounding claim).
- jacobian_at and input_jacobian_at verified against numerical finite
  differences directly on WilsonCowanNetwork, as a permanent regression
  test (previously only checked ad hoc, and only as part of the
  adjoint-gradient debugging work, not as a standalone test of these
  functions in isolation).
"""
import numpy as np
import pytest

from neurosim import simulation


# ---------------------------------------------------------------------
# _sigmoid
# ---------------------------------------------------------------------

class TestSigmoid:
    def test_at_zero_with_default_params(self):
        assert np.isclose(simulation._sigmoid(np.array([0.0])), 0.5)

    def test_bounded_between_0_and_1_realistic_range(self):
        """Realistic range for this project's actual WC dynamics (z
        values typically stay within roughly +-20 given the parameter
        scales used throughout) -- not testing +-100, see the dedicated
        underflow test below for why that range is excluded."""
        x = np.linspace(-20, 20, 1000)
        s = simulation._sigmoid(x)
        assert np.all(s > 0) and np.all(s < 1)

    def test_extreme_values_saturate_to_exact_0_or_1_documented(self):
        """Found while writing the test above, and the exact thresholds
        verified empirically rather than guessed (an initial guess of
        +-500 was itself wrong, see below): saturation to exact 0.0/1.0
        happens via two DIFFERENT mechanisms at very different
        thresholds, not one symmetric "extreme values saturate" story.

        Positive side saturates to exactly 1.0 around x~37: exp(-x)
        becomes smaller than float64's ~1e-16 relative precision, so
        1+exp(-x) rounds to exactly 1.0 -- a PRECISION-LOSS mechanism.

        Negative side only reaches exactly 0.0 around x~-745: this needs
        exp(-x) = exp(+745) to actually OVERFLOW to infinity, so
        1/(1+inf) = 0.0 exactly -- a genuinely different, OVERFLOW
        mechanism, not the same precision-loss story mirrored.

        Neither is a bug -- both are expected float64 behavior -- but the
        asymmetry itself (~37 vs. ~745, a 20x difference in scale, via
        two different numerical mechanisms) is worth having verified and
        recorded rather than assumed symmetric."""
        with pytest.warns(RuntimeWarning, match="overflow encountered in exp"):
            s_negative = simulation._sigmoid(np.array([-745.0]))
        s_positive = simulation._sigmoid(np.array([37.0]))
        assert s_positive[0] == 1.0
        assert s_negative[0] == 0.0

        # Confirm the asymmetry directly: the positive-side threshold is
        # NOT sufficient (mirrored) to saturate the negative side
        s_negative_at_37 = simulation._sigmoid(np.array([-37.0]))
        assert s_negative_at_37[0] != 0.0, (
            "if this becomes exactly 0.0, the mechanisms are no longer "
            "asymmetric as documented above and this test needs revisiting"
        )

    def test_monotonically_increasing(self):
        x = np.linspace(-10, 10, 100)
        s = simulation._sigmoid(x)
        assert np.all(np.diff(s) > 0)

    def test_theta_shifts_midpoint(self):
        theta = 5.0
        assert np.isclose(simulation._sigmoid(np.array([theta]), theta=theta), 0.5)

    def test_a_controls_steepness(self):
        """Larger a should give a steeper transition -- check derivative
        at the midpoint scales with a, per S'(midpoint) = a/4 for the
        standard logistic form."""
        x = np.array([0.0])
        for a in [0.5, 1.0, 2.0, 5.0]:
            eps = 1e-6
            s_plus = simulation._sigmoid(np.array([eps]), a=a)
            s_minus = simulation._sigmoid(np.array([-eps]), a=a)
            deriv = (s_plus - s_minus) / (2 * eps)
            assert np.isclose(deriv, a / 4, rtol=1e-3)

    def test_vectorized_over_array(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        s = simulation._sigmoid(x)
        assert s.shape == x.shape


# ---------------------------------------------------------------------
# WilsonCowanNode
# ---------------------------------------------------------------------

class TestWilsonCowanNode:
    def test_default_params_are_limit_cycle_params(self):
        node = simulation.WilsonCowanNode()
        assert node.w_EE == simulation.WilsonCowanNode.LIMIT_CYCLE_PARAMS["w_EE"]

    def test_custom_params_override_defaults(self):
        node = simulation.WilsonCowanNode(w_EE=99.0)
        assert node.w_EE == 99.0
        assert node.w_IE == simulation.WilsonCowanNode.LIMIT_CYCLE_PARAMS["w_IE"]

    def test_simulate_returns_expected_keys_and_shapes(self):
        node = simulation.WilsonCowanNode()
        result = node.simulate(t_span=(0, 100), n_points=50)
        assert set(result.keys()) == {"t", "E", "I"}
        assert result["t"].shape == (50,)
        assert result["E"].shape == (50,)
        assert result["I"].shape == (50,)

    def test_E_and_I_stay_in_sigmoid_range(self):
        """E, I are driven by a sigmoid nonlinearity with a leak term --
        should stay within a physically sane range, not blow up."""
        node = simulation.WilsonCowanNode()
        result = node.simulate(t_span=(0, 500), n_points=1000)
        assert np.all(result["E"] > -0.5) and np.all(result["E"] < 1.5)
        assert np.all(result["I"] > -0.5) and np.all(result["I"] < 1.5)

    def test_limit_cycle_params_produce_genuine_oscillation(self):
        """The module's own docstring claims LIMIT_CYCLE_PARAMS produces
        a stable limit cycle -- verify this directly rather than just
        trusting the claim: E should show real variance after the
        transient, not settle to a fixed point."""
        node = simulation.WilsonCowanNode()  # defaults ARE LIMIT_CYCLE_PARAMS
        result = node.simulate(t_span=(0, 2000), n_points=4000)
        tail = result["E"][2000:]  # past transient
        assert np.std(tail) > 0.01, (
            "Expected genuine oscillation (nonzero variance) under "
            "LIMIT_CYCLE_PARAMS, found near-constant E -- check params "
            "still produce a limit cycle, not a fixed point."
        )


# ---------------------------------------------------------------------
# input_jacobian_at
# ---------------------------------------------------------------------

class TestInputJacobianAt:
    def _build_net(self, seed=0, N=5):
        rng = np.random.default_rng(seed)
        C = rng.uniform(0, 0.15, (N, N))
        np.fill_diagonal(C, 0)
        params = dict(w_EE=4.0, w_IE=4.0, w_EI=3.0, w_II=2.0,
                      c_E=-2.0, c_I=-2.0, tau_E=10.0, tau_I=20.0)
        return simulation.WilsonCowanNetwork(n_regions=N, C=C, node_params=params)

    def test_shape(self):
        net = self._build_net()
        N = net.n_regions
        E_star, I_star = net.find_fixed_point()
        G = simulation.input_jacobian_at(net, E_star, I_star)
        assert G.shape == (2 * N, N)

    def test_bottom_block_is_zero(self):
        """Control only enters the E equation -- I-block must be exactly
        zero, per the function's own docstring."""
        net = self._build_net()
        N = net.n_regions
        E_star, I_star = net.find_fixed_point()
        G = simulation.input_jacobian_at(net, E_star, I_star)
        assert np.allclose(G[N:, :], 0.0)

    def test_matches_numerical_finite_difference_at_u_equals_0(self):
        """Direct, standalone correctness check against numerical
        differentiation of the actual network ODE -- not previously
        formalized as its own test (only checked indirectly via the
        adjoint-gradient debugging work, using a different, u-aware
        version of this function, not this original one in isolation)."""
        net = self._build_net(seed=1, N=4)
        N = net.n_regions
        E_star, I_star = net.find_fixed_point()
        G_analytic = simulation.input_jacobian_at(net, E_star, I_star)

        y_star = np.concatenate([E_star, I_star])
        eps = 1e-6
        G_numerical = np.zeros_like(G_analytic)
        for j in range(N):
            u_plus = np.zeros(N); u_plus[j] = eps
            u_minus = np.zeros(N); u_minus[j] = -eps
            f_plus = net._ode_network(0.0, y_star, lambda t: u_plus)
            f_minus = net._ode_network(0.0, y_star, lambda t: u_minus)
            G_numerical[:, j] = (f_plus - f_minus) / (2 * eps)

        assert np.allclose(G_analytic, G_numerical, atol=1e-5)

    def test_diagonal_structure_top_block(self):
        """Top block should be purely diagonal (each region's u only
        affects its own dE/dt directly, no cross terms) -- off-diagonal
        entries must be exactly zero."""
        net = self._build_net()
        N = net.n_regions
        E_star, I_star = net.find_fixed_point()
        G = simulation.input_jacobian_at(net, E_star, I_star)
        top = G[:N, :]
        off_diag_mask = ~np.eye(N, dtype=bool)
        assert np.allclose(top[off_diag_mask], 0.0)


# ---------------------------------------------------------------------
# discretize_system / discretize_jacobian
# ---------------------------------------------------------------------

class TestDiscretizeSystemAndJacobian:
    def test_discretize_system_A_matches_discretize_jacobian_exactly(self):
        """Real, provable mathematical property, not previously
        formalized as a test anywhere in this project: the (1,1) block of
        expm([[J,G],[0,0]]*dt) depends ONLY on J (block-triangular
        structure), so discretize_system's A must exactly equal
        discretize_jacobian(J, dt) for the SAME J, regardless of G."""
        rng = np.random.default_rng(0)
        n, m = 8, 3
        J = rng.uniform(-1, 1, (n, n)) * 0.1 - np.eye(n) * 0.5  # damped, arbitrary
        G = rng.uniform(-1, 1, (n, m))
        dt = 2.0

        A_from_system, B = simulation.discretize_system(J, G, dt)
        A_from_jacobian = simulation.discretize_jacobian(J, dt)

        assert np.allclose(A_from_system, A_from_jacobian, atol=1e-10)

    def test_discretize_system_A_independent_of_G_choice(self):
        """Corollary of the above, stated directly: two DIFFERENT G
        matrices with the same J must give the same A."""
        rng = np.random.default_rng(1)
        n = 6
        J = rng.uniform(-1, 1, (n, n)) * 0.1 - np.eye(n) * 0.4
        G1 = rng.uniform(-1, 1, (n, 3))
        G2 = rng.uniform(-1, 1, (n, 5))  # different shape too
        dt = 1.5

        A1, _ = simulation.discretize_system(J, G1, dt)
        A2, _ = simulation.discretize_system(J, G2, dt)
        assert np.allclose(A1, A2, atol=1e-10)

    def test_discretize_jacobian_zero_J_gives_identity(self):
        """expm(0) = I -- a trivial but real sanity check."""
        J = np.zeros((5, 5))
        A = simulation.discretize_jacobian(J, dt=10.0)
        assert np.allclose(A, np.eye(5))

    def test_discretize_system_B_shape(self):
        rng = np.random.default_rng(2)
        n, m = 10, 4
        J = rng.uniform(-1, 1, (n, n)) * 0.1 - np.eye(n) * 0.3
        G = rng.uniform(-1, 1, (n, m))
        A, B = simulation.discretize_system(J, G, dt=2.0)
        assert A.shape == (n, n)
        assert B.shape == (n, m)

    def test_discrete_A_stable_when_continuous_J_stable(self):
        """A stable continuous system (all eigenvalues of J have negative
        real part) should discretize to a stable discrete system
        (rho(A) < 1) -- the standard exp-map stability correspondence."""
        n = 6
        J = -np.eye(n) * 0.5  # clearly stable
        A = simulation.discretize_jacobian(J, dt=2.0)
        rho = np.max(np.abs(np.linalg.eigvals(A)))
        assert rho < 1.0


# ---------------------------------------------------------------------
# WilsonCowanNetwork
# ---------------------------------------------------------------------

class TestWilsonCowanNetwork:
    def _build_stable_net(self, seed=0, N=5):
        rng = np.random.default_rng(seed)
        C = rng.uniform(0, 0.15, (N, N))
        np.fill_diagonal(C, 0)
        params = dict(w_EE=4.0, w_IE=4.0, w_EI=3.0, w_II=2.0,
                      c_E=-2.0, c_I=-2.0, tau_E=10.0, tau_I=20.0)
        return simulation.WilsonCowanNetwork(n_regions=N, C=C, node_params=params)

    def test_init_defaults_to_limit_cycle_params_if_none_given(self):
        net = simulation.WilsonCowanNetwork(n_regions=3, C=np.zeros((3, 3)))
        assert net.params["w_EE"] == simulation.WilsonCowanNode.LIMIT_CYCLE_PARAMS["w_EE"]

    def test_simulate_shapes(self):
        net = self._build_stable_net(N=4)
        result = net.simulate(t_span=(0, 200), n_points=50, seed=1)
        assert result["E"].shape == (4, 50)
        assert result["I"].shape == (4, 50)
        assert result["t"].shape == (50,)

    def test_simulate_reproducible_with_same_seed(self):
        net = self._build_stable_net(N=4)
        r1 = net.simulate(t_span=(0, 100), n_points=20, seed=7)
        r2 = net.simulate(t_span=(0, 100), n_points=20, seed=7)
        assert np.allclose(r1["E"], r2["E"])

    def test_simulate_different_seeds_give_different_trajectories(self):
        net = self._build_stable_net(N=4)
        r1 = net.simulate(t_span=(0, 100), n_points=20, seed=1)
        r2 = net.simulate(t_span=(0, 100), n_points=20, seed=2)
        assert not np.allclose(r1["E"], r2["E"])

    def test_find_fixed_point_residual_is_near_zero(self):
        """The most important correctness property: the returned (E*,I*)
        must actually satisfy dE/dt = dI/dt = 0."""
        net = self._build_stable_net(N=6)
        E_star, I_star = net.find_fixed_point()
        y_star = np.concatenate([E_star, I_star])
        residual = net._ode_network(0.0, y_star)
        assert np.allclose(residual, 0.0, atol=1e-9)

    def test_oscillatory_regime_has_unstable_fixed_point_not_no_fixed_point(self):
        """Corrected understanding, found while writing this test: an
        original version of this test assumed LIMIT_CYCLE_PARAMS would
        make find_fixed_point raise RuntimeError (no fixed point exists).
        That's wrong -- confirmed directly: a genuine fixed point DOES
        exist here (residual ~2e-15), it is simply locally UNSTABLE (max
        Jacobian eigenvalue > 0), the classic Hopf-bifurcation picture
        (unstable fixed point inside a stable limit cycle). A plain
        root-finder has no reason to fail just because the root it finds
        is dynamically unstable -- it only solves f(y)=0. This test
        checks the actually-correct property."""
        rng = np.random.default_rng(0)
        N = 5
        C = rng.uniform(0, 0.15, (N, N))
        np.fill_diagonal(C, 0)
        net = simulation.WilsonCowanNetwork(n_regions=N, C=C)  # LIMIT_CYCLE_PARAMS

        E_star, I_star = net.find_fixed_point()
        y_star = np.concatenate([E_star, I_star])
        residual = net._ode_network(0.0, y_star)
        assert np.allclose(residual, 0.0, atol=1e-9), "should be a genuine root"

        J = net.jacobian_at(E_star, I_star)
        max_eig = np.max(np.linalg.eigvals(J).real)
        assert max_eig > 0, (
            "expected the fixed point in this oscillatory regime to be "
            "locally unstable (Hopf bifurcation), found it stable instead"
        )

        # Confirm genuine oscillation still occurs from a generic IC,
        # consistent with an unstable-fixed-point-inside-limit-cycle picture
        result = net.simulate(t_span=(0, 2000), n_points=4000, seed=1)
        tail_std = np.std(result["E"][0][2000:])
        assert tail_std > 0.01

    def test_find_fixed_point_raises_runtimeerror_on_solver_failure(self, monkeypatch):
        """Tests the error-handling CODE PATH directly via mocking, since
        a naturally non-convergent case could not be reliably constructed
        (tried an extreme initial guess, 1e8, and the solver -- scipy's
        hybr method -- converged anyway). This is a legitimate, standard
        way to test an error path that's hard to trigger organically:
        confirms the `if not result.success: raise RuntimeError` logic
        itself is correct, independent of finding a case that exercises
        it naturally.

        Patches simulation.root, NOT scipy.optimize.root -- simulation.py
        does `from scipy.optimize import root`, which creates its own
        name binding in simulation's namespace at import time. Patching
        scipy.optimize.root would not affect that already-bound name;
        confirmed by checking simulation.py's actual import statement
        before writing this, not assumed."""
        net = self._build_stable_net(N=3)

        class FakeFailedResult:
            success = False
            x = np.zeros(6)

        monkeypatch.setattr(simulation, "root", lambda *a, **k: FakeFailedResult())
        with pytest.raises(RuntimeError, match="did not converge"):
            net.find_fixed_point()

    def test_jacobian_at_matches_numerical_finite_difference(self):
        """Direct correctness check of jacobian_at against numerical
        differentiation of the network ODE -- formalized as a permanent
        test (previously only checked ad hoc during unrelated debugging
        work, never as a standalone test of this function)."""
        net = self._build_stable_net(seed=2, N=4)
        E_star, I_star = net.find_fixed_point()
        J_analytic = net.jacobian_at(E_star, I_star)

        y_star = np.concatenate([E_star, I_star])
        n = len(y_star)
        eps = 1e-6
        J_numerical = np.zeros((n, n))
        f0 = net._ode_network(0.0, y_star)
        for i in range(n):
            y_pert = y_star.copy()
            y_pert[i] += eps
            f1 = net._ode_network(0.0, y_pert)
            J_numerical[:, i] = (f1 - f0) / eps

        assert np.allclose(J_analytic, J_numerical, atol=1e-4)

    def test_jacobian_at_stable_fixed_point_has_negative_eigenvalues(self):
        """A fixed point found by the solver (which uses a generic
        root-finder, not a stability-aware one) isn't guaranteed to be
        LOCALLY stable in general -- but for this network's damped
        parameter regime it should be. Sanity check, not a universal
        guarantee."""
        net = self._build_stable_net(seed=3, N=5)
        E_star, I_star = net.find_fixed_point()
        J = net.jacobian_at(E_star, I_star)
        max_eig = np.max(np.linalg.eigvals(J).real)
        assert max_eig < 0

    def test_simulate_controlled_zero_control_matches_uncontrolled(self):
        """Injecting a zero control function should reproduce the
        uncontrolled trajectory from the same initial condition."""
        net = self._build_stable_net(seed=4, N=4)
        E_star, I_star = net.find_fixed_point()

        def zero_u(t):
            return np.zeros(4)

        controlled = net.simulate_controlled(
            u_func=zero_u, t_span=(0, 50), n_points=30, E0=E_star, I0=I_star,
        )
        # starting exactly at the fixed point with zero control should stay there
        assert np.allclose(controlled["E"][:, -1], E_star, atol=1e-4)
        assert np.allclose(controlled["I"][:, -1], I_star, atol=1e-4)

    def test_simulate_controlled_nonzero_control_moves_state(self):
        """A genuine, nonzero control should actually perturb the
        trajectory away from the fixed point -- confirms the u_func hook
        is real, not silently ignored."""
        net = self._build_stable_net(seed=5, N=4)
        E_star, I_star = net.find_fixed_point()

        def constant_u(t):
            return np.full(4, 2.0)

        controlled = net.simulate_controlled(
            u_func=constant_u, t_span=(0, 50), n_points=30, E0=E_star, I0=I_star,
        )
        assert not np.allclose(controlled["E"][:, -1], E_star, atol=1e-3)

    def test_extract_bold_proxy_shape(self):
        net = self._build_stable_net(N=3)
        result = net.simulate(t_span=(0, 2000), n_points=2000, seed=1)
        bold = net.extract_bold_proxy(result, tr_ms=720.0)
        expected_n_trs = int(2000 / 720.0) + 1  # arange(0, t_max, tr_ms)
        assert bold.shape[0] == 3  # n_regions
        assert bold.shape[1] == len(np.arange(0, 2000, 720.0))

    def test_extract_bold_proxy_values_are_interpolated_not_extrapolated(self):
        """BOLD proxy values should stay within the range of the original
        E trace (pure interpolation, no overshoot)."""
        net = self._build_stable_net(N=2)
        result = net.simulate(t_span=(0, 1000), n_points=1000, seed=1)
        bold = net.extract_bold_proxy(result, tr_ms=200.0)
        assert bold.min() >= result["E"].min() - 1e-9
        assert bold.max() <= result["E"].max() + 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
