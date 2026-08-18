"""
Comprehensive unit tests for neurosim.connectivity.

Covers all 7 public functions: functional_connectivity,
ridge_effective_connectivity, graph_laplacian,
graphnet_effective_connectivity, block_bootstrap_ec,
driver_node_rank_stability, simulate_feedforward_network.

Test sizes kept small (N<=6, T<=200) throughout since
graphnet_effective_connectivity's FISTA solver and block_bootstrap_ec's
repeated re-estimation are the most compute-heavy functions in this
module -- there's no need for realistic problem sizes to verify
correctness.
"""
import numpy as np
import pytest

from neurosim import connectivity


# ---------------------------------------------------------------------
# functional_connectivity
# ---------------------------------------------------------------------

class TestFunctionalConnectivity:
    def test_symmetric(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (5, 100))
        FC = connectivity.functional_connectivity(X)
        assert np.allclose(FC, FC.T)

    def test_zero_diagonal(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (5, 100))
        FC = connectivity.functional_connectivity(X)
        assert np.allclose(np.diag(FC), 0.0)

    def test_values_bounded_in_pearson_range(self):
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (6, 200))
        FC = connectivity.functional_connectivity(X)
        assert np.all(FC >= -1.0 - 1e-9) and np.all(FC <= 1.0 + 1e-9)

    def test_perfectly_correlated_signals_give_near_1(self):
        """Two identical (up to noise) time series should show FC close
        to 1 off-diagonal -- a direct correctness check, not just a shape
        check."""
        rng = np.random.default_rng(3)
        base = rng.normal(0, 1, 200)
        X = np.stack([base, base + rng.normal(0, 1e-6, 200), rng.normal(0, 1, 200)])
        FC = connectivity.functional_connectivity(X, detrend=False)
        assert FC[0, 1] > 0.99

    def test_detrend_changes_output_for_trending_signal(self):
        """A linear trend should inflate correlation if NOT removed --
        confirms the detrend flag is actually doing something, not
        silently ignored."""
        rng = np.random.default_rng(4)
        t = np.arange(200)
        X = np.stack([
            t * 0.1 + rng.normal(0, 1, 200),
            t * 0.1 + rng.normal(0, 1, 200),
            rng.normal(0, 1, 200),
        ])
        FC_detrended = connectivity.functional_connectivity(X, detrend=True)
        FC_raw = connectivity.functional_connectivity(X, detrend=False)
        assert not np.allclose(FC_detrended, FC_raw)


# ---------------------------------------------------------------------
# ridge_effective_connectivity
# ---------------------------------------------------------------------

class TestRidgeEffectiveConnectivity:
    def test_shape(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (5, 100))
        EC = connectivity.ridge_effective_connectivity(X)
        assert EC.shape == (5, 5)

    def test_recovers_directional_structure_on_known_causal_chain(self):
        """The module provides simulate_feedforward_network specifically
        to generate ground-truth causal data -- use it directly: ridge EC
        should recover the correct DIRECTION (EC[i+1,i] > EC[i,i+1]) for
        a genuine feedforward chain, not just produce SOME matrix."""
        X, A_true = connectivity.simulate_feedforward_network(
            n_nodes=4, n_timepoints=3000, causal_weight=0.85, noise_std=0.1, seed=0)
        EC = connectivity.ridge_effective_connectivity(X, alpha=1.0, lag=1)
        for i in range(3):
            assert EC[i + 1, i] > EC[i, i + 1], (
                f"expected recovered causal weight {i}->({i+1}) to exceed "
                f"the reverse direction, given the ground-truth chain only "
                f"has a forward edge here"
            )

    def test_higher_alpha_shrinks_coefficients(self):
        """Ridge regularization should shrink coefficients toward zero as
        alpha increases -- confirms alpha is actually wired to the
        regression, not a no-op parameter."""
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (5, 100))
        EC_low = connectivity.ridge_effective_connectivity(X, alpha=0.01)
        EC_high = connectivity.ridge_effective_connectivity(X, alpha=100.0)
        assert np.sum(EC_high ** 2) < np.sum(EC_low ** 2)

    def test_lag_parameter_changes_output(self):
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (4, 100))
        EC_lag1 = connectivity.ridge_effective_connectivity(X, lag=1)
        EC_lag2 = connectivity.ridge_effective_connectivity(X, lag=2)
        assert not np.allclose(EC_lag1, EC_lag2)


# ---------------------------------------------------------------------
# graph_laplacian
# ---------------------------------------------------------------------

class TestGraphLaplacian:
    def test_unnormalized_row_sums_are_zero(self):
        """L = D - SC: row sums must be exactly zero (degree minus its own
        degree), a basic, exactly-checkable Laplacian property."""
        rng = np.random.default_rng(0)
        SC = rng.uniform(0, 1, (6, 6))
        SC = (SC + SC.T) / 2
        L = connectivity.graph_laplacian(SC, normalised=False)
        assert np.allclose(L.sum(axis=1), 0.0, atol=1e-10)

    def test_enforces_symmetry_even_on_asymmetric_input(self):
        """Function explicitly symmetrizes SC internally -- confirm this
        actually happens for a genuinely asymmetric input."""
        rng = np.random.default_rng(1)
        SC_asymmetric = rng.uniform(0, 1, (5, 5))  # NOT symmetrized
        L = connectivity.graph_laplacian(SC_asymmetric)
        assert np.allclose(L, L.T)

    def test_normalised_diagonal_is_one_where_degree_nonzero(self):
        """Normalized Laplacian: diagonal should be 1 for nodes with
        nonzero degree, per the standard normalized-Laplacian formula."""
        rng = np.random.default_rng(2)
        SC = rng.uniform(0.1, 1, (5, 5))  # all positive, guarantees nonzero degree
        SC = (SC + SC.T) / 2
        np.fill_diagonal(SC, 0)
        L = connectivity.graph_laplacian(SC, normalised=True)
        assert np.allclose(np.diag(L), 1.0, atol=1e-8)

    def test_zero_degree_node_does_not_produce_nan(self):
        """An isolated node (zero degree) should not cause division by
        zero in the normalized case -- the implementation guards this
        with np.maximum(diag(D), 1e-12)."""
        SC = np.zeros((4, 4))
        SC[0, 1] = SC[1, 0] = 1.0  # only nodes 0,1 connected; 2,3 isolated
        L = connectivity.graph_laplacian(SC, normalised=True)
        assert not np.any(np.isnan(L))

    def test_diagonal_forced_to_zero_before_laplacian(self):
        """Function fills the diagonal of SC to 0 before computing D and
        L -- confirm a nonzero self-loop in the input doesn't leak into
        the degree calculation."""
        SC = np.array([[5.0, 1.0], [1.0, 5.0]])  # large, spurious self-loops
        L = connectivity.graph_laplacian(SC, normalised=False)
        # degree should be based on off-diagonal only (1.0), not 6.0
        assert np.isclose(L[0, 0], 1.0)


# ---------------------------------------------------------------------
# graphnet_effective_connectivity
# ---------------------------------------------------------------------

class TestGraphnetEffectiveConnectivity:
    def test_shape(self):
        rng = np.random.default_rng(0)
        N, T = 5, 100
        X = rng.normal(0, 1, (N, T))
        SC = rng.uniform(0, 1, (N, N))
        SC = (SC + SC.T) / 2
        EC = connectivity.graphnet_effective_connectivity(X, SC, max_iter=50)
        assert EC.shape == (N, N)

    def test_no_nan_or_inf_in_output(self):
        rng = np.random.default_rng(1)
        N, T = 6, 150
        X = rng.normal(0, 1, (N, T))
        SC = rng.uniform(0, 1, (N, N))
        SC = (SC + SC.T) / 2
        EC = connectivity.graphnet_effective_connectivity(X, SC, max_iter=100)
        assert np.all(np.isfinite(EC))

    def test_stronger_graph_penalty_pulls_toward_sc_structure(self):
        """As lambda_graph grows (with lambda_ridge fixed), the estimated
        EC should be pulled increasingly toward matching SC's sparsity
        pattern -- confirms the graph Laplacian penalty is actually doing
        something, not a no-op regularizer. Checked via correlation
        between |EC| and SC across entries, which should increase."""
        rng = np.random.default_rng(2)
        N, T = 6, 150
        X = rng.normal(0, 1, (N, T))
        SC = rng.uniform(0, 1, (N, N))
        SC = (SC + SC.T) / 2
        np.fill_diagonal(SC, 0)

        EC_weak_graph = connectivity.graphnet_effective_connectivity(
            X, SC, lambda_ridge=1.0, lambda_graph=0.01, max_iter=200)
        EC_strong_graph = connectivity.graphnet_effective_connectivity(
            X, SC, lambda_ridge=1.0, lambda_graph=50.0, max_iter=200)

        mask = ~np.eye(N, dtype=bool)
        corr_weak = np.corrcoef(np.abs(EC_weak_graph[mask]), SC[mask])[0, 1]
        corr_strong = np.corrcoef(np.abs(EC_strong_graph[mask]), SC[mask])[0, 1]
        assert corr_strong > corr_weak

    def test_convergence_within_max_iter_for_well_conditioned_problem(self):
        """A well-conditioned, small problem should converge (not hit
        max_iter without stabilizing) -- checked indirectly by confirming
        the result is stable under a further increase in max_iter."""
        rng = np.random.default_rng(3)
        N, T = 5, 200
        X = rng.normal(0, 1, (N, T))
        SC = rng.uniform(0, 1, (N, N))
        SC = (SC + SC.T) / 2
        EC_200 = connectivity.graphnet_effective_connectivity(X, SC, max_iter=200)
        EC_500 = connectivity.graphnet_effective_connectivity(X, SC, max_iter=500)
        assert np.allclose(EC_200, EC_500, atol=1e-3)


# ---------------------------------------------------------------------
# block_bootstrap_ec
# ---------------------------------------------------------------------

class TestBlockBootstrapEC:
    @staticmethod
    def _fast_ec_func(X, SC, **kwargs):
        """Cheap stand-in for graphnet_effective_connectivity, so these
        tests run fast -- correctness of the bootstrap MACHINERY is what's
        under test here, not the (separately tested) EC estimator itself."""
        return connectivity.ridge_effective_connectivity(X, alpha=1.0)

    def test_shape(self):
        rng = np.random.default_rng(0)
        N, T = 5, 100
        X = rng.normal(0, 1, (N, T))
        SC = rng.uniform(0, 1, (N, N))
        EC_boot = connectivity.block_bootstrap_ec(
            X, SC, ec_func=self._fast_ec_func, n_boot=10, block_length=10, seed=0)
        assert EC_boot.shape == (10, N, N)

    def test_reproducible_with_same_seed(self):
        rng = np.random.default_rng(1)
        N, T = 4, 80
        X = rng.normal(0, 1, (N, T))
        SC = rng.uniform(0, 1, (N, N))
        EC_boot_1 = connectivity.block_bootstrap_ec(
            X, SC, ec_func=self._fast_ec_func, n_boot=5, block_length=10, seed=42)
        EC_boot_2 = connectivity.block_bootstrap_ec(
            X, SC, ec_func=self._fast_ec_func, n_boot=5, block_length=10, seed=42)
        assert np.allclose(EC_boot_1, EC_boot_2)

    def test_different_seeds_give_different_resamples(self):
        rng = np.random.default_rng(2)
        N, T = 4, 80
        X = rng.normal(0, 1, (N, T))
        SC = rng.uniform(0, 1, (N, N))
        EC_boot_1 = connectivity.block_bootstrap_ec(
            X, SC, ec_func=self._fast_ec_func, n_boot=5, block_length=10, seed=1)
        EC_boot_2 = connectivity.block_bootstrap_ec(
            X, SC, ec_func=self._fast_ec_func, n_boot=5, block_length=10, seed=2)
        assert not np.allclose(EC_boot_1, EC_boot_2)

    def test_raises_when_block_length_exceeds_series_length(self):
        rng = np.random.default_rng(3)
        N, T = 4, 10
        X = rng.normal(0, 1, (N, T))
        SC = rng.uniform(0, 1, (N, N))
        with pytest.raises(ValueError, match="exceeds series length"):
            connectivity.block_bootstrap_ec(
                X, SC, ec_func=self._fast_ec_func, n_boot=5, block_length=20, seed=0)

    def test_defaults_to_graphnet_when_ec_func_not_given(self):
        """Confirms the None-defaults-to-graphnet behavior actually
        happens, using a tiny n_boot/max_iter to keep it fast."""
        rng = np.random.default_rng(4)
        N, T = 3, 60
        X = rng.normal(0, 1, (N, T))
        SC = rng.uniform(0, 1, (N, N))
        EC_boot = connectivity.block_bootstrap_ec(
            X, SC, n_boot=2, block_length=15, seed=0, max_iter=30)
        assert EC_boot.shape == (2, N, N)
        assert np.all(np.isfinite(EC_boot))


# ---------------------------------------------------------------------
# driver_node_rank_stability
# ---------------------------------------------------------------------

class TestDriverNodeRankStability:
    def test_identical_bootstrap_draws_give_perfect_stability(self):
        """If every bootstrap EC is IDENTICAL, rankings must be perfectly
        stable: Kendall's tau = 1.0 exactly, Jaccard = 1.0 exactly."""
        N = 6
        rng = np.random.default_rng(0)
        single_EC = rng.uniform(0, 1, (N, N))
        EC_boot = np.stack([single_EC] * 10)

        def controllability_func(A):
            return np.diag(A)  # arbitrary but deterministic scoring

        result = connectivity.driver_node_rank_stability(EC_boot, controllability_func, top_k=3)
        assert np.isclose(result["kendall_tau_mean"], 1.0)
        assert np.isclose(result["jaccard_topk_mean"], 1.0)
        assert result["kendall_tau_std"] < 1e-10

    def test_independent_random_draws_give_low_stability(self):
        """Genuinely independent, unrelated bootstrap EC matrices should
        show near-chance rank stability, not spuriously high agreement."""
        N = 8
        rng = np.random.default_rng(1)
        EC_boot = rng.uniform(0, 1, (30, N, N))

        def controllability_func(A):
            return np.diag(A)

        result = connectivity.driver_node_rank_stability(EC_boot, controllability_func, top_k=3)
        assert abs(result["kendall_tau_mean"]) < 0.3  # near-chance, not perfect agreement

    def test_returns_expected_keys(self):
        N = 5
        rng = np.random.default_rng(2)
        EC_boot = rng.uniform(0, 1, (5, N, N))
        result = connectivity.driver_node_rank_stability(
            EC_boot, lambda A: np.diag(A), top_k=2)
        expected_keys = {"kendall_tau_mean", "kendall_tau_std",
                          "jaccard_topk_mean", "jaccard_topk_std", "rankings"}
        assert set(result.keys()) == expected_keys

    def test_rankings_shape(self):
        N, n_boot = 6, 8
        rng = np.random.default_rng(3)
        EC_boot = rng.uniform(0, 1, (n_boot, N, N))
        result = connectivity.driver_node_rank_stability(
            EC_boot, lambda A: np.diag(A), top_k=2)
        assert result["rankings"].shape == (n_boot, N)

    def test_jaccard_bounded_0_to_1(self):
        N = 6
        rng = np.random.default_rng(4)
        EC_boot = rng.uniform(0, 1, (10, N, N))
        result = connectivity.driver_node_rank_stability(
            EC_boot, lambda A: np.diag(A), top_k=3)
        assert 0.0 <= result["jaccard_topk_mean"] <= 1.0


# ---------------------------------------------------------------------
# simulate_feedforward_network
# ---------------------------------------------------------------------

class TestSimulateFeedforwardNetwork:
    def test_shapes(self):
        X, A_true = connectivity.simulate_feedforward_network(
            n_nodes=4, n_timepoints=500)
        assert X.shape == (4, 500)
        assert A_true.shape == (4, 4)

    def test_a_true_is_pure_feedforward_chain(self):
        """Only the subdiagonal (i+1 <- i) should be nonzero -- confirms
        the ground-truth structure is exactly a serial chain, not
        anything more complex."""
        _, A_true = connectivity.simulate_feedforward_network(n_nodes=5)
        for i in range(5):
            for j in range(5):
                if j == i - 1:
                    assert A_true[i, j] != 0
                else:
                    assert A_true[i, j] == 0

    def test_causal_weight_sets_subdiagonal_value(self):
        _, A_true = connectivity.simulate_feedforward_network(
            n_nodes=3, causal_weight=0.42)
        assert np.isclose(A_true[1, 0], 0.42)
        assert np.isclose(A_true[2, 1], 0.42)

    def test_reproducible_with_same_seed(self):
        X1, _ = connectivity.simulate_feedforward_network(seed=7)
        X2, _ = connectivity.simulate_feedforward_network(seed=7)
        assert np.allclose(X1, X2)

    def test_downstream_nodes_correlate_with_upstream(self):
        """Direct confirmation the causal structure actually propagates
        into the generated time series, not just present in A_true
        without affecting X."""
        X, _ = connectivity.simulate_feedforward_network(
            n_nodes=3, n_timepoints=5000, causal_weight=0.9, noise_std=0.05, seed=0)
        # node 1 depends on node 0's PAST value -- check via 1-lag cross-correlation
        corr = np.corrcoef(X[0, :-1], X[1, 1:])[0, 1]
        assert corr > 0.3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
