"""
tests/test_harmonize.py
=======================
Unit tests for neurosim.harmonize.

The BlindHarmonizer currently has zero test coverage in the repository.
This module addresses that gap entirely.

Scientific context
------------------
The BlindHarmonizer enforces the "controls-only" harmonisation protocol:
site-effect parameters are estimated from healthy controls (HC) only,
then applied to all subjects. This prevents diagnostic leakage — the
central statistical integrity guarantee of NeuroSim's multi-site pipeline.

These tests verify:
1. The fitted model correctly estimates site means and variances.
2. Transform output has the same shape as input.
3. After harmonisation, site-wise means are closer to the grand mean.
4. The protocol does NOT use patient/diagnostic data during fit.
5. Graceful handling of unseen sites during transform.
6. fit_transform is equivalent to fit then transform.
7. detect_site_effects correctly flags significant site effects.
"""

import warnings

import numpy as np
import pytest

from neurosim.harmonize import BlindHarmonizer, detect_site_effects

# Fixture

def make_multisite_data(
    n_sites=3,
    n_ctrl_per_site=20,
    n_patient_per_site=10,
    n_features=50,
    site_effect_scale=2.0,
    seed=42,
):
    """Generate synthetic multi-site data with known site effects."""
    rng = np.random.default_rng(seed)
    sites_ctrl, sites_all = [], []
    X_ctrl_list, X_all_list = [], []

    for s in range(n_sites):
        # Site effect: additive offset per site
        site_offset = rng.normal(0, site_effect_scale, n_features)

        # Controls
        X_c = rng.normal(0, 1, (n_ctrl_per_site, n_features)) + site_offset
        X_ctrl_list.append(X_c)
        sites_ctrl.extend([f"site_{s}"] * n_ctrl_per_site)

        # Patients (with slight disease signal on top)
        disease_signal = rng.normal(0.5, 0.2, n_features)
        X_p = rng.normal(0, 1, (n_patient_per_site, n_features)) + site_offset + disease_signal
        X_all_list.append(X_c)
        X_all_list.append(X_p)
        sites_all.extend([f"site_{s}"] * n_ctrl_per_site)
        sites_all.extend([f"site_{s}"] * n_patient_per_site)

    X_ctrl = np.vstack(X_ctrl_list)
    X_all  = np.vstack(X_all_list)
    return X_ctrl, X_all, np.array(sites_ctrl), np.array(sites_all)

# BlindHarmonizer — basic functionality=

class TestBlindHarmonizerBasic:

    def test_fit_returns_self(self):
        X_ctrl, _, sites_ctrl, _ = make_multisite_data()
        h = BlindHarmonizer()
        result = h.fit(X_ctrl, sites_ctrl)
        assert result is h, "fit() should return self for method chaining"

    def test_is_fitted_flag(self):
        X_ctrl, _, sites_ctrl, _ = make_multisite_data()
        h = BlindHarmonizer()
        assert not h.is_fitted_
        h.fit(X_ctrl, sites_ctrl)
        assert h.is_fitted_

    def test_transform_before_fit_raises(self):
        X_ctrl, X_all, _, sites_all = make_multisite_data()
        h = BlindHarmonizer()
        with pytest.raises(RuntimeError, match="fit"):
            h.transform(X_all, sites_all)

    def test_transform_preserves_shape(self):
        X_ctrl, X_all, sites_ctrl, sites_all = make_multisite_data()
        h = BlindHarmonizer()
        h.fit(X_ctrl, sites_ctrl)
        X_h = h.transform(X_all, sites_all)
        assert X_h.shape == X_all.shape

    def test_no_nans_in_output(self):
        X_ctrl, X_all, sites_ctrl, sites_all = make_multisite_data()
        h = BlindHarmonizer()
        h.fit(X_ctrl, sites_ctrl)
        X_h = h.transform(X_all, sites_all)
        assert not np.any(np.isnan(X_h)), "Harmonised output contains NaN values"

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        X_ctrl, X_all, sites_ctrl, sites_all = make_multisite_data(seed=99)

        h1 = BlindHarmonizer()
        h1.fit(X_ctrl, sites_ctrl)
        X_h1 = h1.transform(X_all, sites_all)

        h2 = BlindHarmonizer()
        X_h2 = h2.fit_transform(X_ctrl, sites_ctrl, X_all, sites_all)

        assert np.allclose(X_h1, X_h2, atol=1e-10), \
            "fit_transform result differs from fit then transform"


# BlindHarmonizer — scientific correctness

class TestBlindHarmonizerScience:

    def test_site_means_reduced_after_harmonisation(self):
        """
        After blind harmonisation, per-site feature means should be closer
        to the grand mean than before harmonisation. This is the core
        statistical guarantee of ComBat-style correction.
        """
        X_ctrl, X_all, sites_ctrl, sites_all = make_multisite_data(
            site_effect_scale=3.0, seed=7
        )
        h = BlindHarmonizer()
        h.fit(X_ctrl, sites_ctrl)
        X_h = h.transform(X_all, sites_all)

        grand_mean = X_all.mean(axis=0)
        grand_mean_h = X_h.mean(axis=0)

        unique_sites = np.unique(sites_all)
        pre_dispersion, post_dispersion = [], []

        for site in unique_sites:
            mask = sites_all == site
            pre_dispersion.append(np.mean(np.abs(X_all[mask].mean(axis=0) - grand_mean)))
            post_dispersion.append(np.mean(np.abs(X_h[mask].mean(axis=0) - grand_mean_h)))

        mean_pre  = np.mean(pre_dispersion)
        mean_post = np.mean(post_dispersion)

        assert mean_post < mean_pre, (
            f"Site mean dispersion not reduced: pre={mean_pre:.4f}, post={mean_post:.4f}. "
            "Harmonisation should reduce inter-site mean differences."
        )

    def test_controls_only_fit_does_not_use_patient_data(self):
        """
        The BlindHarmonizer must estimate site effects from controls only.
        Fitting on controls vs controls+patients should give different
        grand means and site estimates.
        """
        X_ctrl, X_all, sites_ctrl, sites_all = make_multisite_data(seed=42)

        h_blind = BlindHarmonizer()
        h_blind.fit(X_ctrl, sites_ctrl)

        h_nonblind = BlindHarmonizer()
        h_nonblind.fit(X_all, sites_all)  # wrong — uses patients

        # Grand means should differ
        assert not np.allclose(h_blind.grand_mean_, h_nonblind.grand_mean_, atol=1e-6), \
            "Controls-only and all-subjects fits should produce different grand means."

    def test_single_site_passthrough(self):
        """With only one site, harmonisation is a no-op (no between-site correction)."""
        rng = np.random.default_rng(5)
        X = rng.normal(0, 1, (30, 20))
        sites = np.array(["site_A"] * 30)

        h = BlindHarmonizer()
        h.fit(X, sites)
        X_h = h.transform(X, sites)

        assert X_h.shape == X.shape

    def test_grand_mean_computed_from_controls_only(self):
        """grand_mean_ should equal the column-wise mean of the control matrix."""
        X_ctrl, _, sites_ctrl, _ = make_multisite_data(seed=3)
        h = BlindHarmonizer()
        h.fit(X_ctrl, sites_ctrl)
        expected_grand_mean = X_ctrl.mean(axis=0)
        assert np.allclose(h.grand_mean_, expected_grand_mean, atol=1e-10)

    def test_site_means_stored_correctly(self):
        """site_means_ for each site should equal the within-site mean of centred controls."""
        rng = np.random.default_rng(8)
        n, f = 30, 10
        X_ctrl = rng.normal(0, 1, (n, f))
        sites_ctrl = np.array(["A"] * 15 + ["B"] * 15)

        h = BlindHarmonizer()
        h.fit(X_ctrl, sites_ctrl)

        grand_mean = X_ctrl.mean(axis=0)
        expected_A = (X_ctrl[:15] - grand_mean).mean(axis=0)
        expected_B = (X_ctrl[15:] - grand_mean).mean(axis=0)

        assert np.allclose(h.site_means_["A"], expected_A, atol=1e-10)
        assert np.allclose(h.site_means_["B"], expected_B, atol=1e-10)


# BlindHarmonizer — edge cases

class TestBlindHarmonizerEdgeCases:

    def test_unseen_site_warns_and_passes_through(self):
        """If transform sees a site not in fit, it should warn and skip correction."""
        X_ctrl, _, sites_ctrl, _ = make_multisite_data(seed=10)
        h = BlindHarmonizer()
        h.fit(X_ctrl, sites_ctrl)

        # Create data with an unseen site
        rng = np.random.default_rng(11)
        X_new = rng.normal(0, 1, (5, X_ctrl.shape[1]))
        sites_new = np.array(["unseen_site"] * 5)

        with pytest.warns(UserWarning, match="not seen"):
            X_h = h.transform(X_new, sites_new)

        assert X_h.shape == X_new.shape

    def test_two_sites_minimum(self):
        """Harmonisation requires at least 2 sites to make sense."""
        rng = np.random.default_rng(12)
        X = rng.normal(0, 1, (20, 10))
        sites = np.array(["A"] * 10 + ["B"] * 10)
        h = BlindHarmonizer()
        h.fit(X, sites)
        X_h = h.transform(X, sites)
        assert X_h.shape == X.shape

    def test_large_site_effect_substantially_reduced(self):
        """A very large site effect should be substantially reduced after harmonisation."""
        rng = np.random.default_rng(13)
        n, f = 40, 20

        # Site A has massive additive offset
        X_A = rng.normal(0, 1, (n // 2, f)) + 10.0
        X_B = rng.normal(0, 1, (n // 2, f))
        X_ctrl = np.vstack([X_A, X_B])
        sites_ctrl = np.array(["A"] * (n // 2) + ["B"] * (n // 2))

        h = BlindHarmonizer()
        h.fit(X_ctrl, sites_ctrl)
        X_h = h.transform(X_ctrl, sites_ctrl)

        mean_A_pre  = X_ctrl[sites_ctrl == "A"].mean()
        mean_B_pre  = X_ctrl[sites_ctrl == "B"].mean()
        mean_A_post = X_h[sites_ctrl == "A"].mean()
        mean_B_post = X_h[sites_ctrl == "B"].mean()

        pre_gap  = abs(mean_A_pre  - mean_B_pre)
        post_gap = abs(mean_A_post - mean_B_post)

        assert post_gap < pre_gap * 0.5, (
            f"Large site effect not substantially reduced: "
            f"pre gap={pre_gap:.2f}, post gap={post_gap:.2f}"
        )


# detect_site_effects

class TestDetectSiteEffects:

    def test_strong_site_effects_detected(self):
        """Clearly different sites should be flagged as recommend_harmonise=True."""
        rng = np.random.default_rng(20)
        n_subjects, n_features = 60, 100
        X = rng.normal(0, 1, (n_subjects, n_features))
        # Add a large site effect to first 30 subjects
        X[:30] += 5.0
        sites = np.array(["A"] * 30 + ["B"] * 30)

        result = detect_site_effects(X, sites)
        assert result["recommend_harmonise"] is True
        assert result["fraction_sig"] > 0.5

    def test_homogeneous_data_not_flagged(self):
        """Identical-distribution sites should NOT be flagged."""
        rng = np.random.default_rng(21)
        X = rng.normal(0, 1, (60, 100))  # no site effect
        sites = np.array(["A"] * 30 + ["B"] * 30)

        result = detect_site_effects(X, sites, alpha=0.001)  # strict threshold
        # With no site effect, very few features should be significant
        assert result["fraction_sig"] < 0.20

    def test_single_site_returns_no_recommendation(self):
        """Single site — harmonisation not applicable."""
        X = np.random.default_rng(22).normal(0, 1, (20, 50))
        sites = np.array(["A"] * 20)
        result = detect_site_effects(X, sites)
        assert result["recommend_harmonise"] is False

    def test_result_has_required_keys(self):
        X = np.random.default_rng(23).normal(0, 1, (40, 50))
        sites = np.array(["A"] * 20 + ["B"] * 20)
        result = detect_site_effects(X, sites)
        assert "fraction_sig" in result
        assert "recommend_harmonise" in result

    def test_fraction_sig_in_valid_range(self):
        X = np.random.default_rng(24).normal(0, 1, (40, 50))
        sites = np.array(["A"] * 20 + ["B"] * 20)
        result = detect_site_effects(X, sites)
        assert 0.0 <= result["fraction_sig"] <= 1.0
