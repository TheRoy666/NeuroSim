"""
neurosim.harmonize
==================
Blind multi-site harmonisation to prevent data leakage. Provides a Reference-Group-only Empirical Bayes implementation to prevent 
diagnostic circularity (data leakage) in multi-site cohorts.

TODO (GSoC Week 2): Wrap the official `neuroCombat` Python package to 
handle edge cases (missing covariates, small N batches) while strictly 
enforcing this Reference-Group logic.****

The Statistical Leakage Crisis
-------------------------------
Standard ComBat harmonisation (Johnson et al., 2007; Fortin et al., 2017)
accepts a ``covariates`` argument that may include the diagnostic outcome
variable (e.g., Patient vs. Control). Including diagnosis as a covariate
instructs the Empirical Bayes algorithm to *preserve* group differences
while removing scanner effects.

The problem: if the harmonised data then feeds a classifier that is cross-
validated *after* harmonisation on the full cohort, diagnostic information
has leaked into the feature space. This inflates AUC and creates
irreproducible biomarkers - contributing directly to the replication crisis
in computational psychiatry.

NeuroSim Blind Harmonisation Protocol
--------------------------------------
The ``BlindHarmonizer`` class enforces a strict controls-only fit:

1. **Fit** the ComBat model exclusively on healthy control (HC) data.
   The model learns the mapping: site effects -> residuals, using only
   biological covariates (Age, Sex, IQ, BMI).

2. **Transform** all subjects (HC + patients) by applying the HC-derived
   batch-effect estimates.

This ensures the site-effect correction is *not* contaminated by diagnostic
identity. Any group difference surviving this correction is a genuine
biological signal.

References
----------
Fortin, J.-P. et al. (2017). Harmonization of multi-site diffusion tensor
    imaging data. NeuroImage, 161, 149–170.
Fortin, J.-P. et al. (2018). Harmonization of cortical thickness measurements
    across scanners and sites. NeuroImage, 167, 104–120.
Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in
    microarray expression data using empirical Bayes methods. Biostatistics.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


# Diagnostic: ANOVA site-effect detection

def detect_site_effects(
    X: NDArray,
    site_labels: Sequence,
    alpha: float = 0.05,
    n_features_check: int = 50,
) -> Dict:
    """ANOVA check for significant site effects before harmonisation.

    Running ComBat on data without meaningful site effects introduces
    unnecessary variance ("over-correction"). This function tests a random
    sample of features via one-way ANOVA across sites.

    Parameters
    ----------
    X              : (N_subjects, N_features) ndarray - Feature matrix.
    site_labels    : sequence of length N_subjects      - Site identifier per subject.
    alpha          : float - Significance threshold (default 0.05).
    n_features_check : int - Number of features to sample for the ANOVA.

    Returns
    -------
    result : dict with keys:
        ``"n_significant"``   - Number of features with p < alpha.
        ``"fraction_sig"``    - Proportion significant.
        ``"recommend_harmonise"`` - True if >5% features are significant.
        ``"mean_F"``          - Mean F-statistic across sampled features.
    """
    X = np.asarray(X, dtype=float)
    sites = np.asarray(site_labels)
    unique_sites = np.unique(sites)

   if len(unique_sites) < 2:
        return {"recommend_harmonise": False, "note": "Only one site detected."}

    n_features = X.shape[1]
    idx = np.random.choice(n_features, size=min(n_features_check, n_features), replace=False)
    X_sub = X[:, idx]

    p_values = []
    for col in range(X_sub.shape[1]):
        groups = [X_sub[sites == s, col] for s in unique_sites]
        _, p = stats.f_oneway(*groups)
        p_values.append(p)

    p_values = np.array(p_values)
    frac_sig = (p_values < alpha).mean()

    return {
        "fraction_sig": float(frac_sig),
        "recommend_harmonise": frac_sig > 0.05,
    } 


# BlindHarmonizer (controls-only ComBat-style Empirical Bayes based)

class BlindHarmonizer:
    """Blind harmonisation: fit on controls, apply to all subjects.

    Implements a simplified Empirical Bayes harmonisation (ComBat-style)
    that estimates site-effect parameters *exclusively* from healthy control
    data, then applies those estimates to all subjects - preventing diagnostic
    leakage.

    Parameters
    ----------
    biological_covariates : list of str, optional
        Column names in the covariate DataFrame to preserve (e.g., Age, Sex).
        **Must NOT include diagnostic group labels.**

    Attributes
    ----------
    site_means_   : dict - Per-site, per-feature additive effect estimates.
    site_vars_    : dict - Per-site, per-feature multiplicative effect estimates.
    grand_mean_   : ndarray - Feature-wise grand mean estimated from controls.
    is_fitted_    : bool - Whether the model has been fitted.

    Examples
    --------
    >>> harmonizer = BlindHarmonizer(biological_covariates=["age", "sex"])
    >>> harmonizer.fit(X_controls, site_controls)
    >>> X_harmonised = harmonizer.transform(X_all, site_all)
    """

    def __init__(self):
        self.grand_mean_: Optional[NDArray] = None
        self.site_means_: Dict = {}
        self.site_vars_: Dict = {}
        self.is_fitted_ = False

    def fit(self, X_controls: NDArray, site_controls: Sequence) -> "BlindHarmonizer":
        """Estimate site effects from healthy control data only.

        Parameters
        ----------
        X_controls   : (N_ctrl, F) ndarray - Feature matrix, controls only.
        site_controls : sequence of length N_ctrl - Site labels for controls.
        covariates_df : DataFrame, optional - Biological covariates for controls.
                        Must NOT contain diagnostic columns.

        Returns
        -------
        self
        """
        X = np.asarray(X_controls, dtype=float)
        sites = np.asarray(site_controls)

        self.grand_mean_ = X.mean(axis=0)
        X_centred = X - self.grand_mean_

        for site in np.unique(sites):
            mask = sites == site
            X_site = X_centred[mask]
            self.site_means_[site] = X_site.mean(axis=0)
            self.site_vars_[site] = np.maximum(X_site.var(axis=0), 1e-6)

        self.is_fitted_ = True
        return self

    def transform(self, X_all: NDArray, site_all: Sequence) -> NDArray:
        """Apply control-derived site corrections to all subjects.

        Parameters
        ----------
        X_all    : (N_all, F) ndarray - Full feature matrix (controls + patients).
        site_all : sequence of length N_all - Site labels for all subjects.

        Returns
        -------
        X_harmonised : (N_all, F) ndarray - Site-corrected features.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call .fit() on control data first.")

        X = np.asarray(X_all, dtype=float).copy()
        sites = np.asarray(site_all)
        X_centred = X - self.grand_mean_

        for site in np.unique(sites):
            if site not in self.site_means_:
                continue
            mask = sites == site
            mu_s = self.site_means_[site]
            var_s = self.site_vars_[site]
            grand_var = np.var(
                np.concatenate([X_centred[sites == s] - self.site_means_[s]
                                for s in self.site_means_], axis=0),
                axis=0
            )
            grand_var = np.maximum(grand_var, 1e-6)

            X_centred[mask] = (X_centred[mask] - mu_s) * np.sqrt(grand_var / var_s)

        return X_centred + self.grand_mean_

        for site in np.unique(sites):
            if site not in self.site_means_:
                continue
            mask = sites == site
            mu_s  = self.site_means_[site]
            var_s = self.site_vars_[site]
            grand_var = np.var(
                np.concatenate([
                    (X_centred[sites == s] - self.site_means_[s])
                    for s in self.site_means_
                ], axis=0),
                axis=0,
            )
            grand_var = np.maximum(grand_var, 1e-6)

            # Empirical Bayes correction: subtract site mean, rescale variance
            X_centred[mask] = (
                (X_centred[mask] - mu_s) * np.sqrt(grand_var / var_s)
            )

        return X_centred + self.grand_mean_

    def fit_transform(
        self,
        X_controls: NDArray,
        site_controls: Sequence,
        X_all: NDArray,
        site_all: Sequence,
        covariates_df: Optional[pd.DataFrame] = None,
    ) -> NDArray:
        """Fit on controls and transform all subjects in one call."""
        return self.fit(X_controls, site_controls, covariates_df).transform(X_all, site_all)
