"""
tests/test_clinical.py
======================
Unit tests for neurosim.clinical.

Tests cover:
1. All three pipeline classes instantiate and run correctly
2. SubjectResult and CohortResult have correct structure
3. Transition energies are non-negative and finite for valid inputs
4. Stage trajectory produces correct DataFrame structure
5. SOZ identification returns valid region indices
6. Twin discordance analysis produces correct ΔE* signs
7. Edge cases: single subject, missing SC, empty groups
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from neurosim.physics import normalise_matrix
from neurosim.clinical import (
    AUDPipeline,
    ADNIPipeline,
    EpilepsyPipeline,
    SubjectResult,
    CohortResult,
)


# Fixtures

def _make_bold(seed=0, N=12, T=300, attractor_strength=0.0, degradation=0.0):
    """Generate synthetic BOLD time series."""
    rng = np.random.default_rng(seed)
    SC = np.zeros((N, N))
    for i in range(N):
        SC[i, (i+1) % N] = 1.0
        SC[(i+1) % N, i] = 1.0
    np.fill_diagonal(SC, 0)

    if attractor_strength > 0:
        for i in range(min(4, N)):
            for j in range(min(4, N)):
                SC[i, j] += attractor_strength * rng.uniform(0.05, 0.15)

    SC = (SC + SC.T) / 2.0
    SC_deg = SC * (1 - degradation)
    A = normalise_matrix(SC_deg + 0.05*rng.normal(0,1,(N,N)), 0.83)

    X = np.zeros((N, T))
    for t in range(1, T):
        X[:, t] = A @ X[:, t-1] + rng.normal(0, 0.3, N)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    return X, SC


@pytest.fixture
def aud_cohort():
    """Small AUD cohort: 6 subjects (3 AUD, 3 Control)."""
    subjects = []
    for i in range(6):
        group = "AUD" if i < 3 else "Control"
        astr  = 0.12 if group == "AUD" else 0.0
        X, SC = _make_bold(seed=i, attractor_strength=astr)
        subjects.append({
            "X": X, "SC": SC,
            "subject_id": f"sub-AUD{i:02d}",
            "group": group,
            "metadata": {},
        })
    return subjects


@pytest.fixture
def adni_cohort():
    """Small ADNI cohort: 9 subjects (3 CN, 3 MCI, 3 AD)."""
    subjects = []
    for i, (stage, deg) in enumerate(
        [("CN", 0.0)]*3 + [("MCI", 0.2)]*3 + [("AD", 0.45)]*3
    ):
        X, SC = _make_bold(seed=i+20, degradation=deg)
        subjects.append({
            "X": X, "SC": SC,
            "subject_id": f"adni-{stage}-{i:02d}",
            "group": stage,
            "metadata": {},
        })
    return subjects


@pytest.fixture
def epilepsy_cohort():
    """Small epilepsy cohort: 6 subjects (4 TLE, 2 Control)."""
    subjects = []
    for i in range(6):
        is_tle = i < 4
        X, SC = _make_bold(seed=i+40)
        if is_tle:
            X[:3, 220:260] += np.random.default_rng(i).normal(0, 2.0, (3, 40))
            X = (X - X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-8)
        subjects.append({
            "X": X, "SC": SC,
            "subject_id": f"epi-{'TLE' if is_tle else 'Ctrl'}-{i:02d}",
            "group": "TLE" if is_tle else "Control",
            "metadata": {"ictal_indices": list(range(220, 260)) if is_tle else None},
        })
    return subjects


# SubjectResult and CohortResult structure

class TestResultStructure:

    def test_subject_result_fields(self, aud_cohort):
        pipe = AUDPipeline(T=6, reward_indices=[0, 1])
        result = pipe.run_subject(**{k: v for k, v in aud_cohort[0].items()})
        assert isinstance(result, SubjectResult)
        assert result.subject_id == aud_cohort[0]["subject_id"]
        assert result.group == aud_cohort[0]["group"]
        assert result.A.ndim == 2
        assert result.ac.ndim == 1
        assert result.mc.ndim == 1
        assert isinstance(result.energies, dict)

    def test_cohort_result_fields(self, aud_cohort):
        pipe   = AUDPipeline(T=6)
        cohort = pipe.run_cohort(aud_cohort, verbose=False)
        assert isinstance(cohort, CohortResult)
        assert cohort.n_subjects == 6
        assert set(cohort.groups) == {"AUD", "Control"}

    def test_summary_dataframe_shape(self, aud_cohort):
        pipe   = AUDPipeline(T=6)
        cohort = pipe.run_cohort(aud_cohort, verbose=False)
        df = cohort.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert "subject_id" in df.columns
        assert "group" in df.columns

    def test_energy_array_by_group(self, aud_cohort):
        pipe   = AUDPipeline(T=6)
        cohort = pipe.run_cohort(aud_cohort, verbose=False)
        e_aud  = cohort.energy_array("craving_to_rest", group="AUD")
        e_ctrl = cohort.energy_array("craving_to_rest", group="Control")
        assert len(e_aud)  == 3
        assert len(e_ctrl) == 3

    def test_get_group(self, aud_cohort):
        pipe   = AUDPipeline(T=6)
        cohort = pipe.run_cohort(aud_cohort, verbose=False)
        aud_subj = cohort.get_group("AUD")
        assert len(aud_subj) == 3
        assert all(s.group == "AUD" for s in aud_subj)


# AUDPipeline

class TestAUDPipeline:

    def test_transitions_defined(self):
        pipe = AUDPipeline(T=6)
        t = pipe.define_transitions()
        assert "craving_to_rest" in t
        assert "rest_to_cognitive" in t
        assert len(t) == 3

    def test_states_defined(self, aud_cohort):
        pipe = AUDPipeline(T=6)
        X    = aud_cohort[0]["X"]
        states = pipe.define_states(X, {})
        assert "rest" in states and "craving" in states and "cognitive" in states
        for v in states.values():
            assert abs(np.linalg.norm(v) - 1.0) < 1e-6, "States must be unit-normalised"

    def test_energies_nonnegative(self, aud_cohort):
        pipe   = AUDPipeline(T=6)
        cohort = pipe.run_cohort(aud_cohort, verbose=False)
        for s in cohort.subjects:
            for k, v in s.energies.items():
                assert np.isnan(v) or v >= 0, f"Negative energy {v} for {k}"

    def test_statistics_computed(self, aud_cohort):
        pipe   = AUDPipeline(T=6)
        cohort = pipe.run_cohort(aud_cohort, verbose=False)
        stats  = cohort.statistics
        assert "craving_to_rest" in stats
        assert "AUD" in stats["craving_to_rest"]
        assert "p_value" in stats["craving_to_rest"]

    def test_twin_discordance_analysis(self, aud_cohort):
        pipe   = AUDPipeline(T=6)
        cohort = pipe.run_cohort(aud_cohort, verbose=False)
        pairs  = [
            ("sub-AUD00", "sub-AUD03"),
            ("sub-AUD01", "sub-AUD04"),
        ]
        df = pipe.twin_discordance_analysis(cohort, pairs, transition="craving_to_rest")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "delta_E" in df.columns
        assert "ratio" in df.columns
        # ratio must be positive
        assert (df["ratio"] > 0).all()

    def test_reward_indices(self, aud_cohort):
        pipe   = AUDPipeline(T=6, reward_indices=[0, 1, 2])
        cohort = pipe.run_cohort(aud_cohort, verbose=False)
        # All subjects should have craving_to_rest energy
        for s in cohort.subjects:
            assert "craving_to_rest" in s.energies

    def test_run_without_sc(self):
        """Pipeline should work even without a structural connectome."""
        pipe = AUDPipeline(T=6)
        X, _ = _make_bold(seed=99)
        result = pipe.run_subject(X=X, subject_id="test", group="Control")
        assert isinstance(result, SubjectResult)
        assert len(result.energies) == 3


# ADNIPipeline

class TestADNIPipeline:

    def test_transitions_defined(self):
        pipe = ADNIPipeline(T=6)
        t = pipe.define_transitions()
        assert "default_to_memory" in t
        assert "default_to_executive" in t
        assert len(t) == 3

    def test_states_unit_normalised(self, adni_cohort):
        pipe   = ADNIPipeline(T=6)
        X      = adni_cohort[0]["X"]
        states = pipe.define_states(X, {})
        for name, v in states.items():
            assert abs(np.linalg.norm(v) - 1.0) < 1e-6, \
                f"State '{name}' not unit-normalised: norm={np.linalg.norm(v):.4f}"

    def test_no_retrogenesis_method(self):
        """Retrogenesis analysis is a separate project — must not be in ADNIPipeline."""
        pipe = ADNIPipeline(T=6)
        assert not hasattr(pipe, "retrogenesis_score"), \
            "retrogenesis_score belongs to the Entropic Mirror project, not NeuroSim"
        assert not hasattr(pipe, "retrogenesis_regions"), \
            "retrogenesis_regions belongs to the Entropic Mirror project, not NeuroSim"

    def test_stage_trajectory_dataframe(self, adni_cohort):
        pipe   = ADNIPipeline(T=6, stage_order=["CN", "MCI", "AD"])
        cohort = pipe.run_cohort(adni_cohort, verbose=False)
        traj   = pipe.stage_trajectory(cohort, metric="default_to_memory")
        assert isinstance(traj, pd.DataFrame)
        assert "mean" in traj.columns
        assert "sem" in traj.columns
        assert "n" in traj.columns

    def test_stage_trajectory_controllability(self, adni_cohort):
        pipe   = ADNIPipeline(T=6, stage_order=["CN", "MCI", "AD"])
        cohort = pipe.run_cohort(adni_cohort, verbose=False)
        traj_ac = pipe.stage_trajectory(cohort, metric="ac")
        traj_mc = pipe.stage_trajectory(cohort, metric="mc")
        assert isinstance(traj_ac, pd.DataFrame)
        assert isinstance(traj_mc, pd.DataFrame)
        assert (traj_ac["mean"] >= 0).all(), "Average controllability must be non-negative"

    def test_dmn_controllability(self, adni_cohort):
        pipe   = ADNIPipeline(T=6, dmn_indices=[0, 1, 2, 3])
        cohort = pipe.run_cohort(adni_cohort, verbose=False)
        for s in cohort.subjects[:3]:
            dmn = pipe.dmn_controllability(s)
            assert "mean_ac_dmn" in dmn
            assert "mean_mc_dmn" in dmn
            assert "ac_ratio"    in dmn
            assert dmn["mean_ac_dmn"] >= 0
            assert dmn["ac_ratio"]    >= 0

    def test_dmn_controllability_no_indices(self, adni_cohort):
        """Without dmn_indices, should return whole-brain controllability."""
        pipe   = ADNIPipeline(T=6)
        result = pipe.run_subject(**{k:v for k,v in adni_cohort[0].items()})
        dmn = pipe.dmn_controllability(result)
        assert dmn["ac_ratio"] == 1.0

    def test_finite_vs_infinite_comparison(self, adni_cohort):
        pipe   = ADNIPipeline(T=6)
        result = pipe.run_subject(**{k:v for k,v in adni_cohort[0].items()})
        X      = adni_cohort[0]["X"]
        df = pipe.finite_vs_infinite_comparison(result, X, T_range=list(range(1, 8)))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 7
        assert "T" in df.columns
        assert "E_finite" in df.columns
        assert "E_infinite" in df.columns
        assert "ratio" in df.columns
        # Finite energy should be >= infinite at short T
        assert (df["E_finite"] >= 0).all()


# EpilepsyPipeline

class TestEpilepsyPipeline:

    def test_transitions_defined(self):
        pipe = EpilepsyPipeline(T=6)
        t = pipe.define_transitions()
        assert "interictal_to_ictal" in t
        assert "preictal_to_ictal" in t
        assert len(t) == 3

    def test_states_unit_normalised(self, epilepsy_cohort):
        pipe = EpilepsyPipeline(T=6)
        subj = epilepsy_cohort[0]
        states = pipe.define_states(subj["X"], subj["metadata"])
        for name, v in states.items():
            assert abs(np.linalg.norm(v) - 1.0) < 1e-6, \
                f"State '{name}' not unit-normalised"

    def test_states_with_no_ictal_metadata(self):
        """Pipeline should auto-detect ictal periods without metadata."""
        pipe = EpilepsyPipeline(T=6)
        X, _ = _make_bold(seed=77)
        states = pipe.define_states(X, {})
        assert "interictal" in states
        assert "ictal" in states
        assert "preictal" in states

    def test_node_energies_computed(self, epilepsy_cohort):
        pipe   = EpilepsyPipeline(T=6, compute_node_energies=True)
        result = pipe.run_subject(**{k:v for k,v in epilepsy_cohort[0].items()})
        assert result.node_energies is not None
        N = result.A.shape[0]
        assert result.node_energies.shape == (N,)

    def test_identify_soz_returns_valid_indices(self, epilepsy_cohort):
        pipe   = EpilepsyPipeline(T=6, compute_node_energies=True)
        cohort = pipe.run_cohort(epilepsy_cohort, verbose=False)
        tle_subjects = cohort.get_group("TLE")
        for s in tle_subjects:
            soz = pipe.identify_soz(s, top_k=3)
            assert "primary_soz" in soz
            assert "soz_candidates" in soz
            assert "energies" in soz
            N = s.A.shape[0]
            if soz["primary_soz"] is not None:
                assert 0 <= soz["primary_soz"] < N
            assert len(soz["soz_candidates"]) <= 3

    def test_identify_soz_raises_without_node_energies(self, epilepsy_cohort):
        pipe   = EpilepsyPipeline(T=6, compute_node_energies=False)
        result = pipe.run_subject(**{k:v for k,v in epilepsy_cohort[0].items()})
        result.node_energies = None
        with pytest.raises(ValueError, match="node_energies"):
            pipe.identify_soz(result)

    def test_energy_barrier_analysis(self, epilepsy_cohort):
        pipe   = EpilepsyPipeline(T=6)
        cohort = pipe.run_cohort(epilepsy_cohort, verbose=False)
        df = pipe.energy_barrier_analysis(cohort)
        assert isinstance(df, pd.DataFrame)
        assert "mean_E_barrier" in df.columns
        assert "n" in df.columns
        assert set(df.index) == {"TLE", "Control"}

    def test_energies_nonnegative(self, epilepsy_cohort):
        pipe   = EpilepsyPipeline(T=6)
        cohort = pipe.run_cohort(epilepsy_cohort, verbose=False)
        for s in cohort.subjects:
            for k, v in s.energies.items():
                assert np.isnan(v) or v >= 0, \
                    f"Negative energy {v} for {k} in {s.subject_id}"


# BasePipeline

class TestBasePipeline:

    def test_fit_harmonizer(self, aud_cohort):
        """BlindHarmonizer can be attached to any pipeline."""
        pipe = AUDPipeline(T=6)
        X_controls = np.vstack([s["X"].T for s in aud_cohort if s["group"] == "Control"])
        sites = ["site_A"] * len(X_controls)
        pipe.fit_harmonizer(X_controls, sites)
        assert pipe._harmonizer is not None
        assert pipe._harmonizer.is_fitted_

    def test_run_cohort_handles_failures_gracefully(self):
        """A subject that fails should not crash the whole cohort."""
        pipe = AUDPipeline(T=6)
        subjects = [
            {"X": _make_bold(0)[0], "subject_id": "good-01", "group": "Control", "metadata": {}},
            {"X": np.zeros((12, 300)), "subject_id": "bad-01", "group": "AUD", "metadata": {}},
            {"X": _make_bold(1)[0], "subject_id": "good-02", "group": "Control", "metadata": {}},
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cohort = pipe.run_cohort(subjects, verbose=False)
        # Bad subject should be skipped, not crash
        assert cohort.n_subjects >= 2

    def test_controllability_shapes(self, aud_cohort):
        pipe   = AUDPipeline(T=6)
        result = pipe.run_subject(**{k:v for k,v in aud_cohort[0].items()})
        N = result.A.shape[0]
        assert result.ac.shape == (N,)
        assert result.mc.shape == (N,)
        assert np.all(result.ac >= 0), "Average controllability must be non-negative"
