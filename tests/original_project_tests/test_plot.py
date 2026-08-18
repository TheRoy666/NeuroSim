"""
tests/test_plot.py
==================
Unit tests for neurosim.plot.

Design: all tests use synthetic data and Agg backend (no display required).
Tests verify that every public function:
  1. Returns (fig, ax) or (fig, axes) without raising
  2. Produces output with the expected type and dimensionality
  3. Handles edge cases (small N, single group, missing optional deps)

No image comparison — we test the API contract, not the pixels.
"""

import warnings

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neurosim.physics import normalise_matrix, minimum_energy
from neurosim.connectivity import ridge_effective_connectivity


# Shared fixtures

@pytest.fixture(scope="module")
def synthetic_system():
    """20-region ring connectome + BOLD + states."""
    rng = np.random.default_rng(0)
    N, T = 20, 300

    SC = np.zeros((N, N))
    for i in range(N):
        SC[i, (i+1) % N] = 1.0
        SC[(i+1) % N, i] = 1.0
    np.fill_diagonal(SC, 0)

    A_gen = normalise_matrix(SC + 0.05*rng.normal(0,1,(N,N)), 0.85)
    X = np.zeros((N, T))
    for t in range(1, T):
        X[:, t] = A_gen @ X[:, t-1] + rng.normal(0, 0.3, N)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)

    EC = ridge_effective_connectivity(X, alpha=1.0)
    A  = normalise_matrix(EC, target_rho=0.9)
    B  = np.eye(N)

    gs = X.mean(axis=0)
    x0 = X[:, gs < np.percentile(gs, 33)].mean(1)
    xT = X[:, gs > np.percentile(gs, 67)].mean(1)
    x0 /= (np.linalg.norm(x0) + 1e-8)
    xT /= (np.linalg.norm(xT) + 1e-8)

    return {"N": N, "T": T, "X": X, "A": A, "B": B,
            "SC": SC, "x0": x0, "xT": xT}


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# Imports

def test_import_all_public_functions():
    from neurosim.plot import (
        set_style, plot_state_space, plot_state_trajectory,
        plot_energy_matrix, plot_energy_landscape_1d,
        plot_controllability, plot_stimulation_targets,
        plot_cohort_energy, plot_neurosim_summary,
        save_figure, PALETTE,
    )
    assert callable(set_style)
    assert isinstance(PALETTE, dict)
    assert len(PALETTE) >= 6


def test_palette_has_required_colours():
    from neurosim.plot import PALETTE
    for key in ("blue", "red", "green", "orange", "grey"):
        assert key in PALETTE
        assert PALETTE[key].startswith("#")


# plot_state_space

class TestPlotStateSpace:

    def test_returns_fig_ax(self, synthetic_system):
        from neurosim.plot import plot_state_space
        fig, ax = plot_state_space(synthetic_system["X"], method="pca")
        assert isinstance(fig, plt.Figure)
        assert hasattr(ax, "set_xlabel")

    def test_continuous_colour(self, synthetic_system):
        from neurosim.plot import plot_state_space
        X = synthetic_system["X"]
        T = X.shape[1]
        color_by = np.arange(T, dtype=float)
        fig, ax = plot_state_space(X, color_by=color_by, method="pca")
        assert isinstance(fig, plt.Figure)

    def test_discrete_labels(self, synthetic_system):
        from neurosim.plot import plot_state_space
        X = synthetic_system["X"]
        T = X.shape[1]
        labels = ["A" if t < T//2 else "B" for t in range(T)]
        fig, ax = plot_state_space(X, labels=labels, method="pca")
        assert isinstance(fig, plt.Figure)

    def test_highlight_states(self, synthetic_system):
        from neurosim.plot import plot_state_space
        s = synthetic_system
        fig, ax = plot_state_space(
            s["X"], method="pca",
            highlight_states={"Rest": s["x0"], "Task": s["xT"]},
        )
        assert isinstance(fig, plt.Figure)

    def test_umap_falls_back_to_pca_gracefully(self, synthetic_system):
        from neurosim.plot import plot_state_space
        # UMAP may not be installed — should fall back to PCA with a warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, ax = plot_state_space(synthetic_system["X"], method="umap")
        assert isinstance(fig, plt.Figure)

    def test_existing_axes_used(self, synthetic_system):
        from neurosim.plot import plot_state_space
        _, ax_existing = plt.subplots()
        fig, ax = plot_state_space(synthetic_system["X"], method="pca",
                                   ax=ax_existing)
        assert ax is ax_existing


# plot_state_trajectory

class TestPlotStateTrajectory:

    def test_returns_fig_ax(self, synthetic_system):
        from neurosim.plot import plot_state_trajectory
        s = synthetic_system
        fig, ax = plot_state_trajectory(s["X"], s["x0"], s["xT"], method="pca")
        assert isinstance(fig, plt.Figure)

    def test_with_u_opt(self, synthetic_system):
        from neurosim.plot import plot_state_trajectory
        s = synthetic_system
        _, u_opt = minimum_energy(s["A"], s["B"], s["x0"], s["xT"], T=8)
        fig, ax = plot_state_trajectory(
            s["X"], s["x0"], s["xT"], u_opt=u_opt, method="pca"
        )
        assert isinstance(fig, plt.Figure)


# plot_energy_matrix

class TestPlotEnergyMatrix:

    def test_returns_fig_ax(self, synthetic_system):
        from neurosim.plot import plot_energy_matrix
        s = synthetic_system
        states = np.column_stack([s["x0"], s["xT"], (s["x0"]+s["xT"])/2])
        fig, ax = plot_energy_matrix(s["A"], s["B"], states, T=5)
        assert isinstance(fig, plt.Figure)

    def test_with_labels(self, synthetic_system):
        from neurosim.plot import plot_energy_matrix
        s = synthetic_system
        states = np.column_stack([s["x0"], s["xT"]])
        fig, ax = plot_energy_matrix(
            s["A"], s["B"], states, T=5,
            state_labels=["Rest", "Task"]
        )
        assert isinstance(fig, plt.Figure)

    def test_log_scale_false(self, synthetic_system):
        from neurosim.plot import plot_energy_matrix
        s = synthetic_system
        states = np.column_stack([s["x0"], s["xT"]])
        fig, ax = plot_energy_matrix(s["A"], s["B"], states, T=5, log_scale=False)
        assert isinstance(fig, plt.Figure)


# plot_energy_landscape_1d

class TestPlotEnergyLandscape1d:

    def test_returns_fig_ax(self, synthetic_system):
        from neurosim.plot import plot_energy_landscape_1d
        s = synthetic_system
        fig, ax = plot_energy_landscape_1d(
            s["A"], s["B"], s["x0"], s["xT"],
            T_range=list(range(1, 10))
        )
        assert isinstance(fig, plt.Figure)

    def test_no_infinite_comparison(self, synthetic_system):
        from neurosim.plot import plot_energy_landscape_1d
        s = synthetic_system
        fig, ax = plot_energy_landscape_1d(
            s["A"], s["B"], s["x0"], s["xT"],
            T_range=list(range(1, 8)),
            compare_infinite=False
        )
        assert isinstance(fig, plt.Figure)


# plot_controllability

class TestPlotControllability:

    def test_returns_fig_axes(self, synthetic_system):
        from neurosim.plot import plot_controllability
        fig, axes = plot_controllability(synthetic_system["A"])
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 3

    def test_with_node_labels(self, synthetic_system):
        from neurosim.plot import plot_controllability
        s = synthetic_system
        labels = [f"R{i}" for i in range(s["N"])]
        fig, axes = plot_controllability(s["A"], node_labels=labels)
        assert isinstance(fig, plt.Figure)

    def test_with_network_labels(self, synthetic_system):
        from neurosim.plot import plot_controllability
        s = synthetic_system
        net_labels = np.array([i % 4 for i in range(s["N"])])
        fig, axes = plot_controllability(s["A"], network_labels=net_labels)
        assert isinstance(fig, plt.Figure)


# plot_stimulation_targets

class TestPlotStimulationTargets:

    def test_returns_fig_axes(self, synthetic_system):
        from neurosim.plot import plot_stimulation_targets
        s = synthetic_system
        node_energies = np.array([
            minimum_energy(s["A"],
                           np.eye(s["N"])[:, [i]], s["x0"], s["xT"], T=8)[0]
            for i in range(s["N"])
        ])
        fig, axes = plot_stimulation_targets(node_energies, T=8)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2

    def test_with_labels(self, synthetic_system):
        from neurosim.plot import plot_stimulation_targets
        s = synthetic_system
        energies = np.abs(np.random.default_rng(1).normal(1, 0.5, s["N"]))
        labels   = [f"Region_{i}" for i in range(s["N"])]
        fig, axes = plot_stimulation_targets(energies, node_labels=labels, T=10)
        assert isinstance(fig, plt.Figure)

    def test_with_inf_energies(self, synthetic_system):
        from neurosim.plot import plot_stimulation_targets
        s = synthetic_system
        energies = np.abs(np.random.default_rng(2).normal(1, 0.5, s["N"]))
        energies[3] = np.inf   # unreachable node
        fig, axes = plot_stimulation_targets(energies, T=10)
        assert isinstance(fig, plt.Figure)


# plot_cohort_energy

class TestPlotCohortEnergy:

    def test_two_groups(self):
        from neurosim.plot import plot_cohort_energy
        rng = np.random.default_rng(5)
        energies = {
            "Control": rng.normal(1.5, 0.3, 15),
            "Patient": rng.normal(2.5, 0.5, 15),
        }
        fig, ax = plot_cohort_energy(energies, T=10)
        assert isinstance(fig, plt.Figure)

    def test_three_groups(self):
        from neurosim.plot import plot_cohort_energy
        rng = np.random.default_rng(6)
        energies = {
            "CN":  rng.normal(1.0, 0.2, 10),
            "MCI": rng.normal(1.5, 0.3, 10),
            "AD":  rng.normal(2.2, 0.4, 10),
        }
        fig, ax = plot_cohort_energy(energies, T=10, test=False)
        assert isinstance(fig, plt.Figure)

    def test_no_statistical_test(self):
        from neurosim.plot import plot_cohort_energy
        energies = {
            "A": np.array([1.0, 1.2, 0.9]),
            "B": np.array([2.0, 1.8, 2.1]),
        }
        fig, ax = plot_cohort_energy(energies, test=False)
        assert isinstance(fig, plt.Figure)

    def test_with_nan_values(self):
        from neurosim.plot import plot_cohort_energy
        energies = {
            "A": np.array([1.0, np.nan, 1.2, 0.9]),
            "B": np.array([2.0, 1.8, np.nan, 2.1]),
        }
        fig, ax = plot_cohort_energy(energies, test=False)
        assert isinstance(fig, plt.Figure)


# plot_neurosim_summary

class TestPlotNeuroSimSummary:

    def test_returns_fig_axes_grid(self, synthetic_system):
        from neurosim.plot import plot_neurosim_summary
        s = synthetic_system
        fig, axes = plot_neurosim_summary(s["A"], s["B"], s["X"], s["x0"], s["xT"], T=8)
        assert isinstance(fig, plt.Figure)
        assert axes.shape == (2, 3)

    def test_all_axes_populated(self, synthetic_system):
        from neurosim.plot import plot_neurosim_summary
        s = synthetic_system
        fig, axes = plot_neurosim_summary(s["A"], s["B"], s["X"], s["x0"], s["xT"], T=8)
        for row in axes:
            for ax in row:
                assert ax is not None


# save_figure

class TestSaveFigure:

    def test_saves_png(self, synthetic_system, tmp_path):
        from neurosim.plot import save_figure, plot_controllability
        s = synthetic_system
        fig, _ = plot_controllability(s["A"])
        path = str(tmp_path / "test_fig")
        save_figure(fig, path, dpi=72, formats=("png",))
        assert (tmp_path / "test_fig.png").exists()

    def test_saves_multiple_formats(self, synthetic_system, tmp_path):
        from neurosim.plot import save_figure, plot_controllability
        s = synthetic_system
        fig, _ = plot_controllability(s["A"])
        path = str(tmp_path / "test_fig")
        save_figure(fig, path, dpi=72, formats=("png", "pdf"))
        assert (tmp_path / "test_fig.png").exists()
        assert (tmp_path / "test_fig.pdf").exists()
