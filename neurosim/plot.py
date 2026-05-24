"""
neurosim.plot
=============
Publication-quality visualisation for finite-horizon Network Control Theory.

Five visualisation families:

1. **Brain state space** — UMAP/PCA embedding of BOLD time series into
   2D/3D manifolds. Colours by control energy, clinical group, or brain state.

2. **Energy landscape** — control energy as a function of source and target
   state, rendered as heatmaps, surface plots, or attractor basin diagrams.

3. **Controllability maps** — per-node average and modal controllability
   rendered on glass brain or matrix layouts.

4. **Transition diagrams** — optimal control trajectories through state space,
   with energy annotated along the path.

5. **Benchmark figures** — LTI vs Wilson-Cowan comparison panels, NLCF
   maps, regime analysis. These are the paper figures.

Design principles
-----------------
- Every function returns a ``(fig, axes)`` tuple so callers can customise.
- UMAP is an optional dependency (falls back to PCA gracefully).
- All functions accept both numpy arrays and ``from_arrays()`` dicts.
- Default aesthetics are publication-ready (Nature Neuroscience style).

References
----------
McInnes, L. et al. (2018). UMAP: Uniform manifold approximation and
    projection for dimension reduction. arXiv:1802.03426.
Gu, S. et al. (2015). Controllability of structural brain networks.
    Nature Communications, 6, 8414.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from numpy.typing import NDArray

# Optional heavy deps

def _get_reducer(method: str = "umap", n_components: int = 2, **kwargs):
    """Return a fitted-able dimensionality reducer.

    Tries UMAP first; falls back to PCA with a warning.

    Parameters
    ----------
    method      : ``"umap"`` or ``"pca"``
    n_components: Output dimensionality (2 or 3)
    **kwargs    : Passed to UMAP constructor (e.g. n_neighbors, min_dist)
    """
    if method == "umap":
        try:
            from umap import UMAP
            return UMAP(n_components=n_components, random_state=42, **kwargs)
        except ImportError:
            warnings.warn(
                "umap-learn not installed. Falling back to PCA. "
                "Install with: pip install umap-learn",
                UserWarning, stacklevel=3,
            )
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components)


# Palette

PALETTE = {
    "blue":   "#2E6DA4",
    "red":    "#C0392B",
    "green":  "#27AE60",
    "orange": "#E67E22",
    "purple": "#7D3C98",
    "teal":   "#0F6E56",
    "grey":   "#7F8C8D",
    "yellow": "#F1C40F",
}

_NATURE_STYLE = {
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "lines.linewidth":   2,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
}


def set_style():
    """Apply NeuroSim publication style to all subsequent plots."""
    plt.rcParams.update(_NATURE_STYLE)


# 1. Brain State Space 

def plot_state_space(
    X: NDArray,
    color_by: Optional[NDArray] = None,
    labels: Optional[Sequence[str]] = None,
    method: str = "umap",
    n_components: int = 2,
    title: str = "Brain State Space",
    cmap: str = "viridis",
    color_label: str = "Value",
    alpha: float = 0.6,
    point_size: int = 8,
    highlight_states: Optional[Dict[str, NDArray]] = None,
    ax: Optional[plt.Axes] = None,
    reducer_kwargs: Optional[dict] = None,
    figsize: Tuple[float, float] = (7, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Embed BOLD time series into 2D brain state space.

    Parameters
    ----------
    X           : (N_regions, T_timepoints) — BOLD time series.
    color_by    : (T_timepoints,) — scalar to colour each timepoint by.
                  If None, colours by time index.
    labels      : (T_timepoints,) — discrete group labels (overrides color_by).
    method      : ``"umap"`` or ``"pca"``.
    n_components: 2 (default) for 2D embedding.
    title       : Plot title.
    cmap        : Matplotlib colormap name for continuous colour_by.
    color_label : Colorbar label.
    alpha       : Point transparency.
    point_size  : Scatter point size.
    highlight_states : dict mapping label → (N_regions,) state vector to
                       overlay as named markers on the embedding.
    ax          : Existing axes to plot into. If None, creates new figure.
    reducer_kwargs : Extra keyword arguments for UMAP.
    figsize     : Figure size.

    Returns
    -------
    fig, ax
    """
    set_style()
    X_T = X.T   # (T, N) for sklearn convention

    reducer = _get_reducer(method, n_components, **(reducer_kwargs or {}))
    embedding = reducer.fit_transform(X_T)   # (T, 2)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Discrete labels
    if labels is not None:
        unique_labels = list(dict.fromkeys(labels))
        palette = list(PALETTE.values())
        for i, lbl in enumerate(unique_labels):
            mask = np.array(labels) == lbl
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                color=palette[i % len(palette)],
                label=str(lbl), alpha=alpha, s=point_size**2, zorder=3,
            )
        ax.legend(fontsize=9, framealpha=0.7)

    # Continuous colour
    else:
        c = color_by if color_by is not None else np.arange(len(embedding))
        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=c, cmap=cmap, alpha=alpha, s=point_size**2, zorder=3,
        )
        plt.colorbar(sc, ax=ax, label=color_label, fraction=0.046)

    # Named state overlays
    if highlight_states is not None:
        state_embeddings = reducer.transform(
            np.array(list(highlight_states.values()))
        )
        markers = ["*", "D", "^", "s", "P"]
        for idx, (name, _) in enumerate(highlight_states.items()):
            ax.scatter(
                *state_embeddings[idx],
                marker=markers[idx % len(markers)],
                s=200, color="black", zorder=6,
                label=name, edgecolors="white", linewidths=1.5,
            )
        ax.legend(fontsize=9, framealpha=0.8)

    dim_label = "UMAP" if method == "umap" else "PC"
    ax.set_xlabel(f"{dim_label} 1")
    ax.set_ylabel(f"{dim_label} 2")
    ax.set_title(title)

    return fig, ax


def plot_state_trajectory(
    X: NDArray,
    x0: NDArray,
    xT: NDArray,
    u_opt: Optional[NDArray] = None,
    method: str = "umap",
    title: str = "Optimal Control Trajectory",
    reducer_kwargs: Optional[dict] = None,
    figsize: Tuple[float, float] = (7, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the optimal control trajectory from x0 to xT in state space.

    Embeds the full BOLD time series and overlays the start state,
    target state, and optimal first-step control direction.

    Parameters
    ----------
    X      : (N, T) BOLD time series (background embedding).
    x0     : (N,) Initial brain state.
    xT     : (N,) Target brain state.
    u_opt  : (N,) or (M,) Optimal first-step control input. If provided,
             the control direction is shown as an arrow.
    method : ``"umap"`` or ``"pca"``.

    Returns
    -------
    fig, ax
    """
    set_style()
    X_T = X.T
    reducer = _get_reducer(method, 2, **(reducer_kwargs or {}))
    embedding = reducer.fit_transform(X_T)

    # Embed special states
    states_to_embed = np.vstack([x0[None, :], xT[None, :]])
    special = reducer.transform(states_to_embed)

    fig, ax = plt.subplots(figsize=figsize)

    # Background: state distribution
    ax.scatter(embedding[:, 0], embedding[:, 1],
               c=np.arange(len(embedding)), cmap="Greys",
               alpha=0.3, s=4**2, zorder=1)

    # Start and target
    ax.scatter(*special[0], marker="o", s=16**2,
               color=PALETTE["blue"], zorder=5, label="x₀ (initial)")
    ax.scatter(*special[1], marker="*", s=18**2,
               color=PALETTE["green"], zorder=5, label="xT (target)")

    # Arrow: x0 → xT
    ax.annotate(
        "", xy=special[1], xytext=special[0],
        arrowprops=dict(
            arrowstyle="->", color=PALETTE["blue"],
            lw=2.0, connectionstyle="arc3,rad=0.2"
        ),
        zorder=4,
    )

    # Control direction overlay
    if u_opt is not None:
        # Embed x0 + scaled control perturbation
        scale = np.linalg.norm(xT - x0) * 0.3
        u_norm = u_opt / (np.linalg.norm(u_opt) + 1e-8) * scale
        if len(u_norm) == len(x0):
            x_perturbed = x0 + u_norm
            pt = reducer.transform(x_perturbed[None, :])[0]
            ax.annotate(
                "", xy=pt, xytext=special[0],
                arrowprops=dict(
                    arrowstyle="->", color=PALETTE["orange"],
                    lw=2.0, linestyle="dashed"
                ),
                zorder=4,
            )
            ax.scatter(*pt, marker="^", s=10**2,
                       color=PALETTE["orange"], zorder=5,
                       label="u*(0) direction")

    dim_label = "UMAP" if method == "umap" else "PC"
    ax.set_xlabel(f"{dim_label} 1")
    ax.set_ylabel(f"{dim_label} 2")
    ax.set_title(title)
    ax.legend(fontsize=9, framealpha=0.8)

    return fig, ax


# 2. Energy Landscape 

def plot_energy_matrix(
    A: NDArray,
    B: NDArray,
    states: NDArray,
    T: int,
    state_labels: Optional[List[str]] = None,
    title: str = "Control Energy Matrix",
    cmap: str = "YlOrRd",
    log_scale: bool = True,
    figsize: Tuple[float, float] = (7, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Compute and plot the N×N matrix of pairwise control energies.

    For each pair of states (i, j), computes the minimum energy E*(xᵢ → xⱼ).
    The resulting matrix reveals the energetic topology of the state space.

    Parameters
    ----------
    A       : (N_regions, N_regions) — normalised connectivity matrix.
    B       : (N_regions, M) — input matrix.
    states  : (N_regions, K) — K brain states as columns.
    T       : Finite horizon.
    state_labels : K labels for states.
    log_scale    : Use log₁₀ colour scale.

    Returns
    -------
    fig, ax
    """
    from neurosim.physics import minimum_energy, compute_gramian_doubling

    set_style()
    K = states.shape[1]
    E_mat = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            if i == j:
                E_mat[i, j] = 0.0
            else:
                e, _ = minimum_energy(A, B, states[:, i], states[:, j], T=T)
                E_mat[i, j] = max(e, 1e-12)   # guard against zero/negative

    if log_scale:
        display = np.log10(E_mat + 1e-12)
        cbar_label = "log₁₀(E*)"
    else:
        display = E_mat
        cbar_label = "E* (control energy)"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(display, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046)

    labels = state_labels or [f"State {i}" for i in range(K)]
    ax.set_xticks(range(K)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(labels)
    ax.set_xlabel("Target state")
    ax.set_ylabel("Initial state")
    ax.set_title(f"{title}\n(T={T} TRs)")

    # Annotate cells
    for i in range(K):
        for j in range(K):
            val = E_mat[i, j]
            txt = "0" if i == j else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8,
                    color="white" if display[i, j] > display.mean() else "black")

    return fig, ax


def plot_energy_landscape_1d(
    A: NDArray,
    B: NDArray,
    x0: NDArray,
    xT: NDArray,
    T_range: Optional[List[int]] = None,
    compare_infinite: bool = True,
    title: str = "Control Energy vs Horizon",
    figsize: Tuple[float, float] = (9, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot minimum control energy as a function of finite horizon T.

    Optionally compares against the infinite-horizon (Lyapunov) approximation
    to demonstrate the vanishing cost problem.

    Parameters
    ----------
    A               : Normalised connectivity matrix.
    B               : Input matrix.
    x0              : Initial state.
    xT              : Target state.
    T_range         : List of T values to sweep. Default: 1–30.
    compare_infinite: Overlay the infinite-horizon energy.

    Returns
    -------
    fig, ax
    """
    from neurosim.physics import minimum_energy
    from scipy.linalg import solve_discrete_lyapunov

    set_style()
    if T_range is None:
        T_range = list(range(1, 31))

    energies = [minimum_energy(A, B, x0, xT, T=T)[0] for T in T_range]

    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(T_range, energies, color=PALETTE["blue"], lw=2.5,
                marker="o", ms=5, label="Finite-horizon E*(T)")

    if compare_infinite:
        W_inf = solve_discrete_lyapunov(A, B @ B.T)
        W_inf_inv = np.linalg.pinv(W_inf)
        delta_inf = xT - np.linalg.matrix_power(A, max(T_range)) @ x0
        e_inf = float(delta_inf @ W_inf_inv @ delta_inf)
        ax.axhline(e_inf, color=PALETTE["red"], ls="--", lw=2,
                   label=f"Infinite-horizon ≈ {e_inf:.2e}")
        ax.annotate(
            "Vanishing cost\n(biologically indefensible)",
            xy=(T_range[-1], e_inf),
            xytext=(T_range[len(T_range)//2], energies[0] * 0.1),
            fontsize=9, color=PALETTE["red"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["red"]),
        )

    # Annotate cognitive window
    for t_label, t_val, col in [("1s", 1, PALETTE["grey"]),
                                  ("5s", 7, PALETTE["orange"]),
                                  ("10s", 14, PALETTE["red"])]:
        if t_val in T_range:
            ax.axvline(t_val, color=col, ls=":", lw=1.5, alpha=0.7)
            ax.text(t_val + 0.3, energies[T_range.index(t_val)] * 1.2,
                    t_label, color=col, fontsize=9)

    ax.set_xlabel("Horizon T (TR steps)")
    ax.set_ylabel("Minimum Control Energy (log scale)")
    ax.set_title(title)
    ax.legend(fontsize=9)

    return fig, ax


# 3. Controllability Maps 

def plot_controllability(
    A: NDArray,
    node_labels: Optional[List[str]] = None,
    network_labels: Optional[NDArray] = None,
    title: str = "Controllability Landscape",
    figsize: Tuple[float, float] = (14, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot average and modal controllability side-by-side.

    Parameters
    ----------
    A              : (N, N) normalised connectivity matrix.
    node_labels    : N region labels for x-axis ticks.
    network_labels : (N,) integer array assigning each region to a network.
                     If provided, regions are coloured by network.

    Returns
    -------
    fig, (ax_ac, ax_mc, ax_scatter)
    """
    from neurosim.physics import average_controllability, modal_controllability

    set_style()
    ac = average_controllability(A)
    mc = modal_controllability(A)
    N  = A.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Colour by network if provided
    if network_labels is not None:
        unique_nets = np.unique(network_labels)
        palette = list(PALETTE.values())
        colors = [palette[int(network_labels[i]) % len(palette)] for i in range(N)]
    else:
        colors = PALETTE["blue"]

    # Average controllability
    axes[0].bar(range(N), ac, color=colors, alpha=0.85, edgecolor="white")
    axes[0].axhline(ac.mean(), color=PALETTE["red"], ls="--", lw=1.5,
                    label=f"Mean = {ac.mean():.3f}")
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("Average controllability")
    axes[0].set_title("Average Controllability\n(W∞ diagonal)")
    axes[0].legend(fontsize=9)
    if node_labels and len(node_labels) <= 30:
        axes[0].set_xticks(range(N))
        axes[0].set_xticklabels(node_labels, rotation=90, fontsize=7)

    # Modal controllability
    axes[1].bar(range(N), mc, color=colors, alpha=0.85, edgecolor="white")
    axes[1].axhline(mc.mean(), color=PALETTE["red"], ls="--", lw=1.5,
                    label=f"Mean = {mc.mean():.3f}")
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Modal controllability")
    axes[1].set_title("Modal Controllability\n(hard-to-reach state access)")
    axes[1].legend(fontsize=9)
    if node_labels and len(node_labels) <= 30:
        axes[1].set_xticks(range(N))
        axes[1].set_xticklabels(node_labels, rotation=90, fontsize=7)

    # AC vs MC scatter
    sc = axes[2].scatter(ac, mc,
                          c=range(N) if network_labels is None else network_labels,
                          cmap="tab10" if network_labels is not None else "viridis",
                          s=8**2, alpha=0.8, zorder=3)
    axes[2].set_xlabel("Average controllability")
    axes[2].set_ylabel("Modal controllability")
    axes[2].set_title("Average vs Modal\n(network trade-off)")
    corr = np.corrcoef(ac, mc)[0, 1]
    axes[2].text(0.05, 0.92, f"r = {corr:.3f}",
                 transform=axes[2].transAxes, fontsize=10,
                 color=PALETTE["grey"])

    # Label hub and peripheral nodes
    hub_idx = int(np.argmax(ac))
    per_idx = int(np.argmax(mc))
    lbl_ac = node_labels[hub_idx] if node_labels else f"Node {hub_idx}"
    lbl_mc = node_labels[per_idx] if node_labels else f"Node {per_idx}"
    axes[2].annotate(f"Hub\n{lbl_ac}", xy=(ac[hub_idx], mc[hub_idx]),
                     xytext=(ac[hub_idx] + ac.std()*0.3, mc[hub_idx]),
                     fontsize=8, color=PALETTE["blue"],
                     arrowprops=dict(arrowstyle="->", color=PALETTE["blue"]))
    axes[2].annotate(f"Peripheral\n{lbl_mc}", xy=(ac[per_idx], mc[per_idx]),
                     xytext=(ac[per_idx] - ac.std()*1.2, mc[per_idx] + mc.std()*0.5),
                     fontsize=8, color=PALETTE["purple"],
                     arrowprops=dict(arrowstyle="->", color=PALETTE["purple"]))

    plt.suptitle(title, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig, axes


def plot_stimulation_targets(
    energies_per_node: NDArray,
    node_labels: Optional[List[str]] = None,
    T: int = 10,
    top_k: int = 10,
    title: str = "Stimulation Target Ranking",
    figsize: Tuple[float, float] = (12, 4.5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot per-node control energy (stimulation target ranking).

    Parameters
    ----------
    energies_per_node : (N,) array of single-node E* values.
    node_labels       : N region labels.
    T                 : Horizon used to compute energies (for title).
    top_k             : Number of top targets to show in ranking panel.

    Returns
    -------
    fig, (ax_all, ax_ranked)
    """
    set_style()
    N = len(energies_per_node)
    finite_mask = np.isfinite(energies_per_node) & (energies_per_node < 1e10)
    valid_idx   = np.where(finite_mask)[0]
    best_node   = int(np.where(finite_mask, energies_per_node, np.inf).argmin())
    sorted_idx  = valid_idx[np.argsort(energies_per_node[valid_idx])]

    labels = node_labels or [f"N{i}" for i in range(N)]
    display_e = np.where(finite_mask, energies_per_node,
                          energies_per_node[finite_mask].max() * 1.5)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # All nodes
    colors = [PALETTE["green"] if i == best_node
              else PALETTE["red"] if not finite_mask[i]
              else PALETTE["blue"]
              for i in range(N)]
    axes[0].bar(range(N), display_e, color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Region (stimulation site)")
    axes[0].set_ylabel(f"E* (T={T} TRs)")
    axes[0].set_title("Single-Node Stimulation Energy\n(lower = better target)")
    axes[0].annotate(
        f"Best\n{labels[best_node]}",
        xy=(best_node, display_e[best_node]),
        xytext=(best_node + max(1, N//8), display_e[best_node] * 1.3),
        fontsize=9, color=PALETTE["green"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["green"]),
    )
    if len(labels) <= 30:
        axes[0].set_xticks(range(N))
        axes[0].set_xticklabels(labels, rotation=90, fontsize=7)

    # Ranked
    top_k = min(top_k, len(sorted_idx))
    axes[1].barh(range(top_k),
                 energies_per_node[sorted_idx[:top_k]],
                 color=PALETTE["blue"], alpha=0.85, edgecolor="white")
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels([labels[i] for i in sorted_idx[:top_k]], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(f"E* (T={T} TRs)")
    axes[1].set_title(f"Top {top_k} Stimulation Targets\n(ranked by minimum energy)")
    axes[1].axvline(energies_per_node[finite_mask].mean(),
                    color=PALETTE["red"], ls="--", lw=1.5,
                    label=f"Mean = {energies_per_node[finite_mask].mean():.3f}")
    axes[1].legend(fontsize=9)

    plt.suptitle(title, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig, axes


# 4. Cohort Comparison 

def plot_cohort_energy(
    energies: Dict[str, NDArray],
    T: int = 10,
    transition_name: str = "Rest → Task",
    title: str = "Control Energy by Group",
    figsize: Tuple[float, float] = (8, 5),
    test: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot control energy distributions across clinical groups.

    Parameters
    ----------
    energies : dict mapping group name → (N_subjects,) array of E* values.
    T        : Horizon (for title annotation).
    transition_name : Transition label.
    test     : If True, run Mann-Whitney U test between first two groups.

    Returns
    -------
    fig, ax
    """
    set_style()
    fig, ax = plt.subplots(figsize=figsize)

    palette = list(PALETTE.values())
    group_names = list(energies.keys())
    positions = np.arange(len(group_names))

    for i, (name, vals) in enumerate(energies.items()):
        vals = np.asarray(vals)
        finite = vals[np.isfinite(vals)]

        # Violin
        parts = ax.violinplot([finite], positions=[i], widths=0.6,
                               showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(palette[i % len(palette)])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        # Jitter overlay
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(finite))
        ax.scatter(i + jitter, finite, alpha=0.5, s=4**2,
                   color=palette[i % len(palette)], zorder=3)

        ax.text(i, np.nanmax(finite) * 1.05,
                f"n={len(finite)}\n{np.nanmedian(finite):.3f}",
                ha="center", va="bottom", fontsize=8, color="black")

    # Statistical test between first two groups
    if test and len(group_names) >= 2:
        try:
            from scipy.stats import mannwhitneyu
            g1 = np.asarray(list(energies.values())[0])
            g2 = np.asarray(list(energies.values())[1])
            g1 = g1[np.isfinite(g1)]; g2 = g2[np.isfinite(g2)]
            stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            y_top = max(g1.max(), g2.max()) * 1.12
            ax.plot([0, 1], [y_top, y_top], color="black", lw=1.5)
            ax.text(0.5, y_top * 1.02, f"{sig}\np={p:.3f}",
                    ha="center", va="bottom", fontsize=9)
        except ImportError:
            pass

    ax.set_xticks(positions)
    ax.set_xticklabels(group_names, fontsize=10)
    ax.set_ylabel(f"Minimum Control Energy E*\n(T={T} TRs, {transition_name})")
    ax.set_title(title)

    return fig, ax


# 5. Summary Figure

def plot_neurosim_summary(
    A: NDArray,
    B: NDArray,
    X: NDArray,
    x0: NDArray,
    xT: NDArray,
    T: int = 10,
    method: str = "pca",
    title: str = "NeuroSim Analysis Summary",
    figsize: Tuple[float, float] = (18, 10),
) -> Tuple[plt.Figure, plt.Axes]:
    """Six-panel publication summary figure.

    Panels:
    A — Connectivity matrix (SC or EC)
    B — Controllability landscape (AC + MC)
    C — Brain state space embedding
    D — Energy vs horizon
    E — Optimal control input
    F — Per-node stimulation ranking

    Parameters
    ----------
    A      : Normalised connectivity matrix.
    B      : Input matrix (full rank or single-node).
    X      : BOLD time series (N, T).
    x0, xT : Initial and target states.
    T      : Primary finite horizon.

    Returns
    -------
    fig, axes (2×3 grid)
    """
    from neurosim.physics import (
        average_controllability, modal_controllability, minimum_energy
    )

    set_style()
    N  = A.shape[0]
    B_ = np.eye(N)   # always use full rank for summary

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A: Connectivity 
    ax_a = fig.add_subplot(gs[0, 0])
    im   = ax_a.imshow(A, cmap="RdBu_r", aspect="auto",
                        vmin=-np.abs(A).max(), vmax=np.abs(A).max())
    ax_a.set_title("A: Connectivity Matrix\n(normalised EC)")
    ax_a.set_xlabel("Region (source)")
    ax_a.set_ylabel("Region (target)")
    plt.colorbar(im, ax=ax_a, fraction=0.046, label="Weight")

    # Panel B: Controllability 
    ax_b = fig.add_subplot(gs[0, 1])
    ac   = average_controllability(A)
    mc   = modal_controllability(A)
    ax_b.scatter(ac, mc, c=range(N), cmap="viridis", s=7**2, alpha=0.8)
    ax_b.set_xlabel("Average controllability")
    ax_b.set_ylabel("Modal controllability")
    ax_b.set_title(f"B: Controllability\nHub-Periphery Trade-off")
    r_ac_mc = np.corrcoef(ac, mc)[0, 1]
    ax_b.text(0.05, 0.92, f"r = {r_ac_mc:.3f}",
              transform=ax_b.transAxes, fontsize=10, color=PALETTE["grey"])
    # Mark extremes
    ax_b.scatter(ac[np.argmax(ac)], mc[np.argmax(ac)], marker="*",
                 s=14**2, color=PALETTE["blue"], zorder=5, label="Max AC (hub)")
    ax_b.scatter(ac[np.argmax(mc)], mc[np.argmax(mc)], marker="D",
                 s=10**2, color=PALETTE["purple"], zorder=5, label="Max MC")
    ax_b.legend(fontsize=8)

    # Panel C: State space
    ax_c  = fig.add_subplot(gs[0, 2])
    X_T   = X.T
    from sklearn.decomposition import PCA
    pca   = PCA(n_components=2)
    emb   = pca.fit_transform(X_T)
    gs_   = X.mean(axis=0)
    sc_   = ax_c.scatter(emb[:, 0], emb[:, 1], c=gs_, cmap="coolwarm",
                          alpha=0.4, s=4**2)
    plt.colorbar(sc_, ax=ax_c, fraction=0.046, label="Global signal")
    # Overlay states
    s_emb = pca.transform(np.vstack([x0[None, :], xT[None, :]]))
    ax_c.scatter(*s_emb[0], marker="o", s=14**2, color=PALETTE["blue"],
                 zorder=5, label="x₀")
    ax_c.scatter(*s_emb[1], marker="*", s=16**2, color=PALETTE["green"],
                 zorder=5, label="xT")
    ax_c.annotate("", xy=s_emb[1], xytext=s_emb[0],
                  arrowprops=dict(arrowstyle="->", color=PALETTE["blue"], lw=1.5))
    ax_c.set_xlabel("PC 1"); ax_c.set_ylabel("PC 2")
    ax_c.set_title("C: Brain State Space\n(PCA, coloured by global signal)")
    ax_c.legend(fontsize=8)

    # Panel D: Energy vs horizon
    ax_d    = fig.add_subplot(gs[1, 0])
    T_range = list(range(1, 26))
    energies = [minimum_energy(A, B_, x0, xT, T=T_h)[0] for T_h in T_range]
    ax_d.semilogy(T_range, energies, color=PALETTE["blue"], lw=2.5,
                   marker="o", ms=4)
    ax_d.axvline(T, color=PALETTE["red"], ls="--", lw=1.5, label=f"T={T}")
    ax_d.set_xlabel("Horizon T (TR steps)")
    ax_d.set_ylabel("E* (log scale)")
    ax_d.set_title("D: Control Energy vs T\n(finite-horizon sweep)")
    ax_d.legend(fontsize=9)
    T_idx = T_range.index(T) if T in T_range else -1
    ax_d.annotate(f"E*={energies[T_idx]:.3f}",
                  xy=(T, energies[T_idx]),
                  xytext=(T + 2, energies[T_idx] * 3),
                  fontsize=9, color=PALETTE["red"],
                  arrowprops=dict(arrowstyle="->", color=PALETTE["red"]))

    # Panel E: Optimal control input
    ax_e    = fig.add_subplot(gs[1, 1])
    e_opt, u_opt = minimum_energy(A, B_, x0, xT, T=T)
    colors_u = [PALETTE["green"] if u > 0 else PALETTE["red"] for u in u_opt]
    ax_e.bar(range(N), u_opt, color=colors_u, alpha=0.85, edgecolor="white")
    ax_e.axhline(0, color=PALETTE["grey"], lw=0.8)
    ax_e.set_xlabel("Region")
    ax_e.set_ylabel("Optimal input u*(0)")
    ax_e.set_title(f"E: Optimal Control Input\n(T={T}, E*={e_opt:.4f})")

    # Panel F: Per-node stimulation ranking
    ax_f = fig.add_subplot(gs[1, 2])
    node_energies = []
    for i in range(N):
        B_i = np.zeros((N, 1)); B_i[i] = 1.0
        e_i, _ = minimum_energy(A, B_i, x0, xT, T=T)
        node_energies.append(e_i)
    node_energies = np.array(node_energies)
    finite_mask  = np.isfinite(node_energies) & (node_energies < 1e10)
    valid        = np.where(finite_mask)[0]
    sorted_nodes = valid[np.argsort(node_energies[valid])]
    top_k = min(8, len(sorted_nodes))
    ax_f.barh(range(top_k), node_energies[sorted_nodes[:top_k]],
              color=PALETTE["teal"], alpha=0.85, edgecolor="white")
    ax_f.set_yticks(range(top_k))
    ax_f.set_yticklabels([f"Node {i}" for i in sorted_nodes[:top_k]], fontsize=9)
    ax_f.invert_yaxis()
    ax_f.set_xlabel(f"E* (T={T})")
    ax_f.set_title(f"F: Stimulation Targets\n(Top {top_k} ranked nodes)")
    ax_f.axvline(node_energies[finite_mask].mean(), color=PALETTE["red"],
                 ls="--", lw=1.5, label="Mean")
    ax_f.legend(fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    return fig, np.array([[ax_a, ax_b, ax_c], [ax_d, ax_e, ax_f]])


# Utility

def save_figure(
    fig: plt.Figure,
    path: str,
    dpi: int = 300,
    formats: Tuple[str, ...] = ("png", "pdf"),
) -> None:
    """Save figure in multiple formats for publication.

    Parameters
    ----------
    fig     : Matplotlib figure.
    path    : Base path without extension (e.g. ``"figures/controllability"``).
    dpi     : Resolution for raster formats (default 300).
    formats : Tuple of format strings. Default: PNG + PDF.
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}.{{{','.join(formats)}}}")
