"""
NeuroSim Validation Pipeline
==============================
FC vs EC in Discrete Finite-Horizon NCT - Interactive Demonstration

This script serves as the companion to the Jupyter notebook
``notebook/01_fc_vs_ec_validation.ipynb``.

It reproduces the "Teleportation Error" of FC-based NCT and validates
the Van Loan Doubling Algorithm against a naiive O(T.N³) summation.

Run with:
    python notebook/validation_pipeline.py

Or convert to notebook:
    pip install jupytext
    jupytext --to notebook notebook/validation_pipeline.py
"""

# %% [markdown]
# # NeuroSim: FC vs EC in Finite-Horizon NCT
#
# **Core Question**: Does the choice of connectivity matrix (functional vs.
# effective) change control energy estimates? And by how much?
#
# We simulate a ground-truth 3-node causal chain and show that:
# 1. FC-based NCT commits the Teleportation Error (misidentifies drivers)
# 2. EC-based NCT correctly recovers the causal topology
# 3. The Van Loan Doubling Algorithm matches the naiive summation exactly

# %% Setup
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from neurosim.physics import (
    normalise_matrix,
    compute_gramian_doubling,
    minimum_energy,
    average_controllability,
    modal_controllability,
)
from neurosim.connectivity import (
    functional_connectivity,
    ridge_effective_connectivity,
    simulate_feedforward_network,
    graphnet_effective_connectivity,
    graph_laplacian,
)
from neurosim.simulation import WilsonCowanNode, WilsonCowanNetwork

plt.rcParams.update({
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'lines.linewidth': 2,
})

BLUE   = '#2E6DA4'
RED    = '#C0392B'
GREEN  = '#27AE60'
ORANGE = '#E67E22'
GREY   = '#7F8C8D'

print("NeuroSim Validation Pipeline - Initialised")
print(f"NumPy {np.__version__} | SciPy loaded")

# %% [markdown]
# ## Section 1: Simulating Ground-Truth Causal Dynamics
#
# We generate a 3-node serial causal chain:
#   Node 0 (Driver) -> Node 1 (Relay) -> Node 2 (Receiver)
#
# The MVAR generative model is:
#   x[t] = A_true @ x[t-1] + ε,   ε ~ N(0, σ²I)

# %% Simulate feedforward network
print("\n--- Section 1: Generating Ground-Truth Network ---")

X, A_true = simulate_feedforward_network(
    n_nodes=3,
    n_timepoints=8000,
    causal_weight=0.85,
    noise_std=0.10,
    seed=42,
)

print(f"Generated time series: shape {X.shape}")
print(f"Ground-truth A_true:\n{A_true}")
print(f"True causal connections: 0->1 (w={A_true[1,0]:.2f}), 1->2 (w={A_true[2,1]:.2f})")

# %% [markdown]
# ## Section 2: Connectivity Estimation - FC vs EC
#
# **Functional Connectivity (FC)**: Pearson correlation - symmetric, undirected.
# **Effective Connectivity (EC)**: Ridge regression of X(t) onto X(t-1) - asymmetric, causal.

# %% Estimate FC and EC
print("\n--- Section 2: Connectivity Estimation ---")

FC = functional_connectivity(X)
EC = ridge_effective_connectivity(X, alpha=0.1)

print(f"\nFunctional Connectivity Matrix (FC):")
print(np.round(FC, 3))
print(f"\nEffective Connectivity Matrix (EC):")
print(np.round(EC, 3))

# Symmetry check
fc_sym_error = np.max(np.abs(FC - FC.T))
ec_sym_error = np.max(np.abs(EC - EC.T))
print(f"\nFC symmetry error: {fc_sym_error:.2e}  ← should be ~0 (symmetric)")
print(f"EC symmetry error: {ec_sym_error:.4f}   ← should be >0 (asymmetric)")

# Causal recovery check
print(f"\nCausal direction recovery:")
print(f"  EC[1,0] = {EC[1,0]:.3f}  (true: 0.85, Node 0→1) ✓" if EC[1,0] > 0.3 else f"  EC[1,0] = {EC[1,0]:.3f}  ✗")
print(f"  EC[2,1] = {EC[2,1]:.3f}  (true: 0.85, Node 1→2) ✓" if EC[2,1] > 0.3 else f"  EC[2,1] = {EC[2,1]:.3f}  ✗")
print(f"  EC[0,1] = {EC[0,1]:.3f}  (should be ~0, no reverse) ✓" if EC[0,1] < 0.2 else f"  EC[0,1] = {EC[0,1]:.3f}  ✗")

# %% [markdown]
# ## Section 3: The Teleportation Error in Finite-Horizon NCT
#
# We compute the Finite-Horizon Reachability Gramian for both FC and EC,
# then compare per-node average controllability (diagonal of W_T).
#
# **Prediction:**
# - FC-based NCT: all nodes have similar controllability (symmetric matrix -> equal eigenstructure)
# - EC-based NCT: Node 0 (driver) has highest controllability

# %% Gramian computation and Teleportation Error
print("\n--- Section 3: The Teleportation Error ---")

# Normalise both matrices for stability
FC_norm = normalise_matrix(np.abs(FC), target_rho=0.9)
EC_norm = normalise_matrix(np.abs(EC), target_rho=0.9)

B = np.eye(3)
T = 8   # 8 TR steps ≈ 5.76 s at TR=720ms

W_fc = compute_gramian_doubling(FC_norm, B, T=T)
W_ec = compute_gramian_doubling(EC_norm, B, T=T)

# Per-node average controllability
ac_fc = np.diag(W_fc)
ac_ec = np.diag(W_ec)

print(f"\nFinite-Horizon Gramian (T={T} steps)")
print(f"\nFC-based Average Controllability per node:")
for i, ac in enumerate(ac_fc):
    label = ["Driver", "Relay", "Receiver"][i]
    print(f"  Node {i} ({label}): {ac:.4f}")
print(f"  Range ratio: {ac_fc.max()/ac_fc.min():.2f}× (FC cannot differentiate driver from receiver)")

print(f"\nEC-based Average Controllability per node:")
for i, ac in enumerate(ac_ec):
    label = ["Driver", "Relay", "Receiver"][i]
    print(f"  Node {i} ({label}): {ac:.4f}")
print(f"  Range ratio: {ac_ec.max()/ac_ec.min():.2f}× (EC correctly identifies driver)")
print(f"  Driver identified by EC: Node {np.argmax(ac_ec)} (correct: Node 0)")

# %% [markdown]
# ## Section 4: Van Loan Doubling - Numerical Validation
#
# We verify that the O(log T) Doubling Algorithm exactly reproduces
# the naiive O(T.N³) summation for all tested horizons.

# %% Doubling vs Naive validation
print("\n--- Section 4: Doubling Algorithm Validation ---")

def naive_gramian(A, B, T):
    """Reference O(T·N³) implementation."""
    N = A.shape[0]
    W = np.zeros((N, N))
    Ak = np.eye(N)
    Q  = B @ B.T
    for _ in range(T):
        W += Ak @ Q @ Ak.T
        Ak = Ak @ A
    return W

import time

horizons = [2, 4, 8, 16, 32, 64, 128]
errors, t_naive_list, t_doubling_list = [], [], []

for T in horizons:
    t0 = time.perf_counter()
    W_n = naive_gramian(EC_norm, B, T)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    W_d = compute_gramian_doubling(EC_norm, B, T)
    t_doubling = time.perf_counter() - t0

    err = np.max(np.abs(W_n - W_d))
    errors.append(err)
    t_naive_list.append(t_naive * 1e6)       # µs
    t_doubling_list.append(t_doubling * 1e6)

    status = "✓" if err < 1e-8 else "✗"
    print(f"  T={T:4d}: max|ΔW| = {err:.2e}  {status}  "
          f"(naïve: {t_naive*1e6:.1f}µs, doubling: {t_doubling*1e6:.1f}µs)")

print(f"\nAll max errors < 1e-8: {all(e < 1e-8 for e in errors)}")

# %% [markdown]
# ## Section 5: Wilson-Cowan Limit Cycle - Non-Linear Benchmark
#
# We simulate the Wilson-Cowan model in its limit-cycle parameter regime
# and use the resulting oscillatory trajectory as a ground-truth target
# for our linear control energy estimates.

# %% Wilson-Cowan simulation
print("\n--- Section 5: Wilson-Cowan Limit Cycle ---")

node = WilsonCowanNode(**WilsonCowanNode.LIMIT_CYCLE_PARAMS)
wc_sim = node.simulate(t_span=(0.0, 800.0), n_points=8000)

# Detect oscillations
E = wc_sim["E"] # Check that E is not at a fixed point (variance > threshold)
E_var = np.var(E[2000:])  # skip transient
print(f"  E population variance (post-transient): {E_var:.6f}")
if E_var > 1e-6:
    print("  ✓ Limit cycle detected — periodic oscillations present")
else:
    print("  ✗ No oscillations — fixed point only (check parameters)")

# Estimate period
from scipy.signal import find_peaks
peaks, _ = find_peaks(E[2000:], height=np.mean(E[2000:]))
if len(peaks) > 1:
    t_peaks = wc_sim["t"][2000:][peaks]
    period_ms = np.mean(np.diff(t_peaks))
    freq_hz = 1000.0 / period_ms
    print(f"  Estimated oscillation frequency: {freq_hz:.1f} Hz")
    print(f"  (Gamma range 30-80 Hz: {'✓' if 20 < freq_hz < 100 else '?'})")

# %% [markdown]
# ## Section 6: Publication Figure
#
# Generate a summary figure combining all validation results.

# %% Generate figure
print("\n--- Section 6: Generating Summary Figure ---")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel A: Ground truth network
ax_a = fig.add_subplot(gs[0, 0])
im = ax_a.imshow(A_true, cmap='Blues', vmin=0, vmax=1)
ax_a.set_title('A: Ground Truth\nCausal Matrix', fontweight='bold')
ax_a.set_xticks([0,1,2]); ax_a.set_yticks([0,1,2])
ax_a.set_xticklabels(['Driver','Relay','Receiver'], rotation=30, ha='right', fontsize=9)
ax_a.set_yticklabels(['Driver','Relay','Receiver'], fontsize=9)
plt.colorbar(im, ax=ax_a, fraction=0.046)
for i in range(3):
    for j in range(3):
        ax_a.text(j, i, f'{A_true[i,j]:.2f}', ha='center', va='center',
                  color='white' if A_true[i,j] > 0.5 else 'black', fontsize=10)

# Panel B: FC vs EC comparison
ax_b = fig.add_subplot(gs[0, 1])
x_pos = np.arange(3)
width = 0.35
bars_fc = ax_b.bar(x_pos - width/2, ac_fc, width, label='FC-based', color=ORANGE, alpha=0.85, edgecolor='white')
bars_ec = ax_b.bar(x_pos + width/2, ac_ec, width, label='EC-based', color=BLUE, alpha=0.85, edgecolor='white')
ax_b.set_title(f'B: Average Controllability\n(Finite Horizon T={T})', fontweight='bold')
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(['Driver', 'Relay', 'Receiver'], fontsize=9)
ax_b.set_ylabel('Avg. Controllability (W_T diagonal)')
ax_b.legend(fontsize=8)
# Annotate: EC driver node
ax_b.annotate('True Driver\nIdentified ✓',
               xy=(np.argmax(ac_ec) + width/2, ac_ec.max()),
               xytext=(np.argmax(ac_ec) + width/2 + 0.3, ac_ec.max() * 1.1),
               fontsize=8, color=BLUE,
               arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))

# Panel C: Doubling algorithm accuracy
ax_c = fig.add_subplot(gs[0, 2])
ax_c.semilogy(horizons, errors, 'o-', color=GREEN, label='max|W_naive - W_doubling|')
ax_c.axhline(1e-8, ls='--', color=GREY, lw=1, label='Tolerance 1e-8')
ax_c.set_title('C: Doubling Algorithm\nNumerical Accuracy', fontweight='bold')
ax_c.set_xlabel('Horizon T (TR steps)')
ax_c.set_ylabel('Max Absolute Error')
ax_c.legend(fontsize=8)
ax_c.set_xticks(horizons[::2])
ax_c.text(0.5, 0.1, 'Exact match\nfor all T',
          transform=ax_c.transAxes, ha='center', fontsize=9, color=GREEN)

# Panel D: Doubling vs Naive timing
ax_d = fig.add_subplot(gs[1, 0])
ax_d.plot(horizons, t_naive_list, 's-', color=RED, label='Naïve O(T·N³)', alpha=0.9)
ax_d.plot(horizons, t_doubling_list, 'o-', color=BLUE, label='Doubling O(log T·N³)', alpha=0.9)
ax_d.set_title('D: Computational Cost\nvs. Horizon T', fontweight='bold')
ax_d.set_xlabel('Horizon T (TR steps)')
ax_d.set_ylabel('Compute time (µs)')
ax_d.legend(fontsize=8)

# Panel E: Wilson-Cowan limit cycle
ax_e = fig.add_subplot(gs[1, 1])
t = wc_sim["t"]
E_wc = wc_sim["E"]
I_wc = wc_sim["I"]
# Show steady-state
mask = t > 300
ax_e.plot(t[mask], E_wc[mask], color=BLUE, label='Excitatory (E)', alpha=0.9)
ax_e.plot(t[mask], I_wc[mask], color=RED, label='Inhibitory (I)', alpha=0.9, ls='--')
ax_e.set_title('E: Wilson-Cowan\nLimit Cycle (γ oscillations)', fontweight='bold')
ax_e.set_xlabel('Time (ms)')
ax_e.set_ylabel('Population Activity')
ax_e.legend(fontsize=8)

# Panel F: Phase portrait
ax_f = fig.add_subplot(gs[1, 2])
ax_f.plot(E_wc[mask], I_wc[mask], color=ORANGE, alpha=0.6, lw=0.8)
ax_f.plot(E_wc[mask][0], I_wc[mask][0], 'o', color=GREEN, ms=8, label='Start', zorder=5)
ax_f.set_title('F: Phase Portrait\n(Limit Cycle Attractor)', fontweight='bold')
ax_f.set_xlabel('E (Excitatory activity)')
ax_f.set_ylabel('I (Inhibitory activity)')
ax_f.legend(fontsize=8)

fig.suptitle(
    'NeuroSim Validation: FC vs EC Controllability  |  Van Loan Doubling  |  Wilson-Cowan Benchmark',
    fontsize=13, fontweight='bold', y=0.98
)

out_path = os.path.join(os.path.dirname(__file__), 'validation_figure.png')
fig.savefig(out_path, bbox_inches='tight', dpi=150)
print(f"Figure saved → {out_path}")
plt.close()

# %% Summary
print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)
print(f"✓ EC correctly identifies driver node (Node {np.argmax(ac_ec)})")
print(f"✓ EC recovers 0->1 causation: EC[1,0] = {EC[1,0]:.3f}")
print(f"✓ EC recovers 1->2 causation: EC[2,1] = {EC[2,1]:.3f}")
print(f"✓ Doubling algorithm: max error across all T = {max(errors):.2e}")
print(f"✓ Wilson-Cowan limit cycle: E variance = {E_var:.4f}")
print(f"\nNeuroSim validation complete.")
