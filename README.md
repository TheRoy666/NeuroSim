# NeuroSim 

**A physics-constrained Python toolkit for finite-horizon Network Control Theory in macro-scale neuroimaging.**

> *"The brain is not a continuous-time system with infinite patience.  
>  It is a discrete, finite, energetically constrained machine -  
>  and our models should say the same."*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![GSoC 2026](https://img.shields.io/badge/GSoC-2026%20INCF-orange.svg)](https://summerofcode.withgoogle.com/)
[![iNCF](https://img.shields.io/badge/Project-iNCF%20GSoC%202026-blue)](https://neurostars.org/t/gsoc-2026-project-39-national-brain-research-centre-nbrc-ebrains-neurosim-automating-in-silico-stimulation-for-non-invasive-biomarker-discovery/35619/10/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#running-tests)
[![CI](https://github.com/TheRoy666/NeuroSim/actions/workflows/ci.yml/badge.svg)](https://github.com/TheRoy666/NeuroSim/actions/workflows/ci.yml)

---

## The Problem: An Approximation Crisis in Computational Neuroscience

The application of Network Control Theory (NCT) to neuroimaging has produced profound insights into how white-matter architecture constrains brain state dynamics. Yet current open-source tools share three methodological assumptions that, individually, are defensible; collectively, they constitute a reproducibility crisis.

### Error 1 - Temporal Approximation (Infinite Horizon)

Standard NCT pipelines compute the **Infinite-Horizon Controllability Gramian** W(∞), which satisfies the algebraic Lyapunov equation:

```
AW(∞) + W(∞)Aᵀ + BBᵀ = 0
```

This metric assumes the brain has *unlimited time* to transition between states. In a stable system (ρ(A) < 1), the energy to reach any state approaches **zero** as T -> ∞ - the 'vanishing cost' problem. A schizophrenic patient who can *theoretically* engage executive control *given infinite time* is indistinguishable from a healthy control. The infinite-horizon Gramian is blind to this clinically critical impairment.

**Biologically, cognitive switching occurs in 2–10 seconds. The time horizon T is not a free parameter - it is a physiological constant.**

### Error 2 - Structural Blindness (Binary DTI Masking)

Existing tools apply DTI tractography as a hard binary mask: if no streamline connects Region A to Region B, A[i,j] = 0. This ignores:

- **Polysynaptic pathways**: functional influence propagates A->C->B even without a direct A-B tract.
- **False negatives**: probabilistic tractography systematically underestimates crossing fibres and long-range connections.

The result: a "structurally blind" model with false-zero entries that corrupt control energy estimates.

### Error 3 - Statistical Leakage (Non-Blind Harmonisation)

Multi-site neuroimaging studies routinely use ComBat harmonisation with diagnostic group as a covariate - to "preserve" biological variance. This introduces **data leakage**: the harmonisation algorithm encodes group-level differences into the corrected features *before* the classifier sees them, inflating AUC and creating irreproducible biomarkers.

---

## NeuroSim's Solution: Three Methodological Pivots

### Pivot 1 - Discrete Finite-Horizon Physics (Van Loan Doubling)

NeuroSim models the brain as a **Discrete-Time LTI system**:

```
x[k+1] = A x[k] + B u[k]
```

and computes the **Finite-Horizon Reachability Gramian**:

```
W at time T = Σ Aᵏ BBᵀ (Aᵀ)ᵏ [Summation is done from k=0 to k=(T-1)]
```

This T-parameterised metric directly captures the energetic cost of transitioning states *within a cognitive task window*. Naïve computation costs O(T·N³). NeuroSim implements the **Van Loan Doubling Algorithm**, which reduces complexity to **O(log T · N³)** via iterative squaring:

```python
from neurosim.physics import compute_gramian_doubling, normalise_matrix

A = normalise_matrix(A_dti, target_rho=0.9)   # spectral stability
B = np.eye(N)                                  # full-rank input
W_T = compute_gramian_doubling(A, B, T=10)    # 10 TR steps ≈ 7.2 s
```

### Pivot 2 - GraphNet Regularisation (Soft Structural Prior)

NeuroSim replaces hard DTI masking with the **GraphNet** objective (Grosenick et al., 2013):

```
min_A  ‖X_{t+1} - A X_t‖²_F  +  λ₁‖A‖²_F  +  λ₂ Tr(Aᵀ L_sc A)
```

where L_sc is the Graph Laplacian of the structural connectome. This penalises functional deviations from structural topology *without hard-zeroing any connection*. Connections absent from DTI are penalised - not forbidden.

```python
from neurosim.connectivity import graphnet_effective_connectivity

EC = graphnet_effective_connectivity(
    X_bold,          # (N_regions, T_timepoints) BOLD time series
    SC_dti,          # (N_regions, N_regions) DTI connectome
    lambda_ridge=1.0,
    lambda_graph=2.0,
)
```

### Pivot 3 - Blind Harmonisation (Controls-Only ComBat)

NeuroSim enforces a **controls-only harmonisation** protocol:

```python
from neurosim.harmonize import BlindHarmonizer #Yet to be implemented, postBIDS work

harmonizer = BlindHarmonizer(biological_covariates=["age", "sex"])
harmonizer.fit(X_controls, site_controls)          
X_harmonised = harmonizer.transform(X_all, site_all)
```

Any group difference surviving blind harmonisation is a genuine biological signal, not a preprocessing artifact.

---

## Architecture

```
NeuroSim/
├── .github/
│   └── workflows/
│       └── ci.yml             # Continuous Integration pipeline (GitHub Actions)
├── neurosim/                  # Core library package
│   ├── __init__.py            # Package initialization and API exposure
│   ├── connectivity.py        # Effective/Functional connectivity solvers
│   ├── harmonize.py           # Multi-site data harmonization (Blind neuroCombat)
│   ├── physics.py             # Control engine (Van Loan Doubling, Gramians)
│   └── simulation.py          # Wilson-Cowan neural mass modeling
├── notebook/                  # Documentation and walkthroughs
│   ├── 01_fc_vs_ec_validation.ipynb
│   ├── validation_figure.png
│   └── validation_pipeline.py # Functional companion script for the notebook
├── tests/                     # Unit and integration tests
│   ├── test_connectivity.py
│   └── test_physics.py
├── .gitignore                 # Specifies intentionally untracked files to ignore
├── CONTRIBUTING.md            # Guidelines for contributing to the project
├── LICENSE                    # Project license (Apache 2.0)
├── README.md                  # Main project documentation
├── pyproject.toml             # Build system requirements and metadata
└── requirements.txt           # List of Python dependencies
```

### Module Overview

| Module | Key Contribution | Core Algorithm |
|--------|-----------------|----------------|
| `physics` | Finite-horizon control energy | Van Loan Doubling (O(log T)) |
| `connectivity` | Causal EC estimation | GraphNet FISTA solver |
| `harmonize` | Leakage-free harmonisation | Controls-only Empirical Bayes |
| `simulation` | Non-linear ground truth | Wilson-Cowan neural mass model |

---

## Quickstart

```python
import numpy as np
from neurosim.physics import normalise_matrix, compute_gramian_doubling, minimum_energy
from neurosim.connectivity import graphnet_effective_connectivity, functional_connectivity
from neurosim.simulation import WilsonCowanNode

# 1. Estimate causal EC from BOLD time series + DTI prior
EC = graphnet_effective_connectivity(X_bold, SC_dti, lambda_ridge=1.0, lambda_graph=2.0)

# 2. Ensure system stability
A = normalise_matrix(EC, target_rho=0.9)

# 3. Compute finite-horizon control energy (T = 10 TRs)
B = np.eye(N)   # all regions as control nodes
W_T = compute_gramian_doubling(A, B, T=10)

# 4. Minimum energy for a resting -> task transition
x_rest = ...   # (N,) resting-state BOLD vector
x_task = ...   # (N,) task-state BOLD vector
energy, u_opt = minimum_energy(A, B, x_rest, x_task, T=10)
print(f"Minimum control energy: {energy:.4f}")

# 5. Single-node Wilson-Cowan limit cycle (validation benchmark)
node = WilsonCowanNode(**WilsonCowanNode.LIMIT_CYCLE_PARAMS)
result = node.simulate(t_span=(0, 1000), n_points=10000)
# result["E"] -> excitatory population oscillations ~40 Hz
```

---

## The Teleportation Error: Why FC-Based NCT is Wrong

NeuroSim includes a formal demonstration of the Teleportation Error in existing NCT pipelines. Given a ground-truth feedforward network (Node 0 -> Node 1 -> Node 2):

```python
from neurosim.connectivity import simulate_feedforward_network, functional_connectivity, ridge_effective_connectivity

X, A_true = simulate_feedforward_network(n_nodes=3, n_timepoints=8000)

FC = functional_connectivity(X)   # symmetric -> cannot distinguish driver from receiver
EC = ridge_effective_connectivity(X)   # asymmetric -> correctly identifies Node 0 as driver

# EC recovers ground truth:
# EC[1,0] ≈ 0.85  (Node 0 -> 1, correct)
# EC[0,1] ≈ 0.02  (no reverse causation, correct)
```

FC-based NCT assigns similar "controllability" to all nodes in the chain, because symmetric matrices by definition cannot encode causal asymmetry. EC-based NCT correctly flags Node 0 as the dominant driver - equivalent to identifying a **Seizure Onset Zone** in epilepsy or a **dysregulated hub** in addiction.

---

## Running Tests

```bash
# Clone the repository
git clone https://github.com/TheRoy666/neurosim
cd neurosim

# Install in development mode
pip install -e ".[dev]"
pip install -r requirements.txt

# Run the full test suite
pytest tests/ -v

# Expected output:
# PASSED tests/test_physics.py::test_normalise_target_rho
# PASSED tests/test_physics.py::test_normalise_works_on_unstable_matrix
# PASSED tests/test_physics.py::test_normalise_raises_on_degenerate
# PASSED tests/test_physics.py::test_gramian_positive_semidefinite
# PASSED tests/test_physics.py::test_gramian_symmetric
# PASSED tests/test_physics.py::test_gramian_psd_small_system
# PASSED tests/test_physics.py::test_doubling_matches_naive
# PASSED tests/test_physics.py::test_trivial_transition_zero_energy
# PASSED tests/test_physics.py::test_minimum_energy_nonnegative
# PASSED tests/test_physics.py::test_gramian_raises_on_unstable
# PASSED tests/test_physics.py::test_controllability_shapes
# PASSED tests/test_physics.py::test_doubling_speedup

# PASSED tests/test_connectivity.py::test_fc_is_symmetric
# PASSED tests/test_connectivity.py::test_ec_recovers_causal_direction
# PASSED tests/test_connectivity.py::test_teleportation_error
# Passed tests/test_connectivity.py::test_effective_connectivity_is_directed
# Passed tests/test_connectivity.py::test_graph_laplacian_positive_semidefinite
# Passed tests/test_connectivity.py::test_graph_laplacian_row_sum_zero
# Passed tests/test_connectivity.py::test_graphnet_structural_bias
# Passed tests/test_connectivity.py::test_effective_connectivity_graphnet_is_directed
# Passed tests/test_connectivity.py::test_graphnet_structural_penalty
```

---

## Validation Protocol

NeuroSim's linear engine is benchmarked against the **Wilson-Cowan** neural mass model to quantify error margins of the LTI approximation:

```python
from neurosim.simulation import WilsonCowanNetwork

# Simulate 10-region WC network in limit-cycle regime
C = normalise_matrix(SC_dti, target_rho=0.3)   # inter-regional coupling
wc_net = WilsonCowanNetwork(n_regions=10, C=C)
sim = wc_net.simulate(t_span=(0, 5000), n_points=50000)

# Extract BOLD-proxy at TR=720ms
E_bold = wc_net.extract_bold_proxy(sim, tr_ms=720.0)

# Use as ground-truth states for energy benchmarking
x0 = E_bold[:, 0]    # WC initial state
xT = E_bold[:, -1]   # WC final state
energy_lti, _ = minimum_energy(A, B, x0, xT, T=E_bold.shape[1])
```

---

## Clinical Applications (Planned)

| Cohort | Hypothesis | NCT Biomarker |
|--------|-----------|---------------|
| **AUD** (Alcohol Use Disorder) | Pathological attractor basin in reward circuitry | ↑ Control energy to exit craving state |
| **ADNI** (Alzheimer's Disease) | Retrogenesis - white matter degradation reverses developmental trajectory | Δ Modal controllability along retrogenesis axis |
| **Epilepsy** (TLE) | Facilitator nodes lower seizure-onset energy barrier | ↓ Control energy to enter ictal state |

---

## References

1. Gu, S. et al. (2015). Controllability of structural brain networks. *Nature Communications*, 6, 8414.
2. Van Loan, C. F. (1978). Computing integrals involving the matrix exponential. *IEEE TAC*, 23(3), 395–404.
3. Grosenick, L. et al. (2013). Closed-loop and activity-guided optogenetic control. *Neuron*, 86(1), 106–139.
4. Fortin, J.-P. et al. (2017). Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*, 161, 149–170.
5. Wilson, H. R. & Cowan, J. D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. *Biophysical Journal*, 12(1), 1–24.
6. Srivastava, P. et al. (2020). Models of communication and control for brain networks. *PLOS Computational Biology*.

---

## Author

**Ritam Kanti Roy**  
MSc Biotechnology, Jadavpur University

*GSoC 2026 Proposal: INCF Project #39 - NeuroSim: A Physics-Constrained Model for Finite Horizon Network Control Theory*

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
