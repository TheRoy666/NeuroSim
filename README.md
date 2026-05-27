# NeuroSim

**A physics-constrained Python toolkit for finite-horizon Network Control Theory in macro-scale neuroimaging.**

> *"The brain is not a continuous-time system with infinite patience.
> It is a discrete, finite, energetically constrained machine —
> and our models should say the same."*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![GSoC 2026](https://img.shields.io/badge/GSoC-2026%20INCF-orange.svg)](https://summerofcode.withgoogle.com/)
[![iNCF](https://img.shields.io/badge/Project-iNCF%20GSoC%202026-blue)](https://neurostars.org/t/gsoc-2026-project-39-national-brain-research-centre-nbrc-ebrains-neurosim-automating-in-silico-stimulation-for-non-invasive-biomarker-discovery/35619/10/)
[![Tests](https://img.shields.io/badge/Tests-141%20passing-brightgreen.svg)](#running-tests)
[![CI](https://github.com/TheRoy666/NeuroSim/actions/workflows/ci.yml/badge.svg)](https://github.com/TheRoy666/NeuroSim/actions/workflows/ci.yml)

---

## The Problem: An Approximation Crisis in Computational Neuroscience

The application of Network Control Theory (NCT) to neuroimaging has produced profound insights into how white-matter architecture constrains brain state dynamics. Yet current open-source tools share three methodological assumptions that, individually, are defensible; collectively, they constitute a reproducibility crisis.

### Error 1 — Temporal Approximation (Infinite Horizon)

Standard NCT pipelines compute the **Infinite-Horizon Controllability Gramian** W(∞), which satisfies the algebraic Lyapunov equation:

```
AW(∞) + W(∞)Aᵀ + BBᵀ = 0
```

This metric assumes the brain has *unlimited time* to transition between states. In a stable system (ρ(A) < 1), the energy to reach any state approaches **zero** as T → ∞ — the "vanishing cost" problem. A patient who can *theoretically* engage executive control *given infinite time* is indistinguishable from a healthy control. The infinite-horizon Gramian is blind to this clinically critical impairment.

**Biologically, cognitive switching occurs in 2–10 seconds. The time horizon T is not a free parameter — it is a physiological constant.**

### Error 2 — Structural Blindness (Binary DTI Masking)

Existing tools apply DTI tractography as a hard binary mask: if no streamline connects Region A to Region B, A[i,j] = 0. This ignores:

- **Polysynaptic pathways**: functional influence propagates A→C→B even without a direct A–B tract.
- **False negatives**: probabilistic tractography systematically underestimates crossing fibres and long-range connections.

The result: a "structurally blind" model with false-zero entries that corrupt control energy estimates.

### Error 3 — Statistical Leakage (Non-Blind Harmonisation)

Multi-site neuroimaging studies routinely use ComBat harmonisation with diagnostic group as a covariate — to "preserve" biological variance. This introduces **data leakage**: the harmonisation algorithm encodes group-level differences into the corrected features *before* the classifier sees them, inflating AUC and creating irreproducible biomarkers.

---

## NeuroSim's Solution: Three Methodological Pivots

### Pivot 1 — Discrete Finite-Horizon Physics (Van Loan Doubling)

NeuroSim models the brain as a **Discrete-Time LTI system**:

```
x[k+1] = A x[k] + B u[k]
```

and computes the **Finite-Horizon Reachability Gramian**:

```
W(N) at T = Σ  Aᵏ BBᵀ (Aᵀ)ᵏ [from k = 0 to (N-1)]

```

This T-parameterised metric directly captures the energetic cost of transitioning states *within a cognitive task window*. Naïve computation costs O(T·N³). NeuroSim implements the **Van Loan Doubling Algorithm**, which reduces this to **O(log T · N³)**:

```python
from neurosim.physics import compute_gramian_doubling, normalise_matrix

A   = normalise_matrix(A_ec, target_rho=0.9)   # spectral stability
B   = np.eye(N)                                 # full-rank input
W_T = compute_gramian_doubling(A, B, T=10)      # 10 TR steps ≈ 7.2 s
```

### Pivot 2 — GraphNet Regularisation (Soft Structural Prior)

NeuroSim replaces hard DTI masking with the **GraphNet** objective (Grosenick et al., 2013):

$$\min_{A} \|X_{t+1} - A X_t\|^2_F + \lambda_1 \|A\|^2_F + \lambda_2 \text{Tr}(A^T L_{sc} A)$$

where L_sc is the Graph Laplacian of the structural connectome. Connections absent from DTI are **penalised, not forbidden**.

```python
from neurosim.connectivity import graphnet_effective_connectivity

EC = graphnet_effective_connectivity(
    X_bold,          # (N_regions, T_timepoints) BOLD time series
    SC_dti,          # (N_regions, N_regions) DTI connectome
    lambda_ridge=1.0,
    lambda_graph=2.0,
)
```

### Pivot 3 — Blind Harmonisation (Controls-Only ComBat)

NeuroSim enforces a **controls-only harmonisation** protocol — site-effect parameters are estimated exclusively from healthy controls, then applied to all subjects:

```python
from neurosim.harmonize import BlindHarmonizer

harmonizer = BlindHarmonizer()
harmonizer.fit(X_controls, site_controls)
X_harmonised = harmonizer.transform(X_all, site_all)
```

Any group difference surviving blind harmonisation is a genuine biological signal.

---

## Architecture

```
NeuroSim/
├── .github/
│   └── workflows/
│       └── ci.yml                        # GitHub Actions CI (Python 3.9, 3.10, 3.11)
├── neurosim/                             # Core library
│   ├── __init__.py                       # Public API
│   ├── physics.py                        # Van Loan Doubling, Gramian, control energy
│   ├── connectivity.py                   # GraphNet FISTA solver, Ridge EC, FC baseline
│   ├── harmonize.py                      # BlindHarmonizer, site-effect detection
│   ├── simulation.py                     # Wilson-Cowan neural mass model
│   ├── loader.py                         # BIDS ingestion: BIDSLoader, load_bold, from_arrays
│   ├── plot.py                           # Publication figures: 7 visualisation functions
│   └── clinical.py                       # AUDPipeline, ADNIPipeline, EpilepsyPipeline
├── notebooks/
│   ├── 01_fc_vs_ec_validation.ipynb      # Teleportation Error + doubling accuracy demo
│   ├── 02_minimum_energy_control.ipynb   # Finite-horizon physics end-to-end
│   ├── 03_wilson_cowan_benchmark.ipynb   # LTI vs non-linear validation (NLCF)
│   └── 04_clinical_pipeline_demo.ipynb   # AUD · ADNI · Epilepsy synthetic walkthrough
├── tests/
│   ├── test_physics.py                   # 12 tests — Van Loan, Gramian, energy
│   ├── test_connectivity.py              # 9 tests  — EC direction, Teleportation Error
│   ├── test_harmonize.py                 # 19 tests — BlindHarmonizer scientific correctness
│   ├── test_simulation.py               # 19 tests — limit cycle, BOLD proxy
│   ├── test_loader.py                    # 24 tests — ingestion, validation, connectome loading
│   ├── test_plot.py                      # 27 tests — all visualisation functions
│   └── test_clinical.py                  # 33 tests — AUD, ADNI, Epilepsy pipelines
├── CONTRIBUTING.md
├── LICENSE                               # Apache 2.0
├── pyproject.toml
└── requirements.txt
```

### Module Overview

| Module | Key Contribution | Core Method |
|--------|-----------------|-------------|
| `physics` | Finite-horizon control energy | Van Loan Doubling — O(log T · N³) |
| `connectivity` | Causal EC estimation | GraphNet FISTA solver |
| `harmonize` | Leakage-free harmonisation | Controls-only Empirical Bayes |
| `simulation` | Non-linear ground truth | Wilson-Cowan neural mass model |
| `loader` | BIDS-compliant data ingestion | PyBIDS + NiBabel + Nilearn |
| `plot` | Publication-ready figures | UMAP/PCA state space, energy landscape |
| `clinical` | Clinical validation pipelines | AUD · ADNI · Epilepsy |

---

## Installation

```bash
git clone https://github.com/TheRoy666/NeuroSim
cd NeuroSim

# Core (physics + connectivity + harmonisation + simulation)
pip install -e ".[dev]"

# With neuroimaging data loading (NiBabel, Nilearn, PyBIDS)
pip install -e ".[neuroimaging]"

# Everything
pip install -e ".[all]"
```

---

## Quickstart

```python
import numpy as np
from neurosim.physics import normalise_matrix, compute_gramian_doubling, minimum_energy
from neurosim.connectivity import graphnet_effective_connectivity
from neurosim.harmonize import BlindHarmonizer
from neurosim.simulation import WilsonCowanNode

# 1. Harmonise multi-site data (controls-only protocol)
harmonizer = BlindHarmonizer()
harmonizer.fit(X_controls, site_controls)
X = harmonizer.transform(X_all, site_all)

# 2. Estimate causal EC from BOLD + DTI prior
EC = graphnet_effective_connectivity(X, SC_dti, lambda_ridge=1.0, lambda_graph=2.0)

# 3. Normalise for DT-LTI stability
A = normalise_matrix(EC, target_rho=0.9)
B = np.eye(N)

# 4. Compute finite-horizon Gramian (T = 10 TRs ≈ 7.2 s)
W_T = compute_gramian_doubling(A, B, T=10)

# 5. Minimum energy for a resting → task transition
energy, u_opt = minimum_energy(A, B, x_rest, x_task, T=10)
print(f"Minimum control energy: {energy:.4f}")
print(f"Optimal stimulation target: Node {abs(u_opt).argmax()}")
```

### Loading real neuroimaging data (HCP / ADNI / OpenNeuro)

```python
from neurosim.loader import BIDSLoader, from_arrays

# From a BIDS dataset
loader = BIDSLoader('/data/HCP_S1200', atlas='schaefer400')
X, SC  = loader.load_subject('sub-HCP001', task='rest')
# X.shape = (400, 1200)   SC.shape = (400, 400)

# From your own pre-processed numpy arrays
data = from_arrays(X, SC=SC, subject_id='sub-HCP001')
```

---

## The Teleportation Error: Why FC-Based NCT is Wrong

NeuroSim includes a formal demonstration of the Teleportation Error. In a ground-truth feedforward network (Node 0 → Node 1 → Node 2):

```python
from neurosim.connectivity import simulate_feedforward_network
from neurosim.connectivity import functional_connectivity, ridge_effective_connectivity

X, A_true = simulate_feedforward_network(n_nodes=3, n_timepoints=8000)

FC = functional_connectivity(X)       # symmetric → cannot distinguish driver from receiver
EC = ridge_effective_connectivity(X)  # asymmetric → recovers causal direction

# EC[1,0] ≈ 0.85  (Node 0 → 1, correct)
# EC[0,1] ≈ 0.02  (no reverse causation, correct)
```

The Gramian diagonal ratio: **FC = 1.6×** vs **EC = 1064×**.

FC-based NCT cannot distinguish which node drives the chain. EC-based NCT reveals the full causal hierarchy — equivalent to mapping the propagation order of a seizure network or the reward-circuit hierarchy in addiction.

---

## Wilson-Cowan Benchmark

NeuroSim's linear engine is benchmarked against the Wilson-Cowan neural mass model. The Non-Linear Correction Factor (NLCF) quantifies LTI approximation error:

```python
from neurosim.simulation import WilsonCowanNetwork

C      = normalise_matrix(SC_dti, target_rho=0.3)
wc_net = WilsonCowanNetwork(n_regions=N, C=C)
sim    = wc_net.simulate(t_span=(0, 5000), n_points=50000)

E_bold = wc_net.extract_bold_proxy(sim, tr_ms=720.0)
energy_lti, _ = minimum_energy(A, B, E_bold[:, 0], E_bold[:, -1], T=10)
```

**LTI validity confirmed** for: resting-state BOLD (near-equilibrium), T ≤ 15 TRs, coupling ρ(C) < 0.6.

---

## Clinical Pipelines

Three pipeline classes are implemented and validated on synthetic cohorts. Each accepts real BIDS data via `BIDSLoader` on June 9.

```python
from neurosim.clinical import AUDPipeline, ADNIPipeline, EpilepsyPipeline

# AUD — discordant MZ twin design (HCP S1200)
pipe   = AUDPipeline(T=10, reward_indices=reward_network_idx)
cohort = pipe.run_cohort(subjects)
df     = pipe.twin_discordance_analysis(cohort, pair_ids)
# df["delta_E"] = E*(AUD twin) - E*(Control twin) for craving→rest

# ADNI — disease stage biomarker (ADNI-3)
pipe  = ADNIPipeline(T=10, stage_order=["CN", "MCI", "AD"])
traj  = pipe.stage_trajectory(cohort, metric="default_to_memory")
fvi   = pipe.finite_vs_infinite_comparison(result, X)
# Compares finite-horizon E* vs W∞ approximation across disease stages

# Epilepsy — Seizure Onset Zone identification (TLE cohort)
pipe = EpilepsyPipeline(T=10, compute_node_energies=True)
soz  = pipe.identify_soz(result, top_k=5)
# soz["primary_soz"] = region index with lowest E* for interictal→ictal
```

| Cohort | Hypothesis | Primary metric |
|--------|-----------|----------------|
| **AUD** (HCP S1200 twins) | Reward circuit locked in craving attractor | ΔE* (AUD − Control), craving→rest |
| **Alzheimer's** (ADNI) | Finite-horizon E* tracks disease stage better than W∞ | Stage trajectory: CN→MCI→AD |
| **Epilepsy** (TLE) | Facilitator nodes lower the ictal energy barrier | Per-node E* → SOZ ranking |

---

## Running Tests

```bash
# Clone and install
git clone https://github.com/TheRoy666/NeuroSim
cd NeuroSim
pip install -e ".[dev]"

# Full test suite (141 tests, 2 skipped)
PYTHONPATH=. pytest tests/ -v

# By module
pytest tests/test_physics.py       # 12 tests — physics engine
pytest tests/test_connectivity.py  #  9 tests — EC estimation
pytest tests/test_harmonize.py     # 19 tests — harmonisation
pytest tests/test_simulation.py    # 19 tests — Wilson-Cowan
pytest tests/test_loader.py        # 24 tests — data ingestion
pytest tests/test_plot.py          # 27 tests — visualisation
pytest tests/test_clinical.py      # 33 tests — clinical pipelines
```

CI runs on Python 3.9, 3.10, and 3.11 via GitHub Actions on every push and pull request.

---

## Notebooks

| Notebook | Scientific content |
|----------|-------------------|
| `01_fc_vs_ec_validation` | Teleportation Error (1064× ratio) · Doubling accuracy (max error 6.82×10⁻¹³) · Wilson-Cowan limit cycle |
| `02_minimum_energy_control` | Gramian horizon sweep · State transitions · Vanishing cost proof · Stimulation target ranking |
| `03_wilson_cowan_benchmark` | NLCF computation · Regime analysis · LTI validity map · Jacobian stability |
| `04_clinical_pipeline_demo` | AUD twin discordance · ADNI stage trajectory · Epilepsy SOZ identification |

All notebooks use synthetic data with a clearly marked one-cell swap point for real HCP / ADNI / TLE data.

---

## References

1. Gu, S. et al. (2015). Controllability of structural brain networks. *Nature Communications*, 6, 8414.
2. Van Loan, C. F. (1978). Computing integrals involving the matrix exponential. *IEEE TAC*, 23(3), 395–404.
3. Grosenick, L. et al. (2013). Closed-loop and activity-guided optogenetic control. *Neuron*, 86(1), 106–139.
4. Fortin, J.-P. et al. (2017). Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*, 161, 149–170.
5. Wilson, H. R. & Cowan, J. D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. *Biophysical Journal*, 12(1), 1–24.
6. Srivastava, P. et al. (2020). Models of communication and control for brain networks. *PLOS Computational Biology*, 16(8), e1007826.
7. Beck, A. & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
8. McInnes, L. et al. (2018). UMAP: Uniform manifold approximation and projection. *arXiv:1802.03426*.

---

## Author

**Ritam Kanti Roy**
MSc Biotechnology, Jadavpur University

Mentor: **Dr. Khushbu Agarwal**
Computational Neuroscience Laboratory, NBRC

*GSoC 2026 Contributor — INCF Project #39: NeuroSim: A Physics-Constrained Model for Finite Horizon Network Control Theory*

---

## Contributing

External contributions are welcome from **July 2026 onwards**, once the GSoC coding period core architecture is stable. Until then, please open Issues to report bugs, suggest features, or discuss methodology — all discussion is welcome.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---
