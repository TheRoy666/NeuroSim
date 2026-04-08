# NeuroSim

<p align="center">
  <img src="https://img.shields.io/badge/GSoC-2026-orange?style=for-the-badge&logo=google" alt="GSoC 2026" />
  <img src="https://img.shields.io/badge/INCF-EBRAINS-%23005596?style=for-the-badge" alt="INCF EBRAINS" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/License-Apache2.0-green?style=for-the-badge" alt="MIT License" />
  <img src="https://img.shields.io/badge/Physics--Constrained-Network_Control_Theory-8B00FF?style=for-the-badge" alt="Physics-Constrained" />
</p>

## **NeuroSim: A Physics Constrained Model for Finite Horizon Network Control Theory**
[![iNCF](https://img.shields.io/badge/Project-iNCF%20GSoC%202026-blue)](https://neurostars.org/t/gsoc-2026-project-39-national-brain-research-centre-nbrc-ebrains-neurosim-automating-in-silico-stimulation-for-non-invasive-biomarker-discovery/35619/10)

**Mentor:** Dr. Khushbu Agarwal  
**Contributor:** Ritam Kanti Roy

---

## The Mission

NeuroSim is a **physics-constrained** computational framework that models the brain as a dynamic control system. It answers one core question with rigor:

> **"Can linear network control theory effectively steer non-linear brain dynamics?"**

It enforces strict physical constraints (finite-time discrete physics), statistical safety (blind harmonization), and causal validation against non-linear ground truth (Wilson-Cowan models). The library is designed for in-silico brain stimulation and biomarker discovery in conditions such as Alcohol Use Disorder and Temporal Lobe Epilepsy.

---

## Architecture Evolution — Three Major Pivots

NeuroSim evolved through deliberate architectural improvements based on mathematical validation:

| Version | Approach | Key Limitation Addressed | Outcome |
|---------|----------|---------------------------|---------|
| **v1.0** | Infinite-horizon Gramians + standard LTI | Biologically implausible vanishing costs for short cognitive timescales | Deprecated |
| **v2.0** | Continuous finite-horizon (Van Loan integral) | Category error — fMRI data is discrete (TR ≈ 1–2 s) | Deprecated |
| **v3.0** (Current) | **Discrete Finite-Horizon Physics** via Doubling Algorithm + GraphNet soft priors + Wilson-Cowan benchmarking | All major approximations eliminated | Production-ready rigorous standard |

**Current focus (v3.0):** Discrete Finite-Horizon Gramian (O(N³ log T)), directed Effective Connectivity with GraphNet regularization, blind harmonization, and non-linear ground-truth validation.

---

## Core Design Principles (Original Plan)

- **Finite-horizon control** — Discrete Doubling Algorithm for realistic metabolic costs over short timescales
- **Directed Effective Connectivity** — GraphNet regularized MVAR (L1 + L2 + Laplacian soft priors)
- **Blind harmonization** — neuroCombat fitted exclusively on healthy controls to preserve disease signals
- **Diagnostic-first architecture** — Spectral stability checks, Gramian condition monitoring, ANOVA for site effects
- **Non-linear benchmarking** — Wilson-Cowan limit-cycle simulations to quantify LTI approximation error

---

## Scientific Validation & Documentation

All foundational work is fully documented and open:

- **[FC vs EC Validation](docs/validation/FC_vs_EC_Validation.md)** — Mathematical proof why symmetric Pearson FC leads to "teleportation" errors and misidentification of facilitator nodes (includes reproducible 3-node sandbox)
- **[NeuroSim v3.0 Specification Review](docs/whitepapers/NeuroSim_v3.0_Finite_Horizon_Spec.md)** — Detailed finite-horizon Gramian and GraphNet implementation
- **[Architecture Validation Review](docs/whitepapers/NeuroSim_Architecture_Validation_Review.md)** — Risk analysis and methodological pivots
- **[Engineering Verification Checklist](docs/whitepapers/NeuroSim_Engineering_Verification_Checklist.md)** — Doubling Algorithm, GraphNet objective function, Wilson-Cowan parameters

These documents establish NeuroSim as the rigorously validated alternative to legacy approximations.

---

## Repository Structure

```bash
NeuroSim/
├── docs/
│   ├── validation/          # FC vs EC mathematical validation + sandbox
│   └── whitepapers/         # Architecture specs and verification reports
├── notebooks/
│   └── fullpipeline_v3.0.ipynb   # End-to-end demonstration (in progress)
├── neurosim/                # Core library modules
├── figures/                 # Visual outputs and diagrams
├── tests/                   # Unit tests and physics benchmarks
└── README.md
```
---


---

## Quick Start (Implementation in Progress)

Once the core modules are integrated, typical usage will follow this pattern:

```python
# Blind harmonization (preserves biological signal)
harmonized = blind_harmonize(raw_data, sites, covariates)

# Directed stable A matrix with GraphNet soft priors
A = estimate_effective_connectivity(timeseries, structural_prior)

# Finite-horizon control energy
energy = compute_finite_horizon_energy(A, x0, xT, horizon=T)
```

Full working demo notebook coming soon in notebooks/fullpipeline_v3.0.ipynb.

## Current Status (GSoC 2026)

| Component                    | Status | Deliverable                              |
|------------------------------|--------|------------------------------------------|
| Finite-Horizon Gramian       | ✅     | Doubling Algorithm                       |
| GraphNet Connectivity        | ✅     | Soft priors + FISTA                      |
| Blind Harmonization          | ✅     | Controls-only ComBat                     |
| FC vs EC Validation          | ✅     | Mathematical sandbox                     |
| Wilson-Cowan Benchmark       | ✅     | Limit-cycle unit test                    |

- All validation documents and specifications complete ✓  
- Engineering verification phase in progress  
- Core modules (`harmonization`, `connectivity`, `control`) under active development with strict adherence to the validated architecture  

## Future Directions

NeuroSim is deliberately engineered with a **science-first** philosophy: every architectural decision must first survive mathematical and biological scrutiny before being scaled.

### Short-Term Scientific Priorities (GSoC 2026)
- **Empirical validation on real multi-site datasets** (HCP + ADNI + OpenNeuro AUD/Epilepsy cohorts) using the blind harmonization protocol and finite-horizon metrics.
- **Systematic comparison of LTI approximations against non-linear ground truth**: Extend Wilson-Cowan benchmarking to include seizure-like hypersynchronous regimes and addiction-like deep attractor states.
- **Clinical biomarker mapping**: Correlate modal controllability scores and finite-horizon transition energies with clinician-defined seizure onset zones and craving-related behavioral scores.
- **Uncertainty quantification**: Implement bootstrap resampling and Gramian condition-number-based confidence intervals for all control energy estimates.

### Medium-Term Research Extensions
- **Hybrid linear-nonlinear control**: Develop piecewise-LTI approximations around operating points identified from Wilson-Cowan or neural mass models.
- **Personalized in-silico stimulation**: Use subject-specific parcellations and structural priors to simulate optimal TMS/DBS target selection while respecting individual finite-horizon energy landscapes.
- **Longitudinal disease staging**: Integrate SuStaIn-style trajectory inference with control energy profiles to model progression from healthy → prodromal → clinical states in AUD and epilepsy.
- **Open benchmarking platform**: Release a public 'NeuroSim Validation Suite' containing reproducible FC-vs-EC sandbox, Wilson-Cowan limit-cycle generators, and standardized metrics for the broader NCT community.

### Long-Term Vision
The ultimate goal is to move beyond static connectomics toward **energetic phenotyping** of brain disorders — where "how much energy does it cost to exit a pathological state?" becomes a quantifiable, clinically actionable biomarker. By maintaining strict separation between technical harmonization and biological signal, and by anchoring every metric in finite-time discrete physics, NeuroSim aims to provide the methodological foundation for safe, personalized, non-invasive brain stimulation therapies.

### **Guiding Principle:**  
*Engineering scalability follows only after rigorous scientific validation.*  
No feature is added unless it demonstrably improves physical fidelity or clinical interpretability.

### **License**
Apache2.0 License — see LICENSE for details.

### **Full documentation** is available in the docs/ directory.
Contributions and feedback welcome — especially on the physics and clinical validation aspects.
