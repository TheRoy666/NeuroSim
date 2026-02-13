# NeuroSim
A Computational Framework for Finite Horizon Network Control/ In Silico Model for Brain Stimulation

**Current Version:** v3.0 (Reference Implementation)
**Status:** Engineering Verification Phase

---

## The Mission
NeuroSim is a Python library designed to answer a single, rigorous question:
**"Can linear network control theory effectively steer non-linear brain dynamics?"**

Unlike standard tools that assume infinite time horizons or perfect linearity, NeuroSim acts as a **"simulation/test site"** for brain control models. It enforces strict physical constraints (finite-time physics), statistical safety (blind harmonization), and causal validation against non-linear ground truth (Wilson-Cowan models).

---

## The Evolution of the Architecture

NeuroSim is the result of three major architectural pivots, driven by rigorous peer review.

### v1.0: The "Infinite" Prototype (Deprecated)
* **Approach:** Standard Linear Time-Invariant (LTI) control with Infinite Horizon Gramians.
* **Failure:** Biological brains operate on short timescales ($T < 10s$). Infinite horizon metrics mathematically "converged," but were biologically meaningless for transient states like craving or seizures.

### v2.0: The "Continuous" Attempt (Deprecated)
* **Approach:** Finite-Horizon control using the Van Loan Integral.
* **Failure:** Category Error. fMRI data is discrete ($TR \approx 1-2s$). Applying continuous integration to discrete data introduced aliasing artifacts and numerical instability.

### v3.0: The "Discrete Physics" Standard (Current)
* **Approach:**
    * **Discrete Finite-Horizon Physics:** Uses the **Doubling Algorithm** ($O(\log T) $) to compute exact energy costs for step-by-step transitions.
    * **GraphNet Regularization:** Replaces binary masking with ElasticNet + Laplacian constraints to respect white matter topology without ignoring functional divergence.
    * **Causal Benchmarking:** The only library that self-validates by attempting to control a **Non-Linear Wilson-Cowan Simulation** before analyzing real data.
