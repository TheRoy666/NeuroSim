<div align="center">


# NeuroSim

### When Can Linear Network Control Theory Be Trusted?

*A validity-characterization framework for network control theory in clinical neuroimaging*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![GSoC 2026](https://img.shields.io/badge/GSoC-2026%20INCF-orange.svg)](https://summerofcode.withgoogle.com/)
[![iNCF](https://img.shields.io/badge/Project-iNCF%20GSoC%202026-blue)](https://neurostars.org/t/gsoc-2026-project-39-national-brain-research-centre-nbrc-ebrains-neurosim-automating-in-silico-stimulation-for-non-invasive-biomarker-discovery/35619/10/)
[![Tests](https://img.shields.io/badge/Tests-212%20passed%2C%202%20skipped-brightgreen.svg)](#running-tests)
[![CI](https://github.com/TheRoy666/NeuroSim/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TheRoy666/NeuroSim/actions/workflows/ci.yml)
[![Validated](https://img.shields.io/badge/Validated-N%3D349%20real%20subjects%2C%203%20cohorts-success.svg)](#validation)

</div>

---

## The Question This Project Actually Answers

Picture a researcher planning a neurostimulation target with network
control theory. The method returns a precise-looking number — the
energy required to push a brain network from one state to another. That
number rests on an assumption almost nobody stops to examine: that the
brain, for the purposes of this calculation, behaves linearly. It
doesn't, not universally. The real question was never *whether* the
linear approximation breaks down — in large regions of parameter space,
it demonstrably does. The question is whether anyone can say, in
advance and on real data, exactly where the line is.

That's what this project set out to determine, across three real
clinical cohorts (HCP-YA, ADNI-3, UNAM-TLE; N=349 real human
connectomes) and two independent nonlinear neural mass models. The
answer that emerged is sharper than "sometimes it's fine, sometimes
it isn't": **validity turns out to be a property of the specific
network being modeled — measurable, characterizable, and traceable to
concrete structural properties of that network — not a universal
property of the method itself.**

The clearest illustration of what that means, and of how this project
actually arrived at it, is worth telling as it happened, not just as a
conclusion.

**Reachability-prediction error, plotted against proximity to
bifurcation, doesn't behave the way the standard caveats about linear
approximations would suggest.** It doesn't simply get worse as a
system approaches instability. It traces an interior minimum — error
actually *improves* on the approach, before rising again near the
edge. That pattern held on real per-subject structural connectivity
from ADNI, on both of two independent brain atlases, in roughly 96%
of subjects individually. It looked like a clean, general law.

**Then HCP didn't cooperate.** Only 37% of HCP subjects showed the
same interior minimum; the rest showed error declining nearly
monotonically all the way to the edge of a tightly-bounded stable
regime instead. A less careful project stops here, reports two
cohorts as confirmation and treats HCP as noise. This one asked why,
and kept pulling on the thread until five independent, converging
lines of evidence — real-coupling edge-weight ratios up to 875× the
cohort mean, a stability boundary that stayed identical across two
genuinely independent subject samples, structural rewiring that ran
20–35× slower under a degree-preserving null model, and two
directly-computed, standard network-science metrics (node-strength
heterogeneity and a normalized rich-club coefficient, both markedly
higher in HCP than in either ADNI atlas) — all pointed at the same
answer. HCP's connectomes are measurably more hub-concentrated, and
that structural difference is what the earlier divergence had been
tracking all along.

**That finding earned one more, harder test — the kind that risks
overturning the story you already believe.** If hub concentration is
really the explanation, then a network with HCP's exact degree
sequence and edge-weight distribution, but with every specific
connection scrambled, should reproduce the same pattern even though
the real wiring diagram is gone. Building that test properly meant
implementing degree-preserving network rewiring from scratch —
including finding and fixing a real performance bug that made a
first, textbook-standard implementation impractical at real
connectome scale, before it ever reached a server. The result:
**the null model reproduced each cohort's real pattern almost
exactly** — both ADNI atlases' interior minimum, and HCP's own
edge-pileup, independently, across two different HCP parameter
regimes. That is not the answer that would have made the more
dramatic claim — "this is specific to real brain wiring" — true. It
is a more precise and more defensible one: **the effect depends on a
network's degree and weight statistics, not on which exact neuron
talks to which.** Reporting that correction honestly, rather than
keeping the more publishable-sounding version, is the actual
scientific contribution here.

That same discipline — ask the harder question, test it directly,
report what the data actually says — runs through everything else in
this repository: a bootstrap-based reliability check that found
connectome-derived control-energy estimates are close to unreliable
on realistic clinical scan lengths, externally validated against a
published result rather than taken on faith; a from-scratch adjoint
gradient method that let two independent nonlinear models both be
tested properly instead of approximated; and three clinical case
studies where honest nulls were investigated and explained against
prior literature rather than quietly set aside.

---

## GSoC 2026 Final Work Product

*Everything the [GSoC Work Product Submission Guidelines](https://developers.google.com/open-source/gsoc/help/work-product) ask for, in one place*

| | |
|---|---|
| **Contributor** | Ritam Kanti Roy |
| **Organization** | INCF, GSoC 2026, Project #39 |
| **Mentor** | Dr. Khushbu Agarwal, National Brain Research Centre (NBRC) |
| **Repository** | This one — single-purpose for this project; every commit is this contributor's work |

**Goal:** determine, on real clinical data, exactly when linear network
control theory's approximation can be trusted — and build the
infrastructure to make that determination reproducibly.

**What was done:** three engineering pivots to an existing NCT toolkit
(finite-horizon control energy, sparse directed-EC estimation, blind
site-effect harmonization), validated on three real cohorts (N=349) —
plus, beyond that original scope, the full validity-regime
characterization, null-model comparison, and network-topology
quantification narrated above, cross-model generality testing, an
externally-validated uncertainty-propagation method, and a third
clinical cohort. Full findings in [Validation](#validation).

**Current state:** all code written, tested (212 tests, 2 skipped), and
merged to `main`. Every real-data result referenced in this README has
a corresponding script and committed result file under
[`results/`](results/) — nothing here is aspirational.

**What's left:** see [What's Next](#whats-next).

**Challenges and what was learned:** see
[Challenges & What I Learned](#challenges--what-i-learned).

---

## The Problem: An Approximation Crisis in Computational Neuroscience

Modern network control theory for the brain rests on three assumptions
that are each convenient and each, in specific identifiable regimes,
wrong:

1. **Linearity.** Real neural population dynamics are nonlinear. The
   linear approximation is treated as safe by default rather than
   characterized.
2. **Symmetric, functional connectivity as the control operator.** Real
   information flow is directed. Using symmetric FC where directed EC
   belongs silently changes what "control energy" even means.
3. **Infinite-horizon control energy.** Real stimulation protocols are
   finite-duration. The infinite-horizon simplification can differ from
   the finite-horizon reality by an order of magnitude or more.

Each has a regime where the shortcut is safe and a regime where it
produces a number that looks precise and is wrong. This project
characterizes those regimes directly, rather than assuming the shortcut
travels safely from a textbook network to a real individual human
connectome.

---

## NeuroSim's Solution: Three Methodological Pivots

**1. Finite-horizon control energy**, replacing the infinite-horizon
default — differs from the infinite-horizon estimate by up to 32.8× at
short horizons on real ADNI data, decaying to parity by T≈20.

**2. Sparse, directed effective connectivity (EC)** via a
GraphNet-regularized FISTA solver, replacing symmetric FC as the
control operator — produces control-energy estimates diverging from
FC-based estimates by a median of 3.2–13.5× depending on cohort, on
100% of subjects tested.

**3. Blind, controls-only harmonization** for multi-site batch effects,
replacing naive pooled ComBat — preserves a real clinical signal (CN
vs. MCI direction, consistent across every energy metric) on ADNI-3
while removing scanner-manufacturer confounds that independently
explain 17–32% of variance in this cohort.

*Full implementation details and worked demos in [`notebook/`](notebook/).*

---

## Architecture

```
NeuroSim/
├── .github/workflows/ci.yml               # CI: Python 3.9, 3.10, 3.11
├── neurosim/                              # Core library
│   ├── physics.py                         # Van Loan Doubling, Gramian, control energy, minimum-energy trajectories
│   ├── connectivity.py                    # GraphNet FISTA solver, Ridge EC, FC baseline, EC bootstrap
│   ├── harmonize.py                       # BlindHarmonizer, site-effect detection
│   ├── simulation.py                      # Wilson-Cowan model, system discretization
│   ├── loader.py                          # BIDS ingestion
│   ├── plot.py                            # Publication figures
│   ├── clinical.py                        # AUDPipeline, ADNIPipeline, EpilepsyPipeline
│   └── validation_pipeline.py             # End-to-end validation orchestration
├── scripts/
│   ├── nonlinear_validity_regime/         # When is the linear approximation trustworthy?
│   │   ├── wc_linear_validity_sweep_real_coupling.py
│   │   ├── wc_linear_validity_sweep_null_model.py
│   │   ├── null_model_rewiring.py
│   │   ├── network_topology_characterization.py
│   │   ├── fhn_simulation.py, fhn_broader_grid.py, wc_broader_grid.py
│   │   └── adjoint_gradient.py
│   ├── UNAM_TLE/                          # UNAM epilepsy cohort, full pipeline
│   ├── run_hcp_aud_batch.py, run_adni_nctn_batch.py
│   ├── run_ec_bootstrap_batch.py          # Path B: bootstrap uncertainty propagation
│   ├── frassle_stephan_cross_check.py     # External validation
│   └── operator_divergence_cross_cohort_analysis.py
├── results/                               # Every result in this README, de-identified
├── docs/statistical_sensitivity_addendum.md
├── notebook/                              # 5 walkthrough notebooks
├── tests/                                 # 212 tests
├── CONTRIBUTING.md · LICENSE · pyproject.toml
```

*Full tree with per-file annotations in the repository itself.*

---

## Installation

```bash
git clone https://github.com/TheRoy666/NeuroSim
cd NeuroSim
pip install -e ".[all]"
```

## Quickstart

```python
from neurosim.physics import compute_control_energy
from neurosim.connectivity import estimate_ec_graphnet
from neurosim.simulation import WilsonCowanNetwork

# Estimate directed effective connectivity from BOLD timeseries
EC = estimate_ec_graphnet(bold_timeseries, alpha=0.1, l1_ratio=0.5)

# Finite-horizon control energy under the linear approximation
E_star = compute_control_energy(A=EC, x0=x_rest, xf=x_target, T=20)
```

*Full worked examples in [`notebook/`](notebook/).*

---

## Validation

NeuroSim has been validated on real human neuroimaging data across
**three independent cohorts (N=349 total)**. Full analysis code is in
[`scripts/`](scripts/); underlying de-identified results are in
[`results/`](results/).

### HCP S1200 (N = 238)

- **Teleportation Error**: median **3.22×**, 100% of subjects > 1×
- **Finite-vs-infinite horizon**: median **11.7×** at T=1, decaying to ~1× by T≈20
- Clinical group comparison: honest null (p=0.066); a real reward-network
  reanalysis (real anatomical parcellation vs. the original proxy
  definition) closely replicates both the group-level pattern and the
  weak twin-controlled signal, confirming the finding isn't an artifact
  of reward-network definition

### ADNI-3 CN/MCI (N = 49, two independent atlases)

- **Teleportation Error**: median **12.4–13.5×**, 100% of subjects > 1×
- **Finite-vs-infinite horizon**: median **32.8×** at T=1, decaying to ~1× by T≈20
- **Site effects**: scanner manufacturer explains 17–32% of variance
  (p<0.01) — real and substantial; BlindHarmonizer preserves the CN vs.
  MCI direction throughout while addressing it (p=0.061)

### UNAM-TLE (N = 62)

- **Pre-registered lateralization test**: null (N=28, 46.4% correct,
  p=0.851), directly explained via He et al. (2022)'s positive result
  using a different operator (symmetric SC) on a larger sample — not
  left as an unexplained negative
- **Sex-stratified reanalysis**: a real, Bonferroni-surviving sex effect
  on EC asymmetry was found and checked directly against the clinical
  result — the null holds independently in both sex groups, not a
  pooling artifact

### Validity-Regime Characterization, Null Model, and Network Topology (Path A1)

The full investigative arc is narrated at the top of this README. In
numbers: interior-minimum replication on ~96% of subjects across both
real ADNI atlases; HCP's divergence (37% of subjects) traced to
measurable hub-concentration differences (edge-weight ratios up to
875× mean; node-strength CV 0.805 vs. 0.501/0.541; normalized rich-club
coefficient higher than both ADNI atlases at every threshold tested,
5–30%); and a null-model comparison that reproduces each cohort's real
pattern almost exactly, sharpening "real brain topology matters" into
the more precise "degree and weight statistics matter."

### Oscillatory Regime (Path A2) and Uncertainty Propagation (Path B)

- Neither naive nor LTV linearization dominates consistently in the
  oscillatory regime on either of two independent models (Wilson-Cowan,
  FitzHugh-Nagumo) — a true nonlinear solution is never worse than
  either
- EC-derived control-energy estimates are close to unreliable on
  realistic clinical scan lengths (bootstrap uncertainty propagation,
  externally validated against Frässle & Stephan 2022)
- Directed EC and symmetric FC diverge 3–100× in energy estimates
  across all three cohorts — a real, cross-cohort effect

### Honest Limitations

Full quantitative detail in [`docs/statistical_sensitivity_addendum.md`](docs/statistical_sensitivity_addendum.md).
None of this project's clinical subgroup comparisons were powered to
detect small-to-medium effects (minimum detectable Cohen's d
0.587–1.126) — this does not affect the primary, full-cohort structural
results, only the small-subgroup clinical comparisons. Cross-model
generality rests on two neural mass models, a deliberate scope
boundary. Path B's external validation is confirmed across all three
cohorts — HCP, ADNI-Schaefer400, and ADNI-TianS3 all show the identical
pattern against Frässle & Stephan (2022)'s benchmark.

---

## Challenges & What I Learned

**The hardest moment in this project was realizing part of its planned
contribution wasn't actually new.** A literature check partway through
the summer found that several of the intended engineering
contributions, while sound, weren't independently novel — `nctpy`
already supports finite-horizon transition energy, for instance. The
response that mattered wasn't defending the original plan; it was
asking what the project's real contribution had become, and rebuilding
around that honestly. Everything this README leads with — the interior
minimum, the five-line convergence on HCP's structure, the null-model
correction — exists because of that pivot, not despite it.

**A late-stage diagnostic investigation surfaced a real correctness bug
in the core connectivity estimator, silently affecting results that
had already been reported.** Investigating an unrelated anomaly in one
clinical cohort turned up a suspiciously exact, parameter-independent
result — the kind of finding that's either a genuine discovery or a
sign something's wrong. It was the latter: a one-line control-flow
error in the FISTA solver's convergence check meant that whenever
optimization converged quickly, the function discarded its final
update and silently returned an unintended fallback estimate instead
of the regularized one. Rather than assume the fix mattered or didn't,
every affected cohort was checked directly, not by analogy — some
needed real recomputation, one needed a larger iteration budget, and
the actual downstream numbers were compared old versus new in every
case before anything was trusted again. The corrected results matched
what had already been reported closely enough, in most cases, that no
finding in this project changed as a result — but that conclusion is
earned by direct verification here, not assumed from good intentions.
Found and corrected before submission is the outcome that matters; the
same verification discipline that runs through every other result in
this project is what caught it.

**A textbook-standard algorithm turned out not to be practical at real
scale, and finding that out early mattered.** The null-model rewiring
needed to test the pivot's central claim is a well-established
technique in network science — but a first implementation, correct at
small scale, did not complete in practical time once tested against
real connectome density, not toy density. Rewritten around a more
efficient data structure before it ever touched a server, not after a
failed production run.

**A missing environment-variable fix turned an 11-day compute estimate
into four hours.** Full-cohort real-coupling sweeps were initially
projected to take roughly 11 days for HCP alone. The actual cause was a
missing BLAS thread-capping fix, not a hardware ceiling — once found,
the identical computation completed in under 4.5 hours on existing lab
hardware. No HPC access was ever required for this project's headline
results.

**Honest nulls were more valuable than a positive result would have
been.** UNAM's lateralization null, checked directly against He et al.
(2022)'s positive result under a different operator, became a specific,
defensible statement about what the operator choice actually explains —
not a dead end. Explaining nulls against prior literature, rather than
merely disclosing them, is the throughline of every clinical case study
in this project.

---

## What's Next

- Resolve the open root-mechanism question behind UNAM's raw_rho
  numerical anomaly (its lack of downstream impact on reported metrics
  is already confirmed)
- Manuscript in preparation, targeting *Network Neuroscience* as
  primary venue, built directly on the findings in this repository
- A third neural mass model was considered for broader cross-model
  generality testing and deliberately not pursued this cycle — out of
  proportion in scope to what this project needed, noted here as a
  natural next step rather than a gap

---

## Running Tests

```bash
pip install -e ".[dev]"
PYTHONPATH=. pytest tests/ -v   # 212 tests, 2 skipped
```

CI runs on Python 3.9, 3.10, and 3.11 via GitHub Actions on every push
and pull request.

## Notebooks

Five walkthrough notebooks in [`notebook/`](notebook/), from the
Teleportation Error demo through full real-data validation on HCP and
ADNI.

## References

1. Gu, S. et al. (2015). Controllability of structural brain networks. *Nature Communications*, 6, 8414.
2. Van Loan, C. F. (1978). Computing integrals involving the matrix exponential. *IEEE TAC*, 23(3), 395–404.
3. Wilson, H. R. & Cowan, J. D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. *Biophysical Journal*, 12(1), 1–24.
4. Muldoon, S. F. et al. (2016). Stimulation-based control of dynamic brain networks. *PLOS Computational Biology*, 12(9), e1005076.
5. Lindmark, G. & Altafini, C. (2018). Minimum energy control for complex networks. *Scientific Reports*, 8, 3188.
6. Frässle, S. & Stephan, K. E. (2022). Test-retest reliability of dynamic causal modeling for fMRI. *NeuroImage*, 250, 118928.
7. Opsahl, T. et al. (2008). Prominence and control: the weighted rich-club effect. *Physical Review Letters*, 101(16), 168702.
8. Maslov, S. & Sneppen, K. (2002). Specificity and stability in topology of protein networks. *Science*, 296(5569), 910–913.

*Complete reference list, including cohort-specific clinical citations,
in the repository's full documentation.*

---

## Author

**Ritam Kanti Roy** — GSoC 2026 Contributor, INCF Project #39

## Contributing

External contributions are welcome now that the GSoC 2026 coding period
is complete. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
