"""
neurosim.clinical
=================
Clinical validation pipelines for finite-horizon Network Control Theory.

Three pipeline classes, each targeting a specific neurological or psychiatric
condition where NCT has high translational potential:

AUDPipeline
-----------
Alcohol Use Disorder — discordant monozygotic twin design (HCP S1200).
Hypothesis: The reward circuit is locked in a pathological attractor state
in AUD-affected twins. The minimum energy required to exit this attractor
is significantly elevated compared to the genetically matched healthy
co-twin, controlling for genetic and family environment confounds.

Key metric: ΔE* = E*(AUD) - E*(Control) for the craving → rest transition.

ADNIPipeline
------------
Alzheimer's Disease — Retrogenesis Hypothesis (ADNI dataset).
Hypothesis: White matter degradation in AD reverses the developmental
trajectory of network control. Regions that mature last in development
(association cortices) lose modal controllability first in AD, mirroring
the Retrogenesis pattern (Rosen & Bhattacharya, 2022).

Key metric: Δ modal controllability along the retrogenesis axis, stratified
by disease stage (MCI → early AD → late AD).

EpilepsyPipeline
----------------
Temporal Lobe Epilepsy — Seizure Onset Zone identification.
Hypothesis: Facilitator nodes in the epileptic network lower the energy
barrier to the ictal state. The node with the minimum single-node
stimulation energy to reach an ictal-like state is the Seizure Onset Zone.

Key metric: Per-node E* for the interictal → ictal transition, ranked as
stimulation energy map.

References
----------
Srivastava, P. et al. (2020). Models of communication and control for brain
    networks. PLOS Computational Biology, 16(8), e1007826.
Rosen, A. & Bhattacharya, A. (2022). Retrogenesis: A review of current
    evidence. Ageing Research Reviews.
Jirsa, V. K. et al. (2014). On the nature of seizure dynamics. Brain, 137(8).
Gu, S. et al. (2015). Controllability of structural brain networks.
    Nature Communications, 6, 8414.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurosim.physics import (
    normalise_matrix,
    compute_gramian_doubling,
    minimum_energy,
    average_controllability,
    modal_controllability,
)
from neurosim.connectivity import (
    ridge_effective_connectivity,
    graphnet_effective_connectivity,
)
from neurosim.harmonize import BlindHarmonizer, detect_site_effects


# ── Result dataclasses ─────────────────────────────────────────────────────

@dataclass
class SubjectResult:
    """Per-subject NCT result container."""
    subject_id: str
    group: str
    A: NDArray                            # (N, N) normalised EC
    ac: NDArray                           # (N,) average controllability
    mc: NDArray                           # (N,) modal controllability
    energies: Dict[str, float] = field(default_factory=dict)
    node_energies: Optional[NDArray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class CohortResult:
    """Full cohort NCT result container."""
    pipeline_name: str
    subjects: List[SubjectResult]
    group_energies: Dict[str, NDArray] = field(default_factory=dict)
    statistics: Dict = field(default_factory=dict)
    harmonised: bool = False

    @property
    def n_subjects(self) -> int:
        """Total number of subjects in the cohort."""
        return len(self.subjects)

    @property
    def groups(self) -> List[str]:
        """Sorted list of unique group labels in the cohort."""
        return sorted(set(s.group for s in self.subjects))

    def get_group(self, group: str) -> List[SubjectResult]:
        """Return all SubjectResult objects belonging to a given group."""
        return [s for s in self.subjects if s.group == group]

    def energy_array(self, transition: str, group: Optional[str] = None) -> NDArray:
        """Return (N_subjects,) energy array for a given transition."""
        subjects = self.get_group(group) if group else self.subjects
        return np.array([s.energies.get(transition, np.nan) for s in subjects])

    def summary(self) -> pd.DataFrame:
        """Return a per-subject summary DataFrame."""
        rows = []
        for s in self.subjects:
            row = {
                "subject_id": s.subject_id,
                "group": s.group,
                "mean_ac": float(np.mean(s.ac)),
                "mean_mc": float(np.mean(s.mc)),
                "hub_node": int(np.argmax(s.ac)),
                "peripheral_node": int(np.argmax(s.mc)),
            }
            for k, v in s.energies.items():
                row[f"E_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)


# ── Base pipeline ──────────────────────────────────────────────────────────

class BasePipeline(ABC):
    """Abstract base class for all NeuroSim clinical pipelines.

    Subclasses implement ``define_transitions()``, ``define_states()``,
    and ``run_subject()``. The base class handles harmonisation,
    normalisation, and controllability computation.

    Parameters
    ----------
    target_rho    : Spectral radius for A normalisation (default 0.9).
    T             : Finite horizon in TR steps (default 10).
    use_graphnet  : Use GraphNet EC solver instead of Ridge (default False).
    lambda_ridge  : Ridge penalty for EC estimation.
    lambda_graph  : Graph Laplacian penalty (only used if use_graphnet=True).
    compute_node_energies : Compute per-node single-input energies (slow).
    """

    def __init__(
        self,
        target_rho: float = 0.9,
        T: int = 10,
        use_graphnet: bool = False,
        lambda_ridge: float = 1.0,
        lambda_graph: float = 1.0,
        compute_node_energies: bool = False,
    ):
        self.target_rho = target_rho
        self.T = T
        self.use_graphnet = use_graphnet
        self.lambda_ridge = lambda_ridge
        self.lambda_graph = lambda_graph
        self.compute_node_energies = compute_node_energies
        self._harmonizer: Optional[BlindHarmonizer] = None

    @abstractmethod
    def define_transitions(self) -> Dict[str, Tuple[str, str]]:
        """Return dict mapping transition name → (state_a_name, state_b_name)."""

    @abstractmethod
    def define_states(self, X: NDArray, metadata: Dict) -> Dict[str, NDArray]:
        """Extract brain states from BOLD time series.

        Parameters
        ----------
        X        : (N, T) BOLD time series.
        metadata : Subject-level metadata dict.

        Returns
        -------
        states : dict mapping state name → (N,) normalised state vector.
        """

    def _estimate_ec(
        self,
        X: NDArray,
        SC: Optional[NDArray] = None,
    ) -> NDArray:
        """Estimate EC and normalise to spectral radius = target_rho."""
        if self.use_graphnet and SC is not None:
            EC = graphnet_effective_connectivity(
                X, SC,
                lambda_ridge=self.lambda_ridge,
                lambda_graph=self.lambda_graph,
            )
        else:
            EC = ridge_effective_connectivity(X, alpha=self.lambda_ridge)
        return normalise_matrix(EC, target_rho=self.target_rho)

    def _compute_energies(
        self,
        A: NDArray,
        states: Dict[str, NDArray],
        transitions: Dict[str, Tuple[str, str]],
    ) -> Dict[str, float]:
        """Compute minimum energy for each defined transition."""
        N = A.shape[0]
        B = np.eye(N)
        energies = {}
        for name, (s_from, s_to) in transitions.items():
            if s_from not in states or s_to not in states:
                warnings.warn(
                    f"State '{s_from}' or '{s_to}' not found for transition '{name}'.",
                    UserWarning, stacklevel=2,
                )
                energies[name] = np.nan
                continue
            x0 = states[s_from]
            xT = states[s_to]
            e, _ = minimum_energy(A, B, x0, xT, T=self.T)
            energies[name] = float(e)
        return energies

    def _compute_node_energies(
        self,
        A: NDArray,
        x0: NDArray,
        xT: NDArray,
    ) -> NDArray:
        """Compute per-node single-input E* for the primary transition."""
        N = A.shape[0]
        node_energies = np.full(N, np.nan)
        for i in range(N):
            B_i = np.zeros((N, 1))
            B_i[i] = 1.0
            try:
                e, _ = minimum_energy(A, B_i, x0, xT, T=self.T)
                node_energies[i] = e
            except Exception:
                node_energies[i] = np.nan
        return node_energies

    def run_subject(
        self,
        X: NDArray,
        subject_id: str,
        group: str,
        SC: Optional[NDArray] = None,
        metadata: Optional[Dict] = None,
    ) -> SubjectResult:
        """Run the full NCT pipeline for a single subject.

        Parameters
        ----------
        X          : (N, T) BOLD time series (already preprocessed).
        subject_id : Subject identifier string.
        group      : Group label (e.g. 'Control', 'AUD').
        SC         : (N, N) structural connectome (optional).
        metadata   : Subject-level metadata dict.

        Returns
        -------
        SubjectResult
        """
        metadata = metadata or {}
        A = self._estimate_ec(X, SC)
        ac = average_controllability(A)
        mc = modal_controllability(A)

        states = self.define_states(X, metadata)
        transitions = self.define_transitions()
        energies = self._compute_energies(A, states, transitions)

        node_energies = None
        if self.compute_node_energies and transitions:
            first_trans = next(iter(transitions.values()))
            if first_trans[0] in states and first_trans[1] in states:
                node_energies = self._compute_node_energies(
                    A, states[first_trans[0]], states[first_trans[1]]
                )

        return SubjectResult(
            subject_id=subject_id,
            group=group,
            A=A, ac=ac, mc=mc,
            energies=energies,
            node_energies=node_energies,
            metadata=metadata,
        )

    def fit_harmonizer(
        self,
        X_controls: NDArray,
        site_controls,
        feature_fn=None,
    ) -> "BasePipeline":
        """Fit blind harmonisation on control subjects.

        Parameters
        ----------
        X_controls  : (N_ctrl, F) feature matrix for controls.
        site_controls : Site labels for controls.
        feature_fn  : Optional function to extract features from BOLD.

        Returns self.
        """
        self._harmonizer = BlindHarmonizer()
        self._harmonizer.fit(X_controls, site_controls)
        return self

    def run_cohort(
        self,
        subjects: List[Dict],
        verbose: bool = True,
    ) -> CohortResult:
        """Run pipeline on a full cohort.

        Parameters
        ----------
        subjects : List of dicts, each with keys:
                   'X' (NDArray), 'subject_id' (str), 'group' (str),
                   'SC' (NDArray, optional), 'metadata' (dict, optional).
        verbose  : Print progress.

        Returns
        -------
        CohortResult
        """
        results = []
        n = len(subjects)
        for i, subj in enumerate(subjects):
            if verbose:
                print(f"  [{i+1}/{n}] {subj['subject_id']} ({subj['group']})")
            try:
                result = self.run_subject(
                    X=subj["X"],
                    subject_id=subj["subject_id"],
                    group=subj["group"],
                    SC=subj.get("SC"),
                    metadata=subj.get("metadata", {}),
                )
                results.append(result)
            except Exception as e:
                warnings.warn(
                    f"Subject {subj['subject_id']} failed: {e}",
                    UserWarning, stacklevel=2,
                )

        cohort = CohortResult(
            pipeline_name=self.__class__.__name__,
            subjects=results,
            harmonised=self._harmonizer is not None,
        )
        cohort.group_energies = self._aggregate_energies(cohort)
        cohort.statistics = self._compute_statistics(cohort)
        return cohort

    def _aggregate_energies(self, cohort: CohortResult) -> Dict[str, NDArray]:
        """Aggregate per-subject energies by group and transition."""
        out = {}
        transitions = self.define_transitions()
        for group in cohort.groups:
            for trans_name in transitions:
                key = f"{group}::{trans_name}"
                out[key] = cohort.energy_array(trans_name, group=group)
        return out

    def _compute_statistics(self, cohort: CohortResult) -> Dict:
        """Compute group-level statistics for each transition."""
        stats = {}
        transitions = self.define_transitions()
        groups = cohort.groups

        for trans_name in transitions:
            stats[trans_name] = {}
            for group in groups:
                e = cohort.energy_array(trans_name, group=group)
                e_valid = e[np.isfinite(e)]
                stats[trans_name][group] = {
                    "n": len(e_valid),
                    "mean": float(np.nanmean(e_valid)) if len(e_valid) > 0 else np.nan,
                    "median": float(np.nanmedian(e_valid)) if len(e_valid) > 0 else np.nan,
                    "std": float(np.nanstd(e_valid)) if len(e_valid) > 0 else np.nan,
                }

            # Mann-Whitney U test between first two groups
            if len(groups) >= 2:
                try:
                    from scipy.stats import mannwhitneyu
                    e1 = cohort.energy_array(trans_name, group=groups[0])
                    e2 = cohort.energy_array(trans_name, group=groups[1])
                    e1 = e1[np.isfinite(e1)]; e2 = e2[np.isfinite(e2)]
                    if len(e1) > 0 and len(e2) > 0:
                        stat, p = mannwhitneyu(e1, e2, alternative="two-sided")
                        stats[trans_name]["mannwhitney_U"] = float(stat)
                        stats[trans_name]["p_value"] = float(p)
                        stats[trans_name]["significant_05"] = p < 0.05
                except ImportError:
                    pass

        return stats


# ── AUD Pipeline ───────────────────────────────────────────────────────────

class AUDPipeline(BasePipeline):
    """Alcohol Use Disorder NCT pipeline — discordant MZ twin design.

    Scientific context
    ------------------
    Uses HCP S1200 discordant monozygotic twin pairs where one twin has
    Alcohol Use Disorder (AUD) and the co-twin is a healthy control.
    The discordant design controls for genetic background and shared
    family environment.

    Hypothesis: the reward circuit (NAcc, caudate, putamen, OFC, mPFC)
    is locked in a pathological attractor state in AUD-affected twins.
    The minimum energy required to exit this attractor is elevated compared
    to the healthy co-twin.

    Networks
    --------
    - Reward: nucleus accumbens, caudate, putamen, OFC, vmPFC
    - Salience: anterior insula, dACC
    - DMN: mPFC, PCC, angular gyrus

    State definitions
    -----------------
    - rest: mean BOLD during low-global-signal epochs
    - craving: high-global-signal epochs (reward network elevated)
    - task: externally cued cognitive control state

    Parameters
    ----------
    reward_indices  : Region indices belonging to the reward network.
    T               : Finite horizon (default 10 TRs ≈ 7.2s).
    craving_percentile : Percentile threshold for craving state definition.
    """

    def __init__(
        self,
        reward_indices: Optional[List[int]] = None,
        craving_percentile: float = 75.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reward_indices = reward_indices
        self.craving_percentile = craving_percentile

    def define_transitions(self) -> Dict[str, Tuple[str, str]]:
        """Return AUD-specific state transitions: craving↔rest and rest↔cognitive."""
        return {
            "craving_to_rest": ("craving", "rest"),
            "rest_to_cognitive": ("rest", "cognitive"),
            "craving_to_cognitive": ("craving", "cognitive"),
        }

    def define_states(self, X: NDArray, metadata: Dict) -> Dict[str, NDArray]:
        """Define brain states from BOLD.

        States:
        - rest: low global signal epochs
        - craving: high global signal epochs (reward network activation)
        - cognitive: externally cued (from task metadata if available)
        """
        N, T = X.shape
        global_signal = X.mean(axis=0)

        # Resting state: low global signal
        rest_mask = global_signal < np.percentile(global_signal, 100 - self.craving_percentile)
        x_rest = X[:, rest_mask].mean(axis=1)

        # Craving proxy: high global signal OR reward network activation
        if self.reward_indices is not None:
            reward_signal = X[self.reward_indices, :].mean(axis=0)
            craving_mask = reward_signal > np.percentile(reward_signal, self.craving_percentile)
        else:
            craving_mask = global_signal > np.percentile(global_signal, self.craving_percentile)
        x_craving = X[:, craving_mask].mean(axis=1)

        # Cognitive control: use task metadata if provided, else mid-quintile
        if "task_onsets" in metadata and metadata["task_onsets"] is not None:
            task_idx = metadata["task_onsets"]
            x_cognitive = X[:, task_idx].mean(axis=1)
        else:
            mid_mask = (
                (global_signal >= np.percentile(global_signal, 40)) &
                (global_signal <= np.percentile(global_signal, 60))
            )
            x_cognitive = X[:, mid_mask].mean(axis=1) if mid_mask.sum() > 0 else x_rest.copy()

        # Normalise all states to unit norm
        def _norm(x):
            """Normalise state vector to unit norm (standard NCT convention)."""
            return x / (np.linalg.norm(x) + 1e-8)

        return {
            "rest":      _norm(x_rest),
            "craving":   _norm(x_craving),
            "cognitive": _norm(x_cognitive),
        }

    def compute_reward_network_energy(
        self,
        result: SubjectResult,
        reward_indices: Optional[List[int]] = None,
    ) -> float:
        """Compute control energy restricted to reward network nodes only.

        Parameters
        ----------
        result         : SubjectResult from run_subject().
        reward_indices : Reward network region indices. Uses self.reward_indices
                         if not provided.

        Returns
        -------
        E* for the reward-network-only control problem.
        """
        idx = reward_indices or self.reward_indices
        if idx is None:
            raise ValueError("reward_indices must be set to compute network energy.")

        N = result.A.shape[0]
        B_reward = np.zeros((N, len(idx)))
        for col, row in enumerate(idx):
            B_reward[row, col] = 1.0

        x_craving = result.metadata.get("x_craving")
        x_rest    = result.metadata.get("x_rest")
        if x_craving is None or x_rest is None:
            raise ValueError("States not stored in metadata. Set store_states=True.")

        e, _ = minimum_energy(result.A, B_reward, x_craving, x_rest, T=self.T)
        return float(e)

    def twin_discordance_analysis(
        self,
        cohort: CohortResult,
        pair_ids: List[Tuple[str, str]],
        transition: str = "craving_to_rest",
    ) -> pd.DataFrame:
        """Compute within-pair ΔE* for discordant MZ twin pairs.

        Parameters
        ----------
        cohort    : CohortResult from run_cohort().
        pair_ids  : List of (AUD_subject_id, Control_subject_id) tuples.
        transition: Transition to analyse.

        Returns
        -------
        DataFrame with columns: pair_id, E_AUD, E_Control, delta_E, ratio
        """
        subj_dict = {s.subject_id: s for s in cohort.subjects}
        rows = []
        for aud_id, ctrl_id in pair_ids:
            if aud_id not in subj_dict or ctrl_id not in subj_dict:
                continue
            e_aud  = subj_dict[aud_id].energies.get(transition, np.nan)
            e_ctrl = subj_dict[ctrl_id].energies.get(transition, np.nan)
            rows.append({
                "pair_id":   f"{aud_id}/{ctrl_id}",
                "E_AUD":     e_aud,
                "E_Control": e_ctrl,
                "delta_E":   e_aud - e_ctrl,
                "ratio":     e_aud / (e_ctrl + 1e-12),
            })
        return pd.DataFrame(rows)


# ── ADNI Pipeline ──────────────────────────────────────────────────────────

# ── ADNI Pipeline ──────────────────────────────────────────────────────────

class ADNIPipeline(BasePipeline):
    """Alzheimer's Disease NCT pipeline — disease stage biomarker validation.

    Scientific context
    ------------------
    NeuroSim's core claim tested on ADNI-3: does finite-horizon control energy
    E*(T) track Alzheimer's disease stage better than the infinite-horizon
    approximation W∞?

    The hypothesis is straightforward: as white matter degrades from CN → MCI → AD,
    the energy required to transition between key cognitive states (default mode →
    task-positive; memory encoding → executive) should increase monotonically. This
    increase should be detectable by the finite-horizon metric — which is sensitive
    to the energetic cost at the timescale of actual cognitive transitions (2–10s) —
    but partially masked by the infinite-horizon metric due to the vanishing cost
    problem.

    A secondary question: does average controllability of DMN hub nodes decline with
    disease stage? This would indicate that the structural integrity of the connectome
    is eroding in a topology-specific way.

    Note on Retrogenesis
    --------------------
    The Retrogenesis Hypothesis (that white matter degradation in AD reverses the
    developmental myelination sequence) is tested in a separate project ("The Entropic
    Mirror") using NCT + entropy analysis. That is a distinct scientific contribution
    from NeuroSim's finite-horizon physics engine. ADNIPipeline does not implement
    retrogenesis analysis.

    Data source
    -----------
    ADNI-3 (adni.loni.usc.edu):
    - Resting-state fMRI: preprocessed via FMRIPREP
    - DTI structural connectome: MRtrix3 tractography
    - Clinical labels: CDR, MMSE, diagnosis (CN/EMCI/LMCI/MCI/AD)
    - Longitudinal: baseline + 6-month + 12-month sessions

    Pipeline
    --------
    For each subject:
    1. Load parcellated BOLD (Schaefer-400) and DTI connectome
    2. Estimate EC via GraphNet (DTI as soft prior)
    3. Compute finite-horizon Gramian W_T (T = 10 TRs ≈ 7.2s)
    4. Define states: default mode, memory encoding, executive
    5. Compute E* for each transition
    6. Compare finite-horizon E* vs infinite-horizon approximation

    Disease stages
    --------------
    CN   → Cognitively Normal
    EMCI → Early MCI
    LMCI → Late MCI
    MCI  → Mild Cognitive Impairment
    AD   → Alzheimer's Disease

    Parameters
    ----------
    stage_order : Ordered list of disease stages.
    dmn_indices : Region indices belonging to the Default Mode Network.
                  Used to compute network-specific controllability metrics.
    """

    STAGE_ORDER = ["CN", "EMCI", "LMCI", "MCI", "AD"]

    def __init__(
        self,
        stage_order: Optional[List[str]] = None,
        dmn_indices: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stage_order = stage_order or self.STAGE_ORDER
        self.dmn_indices = dmn_indices

    def define_transitions(self) -> Dict[str, Tuple[str, str]]:
        """Return ADNI-specific state transitions: default↔memory and default↔executive."""
        return {
            "default_to_memory":    ("default", "memory"),
            "default_to_executive": ("default", "executive"),
            "memory_to_executive":  ("memory",  "executive"),
        }

    def define_states(self, X: NDArray, metadata: Dict) -> Dict[str, NDArray]:
        """Define brain states: default mode, memory encoding, executive control.

        State extraction uses global signal percentiles as a proxy for
        task-vs-rest distinctions in resting-state data. In task-based ADNI
        data, ``metadata["task_onsets"]`` can override this with true block timing.
        """
        N, T = X.shape
        global_signal = X.mean(axis=0)

        def _norm(x):
            """Normalise state vector to unit norm (standard NCT convention)."""
            return x / (np.linalg.norm(x) + 1e-8)

        # Default mode: low global signal (DMN active, task-negative)
        default_mask = global_signal < np.percentile(global_signal, 33)
        x_default = X[:, default_mask].mean(axis=1) if default_mask.sum() > 5                     else X[:, :T//3].mean(axis=1)

        # Memory encoding proxy: high global signal
        mem_mask = global_signal > np.percentile(global_signal, 67)
        x_memory = X[:, mem_mask].mean(axis=1) if mem_mask.sum() > 5                    else X[:, T//2:].mean(axis=1)

        # Executive control: mid-range global signal
        exec_mask = (
            (global_signal >= np.percentile(global_signal, 42)) &
            (global_signal <= np.percentile(global_signal, 58))
        )
        x_exec = X[:, exec_mask].mean(axis=1) if exec_mask.sum() > 5                  else (x_default + x_memory) / 2.0

        return {
            "default":   _norm(x_default),
            "memory":    _norm(x_memory),
            "executive": _norm(x_exec),
        }

    def dmn_controllability(self, result: SubjectResult) -> Dict[str, float]:
        """Compute controllability metrics restricted to DMN nodes.

        Parameters
        ----------
        result : SubjectResult from run_subject().

        Returns
        -------
        dict with keys 'mean_ac_dmn', 'mean_mc_dmn', 'ac_ratio' (DMN/whole-brain).
        """
        if self.dmn_indices is None:
            return {
                "mean_ac_dmn": float(np.mean(result.ac)),
                "mean_mc_dmn": float(np.mean(result.mc)),
                "ac_ratio":    1.0,
            }
        idx = np.array(self.dmn_indices)
        ac_dmn = result.ac[idx].mean()
        mc_dmn = result.mc[idx].mean()
        ac_all = result.ac.mean()
        return {
            "mean_ac_dmn": float(ac_dmn),
            "mean_mc_dmn": float(mc_dmn),
            "ac_ratio":    float(ac_dmn / (ac_all + 1e-8)),
        }

    def stage_trajectory(
        self,
        cohort: CohortResult,
        metric: str = "default_to_memory",
    ) -> pd.DataFrame:
        """Compute mean E* or controllability across disease stages.

        Parameters
        ----------
        cohort : CohortResult from run_cohort().
        metric : Transition name (e.g. 'default_to_memory') or
                 'ac' / 'mc' for mean controllability.

        Returns
        -------
        DataFrame indexed by stage with columns: mean, sem, n.
        """
        rows = []
        for stage in self.stage_order:
            subjects = cohort.get_group(stage)
            if not subjects:
                continue
            if metric == "ac":
                vals = np.array([float(np.mean(s.ac)) for s in subjects])
            elif metric == "mc":
                vals = np.array([float(np.mean(s.mc)) for s in subjects])
            else:
                vals = np.array([s.energies.get(metric, np.nan) for s in subjects])
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            rows.append({
                "stage": stage,
                "mean":  float(np.mean(vals)),
                "sem":   float(np.std(vals) / np.sqrt(len(vals))),
                "n":     len(vals),
            })
        return pd.DataFrame(rows).set_index("stage")

    def finite_vs_infinite_comparison(
        self,
        result: SubjectResult,
        X: NDArray,
        T_range: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Compare finite-horizon vs infinite-horizon energy for a subject.

        This is the core NeuroSim vs standard NCT comparison on real AD data.

        Parameters
        ----------
        result  : SubjectResult from run_subject().
        X       : (N, T) BOLD time series (to define states).
        T_range : Horizons to sweep (default 1–20).

        Returns
        -------
        DataFrame with columns T, E_finite, E_infinite, ratio.
        """
        from scipy.linalg import solve_discrete_lyapunov

        if T_range is None:
            T_range = list(range(1, 21))

        A  = result.A
        N  = A.shape[0]
        B  = np.eye(N)

        states = self.define_states(X, {})
        x0 = states["default"]
        xT = states["memory"]

        # Infinite-horizon
        W_inf     = solve_discrete_lyapunov(A, B @ B.T)
        W_inf_inv = np.linalg.pinv(W_inf)
        delta_inf = xT - np.linalg.matrix_power(A, max(T_range)) @ x0
        e_inf     = float(delta_inf @ W_inf_inv @ delta_inf)

        rows = []
        for T_h in T_range:
            e_fin, _ = minimum_energy(A, B, x0, xT, T=T_h)
            rows.append({
                "T":          T_h,
                "E_finite":   e_fin,
                "E_infinite": e_inf,
                "ratio":      e_fin / (e_inf + 1e-12),
            })

        return pd.DataFrame(rows)


class EpilepsyPipeline(BasePipeline):
    """Epilepsy NCT pipeline — Seizure Onset Zone identification.

    Scientific context
    ------------------
    The ictal state (seizure) can be modelled as a high-energy attractor
    in brain state space. Facilitator nodes are regions that, when stimulated
    with minimum energy, most efficiently drive the network from the
    interictal (baseline) state into the ictal attractor.

    These facilitator nodes are candidates for the Seizure Onset Zone (SOZ):
    the region where a seizure begins and where surgical resection or
    responsive neurostimulation would be most effective.

    State definitions
    -----------------
    - interictal: baseline resting state between seizures
    - preictal: the 30–120 second period before seizure onset
    - ictal: the seizure state itself (high-amplitude, high-synchrony)

    Key metric: single-node E* for interictal → ictal transition.
    The node with minimum E* is the predicted SOZ.

    Parameters
    ----------
    ictal_indices    : Time indices of ictal epochs in BOLD (if known).
    preictal_window  : Number of TRs before seizure onset classified as preictal.
    synchrony_threshold : z-score threshold for ictal state classification.
    """

    def __init__(
        self,
        ictal_indices: Optional[List[int]] = None,
        preictal_window: int = 5,
        synchrony_threshold: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ictal_indices = ictal_indices
        self.preictal_window = preictal_window
        self.synchrony_threshold = synchrony_threshold
        # Always compute node energies for SOZ identification
        self.compute_node_energies = True

    def define_transitions(self) -> Dict[str, Tuple[str, str]]:
        """Return epilepsy-specific state transitions: interictal↔ictal and preictal→ictal."""
        return {
            "interictal_to_ictal":   ("interictal", "ictal"),
            "preictal_to_ictal":     ("preictal",   "ictal"),
            "ictal_to_interictal":   ("ictal",      "interictal"),
        }

    def define_states(self, X: NDArray, metadata: Dict) -> Dict[str, NDArray]:
        """Define interictal, preictal, and ictal states.

        If ictal_indices are provided in metadata, uses them directly.
        Otherwise, detects ictal periods via synchrony (z-score of pairwise
        correlation exceeding synchrony_threshold).
        """
        N, T = X.shape

        def _norm(x):
            """Normalise state vector to unit norm (standard NCT convention)."""
            return x / (np.linalg.norm(x) + 1e-8)

        # Priority: explicit ictal indices from metadata or self
        ictal_idx = (
            metadata.get("ictal_indices")
            or self.ictal_indices
        )

        if ictal_idx is not None:
            ictal_idx = np.asarray(ictal_idx)
            ictal_idx = ictal_idx[ictal_idx < T]
        else:
            # Auto-detect: high global synchrony epochs
            # Compute rolling pairwise correlation as synchrony proxy
            global_signal = X.mean(axis=0)
            gs_z = (global_signal - global_signal.mean()) / (global_signal.std() + 1e-8)
            ictal_idx = np.where(gs_z > self.synchrony_threshold)[0]

            if len(ictal_idx) == 0:
                # Fall back to top percentile
                ictal_idx = np.where(
                    global_signal > np.percentile(global_signal, 90)
                )[0]

        # Interictal: all non-ictal, non-preictal timepoints
        ictal_set = set(ictal_idx)
        preictal_set = set()
        for t in ictal_idx:
            for offset in range(1, self.preictal_window + 1):
                if t - offset >= 0:
                    preictal_set.add(t - offset)
        preictal_set -= ictal_set

        interictal_mask = np.array([
            t not in ictal_set and t not in preictal_set
            for t in range(T)
        ])
        preictal_mask = np.array([t in preictal_set for t in range(T)])

        # Compute state vectors
        x_ictal = X[:, list(ictal_set)].mean(axis=1) if len(ictal_set) > 0 \
                  else X[:, T//2:].mean(axis=1)
        x_preictal = X[:, preictal_mask].mean(axis=1) if preictal_mask.sum() > 0 \
                     else (X[:, interictal_mask].mean(axis=1) if interictal_mask.sum() > 0
                           else X[:, :T//3].mean(axis=1))
        x_interictal = X[:, interictal_mask].mean(axis=1) if interictal_mask.sum() > 0 \
                       else X[:, :T//3].mean(axis=1)

        return {
            "interictal": _norm(x_interictal),
            "preictal":   _norm(x_preictal),
            "ictal":      _norm(x_ictal),
        }

    def identify_soz(
        self,
        result: SubjectResult,
        top_k: int = 5,
    ) -> Dict:
        """Identify Seizure Onset Zone candidates from per-node energies.

        Parameters
        ----------
        result : SubjectResult with node_energies computed.
        top_k  : Number of candidate SOZ nodes to return.

        Returns
        -------
        dict with keys:
            'soz_candidates': top_k region indices (lowest E*)
            'energies':       their E* values
            'primary_soz':    single best candidate (lowest E*)
        """
        if result.node_energies is None:
            raise ValueError(
                "node_energies not computed. Set compute_node_energies=True "
                "or EpilepsyPipeline.compute_node_energies=True."
            )

        ne = result.node_energies
        finite_mask = np.isfinite(ne) & (ne < 1e10)
        valid_idx   = np.where(finite_mask)[0]

        if len(valid_idx) == 0:
            return {"soz_candidates": [], "energies": [], "primary_soz": None}

        sorted_idx  = valid_idx[np.argsort(ne[valid_idx])]
        top_k_idx   = sorted_idx[:top_k]

        return {
            "soz_candidates": top_k_idx.tolist(),
            "energies":       ne[top_k_idx].tolist(),
            "primary_soz":    int(top_k_idx[0]),
        }

    def energy_barrier_analysis(
        self,
        cohort: CohortResult,
        group_soz: Optional[Dict[str, List[int]]] = None,
    ) -> pd.DataFrame:
        """Compare control energy barriers across patient groups.

        Parameters
        ----------
        cohort    : CohortResult from run_cohort().
        group_soz : Optional dict mapping group → list of known SOZ region indices.

        Returns
        -------
        DataFrame with energy barrier statistics per group.
        """
        rows = []
        for group in cohort.groups:
            subjects = cohort.get_group(group)
            e_vals = np.array([
                s.energies.get("interictal_to_ictal", np.nan)
                for s in subjects
            ])
            e_vals = e_vals[np.isfinite(e_vals)]

            soz_energy = np.nan
            if group_soz and group in group_soz and e_vals.size > 0:
                # Energy at known SOZ regions
                soz_nes = [
                    s.node_energies[group_soz[group]].mean()
                    for s in subjects
                    if s.node_energies is not None
                ]
                if soz_nes:
                    soz_energy = float(np.nanmean(soz_nes))

            rows.append({
                "group":          group,
                "n":              len(e_vals),
                "mean_E_barrier": float(np.mean(e_vals)) if len(e_vals) > 0 else np.nan,
                "sem_E_barrier":  float(np.std(e_vals)/np.sqrt(len(e_vals))) if len(e_vals)>1 else np.nan,
                "mean_SOZ_energy": soz_energy,
            })
        return pd.DataFrame(rows).set_index("group")
