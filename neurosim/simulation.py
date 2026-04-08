"""
neurosim.simulation
===================
Non-linear ground-truth benchmarking via the Wilson-Cowan neural mass model.

Purpose
-------
The Linear Time-Invariant (LTI) assumption at the heart of NCT is a
mathematical convenience — neural dynamics are inherently non-linear. To
assess the validity and error bounds of NeuroSim's linear control energy
metrics, we benchmark them against a biologically realistic non-linear
simulator: the **Wilson-Cowan** neural mass model (Wilson & Cowan, 1972).

The Wilson-Cowan model describes the mean firing rates of coupled excitatory
(E) and inhibitory (I) neural populations via:

    τ_E · dE/dt = -E + S( w_EE·E - w_IE·I + c_E + P_E(t) )
    τ_I · dI/dt = -I + S( w_EI·E - w_II·I + c_I )

where S(x) = 1/(1 + exp(-x)) is a sigmoidal transfer function.

Critically, for specific parameter regimes (w_EE ≈ 10, w_IE ≈ 12,
w_EI ≈ 8, w_II ≈ 3, c_E ≈ -2, c_I ≈ -3.5), the system settles into
a **stable limit cycle** — periodic oscillations analogous to neural
gamma oscillations (~40 Hz). This provides a rigorous, oscillatory
ground truth against which we can quantify the error of our linearised
NCT metrics.

Validation Protocol
-------------------
1. Simulate the Wilson-Cowan model for N coupled brain regions.
2. Extract the limit-cycle trajectory as ground-truth state sequences.
3. Compute minimum control energy using NeuroSim's finite-horizon engine.
4. Compute a "non-linear correction factor" as the ratio of actual energy
   required in WC dynamics vs. LTI prediction.

References
----------
Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions
    in localized populations of model neurons. Biophysical Journal, 12(1), 1–24.
Breakspear, M. (2017). Dynamic models of large-scale brain activity.
    Nature Neuroscience, 20(3), 340–352.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


# Sigmoid transfer function

def _sigmoid(x: NDArray, a: float = 1.0, theta: float = 0.0) -> NDArray:
    """Generalised sigmoid: S(x) = 1 / (1 + exp(-a(x - θ)))."""
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))


# Wilson-Cowan single-node model

class WilsonCowanNode:
    """Single Wilson-Cowan excitatory-inhibitory unit.

    Parameters
    ----------
    w_EE : float – E→E synaptic weight (recurrent excitation).
    w_IE : float – I→E synaptic weight (inhibitory to excitatory).
    w_EI : float – E→I synaptic weight (excitatory to inhibitory).
    w_II : float – I→I synaptic weight (recurrent inhibition).
    c_E  : float – Excitatory bias current.
    c_I  : float – Inhibitory bias current.
    tau_E : float – Excitatory time constant (ms).
    tau_I : float – Inhibitory time constant (ms).
    """

    # Default parameters producing stable limit cycles (gamma oscillations)
    LIMIT_CYCLE_PARAMS = dict(
        w_EE=10.0, w_IE=12.0, w_EI=8.0, w_II=3.0,
        c_E=-2.0, c_I=-3.5, tau_E=10.0, tau_I=20.0,
    )

    def __init__(
        self,
        w_EE: float = 10.0,
        w_IE: float = 12.0,
        w_EI: float = 8.0,
        w_II: float = 3.0,
        c_E:  float = -2.0,
        c_I:  float = -3.5,
        tau_E: float = 10.0,
        tau_I: float = 20.0,
    ):
        self.w_EE, self.w_IE = w_EE, w_IE
        self.w_EI, self.w_II = w_EI, w_II
        self.c_E,  self.c_I  = c_E,  c_I
        self.tau_E, self.tau_I = tau_E, tau_I

    def _ode(self, t: float, y: NDArray, P_ext: float = 0.0) -> NDArray:
        E, I = y
        dE = (-E + _sigmoid(self.w_EE*E - self.w_IE*I + self.c_E + P_ext)) / self.tau_E
        dI = (-I + _sigmoid(self.w_EI*E - self.w_II*I + self.c_I)) / self.tau_I
        return np.array([dE, dI])

    def simulate(
        self,
        t_span: Tuple[float, float] = (0.0, 1000.0),
        n_points: int = 10000,
        E0: float = 0.5,
        I0: float = 0.5,
        P_ext: float = 0.0,
    ) -> Dict[str, NDArray]:
        """Simulate single-node Wilson-Cowan dynamics.

        Parameters
        ----------
        t_span   : (t_start, t_end) in ms.
        n_points : Number of output time points.
        E0, I0   : Initial conditions for E and I populations.
        P_ext    : External input current to E population.

        Returns
        -------
        dict with keys ``"t"``, ``"E"``, ``"I"``.
        """
        t_eval = np.linspace(*t_span, n_points)
        sol = solve_ivp(
            self._ode,
            t_span,
            [E0, I0],
            t_eval=t_eval,
            args=(P_ext,),
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        return {"t": sol.t, "E": sol.y[0], "I": sol.y[1]}


# Coupled multi-region Wilson-Cowan network

class WilsonCowanNetwork:
    """Coupled multi-region Wilson-Cowan neural mass model.

    Extends the single-node model to N coupled brain regions. Coupling is
    mediated by the effective connectivity matrix C, where C[i,j] represents
    the excitatory drive from region j to region i's E population.

    Parameters
    ----------
    n_regions : int        – Number of brain regions.
    C         : (N,N) ndarray – Inter-regional coupling matrix.
    node_params : dict, optional – Shared node parameters.
    """

    def __init__(
        self,
        n_regions: int,
        C: NDArray,
        node_params: Optional[Dict] = None,
    ):
        self.n_regions = n_regions
        self.C = np.asarray(C, dtype=float)
        self.params = node_params or WilsonCowanNode.LIMIT_CYCLE_PARAMS.copy()

    def _ode_network(self, t: float, y: NDArray) -> NDArray:
        """ODE for the coupled network. y = [E_0,...,E_{N-1}, I_0,...,I_{N-1}]."""
        N = self.n_regions
        E = y[:N]
        I = y[N:]
        p = self.params

        coupling = self.C @ E  # (N,) – net excitatory input from other regions

        dE = (-E + _sigmoid(p["w_EE"]*E - p["w_IE"]*I + p["c_E"] + coupling)) / p["tau_E"]
        dI = (-I + _sigmoid(p["w_EI"]*E - p["w_II"]*I + p["c_I"])) / p["tau_I"]

        return np.concatenate([dE, dI])

    def simulate(
        self,
        t_span: Tuple[float, float] = (0.0, 2000.0),
        n_points: int = 20000,
        E0: Optional[NDArray] = None,
        I0: Optional[NDArray] = None,
        seed: int = 42,
    ) -> Dict[str, NDArray]:
        """Simulate the coupled network dynamics.

        Parameters
        ----------
        t_span   : (t_start, t_end) in ms.
        n_points : Output resolution.
        E0, I0   : Initial conditions (N,). Random if not provided.
        seed     : RNG seed for initial conditions.

        Returns
        -------
        dict with keys ``"t"``, ``"E"`` (N×T), ``"I"`` (N×T).
        """
        N = self.n_regions
        rng = np.random.default_rng(seed)

        if E0 is None:
            E0 = rng.uniform(0.1, 0.6, N)
        if I0 is None:
            I0 = rng.uniform(0.1, 0.6, N)

        y0 = np.concatenate([E0, I0])
        t_eval = np.linspace(*t_span, n_points)

        sol = solve_ivp(
            self._ode_network,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        return {
            "t": sol.t,
            "E": sol.y[:N],
            "I": sol.y[N:],
        }

    def extract_bold_proxy(self, sim_result: Dict, tr_ms: float = 720.0) -> NDArray:
        """Downsample excitatory population to BOLD-like TR resolution.

        The excitatory population E(t) serves as a proxy for the haemodynamic
        BOLD signal. Downsampling to TR resolution allows direct comparison
        with fMRI-derived control energy estimates.

        Parameters
        ----------
        sim_result : dict from ``.simulate()``.
        tr_ms      : Repetition time in ms (default 720 ms = 0.72 s).

        Returns
        -------
        E_bold : (N_regions, T_bold) ndarray – TR-sampled excitatory activity.
        """
        t = sim_result["t"]
        E = sim_result["E"]
        t_max  = t[-1]
        t_bold = np.arange(0, t_max, tr_ms)
        E_bold = np.stack([
            np.interp(t_bold, t, E[i]) for i in range(E.shape[0])
        ])
        return E_bold
