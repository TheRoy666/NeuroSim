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

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import root


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


def input_jacobian_at(net: "WilsonCowanNetwork", E_star: NDArray, I_star: NDArray) -> NDArray:
    """Continuous-time input Jacobian G = df/du at the fixed point.

    Control current u enters only the E-population equation, inside the
    sigmoid (see ``_ode_network``'s u_func hook):

        dE/dt = (-E + S(w_EE E - w_IE I + c_E + coupling + u)) / tau_E

    so a unit of u does NOT map 1:1 onto a state-space push — it is scaled
    by the local sigmoid slope S'(z_E)/tau_E. Using G = [I; 0] (as if u
    were already a direct state perturbation) silently understates or
    overstates the true control effect depending on S'(z_E), and produces
    control trajectories that fail to reach the target when injected into
    the real nonlinear network. This function computes the correct G;
    ``discretize_system`` below combines it with the state Jacobian J via
    the augmented matrix-exponential method for zero-order-hold input.

    Returns
    -------
    G : (2N, N) ndarray - top block = diag(S'(z_E))/tau_E, bottom block = 0.
    """
    N = net.n_regions
    p = net.params
    z_E = p["w_EE"]*E_star - p["w_IE"]*I_star + p["c_E"] + net.C @ E_star
    s_E = _sigmoid(z_E)
    sp_E = s_E * (1 - s_E)
    G_top = np.diag(sp_E / p["tau_E"])
    G_bottom = np.zeros((N, N))
    return np.vstack([G_top, G_bottom])


def discretize_system(J: NDArray, G: NDArray, dt: float) -> Tuple[NDArray, NDArray]:
    """Jointly discretize (J, G) to (A, B) for zero-order-hold input, exact.

    Uses the standard augmented-matrix-exponential trick: for
    dx/dt = Jx + Gu held constant over [0, dt),

        expm([[J, G], [0, 0]] * dt) = [[A, B], [0, I]]

    which gives the exact discrete-time pair (A, B) such that
    x[k+1] = A x[k] + B u[k] matches the continuous dynamics under
    zero-order hold — not the approximate/wrong shortcut of reusing
    ``discretize_jacobian(J, dt)`` for A and assuming B = [I; 0].

    Parameters
    ----------
    J : (n, n) ndarray - continuous state Jacobian (from ``jacobian_at``).
    G : (n, m) ndarray - continuous input Jacobian (from ``input_jacobian_at``).
    dt : float - step size, same time units as J, G.

    Returns
    -------
    A : (n, n) ndarray - discrete state matrix.
    B : (n, m) ndarray - discrete input matrix.
    """
    n = J.shape[0]
    m = G.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = J
    M[:n, n:] = G
    M_exp = expm(M * dt)
    A = M_exp[:n, :n]
    B = M_exp[:n, n:]
    return A, B


def discretize_jacobian(J: NDArray, dt: float) -> NDArray:
    """Discretize a continuous-time Jacobian to the A matrix physics.py expects.

    x[k+1] = A x[k]  <-  A = expm(J * dt), exact for the linearized system
    (not an Euler approximation).

    Parameters
    ----------
    J  : (2N, 2N) ndarray - continuous Jacobian from ``jacobian_at``.
    dt : float - sampling interval in the same time units as J (ms, if
         J came from WilsonCowanNetwork whose tau_E/tau_I are in ms).
         For Path A1, dt = TR in ms.

    Returns
    -------
    A : (2N, 2N) ndarray - discrete-time state matrix. Spectral radius
        reflects the true local stability margin at this operating point;
        do not renormalize with ``normalise_matrix`` for A1 (see Phase 0
        plan) — target_rho there is a diagnostic, not a free rescale.
    """
    return expm(np.asarray(J, dtype=float) * dt)


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

    def _ode_network(self, t: float, y: NDArray, u_func: Optional["Callable"] = None) -> NDArray:
        """ODE for the coupled network. y = [E_0,...,E_{N-1}, I_0,...,I_{N-1}].

        Parameters
        ----------
        u_func : callable, optional
            u_func(t) -> (N,) ndarray of external control current injected
            into each region's E-population equation (same channel as the
            single-node model's ``P_ext``). This is the hook used to drive
            the network with a linear-optimal control trajectory computed
            by ``physics.py`` and check whether it actually steers the
            nonlinear system to the intended target (Path A validation).
        """
        N = self.n_regions
        E = y[:N]
        I = y[N:]
        p = self.params

        coupling = self.C @ E  # (N,) – net excitatory input from other regions
        u = u_func(t) if u_func is not None else 0.0

        dE = (-E + _sigmoid(p["w_EE"]*E - p["w_IE"]*I + p["c_E"] + coupling + u)) / p["tau_E"]
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

    def simulate_controlled(
        self,
        u_func: Callable[[float], NDArray],
        t_span: Tuple[float, float],
        n_points: int,
        E0: NDArray,
        I0: NDArray,
    ) -> Dict[str, NDArray]:
        """Simulate the network under an external control input u_func(t).

        This is the Path A1 validation hook: drive the nonlinear WC network
        with the linear-optimal control trajectory derived from
        ``physics.py`` (via zero-order-hold interpolation of u*[k] into a
        continuous u_func) and check whether the network actually reaches
        the intended target state, and at what realized energy cost.

        Parameters
        ----------
        u_func : callable, u_func(t) -> (N,) ndarray
            External control current, same channel as ``P_ext``.
        t_span, n_points : as in ``.simulate()``.
        E0, I0 : (N,) ndarray - required (no random default, since this is
            meant to be called from a known operating point / fixed point).

        Returns
        -------
        dict with keys "t", "E" (N×T), "I" (N×T).
        """
        N = self.n_regions
        y0 = np.concatenate([np.asarray(E0, dtype=float), np.asarray(I0, dtype=float)])
        t_eval = np.linspace(*t_span, n_points)

        sol = solve_ivp(
            self._ode_network,
            t_span,
            y0,
            t_eval=t_eval,
            args=(u_func,),
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        return {"t": sol.t, "E": sol.y[:N], "I": sol.y[N:]}

    def find_fixed_point(
        self,
        E0_guess: Optional[NDArray] = None,
        I0_guess: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Solve for the network's stable fixed point (E*, I*).

        Only meaningful in a non-oscillating parameter regime (i.e. not
        ``WilsonCowanNode.LIMIT_CYCLE_PARAMS``) — see Phase 0 plan, Path A1.
        Uses ``scipy.optimize.root`` on the autonomous (u=0) network ODE.

        Returns
        -------
        E_star, I_star : (N,) ndarray each.

        Raises
        ------
        RuntimeError
            If the solver does not converge — check that node_params
            actually yields a stable fixed point (not a limit cycle) at
            this coupling strength.
        """
        N = self.n_regions
        if E0_guess is None:
            E0_guess = np.full(N, 0.3)
        if I0_guess is None:
            I0_guess = np.full(N, 0.3)
        y0 = np.concatenate([E0_guess, I0_guess])

        result = root(lambda y: self._ode_network(0.0, y), y0, method="hybr",
                       tol=1e-12)
        if not result.success:
            raise RuntimeError(
                "Fixed-point solver did not converge. If node_params are "
                "close to LIMIT_CYCLE_PARAMS, this regime may genuinely "
                "lack a stable fixed point — use A2 (limit-cycle) design "
                "instead, or reduce coupling/w_EE toward a damped regime."
            )
        E_star, I_star = result.x[:N], result.x[N:]
        return E_star, I_star

    def jacobian_at(self, E_star: NDArray, I_star: NDArray) -> NDArray:
        """Exact analytic Jacobian of the full 2N-dim (E,I) system at (E*,I*).

        Continuous-time Jacobian J (units 1/ms), block form::

            J = [[J_EE, J_EI],
                 [J_IE, J_II]]

        where, with S'(z) = S(z)(1-S(z)) evaluated at each region's fixed
        point drive:

            J_EE = (diag(S'(z_E)) @ (w_EE*I + C) - I) / tau_E
            J_EI = -w_IE/tau_E * diag(S'(z_E))
            J_IE =  w_EI/tau_I * diag(S'(z_I))
            J_II = (-w_II/tau_I * diag(S'(z_I)) - I) / tau_I

        This is exact for the network as implemented — no adiabatic
        elimination of I, no approximation. Use ``discretize_jacobian`` to
        get the discrete-time A matrix physics.py expects.

        Parameters
        ----------
        E_star, I_star : (N,) ndarray - fixed point from ``find_fixed_point``.

        Returns
        -------
        J : (2N, 2N) ndarray - continuous-time Jacobian.
        """
        N = self.n_regions
        p = self.params
        eye = np.eye(N)

        z_E = p["w_EE"]*E_star - p["w_IE"]*I_star + p["c_E"] + self.C @ E_star
        z_I = p["w_EI"]*E_star - p["w_II"]*I_star + p["c_I"]

        s_E = _sigmoid(z_E)
        s_I = _sigmoid(z_I)
        sp_E = np.diag(s_E * (1 - s_E))
        sp_I = np.diag(s_I * (1 - s_I))

        J_EE = (sp_E @ (p["w_EE"]*eye + self.C) - eye) / p["tau_E"]
        J_EI = (-p["w_IE"] / p["tau_E"]) * sp_E
        J_IE = (p["w_EI"] / p["tau_I"]) * sp_I
        J_II = (-p["w_II"]*sp_I - eye) / p["tau_I"]

        return np.block([[J_EE, J_EI], [J_IE, J_II]])

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
