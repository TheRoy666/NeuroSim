"""
Coupled FitzHugh-Nagumo (FHN) network -- the Phase 6 robustness check for
Path A1/A2. Same public interface as simulation.WilsonCowanNetwork
(find_fixed_point, jacobian_at, simulate, simulate_controlled) so it plugs
directly into the existing, verified sweep machinery
(wc_linear_validity_sweep.py's logic, physics.py's minimum_energy_trajectory
and discretize_system) with no changes needed there.

Model (classic 2-variable relaxation-oscillator form, directed coupling
through the fast/voltage-like variable v, analogous to WC's C @ E term):

    dv_i/dt = v_i - v_i^3/3 - w_i + I_ext + kappa * (C @ v)_i
    dw_i/dt = eps * (v_i + a - b*w_i)

Bifurcation parameter: I_ext (external bias current), the standard,
textbook choice for the FHN Hopf bifurcation -- fixed a, b, eps at
well-known reference values (a=0.7, b=0.8, eps=0.08), consistent with
how the WC sweep fixed all node parameters except the swept axis (w_EE).

The exact boundary location is NOT assumed from memory -- verify
numerically via the same continuation method used for WC, same discipline
as the rest of this project.
"""
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import root


FHN_DEFAULT_PARAMS = dict(a=0.7, b=0.8, eps=0.08, kappa=0.1)


class FitzHughNagumoNetwork:
    """Coupled multi-region FitzHugh-Nagumo network.

    Mirrors WilsonCowanNetwork's public interface exactly:
    find_fixed_point, jacobian_at, simulate, simulate_controlled.

    Parameters
    ----------
    n_regions : int
    C : (N,N) ndarray -- coupling matrix (directed, analogous to WC's C).
    I_ext : float -- external bias current, the bifurcation parameter
        (analogous to WC's w_EE in the sense of "the swept axis").
    node_params : dict, optional -- a, b, eps, kappa. Defaults to
        FHN_DEFAULT_PARAMS if not given.
    """

    def __init__(
        self,
        n_regions: int,
        C: NDArray,
        I_ext: float = 0.0,
        node_params: Optional[Dict] = None,
    ):
        self.n_regions = n_regions
        self.C = np.asarray(C, dtype=float)
        self.I_ext = I_ext
        self.params = node_params or FHN_DEFAULT_PARAMS.copy()

    def _ode_network(self, t: float, y: NDArray, u_func: Optional[Callable] = None) -> NDArray:
        """y = [v_0,...,v_{N-1}, w_0,...,w_{N-1}].

        u_func(t) -> (N,) ndarray, external control current injected into
        each region's v equation -- same role as WC's u_func hook, used
        by Path A validation to inject the linear-optimal control
        trajectory and check real reachability.
        """
        N = self.n_regions
        v = y[:N]
        w = y[N:]
        p = self.params

        coupling = p["kappa"] * (self.C @ v)
        u = u_func(t) if u_func is not None else 0.0

        dv = v - (v**3) / 3.0 - w + self.I_ext + coupling + u
        dw = p["eps"] * (v + p["a"] - p["b"] * w)

        return np.concatenate([dv, dw])

    def simulate(
        self,
        t_span: Tuple[float, float] = (0.0, 500.0),
        n_points: int = 5000,
        v0: Optional[NDArray] = None,
        w0: Optional[NDArray] = None,
        seed: int = 42,
    ) -> Dict[str, NDArray]:
        N = self.n_regions
        rng = np.random.default_rng(seed)
        if v0 is None:
            v0 = rng.uniform(-0.5, 0.5, N)
        if w0 is None:
            w0 = rng.uniform(-0.5, 0.5, N)

        y0 = np.concatenate([v0, w0])
        t_eval = np.linspace(*t_span, n_points)
        sol = solve_ivp(self._ode_network, t_span, y0, t_eval=t_eval,
                         method="RK45", rtol=1e-6, atol=1e-8)
        return {"t": sol.t, "v": sol.y[:N], "w": sol.y[N:]}

    def simulate_controlled(
        self,
        u_func: Callable[[float], NDArray],
        t_span: Tuple[float, float],
        n_points: int,
        E0: NDArray,  # named E0/I0 to match WC's call signature exactly
        I0: NDArray,  # (v0, w0) here -- same positional role
    ) -> Dict[str, NDArray]:
        """Same signature as WilsonCowanNetwork.simulate_controlled (E0/I0
        naming kept for drop-in compatibility with the existing sweep
        script's calling convention -- here they are (v0, w0))."""
        N = self.n_regions
        v0, w0 = E0, I0
        y0 = np.concatenate([np.asarray(v0, dtype=float), np.asarray(w0, dtype=float)])
        t_eval = np.linspace(*t_span, n_points)
        sol = solve_ivp(self._ode_network, t_span, y0, t_eval=t_eval,
                         args=(u_func,), method="RK45", rtol=1e-6, atol=1e-8)
        return {"t": sol.t, "E": sol.y[:N], "I": sol.y[N:]}  # E/I keys for drop-in reuse

    def find_fixed_point(
        self,
        v0_guess: Optional[NDArray] = None,
        w0_guess: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray]:
        N = self.n_regions
        if v0_guess is None:
            v0_guess = np.full(N, -1.0)  # near the classic FHN resting point
        if w0_guess is None:
            w0_guess = np.full(N, -0.5)
        y0 = np.concatenate([v0_guess, w0_guess])

        result = root(lambda y: self._ode_network(0.0, y), y0, method="hybr", tol=1e-12)
        if not result.success:
            raise RuntimeError(
                "Fixed-point solver did not converge. If I_ext is close to "
                "the Hopf boundary, this regime may genuinely lack a stable "
                "fixed point (limit cycle instead)."
            )
        v_star, w_star = result.x[:N], result.x[N:]
        return v_star, w_star

    def jacobian_at(self, v_star: NDArray, w_star: NDArray) -> NDArray:
        """Exact analytic Jacobian at (v*, w*). Block form:

            J = [[diag(1 - v*^2) + kappa*C,   -I],
                 [eps*I,                      -eps*b*I]]
        """
        N = self.n_regions
        p = self.params
        eye = np.eye(N)

        J_vv = np.diag(1 - v_star**2) + p["kappa"] * self.C
        J_vw = -eye
        J_wv = p["eps"] * eye
        J_ww = -p["eps"] * p["b"] * eye

        return np.block([[J_vv, J_vw], [J_wv, J_ww]])
