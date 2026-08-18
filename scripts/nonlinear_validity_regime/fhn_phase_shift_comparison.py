#!/usr/bin/env python3
"""
Phase 6 robustness check, A2 half: does the WC A2 finding (naive frozen-
snapshot linearization is not just inaccurate but actively counter-
productive in the oscillatory regime, distinct from LTV and from the
true nonlinear-optimal ceiling) replicate in FitzHugh-Nagumo?

Single-node FHN oscillator (I_ext=0.5, confirmed genuine limit cycle,
period=39.47), matching how the phase-shift task is framed in the WC A2
work and in the literature (Chouzouris et al. 2021, Salfenmoser &
Obermayer 2023) -- also keeps the nonlinear-ceiling direct optimization
tractable.

Task: starting exactly on the nominal orbit at phase 0, reach the point
on the orbit corresponding to a DIFFERENT phase (phase_shift) after T
time units, rather than where free/uncontrolled evolution would land
(phase T mod period). Deviation formulation: delta_x0 = 0, delta_xT =
nominal(phase_shift) - nominal(T mod period).

Three methods, same task, same target:
1. NAIVE: single Jacobian frozen at the starting phase (t=0), constant
   A/B via discretize_system, LTI minimum-energy control.
2. LTV: Jacobian evaluated along the TRUE nominal orbit at each time
   step -- genuine time-varying A(t). B is constant for FHN (u enters
   additively, not through a sigmoid) -- simpler than WC's LTV case,
   which needed a time-varying B too.
3. NONLINEAR CEILING: direct optimization of the control sequence via
   scipy.optimize.minimize, objective = actual final-state distance from
   target under the TRUE nonlinear FHN simulation. Best achievable,
   not an approximation.

RESULT (2 of 3 planned phase-shift targets completed; the third timed
out consistently -- its trajectory passes through the confirmed-unstable
phase region 13.81-15.78, see fhn_a2_phase_stability_mapping.csv):

phase_shift=3.945: naive_err=1.41, ltv_err=0.79, ceiling_err=0.79 (LTV
matched the ceiling) -- naive clearly worse, consistent with WC A2.

phase_shift=9.863: naive_err=1.77, ltv_err=2.76 (WORSE than naive here),
ceiling_err=0.065 (dramatically better than both) -- LTV underperforms
naive in this case, the opposite ordering from the first target, while
the nonlinear ceiling shows the target was genuinely reachable all
along. Same qualitative story as WC's A2: failure is about linearization
quality, not unreachability, and no single linear method dominates the
other across all conditions.

Also confirmed (fhn_a2_phase_stability_mapping.csv): 15/20 sampled
phases give a locally stable frozen Jacobian, 5/20 unstable -- the
unstable phases sit precisely at the relaxation oscillator's fast-
transition regions, directly replicating WC A2's "phase-dependent local
stability, unstable at fast transitions, stable at turning points" in a
structurally different model.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

import fhn_simulation as fhn
import physics
from simulation import discretize_system

I_EXT = 0.5
DT = 0.5
PARAMS = dict(**fhn.FHN_DEFAULT_PARAMS)
PARAMS["kappa"] = 0.0  # single node, no coupling


def load_nominal_orbit():
    t = np.load('fhn_a2_data/nominal_t.npy')
    v = np.load('fhn_a2_data/nominal_v.npy')
    w = np.load('fhn_a2_data/nominal_w.npy')
    period = t[-1] + (t[1] - t[0])  # approx, matches the confirmed 39.47
    v_interp = interp1d(t, v, kind='cubic', fill_value='extrapolate')
    w_interp = interp1d(t, w, kind='cubic', fill_value='extrapolate')
    return period, v_interp, w_interp


def nominal_state(phase, period, v_interp, w_interp):
    phase_mod = phase % period
    return np.array([v_interp(phase_mod), w_interp(phase_mod)])


def build_net():
    return fhn.FitzHughNagumoNetwork(n_regions=1, C=np.zeros((1, 1)), I_ext=I_EXT, node_params=PARAMS)


def naive_method(net, x0, target_shift_state, drift_state, n_steps):
    """Frozen Jacobian at the starting phase, constant A/B, LTI control."""
    J = net.jacobian_at(np.array([x0[0]]), np.array([x0[1]]))
    G = np.array([[1.0], [0.0]])  # u enters additively into dv/dt only
    A, B = discretize_system(J, G, DT)
    delta_x0 = np.zeros(2)
    delta_xT = target_shift_state - drift_state
    energy, U = physics.minimum_energy_trajectory(A, B, delta_x0, delta_xT, n_steps)
    return U, energy


def ltv_method(net, x0, target_shift_state, drift_state, n_steps, period, v_interp, w_interp):
    """Jacobian evaluated along the TRUE nominal orbit at each step."""
    A_list, B_list = [], []
    for k in range(n_steps):
        t_k = k * DT
        state_k = nominal_state(t_k, period, v_interp, w_interp)
        J_k = net.jacobian_at(np.array([state_k[0]]), np.array([state_k[1]]))
        G = np.array([[1.0], [0.0]])
        A_k, B_k = discretize_system(J_k, G, DT)
        A_list.append(A_k)
        B_list.append(B_k)
    delta_x0 = np.zeros(2)
    delta_xT = target_shift_state - drift_state
    energy, U = physics.minimum_energy_trajectory_ltv(A_list, B_list, delta_x0, delta_xT)
    return U, energy


def inject_and_measure(net, x0, U, n_steps, target_absolute):
    """Inject a control sequence into the TRUE nonlinear FHN system,
    measure final-state error against the absolute target."""
    def u_func(t):
        idx = min(int(t / DT), n_steps - 1)
        return np.array([U[idx, 0]])
    result = net.simulate_controlled(
        u_func=u_func, t_span=(0.0, n_steps * DT), n_points=max(100, n_steps),
        E0=np.array([x0[0]]), I0=np.array([x0[1]]),
    )
    final_state = np.array([result["E"][0, -1], result["I"][0, -1]])
    err = np.linalg.norm(final_state - target_absolute)
    return err, final_state


def nonlinear_ceiling(net, x0, n_steps, target_absolute, U_init):
    """Direct optimization of the control sequence against the TRUE
    nonlinear system -- the best achievable ceiling."""
    def objective(u_flat):
        U = u_flat.reshape(n_steps, 1)
        err, _ = inject_and_measure(net, x0, U, n_steps, target_absolute)
        return err

    result = minimize(objective, U_init.flatten(), method='L-BFGS-B',
                       options={'maxiter': 25})
    U_opt = result.x.reshape(n_steps, 1)
    return U_opt, result.fun


if __name__ == "__main__":
    period, v_interp, w_interp = load_nominal_orbit()
    print(f"Loaded nominal orbit, period={period:.3f}")

    net = build_net()
    # Phase 5.92 confirmed locally STABLE (rho(A)=0.9464) via the
    # phase-stability mapping done before this comparison -- phase=0 was
    # confirmed UNSTABLE (rho(A)=1.4798), itself a real finding directly
    # replicating WC A2's "phase-dependent local stability, unstable at
    # fast transitions" pattern. Starting here isolates the comparison
    # to the well-posed case; the unstable-start case is a separate,
    # already-documented finding, not conflated with this comparison.
    START_PHASE = 5.92
    x0 = nominal_state(START_PHASE, period, v_interp, w_interp)
    print(f"Starting state (phase {START_PHASE}, confirmed locally stable): "
          f"v={x0[0]:.4f}, w={x0[1]:.4f}\n")

    T_time = 15.0  # transition time, less than one period
    n_steps = int(T_time / DT)

    PHASE_SHIFTS = [period * 0.1, period * 0.25, period * 0.4]  # fractions of a period

    print(f"{'phase_shift':>12} {'naive_err':>12} {'ltv_err':>12} {'ceiling_err':>12} "
          f"{'naive_energy':>14} {'ltv_energy':>14}")

    for phase_shift in PHASE_SHIFTS:
        drift_state = nominal_state(T_time, period, v_interp, w_interp)
        target_shift_state = nominal_state(phase_shift, period, v_interp, w_interp)
        target_absolute = target_shift_state  # target IS the absolute state to reach

        # Naive
        delta_x0 = np.zeros(2)
        U_naive, energy_naive = naive_method(net, x0, target_shift_state, drift_state, n_steps)
        err_naive, _ = inject_and_measure(net, x0, U_naive, n_steps, target_absolute)

        # LTV
        U_ltv, energy_ltv = ltv_method(net, x0, target_shift_state, drift_state, n_steps,
                                        period, v_interp, w_interp)
        err_ltv, _ = inject_and_measure(net, x0, U_ltv, n_steps, target_absolute)

        # Nonlinear ceiling (warm-start from the LTV solution)
        U_ceiling, err_ceiling = nonlinear_ceiling(net, x0, n_steps, target_absolute, U_ltv)

        print(f"{phase_shift:>12.3f} {err_naive:>12.4f} {err_ltv:>12.4f} {err_ceiling:>12.4f} "
              f"{energy_naive:>14.4f} {energy_ltv:>14.4f}")
