#!/usr/bin/env python3
"""
A2 broader grid, made tractable by the adjoint-gradient method (the
priority engineering task that gated this). Mirrors Path A1's
statistical rigor (multiple starting conditions x multiple targets)
rather than the single-case-strength result the original A2 work was
limited to before this.

8 starting phases (deliberate mix, using the confirmed phase-stability
mapping in fhn_a2_phase_stability_mapping.csv, not arbitrary): 5
confirmed locally-stable (1.97, 5.92, 11.84, 21.70, 29.59) + 3 confirmed
locally-unstable (0.0, 13.81, 35.51) -- mirrors Path A1's 8-seed
convention, and deliberately tests whether starting from an unstable
point causes systematic problems, not just an aside.

5 phase-shift targets: 0.1, 0.2, 0.3, 0.4, 0.5 x period.

For unstable starting phases, the naive method's frozen-LTI Gramian is
not computable (confirmed earlier at phase=0) -- recorded as
"naive_undefined", not silently skipped or crashed on.
"""
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize

import fhn_simulation as fhn
import physics
from simulation import discretize_system
from adjoint_gradient import compute_adjoint_gradient

I_EXT = 0.5
DT = 0.5
T_TIME = 15.0
N_STEPS = int(T_TIME / DT)
PARAMS = dict(**fhn.FHN_DEFAULT_PARAMS)
PARAMS["kappa"] = 0.0

STABLE_PHASES = [1.972697269726973, 5.918091809180918, 11.836183618361837,
                 21.6996699669967, 29.590459045904595]
UNSTABLE_PHASES = [0.0, 13.80888088808881, 35.50855085508551]
START_PHASES = STABLE_PHASES + UNSTABLE_PHASES

PHASE_SHIFT_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]

G_CONST = np.array([[1.0], [0.0]])


def load_nominal_orbit():
    t = np.load('fhn_a2_data/nominal_t.npy')
    v = np.load('fhn_a2_data/nominal_v.npy')
    w = np.load('fhn_a2_data/nominal_w.npy')
    period = t[-1] + (t[1] - t[0])
    v_interp = interp1d(t, v, kind='cubic', fill_value='extrapolate')
    w_interp = interp1d(t, w, kind='cubic', fill_value='extrapolate')
    return period, v_interp, w_interp


def nominal_state(phase, period, v_interp, w_interp):
    phase_mod = phase % period
    return np.array([v_interp(phase_mod), w_interp(phase_mod)])


def inject_and_measure(net, x0, U, n_steps, target_absolute):
    def u_func(t):
        idx = min(int(t / DT), n_steps - 1)
        return np.array([U[idx, 0]])
    result = net.simulate_controlled(
        u_func=u_func, t_span=(0.0, n_steps * DT), n_points=max(100, n_steps),
        E0=np.array([x0[0]]), I0=np.array([x0[1]]),
    )
    final_state = np.array([result["E"][0, -1], result["I"][0, -1]])
    return np.linalg.norm(final_state - target_absolute), final_state


def run_one_combo(net, start_phase, phase_shift, period, v_interp, w_interp,
                   is_stable_start):
    x0 = nominal_state(start_phase, period, v_interp, w_interp)
    drift_state = nominal_state(start_phase + T_TIME, period, v_interp, w_interp)
    target_shift_state = nominal_state(start_phase + phase_shift, period, v_interp, w_interp)
    target_absolute = target_shift_state
    delta_x0 = np.zeros(2)
    delta_xT = target_shift_state - drift_state

    row = {"start_phase": start_phase, "phase_shift": phase_shift,
           "is_stable_start": is_stable_start}

    # Naive -- only computable if the frozen Jacobian at x0 is stable
    if is_stable_start:
        J0 = net.jacobian_at(np.array([x0[0]]), np.array([x0[1]]))
        A, B = discretize_system(J0, G_CONST, DT)
        rho_A = np.max(np.abs(np.linalg.eigvals(A)))
        if rho_A < 1:
            energy_naive, U_naive = physics.minimum_energy_trajectory(
                A, B, delta_x0, delta_xT, N_STEPS)
            err_naive, _ = inject_and_measure(net, x0, U_naive, N_STEPS, target_absolute)
            row["naive_err"] = err_naive
            row["naive_energy"] = energy_naive
        else:
            row["naive_err"] = None
            row["naive_energy"] = None
            row["naive_undefined_reason"] = f"frozen A unstable (rho={rho_A:.3f})"
    else:
        row["naive_err"] = None
        row["naive_energy"] = None
        row["naive_undefined_reason"] = "unstable starting phase"

    # LTV -- always computable, uses the true trajectory's Jacobian at each step
    A_list, B_list = [], []
    for k in range(N_STEPS):
        state_k = nominal_state(start_phase + k * DT, period, v_interp, w_interp)
        J_k = net.jacobian_at(np.array([state_k[0]]), np.array([state_k[1]]))
        A_k, B_k = discretize_system(J_k, G_CONST, DT)
        A_list.append(A_k); B_list.append(B_k)
    energy_ltv, U_ltv = physics.minimum_energy_trajectory_ltv(
        A_list, B_list, delta_x0, delta_xT)
    err_ltv, _ = inject_and_measure(net, x0, U_ltv, N_STEPS, target_absolute)
    row["ltv_err"] = err_ltv
    row["ltv_energy"] = energy_ltv

    # Nonlinear ceiling via adjoint gradient
    def fhn_jacobian_func(x, u=None):
        return net.jacobian_at(x[0:1], x[1:2])
    def fhn_input_jacobian_func(x, u=None):
        return G_CONST
    def objective_with_grad(u_flat):
        U = u_flat.reshape(N_STEPS, 1)
        cost, grad, _ = compute_adjoint_gradient(
            net, x0, target_absolute, U, DT, fhn_jacobian_func,
            fhn_input_jacobian_func,
        )
        return cost, grad.flatten()

    result = minimize(objective_with_grad, U_ltv.flatten(), jac=True,
                       method='L-BFGS-B', options={'maxiter': 100})
    U_ceiling = result.x.reshape(N_STEPS, 1)
    err_ceiling, _ = inject_and_measure(net, x0, U_ceiling, N_STEPS, target_absolute)
    row["ceiling_err"] = err_ceiling
    row["ceiling_n_evals"] = result.nfev

    return row


if __name__ == "__main__":
    period, v_interp, w_interp = load_nominal_orbit()
    net = fhn.FitzHughNagumoNetwork(n_regions=1, C=np.zeros((1, 1)),
                                     I_ext=I_EXT, node_params=PARAMS)
    print(f"Period={period:.3f}, {len(START_PHASES)} starting phases x "
          f"{len(PHASE_SHIFT_FRACTIONS)} targets = "
          f"{len(START_PHASES)*len(PHASE_SHIFT_FRACTIONS)} combos\n")

    t_start = time.time()
    rows = []
    for start_phase in START_PHASES:
        is_stable = start_phase in STABLE_PHASES
        for frac in PHASE_SHIFT_FRACTIONS:
            phase_shift = frac * period
            row = run_one_combo(net, start_phase, phase_shift, period,
                                 v_interp, w_interp, is_stable)
            rows.append(row)
            print(f"  start_phase={start_phase:.2f} ({'stable' if is_stable else 'UNSTABLE'}), "
                  f"shift_frac={frac}: naive={row.get('naive_err')}, "
                  f"ltv={row['ltv_err']:.4f}, ceiling={row['ceiling_err']:.4f} "
                  f"[{time.time()-t_start:.0f}s elapsed]", flush=True)

    elapsed = time.time() - t_start
    df = pd.DataFrame(rows)
    df.to_csv("fhn_broader_grid_results.csv", index=False)
    print(f"\nComplete: {len(df)} combos in {elapsed:.1f}s")
    print(f"Saved to fhn_broader_grid_results.csv")
