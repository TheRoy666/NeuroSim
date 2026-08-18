#!/usr/bin/env python3
"""
WC A2 broader grid -- the direct completion of
wc_oscillatory_linearization_comparison.py's own designed grid (8 seeds
x 5 shifts), now computing the nonlinear ceiling for EVERY combo via the
verified adjoint method, not just 2 representative cases as the original
script's ceiling was too expensive (coarse Powell-based direct
optimization) to run more broadly.

Reuses the original script's verified network setup, reference-
trajectory generation, phase-stability finding, and naive/LTV legs
unchanged -- only the ceiling computation is upgraded.
"""
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import simulation
import physics
import wc_oscillatory_linearization_comparison as wc_a2
from adjoint_gradient import compute_adjoint_gradient
from adjoint_gradient_wc_verification import make_wc_jacobian_func, make_wc_input_jacobian_func

N = 10
DT = 2.0
T_STEPS = 25
PERIOD = 99.96
SEEDS = list(range(8))
SHIFTS_MS = [10, 20, 30, 40, 60]


def run_one_combo(seed, shift_ms, net, t_arr, E_arr, I_arr, t_ref, phase_report):
    E_ref, I_ref = wc_a2.state_at_time(t_arr, E_arr, I_arr, t_ref)
    x_ref = np.concatenate([E_ref, I_ref])
    T_ms = T_STEPS * DT

    E_target, I_target = wc_a2.state_at_time(t_arr, E_arr, I_arr, t_ref + T_ms + shift_ms)
    x_target = np.concatenate([E_target, I_target])
    E_natural, I_natural = wc_a2.state_at_time(t_arr, E_arr, I_arr, t_ref + T_ms)
    x_natural = np.concatenate([E_natural, I_natural])
    target_distance = np.linalg.norm(x_target - x_ref)

    # Naive (unchanged from the original script's verified logic)
    J = net.jacobian_at(E_ref, I_ref)
    G = simulation.input_jacobian_at(net, E_ref, I_ref)
    A, B = simulation.discretize_system(J, G, DT)
    delta_xT_naive = x_target - x_ref
    _, U_naive = physics.minimum_energy_trajectory(A, B, np.zeros(2*N), delta_xT_naive, T_STEPS)
    u_func_naive = physics.zero_order_hold(U_naive, DT)
    ctrl_naive = net.simulate_controlled(u_func=u_func_naive, t_span=(0.0, T_ms),
                                          n_points=max(100, T_STEPS), E0=E_ref, I0=I_ref)
    x_final_naive = np.concatenate([ctrl_naive["E"][:, -1], ctrl_naive["I"][:, -1]])
    err_naive = np.linalg.norm(x_final_naive - x_target) / target_distance

    # LTV (unchanged)
    _, x_final_ltv = wc_a2.run_ltv_control(net, t_arr, E_arr, I_arr, t_ref, T_STEPS,
                                            x_target, x_natural, E_ref, I_ref)
    err_ltv = np.linalg.norm(x_final_ltv - x_target) / target_distance

    # Ceiling via verified adjoint method (the actual upgrade)
    wc_jacobian_func = make_wc_jacobian_func(net)
    wc_input_jacobian_func = make_wc_input_jacobian_func(net)

    def objective_with_grad(u_flat):
        U = u_flat.reshape(T_STEPS, N)
        cost, grad, _ = compute_adjoint_gradient(
            net, x_ref, x_target, U, DT, wc_jacobian_func, wc_input_jacobian_func,
        )
        return cost, grad.flatten()

    result = minimize(objective_with_grad, U_naive.flatten(), jac=True,
                       method='L-BFGS-B', options={'maxiter': 100})
    U_ceiling = result.x.reshape(T_STEPS, N)
    u_func_ceiling = physics.zero_order_hold(U_ceiling, DT)
    ctrl_ceiling = net.simulate_controlled(u_func=u_func_ceiling, t_span=(0.0, T_ms),
                                            n_points=max(100, T_STEPS), E0=E_ref, I0=I_ref)
    x_final_ceiling = np.concatenate([ctrl_ceiling["E"][:, -1], ctrl_ceiling["I"][:, -1]])
    err_ceiling = np.linalg.norm(x_final_ceiling - x_target) / target_distance

    return {
        "seed": seed, "shift_ms": shift_ms, "target_distance": target_distance,
        "rel_err_naive": err_naive, "rel_err_ltv": err_ltv,
        "rel_err_ceiling": err_ceiling,
        "rel_err_do_nothing": np.linalg.norm(x_natural - x_target) / target_distance,
        "ceiling_n_evals": result.nfev,
    }


if __name__ == "__main__":
    print(f"{len(SEEDS)} seeds x {len(SHIFTS_MS)} shifts = {len(SEEDS)*len(SHIFTS_MS)} combos\n")

    t_start = time.time()
    rows = []
    for seed in SEEDS:
        net = wc_a2.build_network(seed)
        t_arr, E_arr, I_arr = wc_a2.get_reference_trajectory(net, seed)
        t_ref, phase_report = wc_a2.find_stable_reference_phase(net, t_arr, E_arr, I_arr, PERIOD)
        if t_ref is None:
            print(f"seed={seed}: no stable phase found, skipping")
            continue

        for shift_ms in SHIFTS_MS:
            row = run_one_combo(seed, shift_ms, net, t_arr, E_arr, I_arr, t_ref, phase_report)
            rows.append(row)
            print(f"  seed={seed} shift_ms={shift_ms}: naive={row['rel_err_naive']:.1%} "
                  f"ltv={row['rel_err_ltv']:.1%} ceiling={row['rel_err_ceiling']:.1%} "
                  f"do_nothing={row['rel_err_do_nothing']:.1%} "
                  f"[{time.time()-t_start:.0f}s elapsed]", flush=True)

    elapsed = time.time() - t_start
    df = pd.DataFrame(rows)
    df.to_csv("wc_broader_grid_results.csv", index=False)
    print(f"\nComplete: {len(df)} combos in {elapsed:.1f}s")
    print(f"Saved to wc_broader_grid_results.csv")
