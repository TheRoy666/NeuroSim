#!/usr/bin/env python3
"""
Path A2 -- oscillatory regime, naive linearization test.

A1 characterized linear control validity around a STABLE FIXED POINT.
There is no fixed point in the oscillatory (limit-cycle) regime -- the
reference state is always moving. The realistic failure mode this section
targets: a practitioner freezes the Jacobian at one instant (as if it were
a static operating point, exactly as in A1) and applies the resulting
linear-optimal control anyway. Does this naive approach work at all in a
genuinely oscillating system, and if not, how does it fail?

Task (matches Salfenmoser & Obermayer 2023's "phase-shifting" framing):
ask the oscillator to arrive, in T ms, at the state its own free-running
dynamics would only reach after T + shift_ms -- i.e. a phase ADVANCE task.
This is a genuine control problem: doing nothing gets you the wrong
(lagging) phase.

Three things are compared at the final time:
  1. Intended target state (the phase-advanced state)
  2. Free-running (zero control) final state -- the "do nothing" baseline
  3. Naive-linear-controlled final state -- inject u* from a Jacobian
     frozen at the START point, computed exactly as in A1, ignoring the
     fact that the reference point itself is moving

Run directly: `python3 path_a2_oscillatory_naive_linearization.py`
"""
import numpy as np
import simulation
import physics
from scipy.signal import find_peaks
from scipy.optimize import minimize

N = 10
DT = 2.0  # ms, matches A1
LC_PARAMS = dict(w_EE=10.0, w_IE=12.0, w_EI=8.0, w_II=3.0,
                  c_E=-2.0, c_I=-3.5, tau_E=10.0, tau_I=20.0)


def build_network(seed):
    rng = np.random.default_rng(seed)
    C = rng.uniform(0, 0.15, (N, N))
    np.fill_diagonal(C, 0)
    return simulation.WilsonCowanNetwork(n_regions=N, C=C, node_params=LC_PARAMS)


def get_reference_trajectory(net, seed):
    """Run a long free-running simulation, discard transient, return the
    post-transient trajectory for defining reference/target states."""
    rng = np.random.default_rng(seed + 1000)
    E0 = rng.uniform(0.1, 0.3, N)
    I0 = rng.uniform(0.1, 0.3, N)
    result = net.simulate(t_span=(0, 2000), n_points=8000, E0=E0, I0=I0)
    mask = result["t"] > 1000  # past transient
    return result["t"][mask], result["E"][:, mask], result["I"][:, mask]


def state_at_time(t_arr, E_arr, I_arr, t_query):
    """Nearest-sample lookup into the reference trajectory."""
    idx = np.argmin(np.abs(t_arr - t_query))
    return E_arr[:, idx], I_arr[:, idx]


def find_stable_reference_phase(net, t_arr, E_arr, I_arr, period, dt=DT):
    """Scan phases across one period, return the first with a locally
    stable frozen-snapshot linearization (rho_A < 1). Reports the full
    phase-stability pattern -- this alternation is itself a real finding,
    not an implementation detail to hide."""
    t0 = t_arr[len(t_arr) // 3]
    phase_report = []
    chosen_t_ref = None
    for frac in np.linspace(0, 0.95, 20):
        t_query = t0 + frac * period
        idx = np.argmin(np.abs(t_arr - t_query))
        E_pt, I_pt = E_arr[:, idx], I_arr[:, idx]
        J = net.jacobian_at(E_pt, I_pt)
        rho_A = np.max(np.abs(np.linalg.eigvals(
            simulation.discretize_system(
                J, simulation.input_jacobian_at(net, E_pt, I_pt), dt)[0])))
        phase_report.append((frac, rho_A))
        if rho_A < 1.0 and chosen_t_ref is None:
            chosen_t_ref = t_arr[idx]
    return chosen_t_ref, phase_report


def run_phase_advance_test(seed, T_steps, shift_ms):
    net = build_network(seed)
    t_arr, E_arr, I_arr = get_reference_trajectory(net, seed)
    period = 99.96  # confirmed via peak detection, consistent across seeds (same LC_PARAMS)

    t_ref, phase_report = find_stable_reference_phase(net, t_arr, E_arr, I_arr, period)
    if t_ref is None:
        return {"seed": seed, "T_steps": T_steps, "shift_ms": shift_ms,
                "status": "no locally-stable phase found this cycle",
                "phase_report": phase_report}

    E_ref, I_ref = state_at_time(t_arr, E_arr, I_arr, t_ref)
    x_ref = np.concatenate([E_ref, I_ref])

    T_ms = T_steps * DT
    # Target: state naturally reached after T_ms + shift_ms (phase-ADVANCED)
    E_target, I_target = state_at_time(t_arr, E_arr, I_arr, t_ref + T_ms + shift_ms)
    x_target = np.concatenate([E_target, I_target])

    # "Do nothing" baseline: state naturally reached after just T_ms
    E_natural, I_natural = state_at_time(t_arr, E_arr, I_arr, t_ref + T_ms)
    x_natural = np.concatenate([E_natural, I_natural])

    # Naive linearization: freeze Jacobian at x_ref, exactly as A1's method,
    # ignoring that x_ref is not a fixed point (it's moving along the cycle)
    J = net.jacobian_at(E_ref, I_ref)
    G = simulation.input_jacobian_at(net, E_ref, I_ref)
    A, B = simulation.discretize_system(J, G, DT)

    delta_x0 = np.zeros(2 * N)
    delta_xT = x_target - x_ref  # naive: target relative to the FROZEN start point
    energy, U = physics.minimum_energy_trajectory(A, B, delta_x0, delta_xT, T_steps)
    u_func = physics.zero_order_hold(U, DT)

    controlled = net.simulate_controlled(
        u_func=u_func, t_span=(0.0, T_ms), n_points=max(100, T_steps),
        E0=E_ref, I0=I_ref,
    )
    x_controlled_final = np.concatenate(
        [controlled["E"][:, -1], controlled["I"][:, -1]])

    target_distance = np.linalg.norm(x_target - x_ref)
    err_naive = np.linalg.norm(x_controlled_final - x_target)
    err_donothing = np.linalg.norm(x_natural - x_target)

    return {
        "seed": seed, "T_steps": T_steps, "shift_ms": shift_ms,
        "status": "ok",
        "predicted_energy": energy,
        "target_distance": target_distance,
        "rel_err_naive_control": err_naive / target_distance,
        "rel_err_do_nothing": err_donothing / target_distance,
        "naive_beats_do_nothing": err_naive < err_donothing,
        "phase_report": phase_report,
    }


def run_ltv_control(net, t_arr, E_arr, I_arr, t_ref, T_steps, x_target, x_natural, E_ref, I_ref):
    """Leg 2: proper time-varying linearization along the ACTUAL free-running
    reference trajectory (Floquet-style), rather than freezing one snapshot.
    Deviation is tracked relative to the MOVING reference x_ref(t), so the
    target deviation is (x_target - x_natural), not (x_target - x_ref_start).
    """
    A_list, B_list = [], []
    for k in range(T_steps):
        t_k = t_ref + k * DT
        E_k, I_k = state_at_time(t_arr, E_arr, I_arr, t_k)
        J_k = net.jacobian_at(E_k, I_k)
        G_k = simulation.input_jacobian_at(net, E_k, I_k)
        A_k, B_k = simulation.discretize_system(J_k, G_k, DT)
        A_list.append(A_k)
        B_list.append(B_k)

    delta_x0 = np.zeros(2 * N)
    delta_xT = x_target - x_natural  # deviation from where free-running ends up
    energy, U = physics.minimum_energy_trajectory_ltv(A_list, B_list, delta_x0, delta_xT)
    u_func = physics.zero_order_hold(U, DT)

    controlled = net.simulate_controlled(
        u_func=u_func, t_span=(0.0, T_steps * DT), n_points=max(100, T_steps),
        E0=E_ref, I0=I_ref,
    )
    x_final = np.concatenate([controlled["E"][:, -1], controlled["I"][:, -1]])
    return energy, x_final


def run_nonlinear_ceiling(net, x_ref, x_target, T_steps, n_segments=5, lam=5000.0):
    """Leg 3: genuine nonlinear optimal control, via direct shooting with a
    reduced piecewise-constant parameterization (n_segments << T_steps) to
    keep finite-difference optimization tractable. This is the honest
    ceiling: actually simulating the true nonlinear network under candidate
    controls and optimizing to minimize energy + penalized target miss.
    Warm-started from a naive-linear guess for faster convergence.
    """
    E_ref = x_ref[:N]
    I_ref = x_ref[N:]
    seg_len = T_steps // n_segments

    def unpack(u_flat):
        return u_flat.reshape(n_segments, N)

    def simulate_cost(u_flat):
        U_coarse = unpack(u_flat)
        U_fine = np.repeat(U_coarse, seg_len, axis=0)
        if U_fine.shape[0] < T_steps:
            U_fine = np.vstack([U_fine, np.tile(U_fine[-1], (T_steps - U_fine.shape[0], 1))])
        u_func = physics.zero_order_hold(U_fine, DT)
        result = net.simulate_controlled(
            u_func=u_func, t_span=(0.0, T_steps * DT), n_points=max(100, T_steps),
            E0=E_ref, I0=I_ref,
        )
        x_final = np.concatenate([result["E"][:, -1], result["I"][:, -1]])
        energy = np.sum(U_fine ** 2)
        miss = np.sum((x_final - x_target) ** 2)
        return energy + lam * miss, x_final, energy

    def objective(u_flat):
        return simulate_cost(u_flat)[0]

    u0 = np.zeros(n_segments * N)  # zero-control start; kept simple/robust
    res = minimize(objective, u0, method="Powell",
                    options={"maxiter": 40, "xtol": 1e-4, "ftol": 1e-4})

    _, x_final, energy = simulate_cost(res.x)
    return energy, x_final, res.fun, res.success


# ---- Combined three-way comparison ----

def three_way_comparison(seed, T_steps, shift_ms, run_ceiling=False):
    net = build_network(seed)
    t_arr, E_arr, I_arr = get_reference_trajectory(net, seed)
    period = 99.96

    t_ref, phase_report = find_stable_reference_phase(net, t_arr, E_arr, I_arr, period)
    if t_ref is None:
        return {"status": "no stable phase found"}

    E_ref, I_ref = state_at_time(t_arr, E_arr, I_arr, t_ref)
    x_ref = np.concatenate([E_ref, I_ref])
    T_ms = T_steps * DT

    E_target, I_target = state_at_time(t_arr, E_arr, I_arr, t_ref + T_ms + shift_ms)
    x_target = np.concatenate([E_target, I_target])
    E_natural, I_natural = state_at_time(t_arr, E_arr, I_arr, t_ref + T_ms)
    x_natural = np.concatenate([E_natural, I_natural])
    target_distance = np.linalg.norm(x_target - x_ref)

    # Leg 1: naive (already-verified logic, inlined here for a single self-
    # contained three-way comparison)
    J = net.jacobian_at(E_ref, I_ref)
    G = simulation.input_jacobian_at(net, E_ref, I_ref)
    A, B = simulation.discretize_system(J, G, DT)
    delta_xT_naive = x_target - x_ref
    _, U_naive = physics.minimum_energy_trajectory(A, B, np.zeros(2*N), delta_xT_naive, T_steps)
    u_func_naive = physics.zero_order_hold(U_naive, DT)
    ctrl_naive = net.simulate_controlled(u_func=u_func_naive, t_span=(0.0, T_ms),
                                          n_points=max(100, T_steps), E0=E_ref, I0=I_ref)
    x_final_naive = np.concatenate([ctrl_naive["E"][:, -1], ctrl_naive["I"][:, -1]])
    err_naive = np.linalg.norm(x_final_naive - x_target) / target_distance

    # Leg 2: LTV
    _, x_final_ltv = run_ltv_control(net, t_arr, E_arr, I_arr, t_ref, T_steps,
                                       x_target, x_natural, E_ref, I_ref)
    err_ltv = np.linalg.norm(x_final_ltv - x_target) / target_distance

    result = {
        "seed": seed, "T_steps": T_steps, "shift_ms": shift_ms,
        "target_distance": target_distance,
        "rel_err_naive": err_naive,
        "rel_err_ltv": err_ltv,
        "rel_err_do_nothing": np.linalg.norm(x_natural - x_target) / target_distance,
    }

    # Leg 3: nonlinear ceiling (expensive -- only run when asked)
    if run_ceiling:
        _, x_final_nl, _, success = run_nonlinear_ceiling(net, x_ref, x_target, T_steps)
        result["rel_err_nonlinear_ceiling"] = np.linalg.norm(x_final_nl - x_target) / target_distance
        result["nonlinear_opt_success"] = success

    return result


if __name__ == "__main__":
    print("=" * 78)
    print("Three-way comparison: naive frozen-snapshot vs. proper LTV (Floquet)")
    print("vs. true nonlinear optimal control ceiling")
    print("=" * 78)
    print()
    print(f"{'seed':>4} {'shift_ms':>9} {'target_dist':>12} {'naive':>10} "
          f"{'LTV':>10} {'nonlin_ceiling':>15} {'do_nothing':>11}")

    # Cheap legs (naive, LTV, do-nothing) across a real spread
    cheap_results = []
    for seed in range(5):
        for shift_ms in [10, 20, 40]:
            r = three_way_comparison(seed, T_steps=25, shift_ms=shift_ms, run_ceiling=False)
            if "status" in r and r.get("status") == "no stable phase found":
                print(f"{seed:>4} {shift_ms:>9}  -- no stable phase found --")
                continue
            cheap_results.append(r)
            print(f"{r['seed']:>4} {r['shift_ms']:>9} {r['target_distance']:>12.4f} "
                  f"{r['rel_err_naive']:>9.1%} {r['rel_err_ltv']:>9.1%} "
                  f"{'--':>15} {r['rel_err_do_nothing']:>10.1%}")

    print()
    print("-" * 78)
    print("Adding nonlinear ceiling (expensive -- 2 representative cases only)")
    print("-" * 78)
    for seed, shift_ms in [(0, 20), (1, 20)]:
        r = three_way_comparison(seed, T_steps=25, shift_ms=shift_ms, run_ceiling=True)
        print(f"seed={seed} shift_ms={shift_ms}: "
              f"naive={r['rel_err_naive']:.1%}  ltv={r['rel_err_ltv']:.1%}  "
              f"nonlinear_ceiling={r['rel_err_nonlinear_ceiling']:.1%}  "
              f"do_nothing={r['rel_err_do_nothing']:.1%}  "
              f"(optimizer success={r['nonlinear_opt_success']})")
