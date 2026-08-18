#!/usr/bin/env python3
"""
Path A1 -- dt calibration.

Purpose
-------
Diagnose and resolve the TR-vs-neural-timescale mismatch found when trying
to discretize the Wilson-Cowan Jacobian at real fMRI TR (e.g. 720ms):
WC's neural time constants (tau_E=10ms, tau_I=20ms) are two orders of
magnitude faster than a TR, so discretizing directly at TR gives a discrete
A matrix with spectral radius ~0 (no memory between steps) -- which makes
any horizon-T sweep scientifically empty.

This script:
  1. Builds a toy WC network in a damped (non-oscillating) regime.
  2. Finds its fixed point and exact Jacobian.
  3. Sweeps dt to show WHERE the spectral radius collapses (reproduces the
     diagnosis from the last session, as a saved/rerunnable artifact).
  4. Picks a working dt (order of the neural time constant) and shows what
     a horizon sweep in THAT dt looks like -- i.e. the actual numbers for
     the fix, not just the diagnosis.
  5. Runs one concrete open-loop check: drive the real nonlinear WC network
     with the linear-optimal trajectory at the chosen dt, and report
     reachability error and energy at a couple of T values.

Run directly: `python3 path_a1_dt_calibration.py`
Everything below is deterministic (fixed seed) and prints its own numbers --
no hidden state, nothing needs to be taken on faith from a prior chat turn.
"""
import numpy as np
import physics
import simulation

np.random.seed(1)


def build_toy_network(N=3):
    """Same toy setup used in the diagnosis step, factored out so it's
    reused identically by every section below."""
    damped_params = dict(
        w_EE=3.0, w_IE=4.0, w_EI=3.0, w_II=2.0,
        c_E=-2.0, c_I=-2.0, tau_E=10.0, tau_I=20.0,
    )
    C = np.random.uniform(0, 0.15, (N, N))
    np.fill_diagonal(C, 0)
    net = simulation.WilsonCowanNetwork(n_regions=N, C=C, node_params=damped_params)
    return net


def section_1_diagnose_dt_collapse(net):
    print("=" * 70)
    print("SECTION 1 -- reproduce the dt-collapse diagnosis")
    print("=" * 70)

    E_star, I_star = net.find_fixed_point()
    J = net.jacobian_at(E_star, I_star)
    eigs = np.linalg.eigvals(J).real
    print(f"Fixed point E*: {E_star}")
    print(f"Jacobian eigenvalues (1/ms): {eigs}")
    print(f"Dominant neural time constant: {-1/eigs.max():.2f} ms")
    print()
    print(f"{'dt (ms)':>10} | {'rho(A)':>12}")
    print("-" * 25)
    dts = [1, 2, 5, 10, 20, 50, 100, 200, 400, 720]
    rhos = []
    for dt in dts:
        A = simulation.discretize_jacobian(J, dt)
        rho = np.max(np.abs(np.linalg.eigvals(A)))
        rhos.append(rho)
        print(f"{dt:>10} | {rho:>12.6f}")
    print()
    print("Interpretation: rho(A) is the fraction of state 'remembered'")
    print("from one discrete step to the next. It collapses to ~0 well")
    print("before real TR values (400-2000ms) -- confirms the mismatch.")
    return E_star, I_star, J


def section_2_choose_working_dt(J):
    print()
    print("=" * 70)
    print("SECTION 2 -- pick a working dt, show what it buys us")
    print("=" * 70)
    # Dominant time constant was ~15ms (Section 1). Pick dt an order of
    # magnitude below that, so successive steps still share real dynamics.
    dt = 2.0  # ms
    A = simulation.discretize_jacobian(J, dt)
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    print(f"Chosen dt = {dt} ms  ->  rho(A) = {rho:.4f}")
    print("(rho well below 1 but not collapsed -- real memory across steps)")
    print()
    print("Horizon T (in steps of this dt) vs. neural time spanned:")
    for T in [10, 25, 50, 100, 200, 500]:
        print(f"  T={T:>4} steps  =  {T*dt:>7.1f} ms of neural time")
    print()
    print("So a sweep over T=10..500 at dt=2ms covers ~20ms-1s of neural")
    print("time -- the actual cognitively-relevant transition window --")
    print("without the TR mismatch. This dt/T pairing is what A1's sweep")
    print("should use; it is decoupled from the real fMRI TR by design.")
    return dt, A


def section_3_open_loop_reachability_check(net, E_star, I_star, dt):
    print()
    print("=" * 70)
    print("SECTION 3 -- concrete open-loop check on the real nonlinear network")
    print("=" * 70)
    print("NOTE: this section went through two rounds of bugfixing before")
    print("producing physically sensible numbers:")
    print("  (1) B must be the correctly-discretized input Jacobian (via")
    print("      discretize_system), not a naive [I; 0] assumption -- u")
    print("      enters through the sigmoid, not as a direct state push.")
    print("  (2) The linear model must operate in DEVIATION coordinates")
    print("      from the fixed point (delta_x = x - x_star), not absolute")
    print("      state -- A only describes how deviations evolve; feeding")
    print("      it absolute x0 makes it think the fixed point itself is")
    print("      decaying, which swamps the real (small) target signal.")
    print()

    N = net.n_regions
    J = net.jacobian_at(E_star, I_star)
    G = simulation.input_jacobian_at(net, E_star, I_star)
    A, B = simulation.discretize_system(J, G, dt)
    x_star = np.concatenate([E_star, I_star])

    print(f"{'target scale':>13} | {'target_move':>12} | {'energy':>12} | "
          f"{'max|u|':>10} | {'rel_error':>10}")
    print("-" * 68)
    for scale in [1.0, 0.1, 0.01, 0.001]:
        target_perturb = np.array([0.05, -0.03, 0.02]) * scale
        delta_x0 = np.zeros(2 * N)
        delta_xT = np.concatenate([target_perturb, np.zeros(N)])

        T = 25
        energy, U = physics.minimum_energy_trajectory(A, B, delta_x0, delta_xT, T)
        u_func = physics.zero_order_hold(U, dt)

        result = net.simulate_controlled(
            u_func=u_func, t_span=(0.0, T * dt), n_points=200,
            E0=E_star, I0=I_star,
        )
        realized_final = np.concatenate([result["E"][:, -1], result["I"][:, -1]])
        xT_absolute = x_star + delta_xT
        target_movement = np.linalg.norm(delta_xT)
        err = np.linalg.norm(realized_final - xT_absolute)
        max_u = np.abs(U).max()

        print(f"{scale:>13} | {target_movement:>12.5f} | {energy:>12.6f} | "
              f"{max_u:>10.5f} | {err/target_movement:>9.2%}")

    print()
    print("Interpretation: energy scales ~quadratically and max|u| ~linearly")
    print("with target distance (as a linear model requires), AND relative")
    print("reachability error SHRINKS as the target shrinks -- the correct")
    print("signature of a linearization: accurate for small perturbations,")
    print("degrading for large ones. This is the actual Path A1 signal --")
    print("target-distance is one of the sweep axes in the Phase 0 plan.")


if __name__ == "__main__":
    net = build_toy_network()
    E_star, I_star, J = section_1_diagnose_dt_collapse(net)
    dt, A = section_2_choose_working_dt(J)
    section_3_open_loop_reachability_check(net, E_star, I_star, dt)
