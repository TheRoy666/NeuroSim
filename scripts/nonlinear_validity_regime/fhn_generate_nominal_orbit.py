#!/usr/bin/env python3
"""
Generates the nominal FHN limit-cycle orbit used by
fhn_a2_phase_shift_comparison.py (nominal_t.npy, nominal_v.npy,
nominal_w.npy in fhn_a2_data/). Previously produced ad hoc and never
saved as a standalone script -- this closes that reproducibility gap.

Single-node FHN oscillator, I_ext=0.5, no coupling (kappa=0). Simulates
past the initial transient, locates the limit cycle via upward zero-
crossings of v relative to its own mean, confirms the period is stable
across multiple cycles (not just a single estimate), then extracts
exactly one clean period as the nominal reference orbit.
"""
import os
import numpy as np
import fhn_simulation as fhn

I_EXT = 0.5
T_SPAN = (0, 500)
N_POINTS = 10000
TAIL_START_IDX = 3000  # t ~ 150, well past the initial transient
OUT_DIR = "fhn_a2_data"


def generate_nominal_orbit(seed=1, verbose=True):
    params = dict(**fhn.FHN_DEFAULT_PARAMS)
    params["kappa"] = 0.0
    net = fhn.FitzHughNagumoNetwork(n_regions=1, C=np.zeros((1, 1)),
                                     I_ext=I_EXT, node_params=params)
    result = net.simulate(t_span=T_SPAN, n_points=N_POINTS, seed=seed)
    v_trace = result["v"][0]
    w_trace = result["w"][0]
    t_trace = result["t"]

    v_tail = v_trace[TAIL_START_IDX:]
    t_tail = t_trace[TAIL_START_IDX:]
    sign_diff = np.diff(np.sign(v_tail - v_tail.mean()))
    crossing_idx = np.where(sign_diff > 0)[0]

    if len(crossing_idx) < 3:
        raise RuntimeError(
            f"Only {len(crossing_idx)} upward crossings found in the tail -- "
            f"not enough to confirm a stable period. Check I_EXT is genuinely "
            f"past the Hopf boundary (confirmed ~0.32-0.325 elsewhere in this "
            f"project) and that T_SPAN/TAIL_START_IDX give enough post-"
            f"transient cycles."
        )

    crossing_times = t_tail[crossing_idx]
    periods = np.diff(crossing_times)
    period = np.mean(periods)
    period_std = np.std(periods)
    if verbose:
        print(f"Confirmed period: {period:.4f} (std across {len(periods)} "
              f"cycles: {period_std:.4f})")
        if period_std > 0.1:
            print("WARNING: period std is larger than expected -- verify "
                  "the system has genuinely settled onto the limit cycle.")

    start_idx = TAIL_START_IDX + crossing_idx[0]
    end_idx = TAIL_START_IDX + crossing_idx[1]
    nominal_t = t_trace[start_idx:end_idx] - t_trace[start_idx]
    nominal_v = v_trace[start_idx:end_idx]
    nominal_w = w_trace[start_idx:end_idx]

    if verbose:
        print(f"Nominal orbit: {len(nominal_t)} points over one period "
              f"({nominal_t[-1]:.2f} time units)")
        print(f"v range: [{nominal_v.min():.3f}, {nominal_v.max():.3f}]")

    return nominal_t, nominal_v, nominal_w, period


if __name__ == "__main__":
    nominal_t, nominal_v, nominal_w, period = generate_nominal_orbit()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "nominal_t.npy"), nominal_t)
    np.save(os.path.join(OUT_DIR, "nominal_v.npy"), nominal_v)
    np.save(os.path.join(OUT_DIR, "nominal_w.npy"), nominal_w)
    print(f"\nSaved to {OUT_DIR}/nominal_{{t,v,w}}.npy")
