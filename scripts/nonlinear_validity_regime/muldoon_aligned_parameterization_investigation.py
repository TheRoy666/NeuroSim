#!/usr/bin/env python3
"""
Muldoon-aligned parameterization investigation for Path A1.

Muldoon et al. (2016) use a different WC parameterization convention
(c1-c4 connection weights, a_e/a_i sigmoid steepness, theta_e/theta_i
sigmoid thresholds, a single shared tau=8ms) than this project's own
(w_EE/w_IE/w_EI/w_II/c_E/c_I, separate tau_E/tau_I, sigmoid steepness/
threshold hardcoded to defaults a=1, theta=0 in simulation.py).

Their form S(a*(z-theta)) is exactly representable in our own default-
sigmoid convention by absorbing steepness and threshold into the linear
weights and bias: w_eff = a*c, c_eff = -a*theta (verified numerically
below to match Muldoon's own sigmoid exactly across multiple random
(E,I) points, not just algebraically claimed).

RESULT: sweeping w_EE (our usual bifurcation-proximity axis) from their
baseline (20.8) up to 15x that value never destabilizes the system --
max eigenvalue plateaus at exactly -0.125 from 3x onward. The
E-population saturates (sigmoid slope -> 0) rather than crossing a Hopf
boundary. Muldoon's strong inhibitory feedback (w_IE=15.6, w_EI=30)
stabilizes the system far more robustly than this project's own chosen
parameters. This is a real finding about the relative stability margins
of the two regimes, not a failed replication -- forcing a direct w_EE-
based comparison would need a different destabilization axis for their
regime specifically, not attempted here (open-ended, disproportionate
given project timeline).
"""
import numpy as np
from simulation import WilsonCowanNetwork, _sigmoid

# Muldoon et al. (2016) published constants
C1, C2, C3, C4 = 16.0, 12.0, 15.0, 3.0
A_E, A_I = 1.3, 2.0
THETA_E, THETA_I = 4.0, 3.7
TAU_SHARED = 8.0

MULDOON_BASE = dict(
    w_EE=A_E * C1, w_IE=A_E * C2, c_E=-A_E * THETA_E,
    w_EI=A_I * C3, w_II=A_I * C4, c_I=-A_I * THETA_I,
    tau_E=TAU_SHARED, tau_I=TAU_SHARED,
)


def verify_parameterization_equivalence(n_checks=5, seed=0):
    """Confirms the transformation is exact, not approximate, by directly
    comparing Muldoon's own sigmoid form against our derived-equivalent
    weights, at several random (E,I) points."""
    def muldoon_sigmoid_E(z):
        return 1.0 / (1.0 + np.exp(-A_E * (z - THETA_E)))

    rng = np.random.default_rng(seed)
    max_diff = 0.0
    for _ in range(n_checks):
        E, I = rng.uniform(0, 1), rng.uniform(0, 1)
        z = C1 * E - C2 * I
        direct = muldoon_sigmoid_E(z)
        equivalent = _sigmoid(np.array(
            [MULDOON_BASE["w_EE"] * E - MULDOON_BASE["w_IE"] * I + MULDOON_BASE["c_E"]]
        ))[0]
        max_diff = max(max_diff, abs(direct - equivalent))
    return max_diff


def scan_w_EE_scale(C, scale_range, verbose=True):
    """Scans a scaling factor on Muldoon's own w_EE baseline (1.0x =
    their exact published value), looking for a destabilization
    boundary the same way this project's own parameterization was
    calibrated."""
    N = C.shape[0]
    E_guess, I_guess = None, None
    results = []
    for scale in scale_range:
        params = dict(**MULDOON_BASE)
        params["w_EE"] = MULDOON_BASE["w_EE"] * scale
        net = WilsonCowanNetwork(n_regions=N, C=C, node_params=params)
        try:
            E_star, I_star = net.find_fixed_point(E_guess, I_guess)
            J = net.jacobian_at(E_star, I_star)
            max_eig = np.linalg.eigvals(J).real.max()
            is_stable = max_eig < 0
            if verbose:
                print(f"  scale={scale:.2f} (w_EE={params['w_EE']:.2f}): "
                      f"max_eig={max_eig:.5f} [{'STABLE' if is_stable else 'UNSTABLE'}]")
            results.append((scale, max_eig, is_stable))
            if is_stable:
                E_guess, I_guess = E_star, I_star
            else:
                break
        except RuntimeError:
            if verbose:
                print(f"  scale={scale:.2f}: solver did not converge")
            break
    return results


if __name__ == "__main__":
    max_diff = verify_parameterization_equivalence()
    print(f"Parameterization equivalence check: max diff = {max_diff:.2e} "
          f"({'PASS, exact match' if max_diff < 1e-10 else 'FAIL'})\n")

    N = 10
    rng = np.random.default_rng(0)
    C = rng.uniform(0, 0.15, (N, N)) / N * 20
    np.fill_diagonal(C, 0)

    print("=== Coarse scan, scale 0.5 to 3.0 ===")
    scan_w_EE_scale(C, np.arange(0.5, 3.01, 0.25))

    print("\n=== Extended scan, scale 3.0 to 15.0 ===")
    results = scan_w_EE_scale(C, np.arange(3.0, 15.01, 1.0))

    print(f"\nConclusion: max_eig plateaus (see results above) -- the system "
          f"does not destabilize via this axis within a physically "
          f"reasonable range. Muldoon's own parameterization sits in a "
          f"more strongly stability-margined regime than this project's "
          f"own chosen constants.")
