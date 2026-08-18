#!/usr/bin/env python3
"""
Verification and efficiency comparison for adjoint_gradient.py -- the
priority engineering task that gates real-N and broader-grid A2 work.

Persisted record of the actual development process, including two real
bugs found and fixed via finite-difference verification (not asserted
correct without checking):

1. Discontinuous-control integration bug: a single solve_ivp call
   spanning the whole piecewise-constant control trajectory let RK45's
   adaptive stepper (which assumes smooth dynamics) straddle segment
   discontinuities inconsistently between perturbed and unperturbed
   runs. Fixed by integrating segment-by-segment, chaining initial
   conditions, guaranteeing each individual solve_ivp call only ever
   sees a truly constant control.

2. Missing dt scaling: the continuous-time formula G(t)^T @ lambda(t)
   is a sensitivity DENSITY (per unit time); the gradient with respect
   to a piecewise-constant control acting over an interval of width dt
   needs the integral over that interval, not the density alone.
   First attempted fix (multiply midpoint evaluation by dt) reduced
   error from ~100% to ~2% -- correct in principle but not tight enough.
   Final fix: augment the adjoint ODE with a gradient-accumulator state,
   so the segment integral is computed exactly by the ODE solver itself,
   not approximated by a single midpoint evaluation. This closed the
   remaining ~2% gap down to ~1e-7, at the level of integration
   tolerance.

Final verification: adjoint gradient matches finite differences to
~1e-7 to 1e-8 relative error across multiple random seeds, U values,
and both with and without lambda_reg (control-effort regularization).
"""
import time
import numpy as np
from scipy.optimize import minimize

import fhn_simulation as fhn
from adjoint_gradient import compute_adjoint_gradient, verify_gradient_against_finite_difference

PARAMS = dict(**fhn.FHN_DEFAULT_PARAMS)
PARAMS["kappa"] = 0.0
NET = fhn.FitzHughNagumoNetwork(n_regions=1, C=np.zeros((1, 1)), I_ext=0.5, node_params=PARAMS)


def fhn_jacobian_func(x, u=None):
    return NET.jacobian_at(x[0:1], x[1:2])


def fhn_input_jacobian_func(x, u=None):
    return np.array([[1.0], [0.0]])


def run_verification_suite():
    x0 = np.array([1.6846, 0.6700])
    target_state = np.array([0.5, -0.3])
    dt = 0.5

    print("=== Verification 1: original test case (10 steps, seed 0/1) ===")
    rng = np.random.default_rng(0)
    U = rng.normal(0, 0.5, (10, 1))
    max_err, results = verify_gradient_against_finite_difference(
        NET, x0, target_state, U, dt, fhn_jacobian_func,
        fhn_input_jacobian_func, n_check=8, seed=1,
    )
    print(f"Max rel error: {max_err:.2e} -- {'PASS' if max_err < 1e-3 else 'FAIL'}")

    print("\n=== Verification 2: different seed/U, no regularization ===")
    rng2 = np.random.default_rng(42)
    U2 = rng2.normal(0, 0.8, (12, 1))
    max_err2, _ = verify_gradient_against_finite_difference(
        NET, x0, target_state, U2, dt, fhn_jacobian_func,
        fhn_input_jacobian_func, n_check=10, seed=99,
    )
    print(f"Max rel error: {max_err2:.2e} -- {'PASS' if max_err2 < 1e-3 else 'FAIL'}")

    print("\n=== Verification 3: nonzero lambda_reg ===")
    rng3 = np.random.default_rng(7)
    U3 = rng3.normal(0, 0.5, (8, 1))
    max_err3, _ = verify_gradient_against_finite_difference(
        NET, x0, target_state, U3, dt, fhn_jacobian_func,
        fhn_input_jacobian_func, lambda_reg=0.3, n_check=8, seed=3,
    )
    print(f"Max rel error: {max_err3:.2e} -- {'PASS' if max_err3 < 1e-3 else 'FAIL'}")

    return max(max_err, max_err2, max_err3)


def run_efficiency_comparison():
    print("\n=== Efficiency test: solving the same problem class that "
          "previously timed out with finite-difference gradients ===")
    x0 = np.array([1.6846, 0.6700])
    target_state = np.array([1.5, 0.6])
    dt = 0.5
    n_steps = 30  # T=15, matching the original A2 phase-shift comparison

    def objective_with_grad(u_flat):
        U = u_flat.reshape(n_steps, 1)
        cost, grad, _ = compute_adjoint_gradient(
            NET, x0, target_state, U, dt, fhn_jacobian_func,
            fhn_input_jacobian_func,
        )
        return cost, grad.flatten()

    U_init = np.zeros(n_steps)
    t0 = time.time()
    result = minimize(objective_with_grad, U_init, jac=True, method='L-BFGS-B',
                       options={'maxiter': 100})
    elapsed = time.time() - t0

    print(f"Final cost: {result.fun:.6f}")
    print(f"Iterations: {result.nit}, function evaluations: {result.nfev}, "
          f"elapsed: {elapsed:.2f}s")
    print(f"Comparison: the earlier finite-difference approach needed up to "
          f"2*n_steps={2*n_steps} extra simulations PER gradient step, up to "
          f"60 iterations before timing out on some targets -- worst case "
          f"~3600 total simulations vs. this run's {result.nfev} evaluations "
          f"(each just 1 forward + 1 backward integration).")

    return result


if __name__ == "__main__":
    max_error = run_verification_suite()
    print(f"\n{'='*70}")
    print(f"Overall max relative error across all verification tests: {max_error:.2e}")
    print(f"{'ALL PASS' if max_error < 1e-3 else 'AT LEAST ONE FAILED'}")
    print(f"{'='*70}")

    run_efficiency_comparison()
