"""
Adjoint (costate) gradient method for nonlinear optimal control.

Priority engineering task, gates real-N and broader-grid A2 work. The
earlier nonlinear-ceiling approach (fhn_a2_phase_shift_comparison.py)
used scipy.optimize.minimize with NO analytic gradient -- L-BFGS-B fell
back to finite-difference gradients, needing ~2 x n_steps extra
simulations per gradient evaluation. This is why a real-N nonlinear-
ceiling run was explicitly flagged "do not attempt" -- finite-difference
gradients don't scale.

The adjoint method computes the EXACT gradient with one forward
simulation + one backward (adjoint) integration, regardless of problem
dimension:

    Cost:     J = (1/2)||x(T) - x_target||^2 + (lambda_reg/2) * sum_k ||u_k||^2 * dt
    Dynamics: dx/dt = f(x, u)
    Adjoint:  dlambda/dt = -[df/dx]^T @ lambda,  lambda(T) = x(T) - x_target
    Gradient: dJ/du_k = [df/du]^T @ lambda(t_k) + lambda_reg * u_k * dt

lambda_reg=0 by default, exactly matching the pure tracking-error
objective used in the earlier direct-optimization work -- this is a
drop-in replacement, not a different problem.

Model-agnostic: works with WilsonCowanNetwork or FitzHughNagumoNetwork
(or any network exposing the same jacobian_at / dynamics interface) via
small per-model wrapper functions, not by hardcoding either model's
specifics into this module.
"""
import numpy as np
from scipy.integrate import solve_ivp


def compute_adjoint_gradient(
    net, x0, target_state, U, dt, jacobian_func, input_jacobian_func,
    lambda_reg=0.0,
):
    """
    Parameters
    ----------
    net : network object with a ._ode_network(t, y, u_func) method
    x0 : (2N,) initial state
    target_state : (2N,) absolute target state to track at t=T
    U : (n_steps, N) piecewise-constant control sequence
    dt : float, duration of each control interval
    jacobian_func : callable(x, u) -> (2N,2N) df/dx at state x AND the
        actual control u (2N-dim stacked state, e.g.
        np.concatenate([E,I]) or [v,w]). Like input_jacobian_func, must
        depend on u whenever the model's true df/dx does (e.g. WC: the
        state Jacobian's sigmoid-slope terms are evaluated at z+u, not
        z alone -- confirmed as a SECOND real bug, found after fixing
        input_jacobian_func alone only partially closed the WC
        verification gap, 40% -> 16% relative error, not to threshold).
    input_jacobian_func : callable(x, u) -> (2N,N) df/du at state x AND
        the ACTUAL control u being applied. MUST depend on u whenever the
        model's true df/du is u-dependent (e.g. WC, where u enters inside
        a sigmoid) -- passing a u-independent function here is only
        correct for models where control enters purely additively (e.g.
        FHN). Confirmed as a real bug via verification: using WC's
        existing input_jacobian_at(net,E,I), which evaluates the sigmoid
        slope assuming u=0 (correct for ITS OWN purpose serving the
        linear naive/LTV methods, wrong for this exact-nonlinear-gradient
        use), gave ~40% relative error against finite differences.
    lambda_reg : float, control-effort regularization weight (0 = pure
        tracking-error objective, matching prior work exactly)

    Returns
    -------
    cost : float, J at the current U
    grad : (n_steps, N) ndarray, exact dJ/dU
    x_final : (2N,) final state under this U (for convenience/logging)

    Implementation note: integrates SEGMENT BY SEGMENT (one solve_ivp
    call per control interval, chaining initial conditions across
    segments), not one call spanning the whole trajectory. A single call
    with a u_func that jumps at each dt boundary lets RK45's adaptive
    stepper -- which assumes smooth dynamics -- straddle discontinuities
    inconsistently between the unperturbed and perturbed trajectories
    used for finite-difference verification, corrupting the gradient.
    Confirmed as a real bug via the verification function below before
    this fix; segment-by-segment integration guarantees each individual
    call only ever sees a truly constant control.
    """
    n_steps, N_ctrl = U.shape

    # --- Forward pass, segment by segment ---
    fwd_sols = []  # dense-output solution object per segment
    x_current = x0.copy()
    for k in range(n_steps):
        u_k = U[k]
        sol_k = solve_ivp(
            lambda t, y: net._ode_network(t, y, lambda t_: u_k),
            (0.0, dt), x_current, method="RK45",
            dense_output=True, rtol=1e-8, atol=1e-10,
        )
        fwd_sols.append(sol_k)
        x_current = sol_k.y[:, -1]
    x_final = x_current

    tracking_error = x_final - target_state
    cost = 0.5 * np.dot(tracking_error, tracking_error)
    if lambda_reg > 0:
        cost += 0.5 * lambda_reg * np.sum(U**2) * dt

    def x_at_global_time(t_global):
        k = min(int(t_global / dt), n_steps - 1)
        t_local = t_global - k * dt
        t_local = min(max(t_local, 0.0), dt)  # clip for float safety
        return fwd_sols[k].sol(t_local)

    # --- Backward (adjoint) pass, segment by segment, in reverse.
    # Augment the adjoint state with a gradient-accumulator so each
    # segment's true integral contribution to the gradient
    # (integral over [0,dt] of G(t)^T @ lambda(t) dt) is computed
    # exactly via the ODE solver's own integration, not approximated by
    # a single midpoint evaluation. The earlier midpoint approximation
    # gave ~2% residual error against finite differences after the dt-
    # scaling fix -- correct, but not tight enough to trust for real
    # work; this closes that gap properly. ---
    lam_current = tracking_error
    adj_sols = []
    grad = np.zeros((n_steps, N_ctrl))
    for k in reversed(range(n_steps)):
        u_k = U[k]

        def augmented_ode(t_local, state, seg=k, u_seg=u_k):
            lam = state[:len(lam_current)]
            x_t = fwd_sols[seg].sol(t_local)
            J_t = jacobian_func(x_t, u_seg)
            G_t = input_jacobian_func(x_t, u_seg)
            dlam_dt = -J_t.T @ lam
            # d(grad_accum)/dt = -(G(t)^T @ lam(t)) so that integrating
            # BACKWARD from dt to 0 accumulates +integral(G^T @ lam dt)
            dgrad_dt = -(G_t.T @ lam)
            return np.concatenate([dlam_dt, dgrad_dt])

        state0 = np.concatenate([lam_current, np.zeros(N_ctrl)])
        sol_k = solve_ivp(
            augmented_ode, (dt, 0.0), state0, method="RK45",
            dense_output=True, rtol=1e-8, atol=1e-10,
        )
        n_lam = len(lam_current)
        lam_current = sol_k.y[:n_lam, -1]
        grad[k] = sol_k.y[n_lam:, -1]
        if lambda_reg > 0:
            grad[k] += lambda_reg * U[k] * dt
        adj_sols.append((k, sol_k))

    return cost, grad, x_final


def verify_gradient_against_finite_difference(
    net, x0, target_state, U, dt, jacobian_func, input_jacobian_func,
    lambda_reg=0.0, eps=1e-6, n_check=5, seed=0,
):
    """Essential correctness check before trusting this module for
    anything: compare the adjoint-computed gradient against finite
    differences at a random sample of control entries. Returns the
    max relative error found -- should be small (<1e-3 is a reasonable
    bar given RK45's own integration tolerance, not machine precision,
    since both the forward and adjoint passes carry their own numerical
    error)."""
    cost0, grad_adjoint, _ = compute_adjoint_gradient(
        net, x0, target_state, U, dt, jacobian_func, input_jacobian_func,
        lambda_reg=lambda_reg,
    )

    n_steps, N_ctrl = U.shape
    rng = np.random.default_rng(seed)
    check_indices = [(rng.integers(n_steps), rng.integers(N_ctrl))
                      for _ in range(n_check)]

    def cost_only(U_test):
        c, _, _ = compute_adjoint_gradient(
            net, x0, target_state, U_test, dt, jacobian_func,
            input_jacobian_func, lambda_reg=lambda_reg,
        )
        return c

    max_rel_error = 0.0
    results = []
    for (k, j) in check_indices:
        U_plus = U.copy(); U_plus[k, j] += eps
        U_minus = U.copy(); U_minus[k, j] -= eps
        cost_plus = cost_only(U_plus)
        cost_minus = cost_only(U_minus)
        fd_grad = (cost_plus - cost_minus) / (2 * eps)
        adj_grad = grad_adjoint[k, j]
        rel_error = abs(fd_grad - adj_grad) / (abs(fd_grad) + 1e-12)
        max_rel_error = max(max_rel_error, rel_error)
        results.append({"k": k, "j": j, "fd_grad": fd_grad,
                         "adjoint_grad": adj_grad, "rel_error": rel_error})

    return max_rel_error, results
