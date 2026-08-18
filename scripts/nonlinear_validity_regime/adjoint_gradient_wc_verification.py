#!/usr/bin/env python3
"""
WC-specific verification of adjoint_gradient.py, at N=10 (genuinely
different scale/model from the FHN single-node case). Persisted record
of a real, two-stage debugging process -- checking at each new
scale/model rather than trusting the FHN verification to carry over,
which is exactly what caught these:

Bug 1: input_jacobian_func was u-independent (using the existing
WC input_jacobian_at(net,E,I), which evaluates the sigmoid slope
assuming u=0 -- correct for that function's existing purpose serving
the linear naive/LTV methods, wrong for the adjoint method's need for
the TRUE df/du at the actual control being tested). Initial WC test:
40% relative error against finite differences.

Bug 2: jacobian_func (the STATE Jacobian, df/dx) has the SAME issue --
the E-equation's sigmoid-slope terms (J_EE, J_EI) are evaluated at z+u
in the true dynamics, but the existing jacobian_at(E,I) assumes u=0
internally, same as the input Jacobian. Fixing only Bug 1 improved
error from 40% to 16% -- real progress, but not to threshold, revealing
a second distinct bug rather than residual numerical noise.

Both fixed by building u-aware versions of both Jacobians (verified to
exactly match the originals at u=0 -- confirmed generalizations, not
different functions) and updating adjoint_gradient.py's core interface
to pass the actual per-segment u into both jacobian_func and
input_jacobian_func, not just one of them.

Final result: ~1e-9 relative error, essentially machine precision.

Note for future model integrations: any network where control enters
inside a nonlinearity (not purely additively, like FHN) needs this same
treatment -- check both Jacobians for u-dependence, don't assume only
one needs it.
"""
import numpy as np
import simulation
from adjoint_gradient import verify_gradient_against_finite_difference

LC_PARAMS = dict(w_EE=10.0, w_IE=12.0, w_EI=8.0, w_II=3.0,
                  c_E=-2.0, c_I=-3.5, tau_E=10.0, tau_I=20.0)
N = 10
DT = 2.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_wc_net(seed=0):
    rng = np.random.default_rng(seed)
    C = rng.uniform(0, 0.15, (N, N))
    np.fill_diagonal(C, 0)
    return simulation.WilsonCowanNetwork(n_regions=N, C=C, node_params=LC_PARAMS)


def make_wc_jacobian_func(net):
    """u-aware STATE Jacobian. Only J_EE, J_EI (from dE/dt's sigmoid)
    depend on u; J_IE, J_II (from dI/dt, which has no u term) do not,
    and are identical to the original jacobian_at's values."""
    def wc_jacobian_func(x, u=None):
        E, I = x[:N], x[N:]
        p = net.params
        if u is None:
            u = np.zeros(N)
        eye = np.eye(N)
        z_E = p["w_EE"]*E - p["w_IE"]*I + p["c_E"] + net.C @ E + u
        z_I = p["w_EI"]*E - p["w_II"]*I + p["c_I"]
        s_E = _sigmoid(z_E); s_I = _sigmoid(z_I)
        sp_E = np.diag(s_E * (1 - s_E))
        sp_I = np.diag(s_I * (1 - s_I))
        J_EE = (sp_E @ (p["w_EE"]*eye + net.C) - eye) / p["tau_E"]
        J_EI = (-p["w_IE"] / p["tau_E"]) * sp_E
        J_IE = (p["w_EI"] / p["tau_I"]) * sp_I
        J_II = (-p["w_II"]*sp_I - eye) / p["tau_I"]
        return np.block([[J_EE, J_EI], [J_IE, J_II]])
    return wc_jacobian_func


def make_wc_input_jacobian_func(net):
    """u-aware INPUT Jacobian -- df/du evaluated at the true current u,
    not assuming u=0."""
    def wc_input_jacobian_func(x, u=None):
        E, I = x[:N], x[N:]
        p = net.params
        if u is None:
            u = np.zeros(N)
        z_E = p["w_EE"]*E - p["w_IE"]*I + p["c_E"] + net.C @ E + u
        s_E = _sigmoid(z_E)
        sp_E = s_E * (1 - s_E)
        return np.vstack([np.diag(sp_E / p["tau_E"]), np.zeros((N, N))])
    return wc_input_jacobian_func


def sanity_check_matches_original_at_u_zero(net):
    """Confirm both u-aware functions exactly reduce to the originals
    at u=0 -- generalizations, not different functions."""
    rng = np.random.default_rng(99)
    E_test, I_test = rng.uniform(0.1, 0.3, N), rng.uniform(0.1, 0.3, N)
    x_test = np.concatenate([E_test, I_test])

    J_original = net.jacobian_at(E_test, I_test)
    J_new = make_wc_jacobian_func(net)(x_test, u=np.zeros(N))
    j_match = np.allclose(J_original, J_new)

    G_original = simulation.input_jacobian_at(net, E_test, I_test)
    G_new = make_wc_input_jacobian_func(net)(x_test, u=np.zeros(N))
    g_match = np.allclose(G_original, G_new)

    return j_match, g_match


if __name__ == "__main__":
    net = build_wc_net(seed=0)

    j_match, g_match = sanity_check_matches_original_at_u_zero(net)
    print(f"State Jacobian matches original at u=0: {j_match}")
    print(f"Input Jacobian matches original at u=0: {g_match}")
    assert j_match and g_match, "u-aware functions are not proper generalizations"

    wc_jacobian_func = make_wc_jacobian_func(net)
    wc_input_jacobian_func = make_wc_input_jacobian_func(net)

    rng = np.random.default_rng(0)
    result = net.simulate(t_span=(0, 2000), n_points=8000,
                           E0=rng.uniform(0.1, 0.3, N), I0=rng.uniform(0.1, 0.3, N))
    mask = result["t"] > 1000
    x0 = np.concatenate([result["E"][:, mask][:, 100], result["I"][:, mask][:, 100]])
    target_state = np.concatenate([result["E"][:, mask][:, 300], result["I"][:, mask][:, 300]])

    n_steps = 10
    rng2 = np.random.default_rng(5)
    U = rng2.normal(0, 0.3, (n_steps, N))

    max_err, results = verify_gradient_against_finite_difference(
        net, x0, target_state, U, DT, wc_jacobian_func, wc_input_jacobian_func,
        n_check=8, seed=2,
    )
    print(f"\nWC (N=10), both Jacobians u-aware: max relative error = {max_err:.2e}")
    print("PASS" if max_err < 1e-3 else "FAIL")
