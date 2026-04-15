"""
Physics validation test for the variable inverted pendulum environment.

What is being tested
--------------------
1. XML generation  — build_mjcf produces valid XML that MuJoCo can load.
2. Physics accuracy — MuJoCo's trajectory matches scipy ODE integration of
                      the exact Lagrangian equations of motion.
3. Energy conservation — total mechanical energy is conserved when no force
                         is applied (confirms MuJoCo timestep is sane).
4. Graph construction — build_graph returns correctly shaped arrays with
                        features in expected ranges.

Physics reference: exact EOM
-----------------------------
For a single uniform rod of mass m, length L attached at a cart of mass M
(frictionless, no force):

Mass matrix  (2x2):
    [ M+m,         (mL/2) cos θ ]
    [ (mL/2) cos θ, mL²/3       ]

RHS vector:
    [ (mL/2) θ̇² sin θ   ]
    [ (mg L/2) sin θ     ]

Solve for [ẍ, θ̈] = M⁻¹ b at each instant, then integrate with scipy RK45.

Why the rod is uniform (not a point mass)
-----------------------------------------
MuJoCo models each rod as a thin capsule whose mass is distributed along its
length.  For a uniform rod the moment of inertia about one end is mL²/3, which
is what appears in the mass matrix above.  The equations here match MuJoCo's
internal inertia computation for a thin capsule (radius 0.01 m ≪ L = 1.0 m).

Test setup
----------
  M_cart  = 5.0 kg
  m_rod   = 0.1 kg
  L       = 1.0 m
  g       = 9.81 m/s²
  θ₀      = 0.05 rad   (slightly off vertical, unstable — will fall)
  θ̇₀      = 0
  ẋ₀, ẍ₀  = 0
  Force    = 0 throughout
  Duration = 0.5 s  (500 MuJoCo steps at dt = 0.001 s)

After 0.5 s the rod has fallen noticeably (~0.17 rad) providing a
non-trivial comparison while keeping the small-angle approximation error low.

Tolerances
----------
  angle      : 5e-3 rad   (≈ 0.3°)
  position   : 1e-3 m
  ang. vel.  : 1e-2 rad/s
  cart vel.  : 5e-3 m/s
  energy     : 0.5% relative drift over the full trajectory

Run with:
  cd Inverted_Pendulum_RL
  python -m pytest tests/test_physics.py -v
  # or directly:
  python tests/test_physics.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
import mujoco
from scipy.integrate import solve_ivp

from env.mujoco_builder import PendulumConfig, build_mjcf
from graph.graph_builder import build_graph, NODE_FEAT_DIM, EDGE_FEAT_DIM
from graph.graph_utils import validate_graph, graph_summary


M_CART = 5.0    # kg
M_ROD  = 0.1    # kg
L      = 1.0    # m
G      = 9.81   # m/s²

THETA_0   = 0.05  # rad — initial angle from upright
DURATION  = 0.5   # s
DT_MUJOCO = 0.001 # s — must match build_mjcf timestep argument
N_STEPS   = int(DURATION / DT_MUJOCO)  # 500

TOL_ANGLE    = 5e-3   # rad
TOL_POS      = 1e-3   # m
TOL_ANGVEL   = 1e-2   # rad/s
TOL_CARTVEL  = 5e-3   # m/s
TOL_ENERGY   = 0.005  # 0.5% relative energy drift


# Equations of motion
def eom(t, y):
    """
    Equations of motion for a single-link inverted pendulum on a frictionless cart.

    State y = [x, θ, ẋ, θ̇]
    Returns  ẏ = [ẋ, θ̇, ẍ, θ̈]

    Derivation: Lagrangian mechanics for a uniform rod (moment of inertia mL²/3
    about the pivot joint), zero applied force.
    """
    _, theta, xdot, thetadot = y

    # 2×2 mass matrix
    M_mat = np.array([
        [M_CART + M_ROD,        (M_ROD * L / 2) * np.cos(theta)],
        [(M_ROD * L / 2) * np.cos(theta), M_ROD * L**2 / 3],
    ])

    # RHS (zero applied force)
    b = np.array([
        (M_ROD * L / 2) * thetadot**2 * np.sin(theta),
        (M_ROD * G * L / 2) * np.sin(theta),
    ])

    acc = np.linalg.solve(M_mat, b)   # [ẍ, θ̈]
    return [xdot, thetadot, acc[0], acc[1]]


def total_energy(x, theta, xdot, thetadot):
    """
    Total mechanical energy E = KE_cart + KE_rod + PE_rod.

    For a uniform rod (I_cm = mL²/12):
      KE_rod = ½m(v_cm_x² + v_cm_z²) + ½ I_cm θ̇²
    where v_cm = (ẋ + (L/2)θ̇cosθ,  -(L/2)θ̇sinθ)

    PE_rod = m g (L/2) cos θ    (height of rod CoM above cart)
    """
    KE_cart = 0.5 * M_CART * xdot**2

    v_cm_x = xdot + (L / 2) * thetadot * np.cos(theta)
    v_cm_z = -(L / 2) * thetadot * np.sin(theta)
    I_cm = M_ROD * L**2 / 12
    KE_rod = 0.5 * M_ROD * (v_cm_x**2 + v_cm_z**2) + 0.5 * I_cm * thetadot**2

    PE_rod = M_ROD * G * (L / 2) * np.cos(theta)

    return KE_cart + KE_rod + PE_rod


# ---------------------------------------------------------------------------
# Reference: scipy integration
# ---------------------------------------------------------------------------

def scipy_reference():
    """
    Integrate the exact EOM with RK45 (tight tolerances) and return the final
    state [x, θ, ẋ, θ̇] at t = DURATION.
    """
    y0 = [0.0, THETA_0, 0.0, 0.0]
    sol = solve_ivp(
        eom,
        t_span=(0.0, DURATION),
        y0=y0,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        dense_output=False,
    )
    assert sol.success, f"scipy ODE solver failed: {sol.message}"
    return sol.y[:, -1]   # [x, θ, ẋ, θ̇] at final time


# ---------------------------------------------------------------------------
# MuJoCo setup helpers
# ---------------------------------------------------------------------------

def build_mujoco_model():
    """Build and return (model, data) for the known test configuration."""
    config = PendulumConfig(
        n_links=1,
        lengths=[L],
        masses=[M_ROD],
        cart_mass=M_CART,
    )
    # max_force controls the actuator range; zero is invalid in MuJoCo.
    # We apply zero force during the test by setting data.ctrl[0] = 0.0.
    xml = build_mjcf(config, rail_limit=10.0, max_force=100.0, timestep=DT_MUJOCO)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data, config


def set_initial_state(model, data):
    """Set the known initial condition on an MjData object."""
    data.qpos[0] = 0.0       # cart x
    data.qpos[1] = THETA_0   # joint angle
    data.qvel[0] = 0.0       # cart velocity
    data.qvel[1] = 0.0       # angular velocity
    mujoco.mj_forward(model, data)


def run_mujoco(model, data, n_steps):
    """Run n_steps with zero force, recording energy at each step."""
    data.ctrl[0] = 0.0
    energies = []
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        x        = data.qpos[0]
        theta    = data.qpos[1]
        xdot     = data.qvel[0]
        thetadot = data.qvel[1]
        energies.append(total_energy(x, theta, xdot, thetadot))
    return energies


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_xml_generation():
    """build_mjcf produces valid XML that MuJoCo can load without error."""
    print("\n--- TEST 1: XML generation ---")
    for n in [1, 2, 3, 4]:
        config = PendulumConfig(
            n_links=n,
            lengths=[0.5 + 0.1 * i for i in range(n)],
            masses=[0.2 + 0.05 * i for i in range(n)],
            cart_mass=1.0,
        )
        xml = build_mjcf(config)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        dof = model.nv
        expected_dof = n + 1   # slider + n hinges
        assert dof == expected_dof, f"n={n}: expected {expected_dof} DOF, got {dof}"
        print(f"  n_links={n}  DOF={dof}  ✓")
    print("PASSED")


def test_physics_accuracy():
    """
    MuJoCo trajectory matches scipy ODE integration within tolerances.

    This is the central physics validation.  A discrepancy here means either:
      - The XML is building the wrong model geometry / mass.
      - The MuJoCo integrator is configured incorrectly.
      - There is a sign/convention mismatch in the EOM.
    """
    print("\n--- TEST 2: Physics accuracy (MuJoCo vs scipy) ---")
    print(f"  Config: M_cart={M_CART}kg, m_rod={M_ROD}kg, L={L}m")
    print(f"  Initial: θ₀={THETA_0} rad, all velocities=0")
    print(f"  Duration: {DURATION}s ({N_STEPS} steps, no force)")

    # Scipy reference
    ref = scipy_reference()
    x_ref, theta_ref, xdot_ref, thetadot_ref = ref
    print(f"\n  scipy reference (final state):")
    print(f"    x={x_ref:.6f} m,  θ={theta_ref:.6f} rad")
    print(f"    ẋ={xdot_ref:.6f} m/s,  θ̇={thetadot_ref:.6f} rad/s")

    # MuJoCo simulation
    model, data, config = build_mujoco_model()
    set_initial_state(model, data)
    run_mujoco(model, data, N_STEPS)

    x_mj        = data.qpos[0]
    theta_mj    = data.qpos[1]
    xdot_mj     = data.qvel[0]
    thetadot_mj = data.qvel[1]

    print(f"\n  MuJoCo result (final state):")
    print(f"    x={x_mj:.6f} m,  θ={theta_mj:.6f} rad")
    print(f"    ẋ={xdot_mj:.6f} m/s,  θ̇={thetadot_mj:.6f} rad/s")

    d_x        = abs(x_mj - x_ref)
    d_theta    = abs(theta_mj - theta_ref)
    d_xdot     = abs(xdot_mj - xdot_ref)
    d_thetadot = abs(thetadot_mj - thetadot_ref)

    print(f"\n  Absolute differences:")
    print(f"    Δx={d_x:.2e} m      (tol {TOL_POS:.2e})")
    print(f"    Δθ={d_theta:.2e} rad  (tol {TOL_ANGLE:.2e})")
    print(f"    Δẋ={d_xdot:.2e} m/s  (tol {TOL_CARTVEL:.2e})")
    print(f"    Δθ̇={d_thetadot:.2e} rad/s (tol {TOL_ANGVEL:.2e})")

    assert d_theta    <= TOL_ANGLE,    f"FAIL: Δθ={d_theta:.4e} exceeds tolerance {TOL_ANGLE}"
    assert d_x        <= TOL_POS,      f"FAIL: Δx={d_x:.4e} exceeds tolerance {TOL_POS}"
    assert d_xdot     <= TOL_CARTVEL,  f"FAIL: Δẋ={d_xdot:.4e} exceeds tolerance {TOL_CARTVEL}"
    assert d_thetadot <= TOL_ANGVEL,   f"FAIL: Δθ̇={d_thetadot:.4e} exceeds tolerance {TOL_ANGVEL}"
    print("PASSED")


def test_energy_conservation():
    """
    Total mechanical energy is conserved (no friction, no force).
    Checks that RK4 integration does not introduce significant energy drift
    over the full 0.5 s trajectory.
    """
    print("\n--- TEST 3: Energy conservation ---")

    model, data, config = build_mujoco_model()
    set_initial_state(model, data)

    # Record energy at t=0
    E0 = total_energy(
        data.qpos[0], data.qpos[1],
        data.qvel[0], data.qvel[1],
    )

    energies = run_mujoco(model, data, N_STEPS)
    energies = np.array(energies)

    E_max_drift = np.max(np.abs(energies - E0))
    E_rel_drift = E_max_drift / (abs(E0) + 1e-12)

    print(f"  E₀ = {E0:.6f} J")
    print(f"  Max absolute drift = {E_max_drift:.2e} J")
    print(f"  Max relative drift = {E_rel_drift * 100:.4f}%  (tol {TOL_ENERGY * 100:.2f}%)")

    assert E_rel_drift <= TOL_ENERGY, (
        f"FAIL: energy drift {E_rel_drift * 100:.4f}% exceeds tolerance "
        f"{TOL_ENERGY * 100:.2f}%"
    )
    print("PASSED")


def test_graph_construction():
    """
    build_graph returns a PendulumGraph with correct shapes and feature values.
    Tests the graph encoding layer independently from MuJoCo.
    """
    print("\n--- TEST 4: Graph construction ---")

    config = PendulumConfig(n_links=1, lengths=[L], masses=[M_ROD], cart_mass=M_CART)
    cart_pos  = 0.3
    cart_vel  = -0.1
    theta     = np.array([THETA_0])
    theta_dot = np.array([0.0])

    graph = build_graph(config, cart_pos, cart_vel, theta, theta_dot)

    print(graph_summary(graph))

    errors = validate_graph(graph, n_links=1)
    assert not errors, f"Graph validation failed:\n" + "\n".join(errors)

    # Cart node checks
    assert graph.node_features[0, 0] == 1.0,  "Cart node: is_cart should be 1"
    assert graph.node_features[0, 1] == 0.0,  "Cart node: is_joint should be 0"
    assert graph.node_features[0, 6] == cart_pos, "Cart node: x not set correctly"
    assert graph.node_features[0, 7] == cart_vel, "Cart node: ẋ not set correctly"

    # End node checks (n_links=1 → node 1 is the end node)
    assert graph.node_features[1, 2] == 1.0,  "End node: is_end should be 1"
    expected_sin = float(np.sin(THETA_0))
    expected_cos = float(np.cos(THETA_0))
    assert np.isclose(graph.node_features[1, 3], expected_sin, atol=1e-6), \
        f"End node: sin(θ) mismatch: {graph.node_features[1, 3]:.6f} vs {expected_sin:.6f}"
    assert np.isclose(graph.node_features[1, 4], expected_cos, atol=1e-6), \
        f"End node: cos(θ) mismatch: {graph.node_features[1, 4]:.6f} vs {expected_cos:.6f}"

    # Edge feature checks
    for e in range(graph.n_edges):
        assert graph.edge_features[e, 0] == L,     f"Edge {e}: length mismatch"
        assert graph.edge_features[e, 1] == M_ROD, f"Edge {e}: mass mismatch"

    # Node feature dtype
    assert graph.node_features.dtype == np.float32, "node_features dtype should be float32"
    assert graph.edge_index.dtype   == np.int64,    "edge_index dtype should be int64"
    assert graph.edge_features.dtype == np.float32, "edge_features dtype should be float32"

    print("PASSED")


def test_graph_for_multi_link():
    """Graph shapes are correct for 2, 3, 4-link configurations."""
    print("\n--- TEST 5: Multi-link graph shapes ---")
    for n in [2, 3, 4]:
        config = PendulumConfig(
            n_links=n,
            lengths=[0.5] * n,
            masses=[0.2] * n,
            cart_mass=1.0,
        )
        angles = np.full(n, 0.02)
        vels   = np.zeros(n)
        graph  = build_graph(config, 0.0, 0.0, angles, vels)
        errors = validate_graph(graph, n_links=n)
        assert not errors, f"n={n} graph invalid:\n" + "\n".join(errors)
        print(f"  n_links={n}: {graph.n_nodes} nodes, {graph.n_edges} edges  ✓")
    print("PASSED")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    all_tests = [
        test_xml_generation,
        test_physics_accuracy,
        test_energy_conservation,
        test_graph_construction,
        test_graph_for_multi_link,
    ]

    # Run visual test separately — no pass/fail, just opens a window.
    run_visual = "--visual" in sys.argv
    passed = 0
    failed = 0
    for test_fn in all_tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR in {test_fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(all_tests)} tests")

    if failed:
        sys.exit(1)
