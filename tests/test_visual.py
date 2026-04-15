# Run: python3.12 tests/test_visual.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
import mujoco
import mujoco.viewer as mjviewer

from env.mujoco_builder import PendulumConfig, build_mjcf

DT = 0.001

config = PendulumConfig(n_links=1, lengths=[1.0], masses=[0.1], cart_mass=5.0)
xml = build_mjcf(config, rail_limit=10.0, max_force=100.0, timestep=DT)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

rng = np.random.default_rng(seed=42)
data.qpos[0] = 0.0
data.qpos[1] = rng.uniform(0.1, 0.4)
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)
data.ctrl[0] = 0.0

print(f"Starting angle: {data.qpos[1]:.3f} rad ({np.degrees(data.qpos[1]):.1f}°)")
print("Window will close after 3 seconds.")

duration = 3.0
with mjviewer.launch_passive(model, data) as viewer:
    t_start = time.time()
    while viewer.is_running() and (time.time() - t_start) < duration:
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        remaining = model.opt.timestep - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)

print(f"Final angle: {data.qpos[1]:.3f} rad ({np.degrees(data.qpos[1]):.1f}°)")
