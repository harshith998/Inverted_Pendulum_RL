
#Programmatic MJCF (MuJoCo XML) generation, reloaded from at each episode reset when the config changes.


from dataclasses import dataclass
from typing import List


@dataclass
class PendulumConfig:
    n_links: int
    lengths: List[float]   # length of each rod in metres, len == n_links
    masses: List[float]    # mass of each rod in kg,     len == n_links
    cart_mass: float       # kg


def build_mjcf(config: PendulumConfig, rail_limit: float = 2.5, max_force: float = 20.0, timestep: float = 0.001) -> str:
    """
    Generate a MuJoCo MJCF XML string for the given pendulum config.

    Coordinate conventions:
    ---------------------------------------
    - Cart slides along the x-axis.
    - Each rod hangs from its parent joint; when all joint angles are zero
      the rods point straight up (+z), i.e. the fully upright configuration.
    - Joint angles are measured from the upright position (theta=0 → vertical).
    - Positive theta rotates the tip toward +x (right-hand rule around +y axis).

    """
    #check num of masses is num links
    assert len(config.lengths) == config.n_links
    assert len(config.masses) == config.n_links

    link_bodies = _build_link_chain(config)

    xml = f"""<mujoco model="variable_pendulum_{config.n_links}link">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="{timestep}"/>

  <default>
    <!-- disable collisions between all geoms — not needed for this task -->
    <geom contype="0" conaffinity="0"/>
    <!-- zero joint damping so energy is conserved when no force is applied -->
    <joint damping="0" frictionloss="0" armature="0"/>
  </default>

  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0"
             limited="true" range="{-rail_limit} {rail_limit}"/>
      <geom name="cart_geom" type="box" size="0.15 0.1 0.05" mass="{config.cart_mass}"/>
{link_bodies}
    </body>
  </worldbody>

  <actuator>
    <motor name="cart_motor" joint="slider" gear="1"
           ctrllimited="true" ctrlrange="{-max_force} {max_force}"/>
  </actuator>
</mujoco>"""
    
    return xml


def _build_link_chain(config: PendulumConfig, indent: int = 6) -> str:
    """
    Recursively build the nested body XML for the pendulum chain.
    Returns the indented XML string for all link bodies.
    """
    pad = " " * indent
    lines = []
    open_tags = []

    for i in range(config.n_links):
        L = config.lengths[i]
        m = config.masses[i]

        lines.append(f'{pad}<body name="link_{i}" pos="0 0 {0 if i == 0 else config.lengths[i-1]:.6f}">')
        lines.append(f'{pad}  <joint name="joint_{i}" type="hinge" axis="0 1 0" pos="0 0 0"/>')
        lines.append(f'{pad}  <geom name="geom_{i}" type="capsule" fromto="0 0 0 0 0 {L:.6f}" size="0.01" mass="{m}"/>')
        open_tags.append(f'{pad}</body>')
        pad += "  "

    # close all open body tags in reverse order
    for close in reversed(open_tags):
        lines.append(close)

    return "\n".join(lines)
