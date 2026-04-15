"""
Reward functions for the variable inverted pendulum environment.

All rewards are computed from raw (unnormalised) state values.
The total reward per step is a weighted sum of components defined in default.yaml.
"""

import numpy as np


def compute_reward(
    joint_angles: np.ndarray,
    cart_pos: float,
    action: float,
    upright_weight: float = 1.0,
    alive_bonus: float = 0.1,
    force_penalty: float = 0.001,
    rail_penalty: float = 0.01,
) -> tuple[float, dict]:
    """
    Compute the per-step reward and a breakdown dict for logging.

    Parameters
    ----------
    joint_angles : (n_links,) array of joint angles in radians from upright.
                   theta=0 means perfectly vertical, so cos(0)=1 is the maximum.
    cart_pos     : cart x position in metres.
    action       : scalar force applied to the cart (Newtons).

    Returns
    -------
    total_reward : float
    components   : dict with individual reward terms for logging/debugging
    """
    # Each link contributes cos(theta): 1.0 when upright, decreases as it falls.
    upright = float(np.sum(np.cos(joint_angles)))

    # Flat survival bonus — encourages staying alive longer.
    alive = alive_bonus

    # Penalise large forces — encourages smooth, efficient control.
    force_pen = -force_penalty * float(action ** 2)

    # Penalise cart drifting from centre — keeps it recoverable.
    rail_pen = -rail_penalty * float(cart_pos ** 2)

    total = upright_weight * upright + alive + force_pen + rail_pen

    components = {
        "upright": upright_weight * upright,
        "alive": alive,
        "force_penalty": force_pen,
        "rail_penalty": rail_pen,
    }

    return total, components
