import numpy as np


def compute_reward(
    joint_angles: np.ndarray,
    cart_pos: float,
    action: float,
    prev_action: float = 0.0,
    cart_vel: float = 0.0,
    joint_vels: np.ndarray | None = None,
    upright_weight: float = 1.0,
    alive_bonus: float = 0.1,
    force_penalty: float = 0.001,
    rail_penalty: float = 0.01,
    action_rate_penalty: float = 0.0,
    angle_penalty: float = 0.0,
    angular_velocity_penalty: float = 0.0,
    cart_velocity_penalty: float = 0.0,
    failure_penalty: float = 0.0,
    angle_margin_weight: float = 0.0,
    rail_margin_weight: float = 0.0,
    termination_angle: float = np.pi / 4,
    rail_limit: float = 2.5,
) -> tuple[float, dict]:
    
    # Each link contributes cos(theta): 1.0 when upright, decreases as it falls.
    upright = float(np.sum(np.cos(joint_angles)))

    # Flat survival bonus — encourages staying alive longer.
    alive = alive_bonus

    # Penalise large forces — encourages smooth, efficient control.
    force_pen = -force_penalty * float(action ** 2)
    action_rate_pen = -action_rate_penalty * float((action - prev_action) ** 2)

    # Penalise cart drifting from centre — keeps it recoverable.
    rail_pen = -rail_penalty * float(cart_pos ** 2)
    angle_pen = -angle_penalty * float(np.sum(joint_angles ** 2))
    if joint_vels is None:
        joint_vels = np.zeros_like(joint_angles)
    angular_vel_pen = -angular_velocity_penalty * float(np.sum(joint_vels ** 2))
    cart_vel_pen = -cart_velocity_penalty * float(cart_vel ** 2)
    max_angle = float(np.max(np.abs(joint_angles))) if joint_angles.size else 0.0
    angle_margin = max(0.0, float(termination_angle) - max_angle) / max(
        float(termination_angle), 1e-6
    )
    rail_margin = max(0.0, float(rail_limit) - abs(float(cart_pos))) / max(
        float(rail_limit), 1e-6
    )
    angle_margin_bonus = angle_margin_weight * angle_margin
    rail_margin_bonus = rail_margin_weight * rail_margin

    total = (
        upright_weight * upright + alive + force_pen + rail_pen
        + action_rate_pen + angle_pen + angular_vel_pen + cart_vel_pen
        + angle_margin_bonus + rail_margin_bonus
    )

    components = {
        "upright": upright_weight * upright,
        "alive": alive,
        "force_penalty": force_pen,
        "action_rate_penalty": action_rate_pen,
        "rail_penalty": rail_pen,
        "angle_penalty": angle_pen,
        "angular_velocity_penalty": angular_vel_pen,
        "cart_velocity_penalty": cart_vel_pen,
        "angle_margin_bonus": angle_margin_bonus,
        "rail_margin_bonus": rail_margin_bonus,
    }

    return total, components
