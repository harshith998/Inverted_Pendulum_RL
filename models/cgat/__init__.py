"""
CGAT model variants — all share the same graph obs space and param count (±2 scalars).

Variants
--------
  base        — scalar β·M̃ per layer  (current best)
  perhead     — per-head β·M̃  (one scalar per attention head)
  directional — β_fwd/β_bwd  (separate scales for root→leaf / leaf→root edges)
  gravity     — scalar β·M̃  +  gravity torque injected into node embeddings
  perc        — scalar β·M̃  +  PERC critic with w_H init=1
  no_physics  — matched graph transformer control, no M̃ bias
  shuffled    — control with M̃ values assigned to the wrong edges
  state_aug   — CGAT + learned raw padded graph feature path
  structured_state — CGAT + compact fixed-3-link physical state path
  feedback    — CGAT + zero-init learned linear feedback action path
  phys_residual — CGAT + zero-init physical-feature residual MLP
  gain_feedback — CGAT + learned parameter-conditioned feedback gains
  action_scale — gain_feedback + learned bounded raw-action scale
  clipped_action_scale — action_scale with clipped encoder edge features
  action_nonlinear — action_scale + learned nonlinear feedback residual
  energy_residual — action_scale + learned energy-feature residual
  norm_feedback — action_scale + normalized learned feedback residual
  linear_action_scale — gain_feedback + tiny learned linear raw-action scale
  nonlinear_feedback — gain_feedback + learned nonlinear feedback residual
  param_residual — gain_feedback + reciprocal/log-parameter residual
  param_action_scale — param_residual + learned bounded action scale
  mass_schedule — param_residual + trainable mass-conditioned raw-mean scale
  mass_force_schedule — param_residual + trainable mass-conditioned force scale
  poly_force_schedule — param_residual + polynomial length/mass force scale
  rbf_force_schedule — param_residual + RBF length/mass force scale
  rbf_gated_specialist — rbf_force_schedule + gated learned residual experts
  rbf_state_force_schedule — rbf_force_schedule + state-dependent rescue scale
  rbf_damping_schedule — rbf_force_schedule + compact learned damping residual
  force_residual — param_residual + bounded learned raw-action residual expert
  heavy_residual — param_residual + learned heavy-mass correction head
  state_param_residual — param_residual + direct physical feature MLP residual
  physical_gain — CGAT + neural physical-unit gain map
  velocity_bounded — gain_feedback with bounded angular-velocity inputs

Usage
-----
    from models.cgat import load_cgat_variant
    policy = load_cgat_variant("base", hidden=128, n_icga_layers=2, n_heads=2)
"""

from .cgat_base        import CGATBasePPOPolicy
from .cgat_perhead     import CGATPerHeadPPOPolicy
from .cgat_directional import CGATDirectionalPPOPolicy
from .cgat_gravity     import CGATGravityPPOPolicy
from .cgat_perc        import CGATPercPPOPolicy
from .cgat_no_physics  import CGATNoPhysicsPPOPolicy
from .cgat_shuffled    import CGATShuffledPPOPolicy
from .cgat_state_aug   import CGATStateAugPPOPolicy
from .cgat_structured_state import CGATStructuredStatePPOPolicy
from .cgat_feedback    import CGATFeedbackPPOPolicy
from .cgat_phys_residual import CGATPhysResidualPPOPolicy
from .cgat_gain_feedback import CGATGainFeedbackPPOPolicy
from .cgat_action_scale import CGATActionScalePPOPolicy
from .cgat_clipped_action_scale import CGATClippedActionScalePPOPolicy
from .cgat_action_nonlinear import CGATActionNonlinearPPOPolicy
from .cgat_energy_residual import CGATEnergyResidualPPOPolicy
from .cgat_norm_feedback import CGATNormFeedbackPPOPolicy
from .cgat_linear_action_scale import CGATLinearActionScalePPOPolicy
from .cgat_nonlinear_feedback import CGATNonlinearFeedbackPPOPolicy
from .cgat_param_residual import CGATParamResidualPPOPolicy
from .cgat_param_action_scale import CGATParamActionScalePPOPolicy
from .cgat_mass_schedule import CGATMassSchedulePPOPolicy
from .cgat_mass_force_schedule import CGATMassForceSchedulePPOPolicy
from .cgat_poly_force_schedule import CGATPolyForceSchedulePPOPolicy
from .cgat_rbf_force_schedule import CGATRBFForceSchedulePPOPolicy
from .cgat_rbf_gated_specialist import CGATRBFGatedSpecialistPPOPolicy
from .cgat_rbf_state_force_schedule import CGATRBFStateForceSchedulePPOPolicy
from .cgat_rbf_damping_schedule import CGATRBFDampingSchedulePPOPolicy
from .cgat_force_residual import CGATForceResidualPPOPolicy
from .cgat_heavy_residual import CGATHeavyResidualPPOPolicy
from .cgat_state_param_residual import CGATStateParamResidualPPOPolicy
from .cgat_physical_gain import CGATPhysicalGainPPOPolicy
from .cgat_velocity_bounded import CGATVelocityBoundedPPOPolicy

VARIANTS: dict = {
    "base":        CGATBasePPOPolicy,
    "perhead":     CGATPerHeadPPOPolicy,
    "directional": CGATDirectionalPPOPolicy,
    "gravity":     CGATGravityPPOPolicy,
    "perc":        CGATPercPPOPolicy,
    "no_physics":  CGATNoPhysicsPPOPolicy,
    "shuffled":    CGATShuffledPPOPolicy,
    "state_aug":   CGATStateAugPPOPolicy,
    "structured_state": CGATStructuredStatePPOPolicy,
    "feedback":    CGATFeedbackPPOPolicy,
    "phys_residual": CGATPhysResidualPPOPolicy,
    "gain_feedback": CGATGainFeedbackPPOPolicy,
    "action_scale": CGATActionScalePPOPolicy,
    "clipped_action_scale": CGATClippedActionScalePPOPolicy,
    "action_nonlinear": CGATActionNonlinearPPOPolicy,
    "energy_residual": CGATEnergyResidualPPOPolicy,
    "norm_feedback": CGATNormFeedbackPPOPolicy,
    "linear_action_scale": CGATLinearActionScalePPOPolicy,
    "nonlinear_feedback": CGATNonlinearFeedbackPPOPolicy,
    "param_residual": CGATParamResidualPPOPolicy,
    "param_action_scale": CGATParamActionScalePPOPolicy,
    "mass_schedule": CGATMassSchedulePPOPolicy,
    "mass_force_schedule": CGATMassForceSchedulePPOPolicy,
    "poly_force_schedule": CGATPolyForceSchedulePPOPolicy,
    "rbf_force_schedule": CGATRBFForceSchedulePPOPolicy,
    "rbf_gated_specialist": CGATRBFGatedSpecialistPPOPolicy,
    "rbf_state_force_schedule": CGATRBFStateForceSchedulePPOPolicy,
    "rbf_damping_schedule": CGATRBFDampingSchedulePPOPolicy,
    "force_residual": CGATForceResidualPPOPolicy,
    "heavy_residual": CGATHeavyResidualPPOPolicy,
    "state_param_residual": CGATStateParamResidualPPOPolicy,
    "physical_gain": CGATPhysicalGainPPOPolicy,
    "velocity_bounded": CGATVelocityBoundedPPOPolicy,
}


def load_cgat_variant(variant: str, **kwargs):
    """Instantiate a CGAT policy by variant name."""
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown CGAT variant '{variant}'. "
            f"Choose from: {list(VARIANTS)}"
        )
    return VARIANTS[variant](**kwargs)
