# Run (via training/train_hgfn.py): python3.12 training/train_hgfn.py --policy hgfn

"""
Hamiltonian Graph Flow Network (HGFN) — actor-critic for PPO.

Three novel components (all operating on the same observations as GNNTransformerPPO):

  1. Recursive Inertia Module (RIM)
     ─────────────────────────────
     Bidirectional message-passing that mirrors the Articulated Body Algorithm:
       • Backward pass (leaf → root): aggregates inertia information up the chain.
       • Forward  pass (root → leaf): propagates control context back down.
     Standard MPNNs send identical messages in both directions; RIM enforces
     the physical directionality of force/inertia propagation.

  2. Inertia-Coupled Graph Attention (ICGA)
     ───────────────────────────────────────
     The normalised inertia coupling M̃ᵢⱼ(q; L, m) is added as an analytic
     physics bias to each attention logit:

         logit_ij = Q·K/√d  +  β · M̃ᵢⱼ  +  w_e · e_ij

     M̃ᵢⱼ = Mᵢⱼ / √(Mᵢᵢ · Mⱼⱼ)  ∈ [-1, 1]

     is the normalised entry of the Lagrangian mass matrix. It encodes how
     strongly joints i and j are kinematically coupled — high when they share
     heavy distal mass, near zero when mechanically decoupled. β is a learned
     scalar initialised to 0 (degrades to standard transformer at β=0).

     The mass matrix is computed analytically from the existing node/edge
     features — no new information enters the observation.

  3. Potential Energy Residual Critic  (PERC)
     ──────────────────────────────────────────
     The value function is decomposed as:

         V(s) = w_H · V̂_pot(q; L, m)  +  f_θ(z_GNN)

     where V̂_pot = g Σᵢ mᵢ hᵢ(q) / H_scale is the analytically-computed
     potential energy (height of each link's CoM), and w_H is a learned scalar
     initialised to 0.  Using V_pot rather than T+V avoids dependence on the
     unobservable cart mass and removes the kinetic ambiguity (high T can mean
     active balancing OR falling fast).  V_pot is always positive, always exact,
     and monotonically correlated with upright-ness — an ideal physics prior.

Math reference (2-link, generalises to n-link):
  Mᵢⱼ(q) = Σₖ≥max(i,j) mₖ · Jᵢₖ(q) · Jⱼₖ(q) + δᵢⱼ Iᵢ
  T       = ½ q̇ᵀ M(q) q̇
  V_pot   = g Σₖ mₖ hₖ(q)   (height of each centre-of-mass above cart)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_ppo import BasePPOPolicy

NODE_FEAT_DIM = 8
EDGE_FEAT_DIM = 2

# ── Denormalisation constants (must match graph/graph_builder.py) ──────────────
_LEN_MIN    = 0.3;  _LEN_RANGE  = 0.9   # L = norm * 0.9 + 0.3
_MASS_MIN   = 0.1;  _MASS_RANGE = 1.9   # m = norm * 1.9 + 0.1
_CART_VEL   = 5.0                        # ẋ normalisation
_ANG_VEL    = 10.0                       # θ̇  normalisation
_G          = 9.81                       # gravitational acceleration
_M_CART_EST = 1.75                       # avg of [0.5, 3.0]; not observable
_H_SCALE    = 20.0                       # rough energy normalisation constant


# ═══════════════════════════════════════════════════════════════════════════════
# Physics helpers  (pure functions, no learned parameters)
# ═══════════════════════════════════════════════════════════════════════════════

def _rod_tensors(obs: dict):
    """
    Extract per-rod (L, m, sin_th, cos_th, theta_dot) and a validity mask.
    Rod i ↔ forward edge 2*i (same features as backward edge 2*i+1).

    Returns
    -------
    L, m, sin_th, cos_th, theta_dot : each (B, max_links)
    rod_valid                        : (B, max_links) bool
    """
    node_feats = obs["node_features"].float()   # (B, max_nodes, 8)
    edge_feats = obs["edge_features"].float()   # (B, max_edges, 2)
    n_nodes    = obs["n_nodes"].long()           # (B, 1)

    B, max_nodes, _ = node_feats.shape
    max_links = max_nodes - 1
    device    = node_feats.device

    # Forward edges are at even positions 0, 2, 4, …
    rod_ef    = edge_feats[:, 0::2, :]           # (B, max_links, 2)
    L         = rod_ef[..., 0] * _LEN_RANGE  + _LEN_MIN
    m         = rod_ef[..., 1] * _MASS_RANGE + _MASS_MIN

    n_links   = n_nodes.squeeze(-1) - 1         # (B,)
    rod_valid = (torch.arange(max_links, device=device)
                 .unsqueeze(0) < n_links.unsqueeze(1))   # (B, max_links) bool

    L = L * rod_valid.float()
    m = m * rod_valid.float()

    sin_th    = node_feats[:, 1:, 3]            # (B, max_links)
    cos_th    = node_feats[:, 1:, 4]            # (B, max_links)
    theta_dot = node_feats[:, 1:, 5] * _ANG_VEL  # (B, max_links)

    return L, m, sin_th, cos_th, theta_dot, rod_valid


def compute_inertia_coupling(obs: dict) -> torch.Tensor:
    """
    Analytically compute the normalised Lagrangian mass matrix M̃(q).

    Entry M̃ᵢⱼ = Mᵢⱼ / √(Mᵢᵢ · Mⱼⱼ) ∈ [-1, 1].
    Node ordering: 0 = cart, 1..n_links = joints.

    Derivation (row i, col j with q = [x, θ₁, …, θₙ]):
      M_{0j} = Lⱼ cos(θⱼ) · [Σₖ≥ⱼ mₖ  − mⱼ/2]
      M_{jj} = Lⱼ² · [mⱼ/3 + Σₖ>ⱼ mₖ]
      M_{jk} = Lⱼ Lₖ cos(θⱼ−θₖ) · [Σₖ'≥ₖ mₖ' − mₖ/2]   (j < k)

    Returns
    -------
    M_tilde : (B, max_nodes, max_nodes) float32
    """
    L, m, sin_th, cos_th, _, rod_valid = _rod_tensors(obs)
    B = L.shape[0]
    max_links = L.shape[1]
    max_nodes = max_links + 1
    device    = L.device

    # Distal mass sums: distal[:, j] = Σₖ≥ⱼ m[:,k]
    distal     = torch.flip(torch.cumsum(torch.flip(m, [1]), 1), [1])  # (B, max_links)
    distal_excl = distal - m                                            # Σₖ>ⱼ mₖ

    M = torch.zeros(B, max_nodes, max_nodes, device=device)

    # M[0,0]
    M[:, 0, 0] = _M_CART_EST + m.sum(dim=1)

    # M[0,j] = M[j,0]
    M_0j = L * cos_th * (distal - m / 2) * rod_valid.float()   # (B, max_links)
    M[:, 0, 1:] = M_0j
    M[:, 1:, 0] = M_0j

    # M[j,j] diagonal
    M_diag = L ** 2 * (m / 3 + distal_excl) * rod_valid.float()
    for j in range(max_links):
        M[:, j + 1, j + 1] = M_diag[:, j]

    # M[j,k] off-diagonal  (j < k)
    for j in range(max_links):
        for k in range(j + 1, max_links):
            cos_jk = cos_th[:, j] * cos_th[:, k] + sin_th[:, j] * sin_th[:, k]
            M_jk   = L[:, j] * L[:, k] * cos_jk * (distal[:, k] - m[:, k] / 2)
            valid  = (rod_valid[:, j] & rod_valid[:, k]).float()
            M_jk   = M_jk * valid
            M[:, j + 1, k + 1] = M_jk
            M[:, k + 1, j + 1] = M_jk

    # Normalise: M̃ᵢⱼ = Mᵢⱼ / √(Mᵢᵢ Mⱼⱼ)
    diag_sqrt = M.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6).sqrt()  # (B, max_nodes)
    denom     = (diag_sqrt.unsqueeze(-1) * diag_sqrt.unsqueeze(-2)).clamp(min=1e-6)
    return M / denom                                                   # (B, max_nodes, max_nodes)


def compute_hamiltonian(obs: dict) -> torch.Tensor:
    """
    Analytically compute normalised potential energy V_pot / H_scale.

    WHY only V_pot, not T + V:
      Kinetic energy T requires cart mass, which is unobservable.  Using a
      fixed estimate (_M_CART_EST) introduces systematic errors of up to ±50%
      during training (cart mass ~ Uniform[0.5, 3.0]).  Worse, high T often
      means the pendulum is moving fast — which is correlated with *bad* states
      (recovering from, or in the process of, falling) just as much as good
      ones.  In practice this causes w_H to learn a negative value, actively
      fighting the critic.

      V_pot = g Σᵢ mᵢ hᵢ(q) is always exact (no cart mass needed) and is
      monotonically correlated with upright-ness: maximum when all cos(θ) = 1
      (perfectly vertical), decreasing as links tilt.  This is a clean,
      noise-free physics prior for the value function.

    Returns
    -------
    V_norm : (B,) float32, positive; maximum (upright) ≈ 1.0
    """
    L, m, _, cos_th, _, rod_valid = _rod_tensors(obs)
    rv = rod_valid.float()

    # h_com[i] = Σⱼ<ᵢ Lⱼ cos(θⱼ)  +  (Lᵢ/2) cos(θᵢ)  (height of CoM above cart)
    L_cos     = L * cos_th
    cum_L_cos = torch.cumsum(L_cos, dim=1) - L_cos     # exclusive prefix sum
    h_com     = cum_L_cos + L * cos_th / 2
    V         = _G * (m * h_com * rv).sum(dim=1)       # (B,)

    return V / _H_SCALE                                 # (B,)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Recursive Inertia Module (RIM)
# ═══════════════════════════════════════════════════════════════════════════════

class RIMLayer(nn.Module):
    """
    One RIM layer: backward pass (leaf→root) then forward pass (root→leaf).

    Backward edges: src > dst  (odd-indexed in our graph convention).
    Forward  edges: src < dst  (even-indexed).

    Each pass:
      message  = ReLU( W_msg  [ h_src ∥ e_enc ] )   ← e_enc is tanh-bounded
      agg      = mean of incoming messages at each destination node
      h_new    = ReLU( W_upd  [ h ∥ agg ] )
      h        = LayerNorm( h + h_new )               ← residual

    edge_enc_dim: dimensionality of the pre-encoded (tanh-bounded) edge features.
    Raw normalised edge features (norm_len, norm_mass) go out of [0,1] for OOD
    parameters and cause nonlinear ReLU activation patterns to collapse.  The
    caller encodes them with Tanh first so messages stay well-behaved OOD.
    """

    def __init__(self, hidden: int, edge_enc_dim: int):
        super().__init__()
        # Backward (leaf→root)
        self.back_msg  = nn.Linear(hidden + edge_enc_dim, hidden)
        self.back_upd  = nn.Linear(hidden + hidden, hidden)
        self.back_norm = nn.LayerNorm(hidden)

        # Forward (root→leaf)
        self.fwd_msg   = nn.Linear(hidden + edge_enc_dim, hidden)
        self.fwd_upd   = nn.Linear(hidden + hidden, hidden)
        self.fwd_norm  = nn.LayerNorm(hidden)

    def _scatter_pass(self, h, edge_features, src_idx, dst_idx, which_mask,
                      msg_fn, upd_fn, norm_fn):
        """Generic directed scatter-aggregate-update."""
        B, max_nodes, hidden = h.shape
        max_edges = src_idx.shape[1]

        src_exp  = src_idx.unsqueeze(-1).expand(B, max_edges, hidden)
        h_src    = h.gather(1, src_exp)                              # (B, E, hidden)

        msgs = F.relu(msg_fn(torch.cat([h_src,
                                        edge_features.float()], dim=-1)))
        msgs = msgs * which_mask.float().unsqueeze(-1)               # zero pad edges

        agg    = torch.zeros_like(h)
        counts = torch.zeros(B, max_nodes, 1, device=h.device)
        dst_e  = dst_idx.unsqueeze(-1).expand_as(msgs)
        agg.scatter_add_(1, dst_e, msgs)
        counts.scatter_add_(1, dst_idx.unsqueeze(-1),
                             which_mask.float().unsqueeze(-1))
        agg = agg / counts.clamp(min=1.0)

        h_new = F.relu(upd_fn(torch.cat([h, agg], dim=-1)))
        return norm_fn(h + h_new)

    def forward(self, h, edge_features, edge_index, n_edges):
        """
        h            : (B, max_nodes, hidden)
        edge_features: (B, max_edges, 2)
        edge_index   : (B, 2, max_edges)
        n_edges      : (B, 1)
        Returns      : (B, max_nodes, hidden)
        """
        B, max_nodes, _ = h.shape
        max_edges = edge_index.shape[2]
        device    = h.device

        src_idx   = edge_index[:, 0, :]                              # (B, max_edges)
        dst_idx   = edge_index[:, 1, :]

        edge_mask = (torch.arange(max_edges, device=device).unsqueeze(0)
                     < n_edges.squeeze(-1).unsqueeze(-1))            # (B, max_edges)

        back_mask = edge_mask & (src_idx > dst_idx)  # leaf→root
        fwd_mask  = edge_mask & (src_idx < dst_idx)  # root→leaf

        # Pass 1: backward
        h = self._scatter_pass(h, edge_features, src_idx, dst_idx,
                                back_mask, self.back_msg, self.back_upd, self.back_norm)
        # Pass 2: forward
        h = self._scatter_pass(h, edge_features, src_idx, dst_idx,
                                fwd_mask, self.fwd_msg, self.fwd_upd, self.fwd_norm)
        return h


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Inertia-Coupled Graph Attention (ICGA)
# ═══════════════════════════════════════════════════════════════════════════════

class ICGALayer(nn.Module):
    """
    Graph attention layer with an analytic physics bias on attention logits:

        logit_ij = Q_i · K_j / √d_head  +  β · M̃ᵢⱼ  +  w_e · e_ij
                                              ↑ physics (analytic)

    β is a single learned scalar shared across all heads (init=0 → degrades
    to standard graph transformer at the start of training).

    Numerically stable softmax via the log-sum-exp trick (identical to
    gnn_transformer_ppo.py to avoid NaN on overflow).
    """

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = hidden // n_heads

        self.W_q = nn.Linear(hidden, hidden, bias=False)
        self.W_k = nn.Linear(hidden, hidden, bias=False)
        self.W_v = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(hidden, hidden)

        self.edge_bias    = nn.Linear(EDGE_FEAT_DIM, n_heads, bias=False)
        # β: physics-coupling scale — init=0 so model starts as standard transformer
        self.physics_beta = nn.Parameter(torch.zeros(1))

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff    = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, h, edge_features, edge_index, n_edges, M_tilde):
        """
        h            : (B, max_nodes, hidden)
        edge_features: (B, max_edges, 2)
        edge_index   : (B, 2, max_edges)
        n_edges      : (B, 1)
        M_tilde      : (B, max_nodes, max_nodes)  analytic inertia coupling
        Returns      : (B, max_nodes, hidden)
        """
        B, max_nodes, hidden = h.shape
        max_edges = edge_index.shape[2]
        H, D = self.n_heads, self.d_head
        device = h.device

        src_idx = edge_index[:, 0, :]   # (B, max_edges)
        dst_idx = edge_index[:, 1, :]   # (B, max_edges)

        # Project nodes to Q, K, V
        Q = self.W_q(h).view(B, max_nodes, H, D)
        K = self.W_k(h).view(B, max_nodes, H, D)
        V = self.W_v(h).view(B, max_nodes, H, D)

        # Gather per-edge Q (dst), K (src), V (src)
        src_e = src_idx.unsqueeze(-1).unsqueeze(-1).expand(B, max_edges, H, D)
        dst_e = dst_idx.unsqueeze(-1).unsqueeze(-1).expand(B, max_edges, H, D)
        Q_e   = Q.gather(1, dst_e)    # (B, E, H, D)
        K_e   = K.gather(1, src_e)
        V_e   = V.gather(1, src_e)

        # Scaled dot-product logits
        logits = (Q_e * K_e).sum(-1) * (D ** -0.5)    # (B, E, H)

        # Learned edge-feature bias (existing technique)
        logits = logits + self.edge_bias(edge_features.float())  # (B, E, H)

        # ── Physics bias: β · M̃ᵢⱼ ──────────────────────────────────────────
        # Look up M̃[b, src, dst] for every edge (b, src→dst)
        # M_tilde: (B, max_nodes, max_nodes)
        # We need M_tilde[b, src_idx[b,e], dst_idx[b,e]] for each (b, e)
        src_flat = src_idx * max_nodes + dst_idx   # (B, E) — linear index into N×N
        M_flat   = M_tilde.view(B, max_nodes * max_nodes)
        M_edge   = M_flat.gather(1, src_flat)      # (B, E)  — M̃ per edge
        # Add as head-shared bias (broadcast across H).
        # tanh bounds β ∈ (-1, 1) — prevents attention from becoming hypersensitive
        # to M̃ values and destabilising when M̃ distribution shifts OOD.
        logits   = logits + torch.tanh(self.physics_beta) * M_edge.unsqueeze(-1)
        # ──────────────────────────────────────────────────────────────────────

        # Numerically stable scatter-softmax (log-sum-exp trick)
        edge_mask = (torch.arange(max_edges, device=device).unsqueeze(0)
                     < n_edges.squeeze(-1).unsqueeze(-1))            # (B, E)
        dst_h     = dst_idx.unsqueeze(-1).expand(B, max_edges, H)   # (B, E, H)

        logits    = logits.masked_fill(~edge_mask.unsqueeze(-1), float("-inf"))
        node_max  = torch.full((B, max_nodes, H), float("-inf"), device=device)
        node_max.scatter_reduce_(1, dst_h, logits, reduce="amax", include_self=True)
        node_max  = node_max.clamp(min=-1e9)
        max_edge  = node_max.gather(1, dst_h)

        logits_exp = (logits - max_edge).exp() * edge_mask.unsqueeze(-1).float()
        denom      = torch.zeros(B, max_nodes, H, device=device)
        denom.scatter_add_(1, dst_h, logits_exp)
        alpha      = logits_exp / denom.gather(1, dst_h).clamp(min=1e-6)  # (B, E, H)

        # Weighted value aggregation
        weighted = alpha.unsqueeze(-1) * V_e                         # (B, E, H, D)
        out      = torch.zeros(B, max_nodes, H, D, device=device)
        dst_hd   = dst_idx.unsqueeze(-1).unsqueeze(-1).expand(B, max_edges, H, D)
        out.scatter_add_(1, dst_hd, weighted)
        out      = self.W_o(out.reshape(B, max_nodes, hidden))

        h = self.norm1(h + out)
        h = self.norm2(h + self.ff(h))
        return h


# ═══════════════════════════════════════════════════════════════════════════════
# HGFN Encoder: node_embed → RIM → ICGA layers
# ═══════════════════════════════════════════════════════════════════════════════

class HGFNEncoder(nn.Module):
    """
    node_embed  →  RIM (with tanh-encoded edge features)  →  ICGA layers

    Edge features are encoded separately for RIM vs ICGA:
      • RIM uses a tanh-bounded encoder (hidden//4 dims).  Raw norm features go
        OOD for unseen link lengths (norm_len > 1), causing ReLU activation
        collapse in RIM messages.  Tanh saturates gracefully instead.
      • ICGA uses raw features directly in its linear edge_bias — identical to
        GNNTransformerPPO, which already generalises well OOD with this pattern.
    """

    def __init__(self, hidden: int, n_icga_layers: int, n_heads: int):
        super().__init__()
        self.node_embed  = nn.Linear(NODE_FEAT_DIM, hidden)

        # Tanh-bounded edge encoder for RIM.
        # Output is bounded regardless of raw input magnitude → OOD-safe.
        edge_enc_dim = max(hidden // 4, 4)
        self.rim_edge_enc = nn.Sequential(
            nn.Linear(EDGE_FEAT_DIM, edge_enc_dim),
            nn.Tanh(),
        )
        self.rim         = RIMLayer(hidden, edge_enc_dim)

        self.icga_layers = nn.ModuleList(
            [ICGALayer(hidden, n_heads) for _ in range(n_icga_layers)]
        )

    def forward(self, obs: dict, M_tilde: torch.Tensor) -> torch.Tensor:
        """Returns global graph embedding of shape (B, hidden)."""
        node_features = obs["node_features"].float()   # (B, max_nodes, 8)
        edge_index    = obs["edge_index"].long()        # (B, 2, max_edges)
        edge_features = obs["edge_features"].float()   # (B, max_edges, 2)
        n_nodes       = obs["n_nodes"].long()           # (B, 1)
        n_edges       = obs["n_edges"].long()           # (B, 1)

        B, max_nodes, _ = node_features.shape

        h = self.node_embed(node_features)              # (B, max_nodes, hidden)

        # RIM: tanh-encoded edge features (OOD-safe)
        edge_enc = self.rim_edge_enc(edge_features)     # (B, max_edges, edge_enc_dim)
        h = self.rim(h, edge_enc, edge_index, n_edges)

        # ICGA: raw edge features for linear bias (same as GNNTransformerPPO)
        for layer in self.icga_layers:
            h = layer(h, edge_features, edge_index, n_edges, M_tilde)

        # Masked mean pool over real nodes
        node_mask = (torch.arange(max_nodes, device=h.device).unsqueeze(0)
                     < n_nodes.squeeze(-1).unsqueeze(-1))            # (B, max_nodes)
        h = h * node_mask.unsqueeze(-1).float()
        return h.sum(dim=1) / n_nodes.float()                        # (B, hidden)


# ═══════════════════════════════════════════════════════════════════════════════
# HGFN PPO Policy
# ═══════════════════════════════════════════════════════════════════════════════

class HGFNPPOPolicy(BasePPOPolicy):
    """
    Full HGFN actor-critic.

    Inherits the actor/critic trunk and action-sampling logic from BasePPOPolicy.
    Overrides get_value and get_action_and_value to inject the Hamiltonian
    residual into the value estimate:

        V(s) = value_head( critic_trunk( z_GNN ) )  +  w_H · Ĥ(s)
                ↑ learned from rollout rewards              ↑ analytic physics
    """

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 4,
                 max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.encoder = HGFNEncoder(hidden, n_icga_layers, n_heads)
        # w_H: Hamiltonian mixing weight — init=0, learns from data
        self.w_H = nn.Parameter(torch.zeros(1))

    # BasePPOPolicy.encode is required; we satisfy the abstract interface here
    def encode(self, obs: dict) -> torch.Tensor:
        M_tilde = compute_inertia_coupling(obs)
        return self.encoder(obs, M_tilde)

    # ── Override to inject Hamiltonian residual ──────────────────────────────

    def get_value(self, obs: dict) -> torch.Tensor:
        M_tilde = compute_inertia_coupling(obs)
        z       = self.encoder(obs, M_tilde)
        H_norm  = compute_hamiltonian(obs)                 # (B,)
        v_gnn   = self.value_head(self.critic_trunk(z))   # (B, 1)
        return v_gnn + self.w_H * H_norm.unsqueeze(-1)    # (B, 1)

    def get_action_and_value(self, obs: dict, action=None):
        """
        Identical to BasePPOPolicy.get_action_and_value except:
          • Physics ops are computed once and shared between actor + critic.
          • Value = GNN value + w_H · Ĥ(s).
        """
        import numpy as np

        M_tilde  = compute_inertia_coupling(obs)
        z        = self.encoder(obs, M_tilde)
        H_norm   = compute_hamiltonian(obs)                # (B,)

        actor_h  = self.actor_trunk(z)
        critic_h = self.critic_trunk(z)

        raw_mean = self.mean_head(actor_h)                 # (B, 1)

        from models.base_ppo import LOG_STD_MIN, LOG_STD_MAX
        log_std  = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std      = log_std.exp().expand_as(raw_mean)

        dist     = torch.distributions.Normal(raw_mean, std)

        if action is None:
            raw_action = dist.rsample()
        else:
            a_norm     = (action / self.max_force).clamp(-1 + 1e-6, 1 - 1e-6)
            raw_action = torch.atanh(a_norm)

        squashed = torch.tanh(raw_action) * self.max_force

        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - torch.log(
            self.max_force * (1.0 - torch.tanh(raw_action).pow(2)) + 1e-6
        )
        log_prob = log_prob.squeeze(-1)

        entropy  = dist.entropy().squeeze(-1)

        # Hamiltonian residual critic
        v_gnn    = self.value_head(critic_h)               # (B, 1)
        value    = v_gnn + self.w_H * H_norm.unsqueeze(-1)

        return squashed, log_prob, entropy, value
