"""
Trajectory definitions for flow matching in box state space.

RiemannianTrajectory (proposed):
    Operates in state space [cx, cy, log_w, log_h].
    Linear interpolation in this space = geodesic on ℝ² × ℝ₊²
    because log-scale makes the scale dimensions Euclidean.

LinearTrajectory (Euclidean baseline):
    Operates in normalized cxcywh space.
    Linear interpolation in cxcywh = Euclidean straight line, NOT a geodesic
    on ℝ₊² for the width/height dimensions.
    Vector field converted to state space for model training.

OTCoupledTrajectory (structural coupling fix, phase 3 P3-OT):
    Wrapper that Hungarian-matches (b₀, b₁) pairs **per image** before
    interpolation. Resolves the "crossing trajectories → null marginal field"
    pathology observed in e2 when prior support overlaps target support
    (see docs/ISSUES.md, docs/plans/ot_coupling_plan.md).

API contract (every base trajectory class must expose):
    sample(b1, t)           → (b_t, u_t, b0)   — b0 in the class's native space
    _make_trajectory(b0, b1, t) → (b_t, u_t)   — pure formula from (b0, b1, t)
    _b0_native_space         → "state" | "cxcywh"   — tells OT wrapper the space
                                                      of b0 so cost is consistent
    init_noise, ode_step     → inference helpers (OT wrapper passes them through)
"""

import torch
from dataset.box_ops import cxcywh_to_state, state_to_cxcywh


class RiemannianTrajectory:
    """
    Geodesic on ℝ² × ℝ₊² — linear interpolation in log-scale state space.

    b₀   ~ N(0, I) in state space [cx, cy, log_w, log_h]
    b₁   = cxcywh_to_state(boxes_gt)
    b_t  = (1-t)·b₀ + t·b₁   (geodesic in ℝ²×ℝ₊²)
    u_t* = b₁ − b₀            (constant vector field in state)
    """

    _b0_native_space = "state"

    def init_noise(self, B: int, Q: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.randn(B, Q, 4, device=device, dtype=dtype)

    def _make_trajectory(
        self, b0: torch.Tensor, b1: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure formula: given (b0, b1, t) compute (b_t, u_t)."""
        t_  = t[:, None, None]
        b_t = (1.0 - t_) * b0 + t_ * b1
        u_t = b1 - b0
        return b_t, u_t

    def sample(
        self, b1: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs:
            b1: [B, N, 4] target in state space
            t:  [B]       timestep in [0, 1]
        Outputs:
            b_t, u_t, b0  — all [B, N, 4] in state space
        """
        b0 = torch.randn_like(b1)
        b_t, u_t = self._make_trajectory(b0, b1, t)
        return b_t, u_t, b0

    def ode_step(
        self, b_t: torch.Tensor, v_t: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        return b_t + dt * v_t


class LinearTrajectory:
    """
    Euclidean baseline — linear interpolation in normalized cxcywh space.
    Vector field converted to state space for the model.

    b₀ ~ N(0, I) in state → state_to_cxcywh → b₀_cx (log-normal)
    b_t_cx = (1-t)·b₀_cx + t·b₁_cx           (linear in cxcywh)
    b_t    = cxcywh_to_state(b_t_cx)           (state for model input)
    u_t*   = [Δcx, Δcy, Δw/w_t, Δh/h_t]        (time-dependent in state)

    Returned b0 is in **cxcywh** space (not state) — OT wrapper matches
    in cxcywh, which is the space where interpolation happens.
    """

    _b0_native_space = "cxcywh"

    def init_noise(self, B: int, Q: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.randn(B, Q, 4, device=device, dtype=dtype)

    def _make_trajectory(
        self, b0_cx: torch.Tensor, b1_cx: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_     = t[:, None, None]
        b_t_cx = (1.0 - t_) * b0_cx + t_ * b1_cx
        b_t    = cxcywh_to_state(b_t_cx)

        dcx = b1_cx[..., 0] - b0_cx[..., 0]
        dcy = b1_cx[..., 1] - b0_cx[..., 1]
        w_t = b_t_cx[..., 2].clamp(min=1e-6)
        h_t = b_t_cx[..., 3].clamp(min=1e-6)
        dw  = b1_cx[..., 2] - b0_cx[..., 2]
        dh  = b1_cx[..., 3] - b0_cx[..., 3]
        u_t = torch.stack([dcx, dcy, dw / w_t, dh / h_t], dim=-1)
        return b_t, u_t

    def sample(
        self, b1_cx: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs:
            b1_cx: [B, N, 4] target in normalized cxcywh
            t:     [B]       timestep in [0, 1]
        Outputs:
            b_t:   [B, N, 4] interpolated state (state space)
            u_t:   [B, N, 4] target vector field (state space)
            b0_cx: [B, N, 4] prior sample in cxcywh
        """
        b0    = torch.randn_like(b1_cx)     # state
        b0_cx = state_to_cxcywh(b0)         # cxcywh (log-normal w/h)
        b_t, u_t = self._make_trajectory(b0_cx, b1_cx, t)
        return b_t, u_t, b0_cx

    def ode_step(
        self, b_t: torch.Tensor, v_t: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        return b_t + dt * v_t


class RiemannianTrajectoryArbPrior:
    """
    Riemannian (state interp) + arbitrary cxcywh clipped Gaussian prior.
    """

    _b0_native_space = "state"

    def __init__(
        self, mu: float = 0.5, sigma: float = 1.0 / 6.0, eps: float = 0.02,
    ):
        self.mu, self.sigma, self.eps = mu, sigma, eps

    def _sample_b0_cx(self, shape, device, dtype) -> torch.Tensor:
        b0 = self.mu + self.sigma * torch.randn(shape, device=device, dtype=dtype)
        return b0.clamp(min=self.eps, max=1.0)

    def init_noise(self, B: int, Q: int, device, dtype=torch.float32) -> torch.Tensor:
        b0_cx = self._sample_b0_cx((B, Q, 4), device, dtype)
        return cxcywh_to_state(b0_cx)

    def _make_trajectory(
        self, b0: torch.Tensor, b1: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same formula as RiemannianTrajectory — b0 here is already in state."""
        t_  = t[:, None, None]
        b_t = (1.0 - t_) * b0 + t_ * b1
        u_t = b1 - b0
        return b_t, u_t

    def sample(
        self, b1: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b0_cx = self._sample_b0_cx(b1.shape, b1.device, b1.dtype)
        b0    = cxcywh_to_state(b0_cx)
        b_t, u_t = self._make_trajectory(b0, b1, t)
        return b_t, u_t, b0

    def ode_step(
        self, b_t: torch.Tensor, v_t: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        return b_t + dt * v_t


class LinearTrajectoryArbPrior:
    """
    Euclidean (cxcywh interp) + arbitrary cxcywh clipped Gaussian prior.
    """

    _b0_native_space = "cxcywh"

    def __init__(
        self, mu: float = 0.5, sigma: float = 1.0 / 6.0, eps: float = 0.02,
    ):
        self.mu, self.sigma, self.eps = mu, sigma, eps

    def _sample_b0_cx(self, shape, device, dtype) -> torch.Tensor:
        b0 = self.mu + self.sigma * torch.randn(shape, device=device, dtype=dtype)
        return b0.clamp(min=self.eps, max=1.0)

    def init_noise(self, B: int, Q: int, device, dtype=torch.float32) -> torch.Tensor:
        b0_cx = self._sample_b0_cx((B, Q, 4), device, dtype)
        return cxcywh_to_state(b0_cx)

    def _make_trajectory(
        self, b0_cx: torch.Tensor, b1_cx: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_     = t[:, None, None]
        b_t_cx = (1.0 - t_) * b0_cx + t_ * b1_cx
        b_t    = cxcywh_to_state(b_t_cx)

        dcx = b1_cx[..., 0] - b0_cx[..., 0]
        dcy = b1_cx[..., 1] - b0_cx[..., 1]
        w_t = b_t_cx[..., 2].clamp(min=1e-6)
        h_t = b_t_cx[..., 3].clamp(min=1e-6)
        dw  = b1_cx[..., 2] - b0_cx[..., 2]
        dh  = b1_cx[..., 3] - b0_cx[..., 3]
        u_t = torch.stack([dcx, dcy, dw / w_t, dh / h_t], dim=-1)
        return b_t, u_t

    def sample(
        self, b1_cx: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b0_cx = self._sample_b0_cx(b1_cx.shape, b1_cx.device, b1_cx.dtype)
        b_t, u_t = self._make_trajectory(b0_cx, b1_cx, t)
        return b_t, u_t, b0_cx

    def ode_step(
        self, b_t: torch.Tensor, v_t: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        return b_t + dt * v_t


# ──────────────────────────────────────────────────────────────────────────────
#  OT coupling wrapper
# ──────────────────────────────────────────────────────────────────────────────

class OTCoupledTrajectory:
    """
    Mini-batch OT-coupled trajectory wrapper.

    Given any base trajectory (Riemannian/Linear [+ArbPrior]), this wrapper:
      1) samples an independent (b₀, b₁) as usual,
      2) solves a **per-image Hungarian assignment** between b₀ and b₁ in the
         base's native space (state for Riemannian, cxcywh for Linear),
      3) reorders b₀ along the query dim accordingly,
      4) re-evaluates the trajectory formula with the reordered b₀.

    Effect: removes "crossing trajectories" at the source (=matches each GT
    query q with its nearest b₀), restoring a well-posed marginal vector
    field `v*(x, t)` even when prior support overlaps target support.

    Plan: docs/plans/ot_coupling_plan.md.
    Resolves: docs/ISSUES.md "arb_prior 가 학습 안 됨".

    Notes:
    - Inference is unchanged (`init_noise` / `ode_step` pass-through).
    - Hungarian cost: L2 in the base's native space.
    - Training-only behavior. `scipy.optimize.linear_sum_assignment` runs on CPU;
      for Q≤300 the cost is negligible.
    """

    def __init__(self, base):
        self.base = base

    # --- inference helpers (delegate) ------------------------------------------
    def init_noise(self, *args, **kwargs):
        return self.base.init_noise(*args, **kwargs)

    def ode_step(self, *args, **kwargs):
        return self.base.ode_step(*args, **kwargs)

    # --- training sample -------------------------------------------------------
    def sample(
        self, b1: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b_t, u_t, b0 = self.base.sample(b1, t)       # independent first
        b0_ot = self._hungarian_reorder(b0, b1)
        b_t_ot, u_t_ot = self.base._make_trajectory(b0_ot, b1, t)
        return b_t_ot, u_t_ot, b0_ot

    # --- Hungarian per image ---------------------------------------------------
    @staticmethod
    def _hungarian_reorder(
        b0: torch.Tensor, b1: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each batch element b, find the permutation π over the Q query slots
        that minimizes Σ_q ‖b0[π(q)] − b1[q]‖₂. Returns b0 reordered such that
        reordered[b, q] = b0[b, π(q)].

        Inputs:
            b0, b1: [B, Q, 4]  (same space)
        Output:
            b0_ot:  [B, Q, 4]
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as e:
            raise RuntimeError(
                "OTCoupledTrajectory needs scipy. pip install scipy."
            ) from e
        import numpy as np

        B, Q, _ = b0.shape
        # L2 cost
        cost = torch.cdist(b0, b1, p=2.0)       # [B, Q, Q]
        cost_np = cost.detach().cpu().numpy()

        perm = np.zeros((B, Q), dtype=np.int64)
        for i in range(B):
            # row[j] = b0 index, col[j] = b1 index it's matched to
            row, col = linear_sum_assignment(cost_np[i])
            # For each b1 slot q, find which b0 index was assigned:
            #   inv[col[j]] = row[j]
            inv = np.empty(Q, dtype=np.int64)
            inv[col] = row
            perm[i] = inv
        perm_t = torch.from_numpy(perm).to(b0.device)               # [B, Q]
        idx    = perm_t.unsqueeze(-1).expand(-1, -1, b0.shape[-1])  # [B, Q, 4]
        return torch.gather(b0, dim=1, index=idx)

    # --- expose native space (in case callers need it) -------------------------
    @property
    def _b0_native_space(self):
        return self.base._b0_native_space


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== trajectory.py sanity check ===")
    from dataset.box_ops import cxcywh_to_state

    B, N = 2, 5
    boxes_gt = torch.rand(B, N, 4) * 0.5 + 0.1
    t = torch.rand(B)

    # ── RiemannianTrajectory ─────────────────────────────────────────────────
    traj = RiemannianTrajectory()
    b1   = cxcywh_to_state(boxes_gt)

    b_t, u_t, b0 = traj.sample(b1, t)
    assert b_t.shape == (B, N, 4) and u_t.shape == (B, N, 4) and b0.shape == (B, N, 4)
    print(f"RiemannianTrajectory sample: b_t {b_t.shape}  ✓")

    b_t1, _, _ = traj.sample(b1, torch.ones(B))
    assert torch.allclose(b_t1, b1, atol=1e-5)
    assert torch.allclose(u_t, b1 - b0)
    print("  endpoint check + constant field  ✓")

    # ── LinearTrajectory — now returns b0_cx ────────────────────────────────
    lin_traj = LinearTrajectory()
    b_t_lin, u_t_lin, b0_lin = lin_traj.sample(boxes_gt, t)
    assert b_t_lin.shape == (B, N, 4) and b0_lin.shape == (B, N, 4)
    assert (b0_lin[..., 2:] > 0).all(), "b0_cx w/h must be > 0 (log-normal)"
    print(f"LinearTrajectory sample: b_t {b_t_lin.shape}, b0_cx {b0_lin.shape}  ✓")

    # ── LinearTrajectoryArbPrior — returns b0_cx clipped ────────────────────
    arb = LinearTrajectoryArbPrior(mu=0.5, sigma=1.0 / 6.0, eps=0.02)
    b_t_a, u_t_a, b0_a = arb.sample(boxes_gt, t)
    assert b0_a.min() >= 0.02 - 1e-6 and b0_a.max() <= 1.0 + 1e-6
    print(f"LinearArbPrior: b0_cx in [{b0_a.min():.3f}, {b0_a.max():.3f}]  ✓")

    # ── RiemannianTrajectoryArbPrior ─────────────────────────────────────────
    rie_arb = RiemannianTrajectoryArbPrior()
    b_t_r, u_t_r, b0_r = rie_arb.sample(b1, t)
    assert torch.allclose(u_t_r, b1 - b0_r)
    print("RiemannianArbPrior: constant u_t  ✓")

    # ── OTCoupledTrajectory — cost monotone, permutation validity ───────────
    print("\n--- OT coupling ---")
    Q = 10
    torch.manual_seed(42)
    b1_big = torch.rand(B, Q, 4) * 0.5 + 0.1
    t_big  = torch.rand(B)

    # wrap Riemannian
    ot_rm = OTCoupledTrajectory(RiemannianTrajectory())
    b1_state = cxcywh_to_state(b1_big)
    b_t_ot, u_t_ot, b0_ot = ot_rm.sample(b1_state, t_big)
    assert b_t_ot.shape == (B, Q, 4)

    # compare costs before/after OT on a fresh sample
    traj_plain = RiemannianTrajectory()
    torch.manual_seed(42)
    _, _, b0_plain = traj_plain.sample(b1_state, t_big)
    b0_ot_ref = OTCoupledTrajectory._hungarian_reorder(b0_plain, b1_state)
    cost_plain = (b0_plain - b1_state).norm(dim=-1).sum(dim=-1)   # [B]
    cost_ot    = (b0_ot_ref - b1_state).norm(dim=-1).sum(dim=-1)
    assert (cost_ot <= cost_plain + 1e-6).all(), \
        f"OT should not increase cost: plain={cost_plain} ot={cost_ot}"
    print(f"  Riemannian OT:  cost(plain)={cost_plain.tolist()}  cost(OT)={cost_ot.tolist()}  ✓ (OT ≤ plain)")

    # permutation validity — every b0_plain row used exactly once
    for bi in range(B):
        # find which b0_plain row maps to b0_ot_ref[bi, q] by exact match
        matches = (b0_plain[bi].unsqueeze(0) == b0_ot_ref[bi].unsqueeze(1)).all(dim=-1)
        used = matches.any(dim=0).tolist()
        assert all(used) and sum(used) == Q
    print("  permutation validity (all Q queries used exactly once)  ✓")

    # wrap Linear
    ot_lin = OTCoupledTrajectory(LinearTrajectory())
    b_t_ol, u_t_ol, b0_ol = ot_lin.sample(b1_big, t_big)
    assert b_t_ol.shape == (B, Q, 4)
    print(f"  Linear OT:      b_t {b_t_ol.shape}  ✓")

    # wrap LinearArbPrior
    ot_arb = OTCoupledTrajectory(LinearTrajectoryArbPrior())
    b_t_oa, u_t_oa, b0_oa = ot_arb.sample(b1_big, t_big)
    assert b_t_oa.shape == (B, Q, 4)
    print(f"  LinearArbPrior OT: b_t {b_t_oa.shape}  ✓")

    # inference pass-through
    init = ot_rm.init_noise(B, Q, device=boxes_gt.device)
    assert init.shape == (B, Q, 4)
    print(f"  init_noise pass-through: {init.shape}  ✓")

    print("\nAll checks passed.")
