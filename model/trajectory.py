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
"""

import torch
from dataset.box_ops import cxcywh_to_state


class RiemannianTrajectory:
    """
    Geodesic on ℝ² × ℝ₊² — linear interpolation in log-scale state space.

    b₀ ~ N(0, I) in state space (noise)
    b₁ = cxcywh_to_state(boxes_gt)  (target)
    b_t = (1-t)·b₀ + t·b₁          (linear = geodesic in log-space)
    u_t* = b₁ − b₀                  (constant vector field)
    """

    def init_noise(self, B: int, Q: int, device, dtype=torch.float32) -> torch.Tensor:
        """
        Inference-time b₀ sampler. **Must match the training prior** to keep
        train/infer distributions aligned.
        Output: [B, Q, 4] in state space [cx, cy, log_w, log_h].
        """
        return torch.randn(B, Q, 4, device=device, dtype=dtype)

    def sample(
        self,
        b1: torch.Tensor,
        t:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Purpose: Sample a noisy interpolated state and its target vector field.
        Inputs:
            b1: [B, N, 4], float32 — target in state space [cx, cy, log_w, log_h]
            t:  [B],        float32 — timestep in [0, 1]
        Outputs:
            b_t:   [B, N, 4] — interpolated state at time t
            u_t:   [B, N, 4] — target vector field (constant, = b1 − b0)
            b0:    [B, N, 4] — noise sample
        """
        b0  = torch.randn_like(b1)
        t_  = t[:, None, None]           # [B, 1, 1] for broadcasting
        b_t = (1.0 - t_) * b0 + t_ * b1
        u_t = b1 - b0                    # constant vector field
        return b_t, u_t, b0

    def ode_step(
        self,
        b_t: torch.Tensor,
        v_t: torch.Tensor,
        dt:  float,
    ) -> torch.Tensor:
        """
        Purpose: Euler ODE step: b_{t+dt} = b_t + dt * v_t.
        Inputs:
            b_t: [B, N, 4] — current state
            v_t: [B, N, 4] — predicted vector field
            dt:  float
        Outputs:
            b_next: [B, N, 4]
        """
        return b_t + dt * v_t


class LinearTrajectory:
    """
    Euclidean baseline — linear interpolation in normalized cxcywh space.
    The vector field is NOT constant in state space (time-dependent).

    b₀ ~ Uniform([0.05, 0.95]^4) in cxcywh (random noise boxes)
    b₁ = boxes_gt in cxcywh
    b_t_cx = (1-t)·b₀ + t·b₁  (cxcywh space)
    → Convert b_t_cx to state space for model input
    → Vector field in state space: d/dt[log(w_t)] = (w₁-w₀) / w_t  (time-dependent)
    """

    def init_noise(self, B: int, Q: int, device, dtype=torch.float32) -> torch.Tensor:
        """
        Inference-time b₀ sampler — matches training prior: uniform in cxcywh
        then converted to state space. Without this, train/infer distributions
        don't align (randn in state space sees log_w ∈ [~-3, 3] but training
        log_w lives in [~-3, 0]).
        Output: [B, Q, 4] in state space [cx, cy, log_w, log_h].
        """
        b0_cx = torch.rand(B, Q, 4, device=device, dtype=dtype) * 0.9 + 0.05
        return cxcywh_to_state(b0_cx)

    def sample(
        self,
        b1_cx: torch.Tensor,
        t:     torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Purpose: Sample interpolated state (in state space) and target vector field.
        Inputs:
            b1_cx: [B, N, 4], float32 — target in normalized cxcywh
            t:     [B],        float32 — timestep in [0, 1]
        Outputs:
            b_t:   [B, N, 4] — interpolated state (state space)
            u_t:   [B, N, 4] — target vector field (state space, time-dependent)
        """
        # Sample noise boxes uniformly in cxcywh (ensures w,h > 0)
        b0_cx = torch.rand_like(b1_cx) * 0.9 + 0.05   # [B, N, 4] in (0.05, 0.95)

        t_    = t[:, None, None]
        b_t_cx = (1.0 - t_) * b0_cx + t_ * b1_cx       # [B, N, 4] in cxcywh

        # Model input: convert cxcywh → state space
        b_t = cxcywh_to_state(b_t_cx)

        # Target vector field in state space: d/dt[b_t_state]
        # For cx, cy: d/dt[cx_t] = cx₁ − cx₀  (constant)
        # For w, h:   d/dt[log(w_t)] = (w₁ − w₀) / w_t  (time-dependent)
        dcx = b1_cx[..., 0] - b0_cx[..., 0]   # [B, N]
        dcy = b1_cx[..., 1] - b0_cx[..., 1]
        w_t = b_t_cx[..., 2].clamp(min=1e-6)
        h_t = b_t_cx[..., 3].clamp(min=1e-6)
        dw  = b1_cx[..., 2] - b0_cx[..., 2]
        dh  = b1_cx[..., 3] - b0_cx[..., 3]

        u_t = torch.stack([dcx, dcy, dw / w_t, dh / h_t], dim=-1)
        return b_t, u_t

    def ode_step(
        self,
        b_t: torch.Tensor,
        v_t: torch.Tensor,
        dt:  float,
    ) -> torch.Tensor:
        """Euler ODE step in state space."""
        return b_t + dt * v_t


if __name__ == "__main__":
    print("=== trajectory.py sanity check ===")
    from dataset.box_ops import cxcywh_to_state, state_to_cxcywh

    B, N = 2, 5
    boxes_gt = torch.rand(B, N, 4) * 0.5 + 0.1   # normalized cxcywh
    t = torch.rand(B)

    # ── RiemannianTrajectory ──────────────────────────────────────────────────
    traj = RiemannianTrajectory()
    b1   = cxcywh_to_state(boxes_gt)

    b_t, u_t, b0 = traj.sample(b1, t)
    assert b_t.shape == (B, N, 4), f"b_t shape: {b_t.shape}"
    assert u_t.shape == (B, N, 4)
    assert b0.shape  == (B, N, 4)
    print(f"RiemannianTrajectory sample: b_t {b_t.shape}, u_t {u_t.shape}  ✓")

    # Verify interpolation endpoint at t=1 → b_t ≈ b1
    b_t1, _, _ = traj.sample(b1, torch.ones(B))
    assert torch.allclose(b_t1, b1, atol=1e-5), "t=1 should give b_t ≈ b1"
    print("  t=1 endpoint check  ✓")

    # Verify u_t* = b1 - b0 (constant)
    assert torch.allclose(u_t, b1 - b0), "u_t should be b1 - b0"
    print("  constant vector field check  ✓")

    # ODE step
    v = torch.randn_like(b_t)
    b_next = traj.ode_step(b_t, v, dt=0.1)
    assert b_next.shape == b_t.shape
    print(f"  ode_step: {b_next.shape}  ✓")

    # ── LinearTrajectory ─────────────────────────────────────────────────────
    lin_traj = LinearTrajectory()
    b_t_lin, u_t_lin = lin_traj.sample(boxes_gt, t)
    assert b_t_lin.shape == (B, N, 4)
    assert u_t_lin.shape == (B, N, 4)
    print(f"LinearTrajectory sample: b_t {b_t_lin.shape}, u_t {u_t_lin.shape}  ✓")

    print("All checks passed.")
