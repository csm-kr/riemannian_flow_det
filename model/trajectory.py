# File: model/trajectory.py
# Role: box state space 위 flow matching 궤적 정의 (linear / sin / sincos)

import math
import torch


class LinearTrajectory:
    """
    Purpose: linear interpolation in box state space (기본값).

    b_t = (1 - t) * b0 + t * b1
    u_t = b1 - b0   (constant vector field)

    LayoutFlow LinearScheduler 와 동일 방식.
    """

    def interpolate(
        self,
        b0: torch.Tensor,
        b1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Purpose: t 시점의 보간 상태 계산.
        Inputs:
            b0: [B, N, 4], float32 — Gaussian noise
            b1: [B, N, 4], float32 — GT log-scale state
            t:  [B],       float32 — time ∈ [0, 1]
        Outputs:
            b_t: [B, N, 4], float32
        """
        t_ = t[:, None, None]                 # [B, 1, 1] for broadcast
        return (1 - t_) * b0 + t_ * b1

    def target_field(
        self,
        b0: torch.Tensor,
        b1: torch.Tensor,
        t: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Purpose: target vector field (linear은 t 무관, 상수).
        Inputs:
            b0: [B, N, 4], float32
            b1: [B, N, 4], float32
            t:  [B] (미사용 — linear는 상수 vector field)
        Outputs:
            u_t: [B, N, 4], float32 — b1 - b0
        """
        return b1 - b0


class SinTrajectory:
    """
    Purpose: sin 궤적.

    b_t = (1 - sin(t·π/2)) * b0 + sin(t·π/2) * b1
    u_t = (π/2) * cos(t·π/2) * (b1 - b0)
    """

    def interpolate(self, b0, b1, t):
        """
        Inputs:  b0,b1 [B,N,4] / t [B]
        Outputs: b_t   [B,N,4]
        """
        s = torch.sin(t * math.pi / 2)[:, None, None]
        return (1 - s) * b0 + s * b1

    def target_field(self, b0, b1, t):
        """
        Inputs:  b0,b1 [B,N,4] / t [B]
        Outputs: u_t   [B,N,4]
        """
        c = torch.cos(t * math.pi / 2)[:, None, None]
        return (math.pi / 2) * c * (b1 - b0)


class SinCosTrajectory:
    """
    Purpose: sincos 궤적.

    b_t = cos(t·π/2) * b0 + sin(t·π/2) * b1
    u_t = (π/2) * (cos(t·π/2) * b1 - sin(t·π/2) * b0)
    """

    def interpolate(self, b0, b1, t):
        """
        Inputs:  b0,b1 [B,N,4] / t [B]
        Outputs: b_t   [B,N,4]
        """
        s = torch.sin(t * math.pi / 2)[:, None, None]
        c = torch.cos(t * math.pi / 2)[:, None, None]
        return c * b0 + s * b1

    def target_field(self, b0, b1, t):
        """
        Inputs:  b0,b1 [B,N,4] / t [B]
        Outputs: u_t   [B,N,4]
        """
        s = torch.sin(t * math.pi / 2)[:, None, None]
        c = torch.cos(t * math.pi / 2)[:, None, None]
        return (math.pi / 2) * (c * b1 - s * b0)


def build_trajectory(name: str = "linear"):
    """config에서 궤적 이름으로 생성."""
    trajectories = {
        "linear": LinearTrajectory,
        "sin":    SinTrajectory,
        "sincos": SinCosTrajectory,
    }
    assert name in trajectories, f"Unknown trajectory: {name}. Choose from {list(trajectories)}"
    return trajectories[name]()


# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== trajectory.py sanity check ===")
    B, N = 2, 5

    b0 = torch.zeros(B, N, 4)
    b1 = torch.ones(B, N, 4)
    t0 = torch.zeros(B)
    t1 = torch.ones(B)
    th = torch.full((B,), 0.5)

    for name, traj in [
        ("linear", LinearTrajectory()),
        ("sin",    SinTrajectory()),
        ("sincos", SinCosTrajectory()),
    ]:
        bt_0 = traj.interpolate(b0, b1, t0)
        bt_1 = traj.interpolate(b0, b1, t1)
        assert torch.allclose(bt_0, b0, atol=1e-5), f"{name}: t=0 should be b0"
        assert torch.allclose(bt_1, b1, atol=1e-5), f"{name}: t=1 should be b1"

        ut = traj.target_field(b0, b1, th)
        assert ut.shape == (B, N, 4), f"{name}: u_t shape {ut.shape}"
        print(f"{name:8s}: interpolate ✓  target_field {ut.shape} ✓")

    print("All checks passed.")
