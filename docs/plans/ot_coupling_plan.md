# Plan: Mini-batch OT Coupling for Flow Matching Trajectories

**목적**: `trajectory.sample()` 의 **independent coupling** 때문에 prior/target support overlap 시 marginal vector field 가 붕괴하는 문제 (→ [`docs/ISSUES.md`](../ISSUES.md) "arb_prior 가 학습 안 됨") 를 **mini-batch OT assignment** 로 해결. Phase 3 `train.py` 를 착수하기 전 선행 작업.

---

## 1. Motivation

### 현재 문제 (e2 에서 드러남)

Flow matching 이 regression 하는 대상:

`v*(x, t) = E[u_t | γ(t; b₀, b₁) = x]`

`trajectory.sample()` 이 `(b₀, b₁)` 을 **독립 샘플** → 같은 `(x, t)` 를 반대 방향에서 지나는 trajectory 쌍의 target 이 상쇄. Independent coupling 의 marginal field 는 **target/prior support overlap 영역에서 0 으로 붕괴**.

실측 (e2 multi-seed): `riemannian_arb_prior` 에서 box 10 개가 캔버스 중앙에 뭉쳐 digit 에 도달 못 함. baseline `riemannian` 은 support 가 분리돼 있어 우연히 이 문제를 피함.

### OT coupling 의 해법

배치 내 `B × Q` 개의 `b₀` 와 `b₁` 을 **가까운 것끼리 짝** → pair 가 교차하지 않게 만들어 marginal field 의 평균이 그대로 transport 방향을 가리키게 한다.

참고 문헌:
- Tong, Malkin, Kilgour et al. — "Improving and generalizing flow-based generative models with minibatch optimal transport" (2023).
- Pooladian, Ben-Hamu, Domingo-Enrich et al. — "Multisample flow matching: Straightening flows with minibatch couplings" (ICML 2023).

---

## 2. 설계 원칙

1. **Trajectory API 는 유지** — 기존 `RiemannianTrajectory`, `LinearTrajectory`, `*ArbPrior` 는 independent coupling 으로 그대로 남김. OT 는 **optional wrapper** 로 추가.
2. **Training-only 기능** — `init_noise()` (inference) 는 영향 없음. `sample()` 내 assignment 만 변경.
3. **Per-image OT, per-batch 아님** — Detection 에서 각 이미지의 `Q × Q` 박스 매칭이 중요. Multi-image 배치 전체를 flat 해 OT 하면 cross-image leakage 발생.
4. **Query-index preservation** — class-indexed 1-to-1 (MNIST Box) 에선 `boxes[i]` = GT class `i` 로 query i 와 고정 묶임. OT 는 query-target 짝을 재배치하는 게 아니라 **query-b₀ 짝을 재배치**.
5. **CPU-cheap, GPU-interop** — `scipy.optimize.linear_sum_assignment` (Hungarian) 혹은 POT 라이브러리 없이 단순 구현. `Q ≤ 300` 이라 `O(Q³)` 도 1 ms 내.

---

## 3. 인터페이스 변경

### 3.1 Trajectory wrapper

신규 클래스 `OTCoupledTrajectory` (또는 기존 클래스에 `ot_coupling: bool` 파라미터 추가 — 택일, 전자 선호) 가 기존 trajectory 를 감싼다:

```python
class OTCoupledTrajectory:
    """
    Wraps any trajectory to use mini-batch OT coupling between b₀ and b₁.
    Training-only — sample() reassigns (b₀, b₁) pairs per-image by Hungarian match.
    init_noise() is passed through unchanged.
    """
    def __init__(self, base: RiemannianTrajectory | LinearTrajectory | ..., cost: str = "l2"):
        self.base = base
        self.cost = cost   # "l2" in state, "l2_cxcywh", "iou", etc.

    def sample(self, b1, t):
        # 1. base.sample() 로 (b_t, u_t[, b₀]) 샘플 — 단 여기서 b₀ 를 명시적으로 얻어온다
        # 2. 각 image 의 b₀ [Q, 4] 와 b₁ [Q, 4] 사이 cost matrix 계산
        # 3. Hungarian assignment → permutation π
        # 4. b₀ 를 π 에 따라 재배치 후 b_t, u_t 재계산
        # 5. 반환 형식은 base 와 동일
        ...

    def init_noise(self, B, Q, device, dtype):
        return self.base.init_noise(B, Q, device, dtype)

    def ode_step(self, b_t, v_t, dt):
        return self.base.ode_step(b_t, v_t, dt)
```

### 3.2 Config + build_model

`configs/*.yaml` 에 새 필드:
```yaml
ot_coupling: false          # bool
ot_cost: l2                 # l2 | l2_cxcywh | giou   (확장용)
```

`model/__init__.py::build_model`:
```python
if cfg.get("ot_coupling", False):
    model.trajectory = OTCoupledTrajectory(model.trajectory, cost=cfg.get("ot_cost", "l2"))
```

### 3.3 기존 trajectory.sample 리팩토링

`RiemannianTrajectory.sample` 은 이미 `b₀` 를 반환한다 (`b_t, u_t, b₀`). `LinearTrajectory.sample` 은 반환 형식이 `(b_t, u_t)` 로 b₀ 를 숨긴다 — wrapper 가 접근 못함.

**해결**: `LinearTrajectory.sample` 도 tuple 에 `b₀_cx` 를 추가해 일관성 확보. 기존 호출부 (`flow_matching.py::forward_train`) 는 `_` 로 무시하면 됨.

```python
# LinearTrajectory.sample 시그니처 변경
def sample(self, b1_cx, t) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """... returns (b_t, u_t, b0_cx). b0_cx 는 OT wrapper 에서 재배치용."""
```

---

## 4. 구현 세부

### 4.1 Hungarian assignment (기본)

```python
from scipy.optimize import linear_sum_assignment

def _ot_assign(b0: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    """
    Hungarian match per image.
    Inputs:
        b0, b1: [B, Q, 4]  (state or cxcywh, same space)
    Returns:
        perm: [B, Q]  int64 — b0[perm[b]] 을 사용하면 OT-paired
    """
    B, Q, _ = b0.shape
    # L2 cost in given space
    cost = torch.cdist(b0, b1, p=2.0)   # [B, Q, Q]
    cost_np = cost.detach().cpu().numpy()
    perms = np.zeros((B, Q), dtype=np.int64)
    for b in range(B):
        row, col = linear_sum_assignment(cost_np[b])
        # row == arange(Q), col[i] = 매칭된 b1 index
        # 우리는 b₀ 의 순서를 재배치 → inverse: b1 index i 에 매칭된 b₀ 의 원본 index
        inv = np.empty(Q, dtype=np.int64)
        inv[col] = row
        perms[b] = inv
    return torch.from_numpy(perms).to(b0.device)
```

> **중요**: MNIST Box 의 class-indexed 세팅에서 `b1[q]` 는 digit `q` 의 GT 박스로 **query 와 묶여 있다**. Query `q` 가 항상 digit `q` 를 예측해야 하므로 permutation 은 `b₀` 에만 적용, `b₁` 는 고정.

### 4.2 재계산 파이프라인

```python
def sample(self, b1, t):
    b_t, u_t, b0 = self.base.sample(b1, t)   # independent first
    perm = _ot_assign(b0, b1)                # [B, Q]
    # b₀ 를 perm 순서로 재배치
    b0_reordered = torch.gather(b0, 1, perm.unsqueeze(-1).expand(-1, -1, 4))
    # b_t, u_t 재계산 (base 의 수식을 재사용)
    return self.base._recompute_with_b0(b1, b0_reordered, t)
```

base class 들에 `_recompute_with_b0(b1, b0, t)` helper 추가 — 이미 각 sample 이 수행하는 수식을 외부에서 재호출 가능하게 추출한다.

### 4.3 Cost function 옵션 (확장)

- `l2` (기본) — state 공간 L2. Trajectory 공간과 일치.
- `l2_cxcywh` — cxcywh 공간 L2. log-scale 영향을 빼고 싶을 때.
- `giou` — box IoU 기반. Detection 에 더 자연스러움. Phase 4 에서 옵션 추가.

### 4.4 주의사항

- **Gradient flow 없음** — Hungarian 은 결정론적 permutation. b₀ 재배치만 하므로 b₀ 는 여전히 gradient 를 받지 않고 (prior sample) b₁ 은 GT 로 leaf. OT assignment 는 학습 path 에 들어가지 않는다.
- **Q=10 기준 cost**: `Q³` = 1000, per-image 1 ms 미만. Batch 16 → 16 ms. GPU forward 대비 무시 가능.
- **Coupling 은 per step**: t 샘플링처럼 매 step 새로 `(b₀, b₁, π)` 결정. deterministic 하지 않음 (b₀ 가 랜덤).
- **Inference 는 그대로** — `init_noise()` 는 독립. 이는 의도된 동작 — 추론 시엔 GT 없어 매칭 불가, 독립 prior 샘플이 맞다.

---

## 5. Critical Files (수정 대상)

| 파일 | 변경 |
|---|---|
| `model/trajectory.py` | `OTCoupledTrajectory` 추가. 기존 `*Trajectory.sample` 에 `_recompute_with_b0` helper 추출. `LinearTrajectory.sample` 반환 시그니처에 `b0_cx` 추가. |
| `model/flow_matching.py` | `forward_train` 에서 `sample()` 반환 tuple 의 `b0` 를 `_` 로 받아 무시 (후방 호환). |
| `model/__init__.py` | `build_model` 에서 `ot_coupling` cfg 확인 후 wrapper 적용. |
| `model/trajectory.py::__main__` | sanity check 추가 — OT wrapper 가 base 와 동일 shape 반환, permutation 이 cost 를 줄이는지 assert. |
| `configs/mnist_box.yaml` + `experiments/e2*/variants/*.yaml` | `ot_coupling: false` default. 실험에서 true 로 override. |

---

## 6. Verification

### V1. Unit test — Hungarian assignment 정당성

`model/trajectory.py::__main__` 에:
- 무작위 `(b₀, b₁)` 생성 → `_ot_assign` 전후 total cost 가 **monotone 감소** 확인.
- `perm` 이 permutation 인지 (`sorted(perm) == arange(Q)`) 확인.

### V2. Marginal SNR 회복 측정

새 스크립트 `experiments/e3_ot_coupling/measure_marginal_snr.py`:
- `riemannian_arb_prior` 에서 independent vs OT coupling 의 `v*(x, t)` SNR 비교.
- 공간 bin 내 `‖mean u_t‖ / std u_t` → OT 에서 center bin 의 SNR 이 0 → 유의미 값으로 올라오면 가설 직접 확증.

### V3. e2 재실행 (회귀 검증)

기존 `run_multiseed.sh` 를 `ot_coupling: true` 로 재실행. 예상:
- `riemannian_arb_prior`: mean_err [35, 32, 2.6] → [~3, ~3, ~3] px (variance 대폭 감소, baseline 수준)
- `euclidean_arb_prior`: 유사 개선.
- `riemannian` baseline: 변화 없음 (support 분리 → OT 해도 동일).

### V4. Coupling 안정성

3 seed × 4 variant multi-seed 결과의 **std / mean** 비율이 모든 variant 에서 `<30%` 여야 OT 가 structural 해결이라 할 수 있음.

---

## 7. Timeline + Phase 연계

- **즉시 (Phase 2.5 cleanup)**: 이 plan 만 작성 · ISSUES.md 에 연결. 구현은 Phase 3 로.
- **Phase 3 task P3-OT**: `train.py` 작성 직전에 `OTCoupledTrajectory` + config 필드 구현 · V1/V3 테스트.
- **Phase 3 task P3-CFG**: config schema 에 `ot_coupling`, `ot_cost` 추가. 기본 `true` 로 (OT 가 default → prior 변경 에 robust).
- **Phase 4 E1 ablation**: `ot_coupling=on/off` 2-way 비교. VOC/COCO mAP 에서 prior-agnostic robustness 확인.

---

## 8. 미해결 / 결정 필요 사항

- **Cost 기본값**: `l2` (state) vs `l2_cxcywh` (normalized). state L2 가 Riemannian trajectory 수식과 정합이라 우선 채택.
- **CPU↔GPU 전이**: Hungarian 이 CPU 이므로 `b₀, b₁` 를 detach·cpu 후 solve. 작은 Q 라 overhead 무시 가능하나, Phase 4 에서 `Q=300` 이면 재검토.
- **OT assignment 를 softmax 화**: Sinkhorn 같은 soft matching 이 gradient 로 흐르게 하면 더 고차 개선 가능 (Pooladian 2023 방식). Phase 4 까지는 Hungarian 고정, 필요 시 upgrade.
- **`num_queries > num_gt`** 시나리오: Phase 4 real detection 에선 padding 된 "no-object" slot 이 생긴다. OT 는 `b₁` 의 valid 만 매칭, 나머지 b₀ 는 "no-target" 처리. 세부는 Phase 4 에서.

---

## 9. 요약

| 항목 | 값 |
|---|---|
| 목적 | independent coupling ill-posedness 해소 → flow matching 이 prior 변경에 robust |
| 범위 | `trajectory.sample()` 만 바뀜, inference · loss · head 는 무관 |
| 신규 클래스 | `model/trajectory.py::OTCoupledTrajectory` (wrapper) |
| 비용 | per-image Hungarian, `O(Q³)`. Q=10 → <1 ms. Q=300 → 수 ms. |
| 기대 효과 | arb_prior 계열 variance [13×→1×], mean_err 회복 (baseline 수준) |
| Phase | **Phase 3 P3-OT** 로 예약. Phase 2.5 에서는 plan 만 |
| 관련 이슈 | [`docs/ISSUES.md`](../ISSUES.md) "arb_prior 가 학습 안 됨" |
