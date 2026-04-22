# SimpleConvNet — Model Diagram

> Generated: 2026-04-17 05:27  
> Source: `/tmp/test_model.py` (Local file)  
> Analysis mode: dynamic  
> Parameters: 110.6 K (trainable: 110.6 K)

---

## Module Definitions

### `MLP`

```
├── fc1: Linear(128→256)                [1, 128] → [1, 256]
├── act: GELU                           [1, 256] → [1, 256]
├── norm: LayerNorm(256, elementwise_)  [1, 256] → [1, 256]
├── fc2: Linear(256→4)                  [1, 256] → [1, 4]
└── drop: Dropout(p=0.1)                [1, 256] → [1, 256]
```

## Model Tree

```
SimpleConvNet                                   [1, 3, 64, 64] → [1, 4]
├── conv1: Conv2d(3, 64, 3×3, s=1, p=1)         [1, 3, 64, 64] → [1, 64, 64, 64]
├── bn1: BatchNorm2d(64)                        [1, 64, 64, 64] → [1, 64, 64, 64]
├── relu: ReLU                                  [1, 128, 32, 32] → [1, 128, 32, 32]
├── pool: MaxPool2d(2×2, s=2, p=0)              [1, 64, 64, 64] → [1, 64, 32, 32]
├── conv2: Conv2d(64, 128, 3×3, s=1, p=1)       [1, 64, 32, 32] → [1, 128, 32, 32]
├── bn2: BatchNorm2d(128)                       [1, 128, 32, 32] → [1, 128, 32, 32]
└── head: MLP                                   [1, 128] → [1, 4]
    ├── fc1: Linear(128→256)                    [1, 128] → [1, 256]
    ├── act: GELU                               [1, 256] → [1, 256]
    ├── norm: LayerNorm(256, elementwise_)      [1, 256] → [1, 256]
    ├── fc2: Linear(256→4)                      [1, 256] → [1, 4]
    └── drop: Dropout(p=0.1)                    [1, 256] → [1, 256]
```

## Summary

| 항목 | 값 |
|------|-----|
| 소스 | Local file |
| 총 파라미터 | 110.6 K |
| Trainable | 110.6 K |
| 분석 모드 | dynamic |
| Conv 계층 수 | 2 |
| Linear 계층 수 | 2 |
| Activation | 2 |
| Norm 계층 수 | 3 |
| 사용자 정의 모듈 | 1 |
