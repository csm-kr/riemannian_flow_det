# 완료된 작업 기록

구현 완료된 항목을 날짜순으로 기록. 설계 결정 및 주요 선택지도 함께 남긴다.

---

## Dataset Phase

### ✅ D0-1 박스 포맷 확정
- **결정**: 기준 포맷 `cxcywh` normalized
- **이유**: 다른 모델과의 실험 비교 용이성

### ✅ D0-2 로딩 방식 확정
- **결정**: `pycocotools` 직접 사용 + VOC는 `torchvision.datasets.VOCDetection`
- **Augmentation**: Detectron2 `AugmentationList` 사용 (resize/flip)

### ✅ D1 — `dataset/box_ops.py`

구현 함수:
| 함수 | 입력 | 출력 |
|------|------|------|
| `xyxy_to_cxcywh` | `[N,4]` xyxy | `[N,4]` cxcywh |
| `cxcywh_to_xyxy` | `[N,4]` cxcywh | `[N,4]` xyxy |
| `normalize_boxes` | cxcywh pixel | normalized [0,1] |
| `denormalize_boxes` | normalized | cxcywh pixel |
| `cxcywh_to_state` | normalized cxcywh | `[cx,cy,log_w,log_h]` |
| `state_to_cxcywh` | `[cx,cy,log_w,log_h]` | normalized cxcywh |
| `clip_boxes` | xyxy pixel | xyxy clipped |
| `box_area` | `[N,4]` cxcywh | `[N]` |
| `box_iou` | `[N,4]`, `[M,4]` xyxy | `[N,M]` |

테스트: `python dataset/box_ops.py`

### ✅ D3-3 — `dataset/voc.py`

- `VOCDetection(root, year, split, download, ...)` — `torch.utils.data.Dataset` 서브클래스
- `torchvision.datasets.VOCDetection` 으로 다운로드 + XML 파싱
- Detectron2 `AugmentationList` 으로 resize/flip
- `visualize_sample()` — denormalize + matplotlib 박스 시각화
- 출력 포맷: `image [3,H,W]`, `boxes [N,4] normalized cxcywh`, `labels [N]`

테스트 및 시각화:
```bash
# 다운로드 + 시각화 (plt.show)
python dataset/voc.py --root data/voc --year 2007 --split val --download --vis_idx 0

# 이미지 파일로 저장
python dataset/voc.py --root data/voc --save outputs/figures/voc_sample_0.png
```

---

## 미완료 (plan_todo.md 참조)

- [ ] D2 — `dataset/transforms.py`
- [ ] D3-1, D3-2 — `dataset/coco.py`
- [ ] D4 — `dataset/collate.py`
- [ ] D5 — `dataset/__init__.py`
- [ ] model/ 전체
- [ ] script/ 전체
- [ ] utils/ 전체
