# Docker 기본 정리

이 프로젝트(`riemannian_flow_det`) 기준으로 설명합니다.

---

## Docker Compose에 필요한 파일

```
프로젝트 루트/
├── Dockerfile          ← 이미지 빌드 설계도
├── docker-compose.yml  ← 컨테이너 실행 방법 정의
└── requirements.txt    ← (Dockerfile에서 참조 — 실제론 Dockerfile 안에 pip install로 고정)
```

- `Dockerfile`과 `docker-compose.yml` 두 파일이 **핵심**.
- `docker-compose.yml`에 `build: .`가 있으면 같은 디렉토리의 `Dockerfile`을 자동으로 읽음.

---

## 핵심 개념: 이미지 vs 컨테이너

| 개념 | 비유 | 설명 |
|------|------|------|
| **이미지** | 설치 파일(.iso) | 환경이 고정된 스냅샷. 읽기 전용. |
| **컨테이너** | 실행 중인 프로그램 | 이미지를 기반으로 실제로 돌아가는 인스턴스. |

- 이미지 1개 → 컨테이너 여러 개 실행 가능
- 컨테이너를 종료해도 이미지는 남음

---

## build vs up vs run — 차이

### `docker compose build`
- `Dockerfile`을 읽어서 **이미지만 만든다**. 컨테이너는 실행 안 함.
- 언제 씀: Dockerfile 또는 requirements를 수정한 뒤 이미지만 갱신할 때.

```bash
docker compose build
```

---

### `docker compose up`
- 이미지가 없으면 빌드 후 **컨테이너를 실행**한다.
- 이미지가 이미 있으면 빌드 없이 바로 실행.
- `--build` 플래그를 붙이면 **항상 새로 빌드 후 실행**.

```bash
docker compose up           # 이미지 있으면 그냥 실행
docker compose up --build   # 항상 빌드 후 실행 (Dockerfile 바꿨을 때)
docker compose up -d        # 백그라운드(detach) 실행
```

이 프로젝트에서의 권장 흐름:

```bash
# 처음 or Dockerfile 변경 시
docker compose up --build

# 이후 (코드만 바뀐 경우 — volume 마운트라 빌드 불필요)
docker compose up
```

---

### `docker compose run`
- `docker-compose.yml`에 정의된 서비스로 **일회성 명령을 실행**한다.
- 컨테이너를 올린 채 두는 게 아니라, 명령 실행 후 종료.

```bash
docker compose run rflow python script/train.py --config configs/coco.yaml
```

> `up`은 장기 실행, `run`은 일회성 작업에 적합.

---

## 자주 쓰는 명령어 모음

```bash
# 실행 중인 컨테이너 접속 (bash 셸)
docker compose exec rflow bash

# 컨테이너 상태 확인
docker compose ps

# 컨테이너 종료 (이미지는 유지)
docker compose down

# 이미지까지 삭제
docker compose down --rmi all

# 로그 확인
docker compose logs -f rflow
```

---

## 이 프로젝트의 Volume 마운트 구조

`docker-compose.yml`에 다음이 설정되어 있음:

```yaml
volumes:
  - .:/workspace/riemannian_flow_det        # 프로젝트 코드 전체
  - ./data:/workspace/riemannian_flow_det/data
  - ./outputs:/workspace/riemannian_flow_det/outputs
```

- **코드를 호스트에서 수정하면 컨테이너 안에 즉시 반영된다** (재빌드 불필요).
- `data/`, `outputs/`도 호스트와 공유 → 체크포인트, 로그가 컨테이너 종료 후에도 남음.

---

## 언제 `--build`가 필요한가?

| 상황 | `--build` 필요? |
|------|:--------------:|
| Python 코드(.py) 수정 | 불필요 (volume 마운트) |
| `Dockerfile` 수정 | **필요** |
| 새 pip 패키지 추가 (`Dockerfile` 수정) | **필요** |
| `docker-compose.yml` 수정 | 불필요 (대부분) |
| 처음 실행 (이미지 없음) | 자동 빌드됨 |

---

## 흐름 요약

```
Dockerfile  ──build──▶  이미지(rflow)  ──up──▶  컨테이너(rflow_dev)
                                                      │
                                              volume 마운트로
                                              호스트 코드 연결
```
