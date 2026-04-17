---
name: ppt-generator
description: >
  Automatically generates a .pptx presentation file from a topic, audience, and
  duration. Use this skill whenever the user asks to create slides, a presentation,
  or a PPT/PPTX file — even if they only give a topic and no other details.
  Trigger phrases include: "PPT 만들어줘", "발표자료 작성해줘", "발표용 슬라이드 구성해줘",
  "이 주제로 발표자료 만들어줘", "슬라이드 만들어줘", "프레젠테이션 만들어줘",
  "make a presentation", "create slides", "build a PPT".
  Always use this skill for presentation creation tasks — do not try to wing it
  without the skill, even for simple requests.
---

# PPT 자동 생성 스킬

발표 주제·청중·시간을 입력받아 **슬라이드 기획 → JSON 플랜 작성 → .pptx 파일 생성**
순서로 완성된 발표자료를 만든다.

---

## 1. 입력 수집

필수 정보가 없으면 **한 번에 모아서** 물어본다 (질문을 여러 번 나눠 하지 않는다).

| 항목 | 필수 | 기본값 |
|------|------|--------|
| 발표 주제 | ✅ | — |
| 대상 청중 | ✅ | — |
| 발표 시간 (분) | ✅ | — |
| 발표 목적 | ⬜ | 정보 전달 |
| 톤앤매너 | ⬜ | 전문적·간결 |
| 반드시 포함할 핵심 내용 | ⬜ | 없음 |
| 테마 | ⬜ | business |

**테마 선택지:** `business` (기본, 네이비 블루), `dark` (다크모드), `minimal` (흑백 미니멀)

정보가 충분하면 즉시 초안을 만든다. 완벽한 정보를 기다리지 말 것.

---

## 2. 슬라이드 수 설계 공식

발표 시간에 따라 적정 슬라이드 수를 결정한다:

- 슬라이드당 평균 1.5–2분 → `raw = round(duration_min / 1.75)`
- **반드시 클램프 적용: `slides = max(5, min(25, raw))`**
  - 예) 45분 → raw=26 → 클램프 후 **25장** (26이 아님)
  - 예) 3분 → raw=2 → 클램프 후 **5장**
- **구조 배분:**
  - 도입부: 전체의 15% (타이틀 1장 + 개요 1장)
  - 본론: 전체의 70% (섹션 헤더 + 내용 슬라이드)
  - 결론: 전체의 15% (요약 + 클로징)

청중 수준에 따른 조정:
- 전문가: 슬라이드당 내용 밀도↑, 용어 제한 없음
- 일반인: 슬라이드당 불릿 ≤ 4개, 전문 용어 풀어 설명
- 경영진/임원: 핵심 메시지 강조, 슬라이드 수 최소화

---

## 3. 슬라이드 플랜 JSON 작성

아래 스키마로 `slide_plan.json`을 작성한다. 파일 저장 위치는 현재 작업 디렉토리.

```json
{
  "title": "전체 발표 제목",
  "author": "선택",
  "theme": "business",
  "slides": [
    {
      "type": "title",
      "title": "발표 제목",
      "key_message": "한 줄 핵심 메시지",
      "bullets": [],
      "notes": "발표자 노트",
      "visual_hint": "권장 시각 자료 설명"
    }
  ]
}
```

**슬라이드 타입:**
- `title` — 첫 슬라이드 전용 (풀스크린 어두운 배경)
- `section` — 파트 구분 헤더 (어두운 배경, 큰 텍스트)
- `content` — 일반 내용 슬라이드 (흰 배경, 제목바 + 불릿)
- `closing` — 마지막 슬라이드 (타이틀과 유사)

**각 슬라이드 필드 작성 원칙:**
- `key_message`: 한 문장, 청중이 이 슬라이드에서 꼭 기억할 것
- `bullets`: 각 항목은 완결된 문장 또는 짧은 구 (슬라이드당 3–6개 권장)
- `notes`: 말할 내용, 발표자만 보는 메모 (완전한 문장으로)
- `visual_hint`: "막대 차트 — 연도별 성장률 비교", "아이콘 3개 가로 배치" 등 구체적으로

---

## 4. .pptx 파일 생성

JSON 플랜이 완성되면 번들 스크립트를 실행한다:

```bash
# 스크립트 경로: <skill-dir>/scripts/create_pptx.py
python <skill-dir>/scripts/create_pptx.py slide_plan.json <출력파일명>.pptx
```

- `<skill-dir>`는 이 SKILL.md 파일이 위치한 디렉토리
- 출력 파일명은 발표 주제를 기반으로 의미있게 지정 (예: `ai_intro_presentation.pptx`)
- `python-pptx`가 없으면 스크립트가 자동 설치함

---

## 5. 텍스트 요약 출력 (필수)

.pptx 생성 후 **반드시** 대화에 아래 항목을 텍스트로 정리한다.
이 단계를 건너뛰면 안 된다 — 검증 체크리스트도 여기서 출력한다.

### 5a. 전체 슬라이드 개요

```
슬라이드 1 [title]    : 제목
슬라이드 2 [content]  : 주제 개요
  핵심 메시지: ___
  ...
```

### 5b. 발표자 노트 요약

각 슬라이드의 `notes` 필드 내용을 번호와 함께 나열

### 5c. 발표용 요약 메시지

전체 발표를 3–5문장으로 압축한 엘리베이터 피치

### 5d. 검증 체크리스트 (섹션 6과 동일)

아래 섹션 6의 체크리스트를 텍스트로도 출력한다.

---

## 6. 자체 검증 체크리스트

파일 생성 후 반드시 확인하고 결과를 표로 출력:

| 검증 항목 | 결과 |
|-----------|------|
| 발표 시간 내 소화 가능한 슬라이드 수인가? | ✅/⚠️ |
| 도입→본론→결론 흐름이 자연스러운가? | ✅/⚠️ |
| 중복 슬라이드 없는가? | ✅/⚠️ |
| 모든 슬라이드에 핵심 메시지가 있는가? | ✅/⚠️ |
| 시각 자료 지시사항이 구체적인가? | ✅/⚠️ |
| .pptx 파일이 정상 생성되었는가? | ✅/❌ |

---

## 빠른 참조 — 발표 시간별 구성 예시

| 시간 | 슬라이드 수 | 본론 파트 수 |
|------|------------|-------------|
| 5분  | 5장        | 2파트        |
| 10분 | 7장        | 3파트        |
| 20분 | 12장       | 4파트        |
| 30분 | 17장       | 5파트        |
| 45분 | 20장       | 6파트        |
