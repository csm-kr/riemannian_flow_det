Problem Definition

객체 탐지는 이미지 내 객체의 위치를 정확한 bounding box로 국소화하는 문제이며, 최종 성능은 일반적으로 IoU 기반 기준으로 평가된다. 그러나 대부분의 기존 검출기는 박스 예측을 본질적으로 endpoint regression 문제로 다룬다. 즉, 초기 박스 상태나 참조 상태로부터 최종 target box를 직접 예측하거나, 여러 번 보정하더라도 궁극적으로는 마지막 endpoint의 정확도만을 학습 목표로 삼는다. 이 방식은 실용적으로 효과적이지만, 박스가 초기 상태에서 목표 상태로 어떤 공간 위에서, 어떤 경로를 따라, 어떤 동역학으로 이동해야 하는지에 대해서는 명시적으로 정의하지 못한다.

이 한계는 객체 탐지를 iterative refinement, diffusion-style denoising, 또는 flow matching 관점에서 바라볼 때 더 분명해진다. 이러한 접근은 단순히 최종 박스 하나만 잘 맞추는 것이 아니라, 초기 박스에서 목표 박스로 이어지는 연속적인 trajectory와, 그 trajectory를 따르는 vector field를 필요로 한다. 하지만 기존 box parameterization은 안정적인 decoding과 regression target 설계에는 적합해도, continuous transport를 위한 상태공간이나 경로 구조를 제공하지 않는다.

문제의 핵심은 bounding box가 단순한 4차원 유클리드 벡터가 아니라는 점이다. 박스의 center coordinate는 translation 성질을 가지는 반면, width와 height는 양수 제약을 갖는 scale 변수이다. 그럼에도 이를 균일한 Euclidean space에서 동일한 방식으로 다루면, translation과 scale의 차이가 무시되고, 특히 작은 객체에서의 scale-sensitive localization error를 적절히 반영하기 어렵다. 또한 iterative refinement 과정에서는 intermediate box state의 품질이 이후 업데이트 안정성과 최종 성능에 직접적인 영향을 미치므로, 단순한 Euclidean interpolation은 detection metric과 불일치하는 비효율적 경로를 유도할 수 있다.

따라서 본 연구의 문제는 다음과 같이 정의된다.
객체 탐지를 endpoint-only box regression 문제가 아니라, 구조화된 box state space 상에서의 continuous refinement 문제로 재정의할 수 있는가?
이를 위해서는
(1) detection box에 적절한 state space를 정의하고,
(2) 초기 박스와 목표 박스를 연결하는 principled trajectory를 설계하며,
(3) 그 trajectory를 실현하는 target vector field를 학습 가능하게 만들고,
(4) 이러한 geometry-aware dynamics가 실제 localization quality와 detector refinement를 개선하는지 검증해야 한다.

본 연구의 출발점은 bounding box를

B=R
2
×R
+
2
	​


형태의 구조적 상태로 보는 것이다. 여기서 R
2
는 box center를, R
+
2
	​

는 양수 제약을 갖는 width와 height를 나타낸다. 이 관점에서 객체 탐지는 단순히 최종 박스를 회귀하는 문제가 아니라, 이미지 조건부 특징 하에서 초기 box state를 target box state로 이동시키는 geometry-aware transport dynamics를 학습하는 문제로 바뀐다.

결국 본 연구가 다루는 핵심 질문은 다음과 같다.
첫째, continuous refinement에 적합한 box state space는 무엇인가?
둘째, 초기 box와 target box를 연결하는 가장 적절한 trajectory는 무엇인가?
셋째, detector가 학습해야 할 target vector field는 어떻게 정의되어야 하는가?
넷째, 이러한 dynamics가 localization metric, assignment, multi-step refinement와 어떻게 정합되어야 하는가?
다섯째, 이 formulation이 특히 small object나 scale-sensitive case에서 실제 검출 성능 향상으로 이어지는가?

요약하면, 이 연구의 문제정의는 다음 한 문장으로 정리할 수 있다.

Object detection should be reformulated from endpoint-only box regression into geometry-aware continuous box transport in a structured box space.