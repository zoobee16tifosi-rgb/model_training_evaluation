
---

# Home Credit Default Risk Prediction (Team6 - Six Sigma)

**팀원**: 박주비(팀장), 이정화, 고준영  
**프로젝트 기간**: 2025.10.15 ~ 2025.10.20  
**데이터 출처**: [Kaggle - Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)  

---

## 1. 프로젝트 개요

**목적**: Home Credit 데이터를 활용하여 채무불이행 위험 고객을 예측하고, 다양한 머신러닝 알고리즘 비교를 통해 최적 모델 선정

**예측 대상**: TARGET (0=정상, 1=불이행)

**기대 효과**:

* 금융 손실 감소
* 심사 효율성 향상
* 고객 서비스 개선

---

## 2. 데이터 전처리

**데이터셋 요약**

| 항목     | 내용                 |
| ------ | ------------------ |
| 원본 데이터 | 307,511개, 122개 특성  |
| 최종 특성  | 20개                |
| 학습/테스트 | 246,008 / 61,503개  |
| 클래스 분포 | 정상 91.9%, 불이행 8.1% |

**전처리 주요 내용**:

* 결측치 처리: 수치형 → 중앙값, 범주형 → 최빈값
* 변수 변환: DAYS_BIRTH → AGE, DAYS_EMPLOYED → YEARS_EMPLOYED
* 금액 변수 로그 변환: AMT_CREDIT, AMT_INCOME_TOTAL
* 인코딩: 순서형 → Label Encoding, 비순서형 → One-Hot Encoding
* 특성 선택: 상관관계 제거 + 중요도 0 변수 제거

---

## 3. 모델링 전략

**알고리즘 및 담당자**

| 모델                  | 담당자 | 특징                   |
| ------------------- | --- | -------------------- |
| Random Forest       | 박주비 | 특성 중요도 분석 가능, 과적합 방지 |
| Decision Tree       | 박주비 | 해석 용이, 비교 기준         |
| Logistic Regression | 이정화 | 빠른 학습, 확률 해석 가능      |
| K-Nearest Neighbor  | 이정화 | 거리 기반 분류             |
| SVM                 | 고준영 | 고차원 데이터 처리, 커널 활용    |

**데이터 분리**

```python
train_test_split(test_size=0.2, stratify=y, random_state=42)
```

**클래스 불균형 대응**

* `class_weight='balanced'`
* Threshold 조정
* SMOTE(실험)

---

## 4. 모델별 최종 성능

| 모델                  | Recall   | Precision | F1 Score | ROC-AUC | 담당자 |
| ------------------- | -------- | --------- | -------- | ------- | --- |
| Random Forest       | 74.60%   | 10.79%    | 0.1885   | 66.42%  | 박주비 |
| Logistic Regression | 77.50%   | 9.56%     | 0.1702   | 60.46%  | 이정화 |
| SVM                 | 92.27% ⭐ | 8.60%     | 0.1574   | 57.96%  | 고준영 |

**최종 모델 선정**: **SVM**

* 목표: 소수 클래스(불이행) 탐지 최대화
* Recall 92.27%로 가장 높음
* 일부 오탐(FP) 감수, 리스크 관리에 적합

---

## 5. 최종 모델 하이퍼파라미터 (SVM)

```python
SVC(
    kernel='linear',
    C=1.0,
    tol=1e-3,
    max_iter=10000,
    class_weight={0:1, 1:3.5},
    shrinking=True,
    cache_size=2000,
    random_state=42
)
```

**클래스별 성능**

| Class   | Precision | Recall | F1-Score |
| ------- | --------- | ------ | -------- |
| 0 (정상)  | 95.34%    | 13.90% | 24.27%   |
| 1 (불이행) | 8.60%     | 92.27% | 15.74%   |

---

## 6. 핵심 교훈

* class_weight가 SMOTE보다 효과적
* Threshold 조정이 모델 성능 개선의 핵심
* 단순 데이터 확장(Bureau JOIN 등)이 성능 향상에 항상 도움되는 것은 아님
* 인프라 제약 고려: SVM의 경우 연산량/커널 선택 중요
