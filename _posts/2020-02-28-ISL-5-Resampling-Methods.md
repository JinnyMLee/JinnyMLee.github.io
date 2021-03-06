---
layout: post
title: ISL 5. Resampling Methods
subtitle: An Introduction to Statistical Learning
tags: [ISL]
use_math: true
---





**리샘플링(resampling)**은 훈련 셋에서 반복적으로 샘플을 뽑아 각 샘플을 모델에 재적합(refit)하는 것을 말한다. 리샘플링을 통해 모델에 딱 한 번만 적합시켰을 때는 얻을 수 없는 정보들을 얻을 수 있다. 예를 들어, 선형 회귀에서 적합의 변산도(variability)를 추정할 수 있다. 이번 장에서는 가장 흔히 사용되는 리샘플링 방법인 **교차검증(cross-validation)**과 **부트스트랩(bootstrap)**에 대하여 알아본다. 이러한 리샘플링 방법은 모델의 성능 평가(model assessment)나 모델 선택(model selection)에 사용될 수 있다.



## 5.1 교차검증

테스트 오류(test error)란 통계 학습을 통해 "새로운" 관측치에 대한 반응변인을 예측한 결과의 평균 오류를 말한다. 훈련 오류(training error)는 쉽게 계산될 수 있지만 대개 테스트 오류를 매우 과소평가한다. 직접 테스트 오류를 계산할 수 있는 테스트 셋이 부재하는 경우, 이미 가지고 있는 훈련 데이터를 사용해 테스트 오류를 추정할 수 있다. 훈련 오류율을 수학적으로 조정하여 테스트 오류를 추정하는 방법이 있는데, 이 방법에 대해서는 챕터 6에서 다룰 것이다. 이번 절에서 학습할 **교차검증(cross-validation)**에서는 적합 과정에서 훈련 관측치들의 서브셋을 따로 떼어뒀다가 나중에 통계 학습 방법을 적용하는 방식으로 테스트 오류율을 추정한다.



### 5.1.1 검증 셋 접근법

검증 셋 접근법에서는 사용 가능한 관측치들을 훈련 셋(training set)과 검증 셋(validation set)의 두 부분으로 나눈다. 훈련 셋을 모델에 적합시키고, 적합된 모델은 검증 셋 관측치들의 반응변인을 예측하는 데 사용된다. 이렇게 계산된 검증 셋 오류율이 테스트 오류율의 추정치가 되는 것이다.

이 방법은 개념적으로 간단하며 이행하기 쉽다는 장점이 있지만, 훈련 셋과 검증 셋이 어떻게 나뉘는지에 따라 검증 셋 오류가 매우 변산이 크며, 모델에 적합시키는 데 사용되는 관측치의 개수가 적어진다는 단점이 있다.



### 5.1.2 Leave-One-Out 교차검증

**Leave-One-Out 교차검증(LOOCV)**에서는 하나의 관측치 $$(x_1, y_1)$$을 검증 셋으로 사용하고 나머지 관측치 $$\left\{(x_2, y_2), . . . , (x_n, y_n)\right\}$$을 훈련 셋으로 한다. $$n-1$$의 훈련 관측치들을 통계적 학습법에 적합시키고, 따로 떼어놓았던 $$x_1$$의 값을 이용해 $$\hat{y}_1$$을 예측한다. $$MSE_1 =(y_1 − \hat{y}_1)^2$$은 테스트 오류에 대한 비편향 추정치지만, 단일 관측치에 의존하고 있기 때문에 매우 변산이 높다. 따라서 이러한 과정을 $$n$$번 반복해 $$n$$개의 제곱 오차 $$MSE_1, . . . , MSE_n$$을 구한다. 테스트 MSE에 대한 LOOCV 추정치는 이 $$n$$개의 테스트 오류 추정치들의 평균이다.



$$
CV_{(n)} = \frac 1 n \sum^n_{i=1} MSE_i
$$



LOOCV는 검증 셋 접근법에 비해 여러 장점을 가진다. 먼저, LOOCV는 검증 셋 접근법에 비해 낮은 편향을 가진다. 즉, 검증 셋 접근에 비해 훈련 셋의 크기가 줄어들지 않기 때문에 테스트 오류율을 덜 과대평가한다. 또한, LOOCV는 반복해서 시행하더라도 항상 같은 결과를 낸다. 그러나 LOOCV는 모델이 $$n$$번 적합되어야 하기 때문에 시행 비용이 크다는 단점이 있다. 최소 자승 선형 혹은 다항 회귀에 관해서는 아래의 공식을 사용하면 LOOCV의 시행 비용을 단일 모델 적합 수준으로 감축할 수 있다.



$$
CV_{(n)} = \frac 1 n \sum^n_{i=1} \left( \frac {y_i-\hat{y}_i} {1-h_i} \right)^2
$$



이때, $$\hat{y}_i$$은 원래 최소 자승 적합에서 $$i$$번째의 적합된 값이며, $$h_i$$는 레버리지(leverage; 챕터 3 참고)이다. 보통의 MSE와 같은 공식이나, $$i$$번째 잔차가 $$1 − h_i$$에 의해 나눠진다는 점만 다르다. 레버리지는 $$1/n$$과 $$1$$ 사이에 존재하며 한 관측치가 그것의 적합에 영향을 미치는 정도를 반영한다. 따라서 위 공식에서 레버리지가 높은 포인트에 대한 잔차가 더 크게 계산된다.



### 5.1.3 k-폴드 교차검증

**k-폴드 교차검증(k-Fold Cross-Validation)**에서는 관측치들을 $$k$$개의 동일한 사이즈의 집단들 혹은 폴드들(folds)로 나눈다. 먼저, 첫 번째 폴드를 검증 셋으로 두고 남은 $$k-1$$개의 폴드들로 모델을 적합시킨다. 그리고 검증 셋을 사용해 첫 번째 평균 제곱 오차 $$MSE_1$$를 계산한다. 이러한 절차를 $$k$$번 반복하며 그때마다 다른 폴드를 검증 셋으로 하여 $$k$$개의 테스트 오류에 대한 추정치, $$MSE_1$$, $$MSE_2$$,  ... , $$MSE_k$$를 생성한다. k-폴드 CV 추정치는 이 값들의 평균으로 계산된다.



$$
CV_{(k)} = \frac 1 k \sum^k_{i=1}MSE_i
$$



위에서 배웠던 LOOCV는 이러한 k-폴드 CV에서 $$k$$가 $$n$$과 같을 때의 특수한 경우라고 할 수 있다. 실제로는 $$k = 5$$나 $$k = 10$$의 값을 사용하여 k-폴드 CV를 수행하는 것이 보편적이다. 이 방법이 LOOCV보다 훨씬 효율적이며 다음 절에 설명할 편향-분산 트레이드오프와 관련된 강점을 가진다. 어떻게 관측치들이 쪼개지는지에 따라 변산성이 커질 수 있지만 검증 셋 접근법에 비해서는 훨씬 낮다.



### 5.1.4 k-폴드 교차검증에서의 편향-분산 트레이드오프

k-폴드 CV는 편향-분산 트레이드오프 측면에서 LOOCV에 비해 더 정확한 테스트 오류율의 추정치를 제공한다. LOOCV에서는 훈련 셋의 크기가 $$n-1$$이기 때문에 편향을 최소화할 수 있는 반면, k-폴드 CV는 그에 비해 훈련 셋의 크기가 작아지므로 편향이 커진다. 그러나 $$k < n$$일 때, LOOCV는 k-폴드 CV에 비해 분산이 높다. LOOCV는 $$n$$개의 적합된 모델에서의 결과를 평균 내는데, 이때 모델은 서로 매우 큰 상관을 가진다. 반면, k-폴드 CV에서는 $$k$$개의 적합된 모델에서의 결과를 평균 내는데, 이 모델들은 서로 더 적은 상관을 가지기 때문이다. k-폴드 CV에서 $$k$$의 개수를 어떻게 하는가에 따라 편향-분산 트레이드오프 역시 달라진다. $$k = 5$$ 혹은 $$k = 10$$은 편향과 분산 모두 지나치게 높지 않게 테스트 오류율을 추정할 수 있는 값이다. 



### 5.1.5 분류 문제에서의 교차검증

교차검증은 분류 상황에서 역시 유용하게 사용될 수 있다. 위에서 설명했던 것과 모두 같지만 MSE 대신 잘못 분류된 관측치의 개수를 사용한다는 점만 다르다. 예를 들어, 분류 문제에서 LOOCV 오류율은 다음과 같다.



$$
CV_{(n)} = \frac 1 n \sum^n_{i=1} Err_i
$$



이때, $$Err_i = I(y_i \ne \hat{y}_i)$$이다. k-폴드 교차검증 오류율과 검증 셋 오류율 역시 같은 방식으로 정의된다.



## 5.2 부트스트랩

**부트스트랩(bootstrap)**은 주어진 추정량 혹은 통계 학습 방법과 연관된 "불확실성"을 수량화할 수 있는 매우 효과적인 방법이다. 부트스트랩에서는 원래의 데이터셋에서 관측치들을 반복적으로 샘플링하여 마치 모집단에서 여러 번 표본을 추출하는 것과 같이 복수의 데이터셋들을 얻어낸다.

간단한 예를 가지고 살펴보자. 각각 $$X$$와 $$Y$$라는 수익이 나는 두 개의 금융 자산이 있다고 하자. 가진 돈 중 $$α$$만큼을 $$X$$에 투자하고, 남은 $$1 − α$$만큼을 $$Y$$에 투자하려고 한다. 우리는 투자의 총 위험, 즉 분산 $$Var(αX +(1 −α)Y )$$을 최소화하는 $$α$$를 선택하고 싶다. 그 값은 아래와 같이 계산될 수 있다.



$$
α = \frac {σ^2_Y − σ_{XY}} {σ^2_X + σ^2_Y − 2σ_{XY}}
$$



이때, $$σ^2_X = Var(X)$$, $$σ^2_Y = Var(Y)$$, $$σ_{XY} = Cov(X, Y)$$이며, 실제로 알려지지 않은 값들이다. 이 값들의 추정치를 아래의 공식에 넣어 투자의 분산을 최소화하는 $$α$$를 추정할 수 있을 것이다.



$$
\hat{α} = \frac {\hat{σ}^2_Y − \hat{σ}_{XY}} {\hat{σ}^2_X + \hat{σ}^2_Y − 2\hat{σ}_{XY}}
$$



우리가 가지고 있는 데이터는 관측치의 개수가 $$n=3$$인 $$Z$$라는 간단한 데이터셋이다. 복원 추출 방법으로 이 데이터셋에서 $$n$$개의 관측치를 무선적으로 선택하여 부트스트랩 데이터셋 $$Z^{∗1}$$을 만든다. 이 $$Z^{∗1}$$을 사용해서 $$α$$에 대한 부트스트랩 추정치 $$\hat{α}^{∗1}$$을 생성할 수 있다. 이 과정을 $$B$$번 반복하여 $$B$$개의 서로 다른 부트스트랩 데이터셋 $$Z^{∗1}$$, $$Z^{∗2}$$, ..., $$Z^{∗B}$$와 $$B$$개의 $$α$$ 추정치 $$\hat{α}^{∗1}$$, $$\hat{α}^{∗2}$$, ..., $$\hat{α}^{∗B}$$를 얻을 수 있다. 그리고 아래의 공식에 따라 부트스트랩 추정치들의 표준오차(standard error)를 계산할 수 있다.



$$
\mathrm{SE}_B(\hat{α}) = \sqrt { \frac 1 {B − 1} \sum^B_{r=1} \left(\hat{α}^{∗r} − \frac 1 {B} \sum^B_{r'=1} \hat{α}^{∗r'}\right)^2}
$$



이 값은 원래의 데이터셋에서 추정된 $$\hat{α}$$의 표준오차에 대한 추정치이다.