---
layout: post
title: ISL 2. Statistical Learning
subtitle: An Introduction to Statistical Learning
tags: [ISL]
use_math: true
---



## 1 통계 학습이란 무엇인가?

통계 학습에서는 **예측변인(predictor)** (혹은 입력변수, 독립변인, feature) $$X$$와 **반응변인(response)** (혹은 출력변수, 종속변인) $$Y$$ 사이에 어떤 관계가 있음을 가정하며 이러한 관계는 아래의 식과 같이 표현될 수 있다.



$$
Y = f(X) + \epsilon
$$



위의 식에서 $$\epsilon$$ (epsilon)은 무선적인 오차항(random error term)이며, $$X$$와 독립이고 평균은 0이다. 결국 통계 학습이란 주어진 데이터를 가지고 위의 식에서 $$f$$를 추정하는 것을 뜻한다.



### 1.1 왜?

통계 학습의 목적은 아래의 두 가지, 예측과 추론으로 정리될 수 있다.



#### 예측

오차항은 평균이 0이기 때문에 $$Y$$를 예측하기 위해서는 오차항을 빼고 단순히 $$f$$의 추정치에 $$X$$를 넣어주면 된다. 이는 아래와 같은 식으로 표현된다.


$$
\hat{Y} = \hat{f}(X)
$$

만일 예측만이 목적이라면 $$Y​$$를 정확하게 예측하기만 하면 $$\hat{f}​$$의 정확한 형태에 대해서는 별로 상관하지 않아도 된다. 예측의 정확도는 크게 두 가지 요소에 따라 좌우된다.

​	첫째, $$\hat{f}$$이 실제 $$f$$가 아닌 것에서 기인하는 **reducible error**이다. 추정의 과정에서 적절한 테크닉을 사용하면 줄일 수 있기 때문에 "reducible"이라는 이름이 붙었다.

​	둘째, $$X$$를 통해 $$Y$$를 완전히 예측할 수 없는 것에서 기인하는 **irreducible error**이다. $$\epsilon$$은 $$Y$$를 예측하는 데 도움이 될 수 있지만 측정되지 않은 변인의 영향을 포함하고 있다. 따라서 irreducible error는 $$\epsilon$$의 분산에 해당한다.



$$\hat{f}$$과 $$X$$가 고정되어 있다면 아래와 같은 식이 성립한다.


$$
E(Y-\hat{Y})^2 = E[f(X)+\epsilon-\hat{f}(X)]^2 \\
\qquad \qquad \quad \quad \ \ =[f(X)-\hat{f}(X)]^2+Var(\epsilon)
$$



위 식에서 좌변의 $$E(Y − \hat{Y} )^2$$은 실제 $$Y$$와 예측된 $$Y$$의 차이의 제곱의 기댓값이다. 우변의 $$[f(X)-\hat{f}(X)]^2$$은 reducible error에, $$Var(\epsilon)$$은 irreducible error에 해당한다. Irreducible error는 말 그대로 줄일 수 없는 오차이기 때문에 $$Y$$에 대한 예측에서 정확도의 상한선을 정해준다. 우리의 목표는 이 상한선 내에서 reducible error를 줄여 최대한의 예측의 정확도를 달성하는 것이다.



#### 추론

예측과 달리 추론이 목적이라면 $$\hat{f}$$의 정확한 형태를 파악해야 한다. 즉, $$X$$에 따라 $$Y$$가 어떻게 변화하는지 그 관계에 대하여 이해해야 한다.

어떤 예측변인이 반응변인과 어떻게 관련되어 있는지, 두 변인의 관계가 선형적인지 아닌지 등과 같은 질문이 이에 해당할 것이다.



### 1.2 어떻게?

$$f$$를 추정하기 위하여 사용하는 데이터를 **training data**라고 한다. 이 데이터 내에서 $$x_{ij}$$라 함은 $$i$$번째 관측치($$i=1,2,...,n$$)의 $$j$$번째 예측변인($$j=1,2,...,p$$)에 해당하는 값을 의미한다. 한편, $$y_i$$라 함은 $$i$$번째 관측치에 대한 반응변인을 말한다. 그렇다면 training data는 $$\left\{(x_1, y_1), (x_2, y_2), ... , (x_n, y_n) \right\}$$의 형태로 표현할 수 있을 것이다. ($$x_i = (x_{i1}, x_{i2}, ... , x_{ip})^T$$)



우리의 목표는 각 관측치의 $$(X, Y)$$ 쌍에 대하여 $$Y ≈ \hat{f}(X)$$를 충족하는 $$\hat{f}$$을 찾는 것이다. 이를 위해서는 크게 모수적 방법(parametric methods)과 비모수적 방법(non-parametric methods)을 사용할 수 있다.



#### 모수적 방법

모수적 방법에서는 $$f$$의 함수적 형태를 미리 가정한 다음 training data를 사용해 모델을 훈련하고 파라미터를 추정한다. 이 방법의 장점은 $$f$$를 추정하는 문제를 파라미터를 추정하는 문제로 축소할 수 있다는 것이다. 단점은 $$f$$의 진짜 형태를 모르는 상태에서 우리가 선택한 모델이 그에 잘 맞지 않을 수 있다는 것인데, 유연한(flexible) 모델을 선택함으로써 어느 정도 이러한 문제를 해소할 수 있다. 그러나 유연한 모델은 항상 과적합(overfitting) 문제를 안고 있다는 점에 유의하여야 한다.



#### 비모수적 방법

모수적 방법과 달리 비모수적 방법에서는 $$f$$의 함수적 형태를 미리 가정하지 않은 상태에서 데이터포인트들을 잘 반영할 수 있는 $$f$$의 추정치를 찾는다. 장점은 $$f$$에 보다 정확하게 모델을 적합시킬 수 있다는 것이지만, 정확한 추정을 위해서는 매우 많은 수의 관측치들이 필요하다고 한다.



### 1.3 예측 정확도와 해석 용이성 간의 트레이드오프

만일 통계 학습에서 추론을 목적으로 하고 있다면, 심플한 모델을 사용하는 것이 해석의 용이성을 높인다. 지나치게 유연한 모델을 사용하면 $$f$$가 매우 복잡한 형태로 추정되어 개별 예측변인들이 반응변인과 어떤 연관성을 가지는지 이해하기 어려워지기 때문이다.

그러나 예측을 목적으로 할 때도 유연한 모델이 항상 좋은 성능을 보이는 것은 아니다. 과적합 때문에 오히려 예측의 정확도가 떨어질 수도 있다.



### 1.4 회귀 대 분류 문제

반응변인은 quantitative일 수도 있고 qualitative (categorical)일 수도 있다. 전자의 경우는 **회귀(regression)** 문제에 해당하고 후자의 경우는 **분류(classification)** 문제에 해당한다.



## 2 모델의 정확도 평가



### 2.1 적합도 측정하기

예측값이 실제 데이터에 얼마나 잘 부합하는지 평가하기 위하여 회귀 상황에서 가장 흔히 사용하는 척도는 **mean squared error (MSE)**이다.


$$
MSE = \frac{1}{n} \sum_{i=1}^n(y_i-\hat{f}(x_i))^2
$$



Training data에서 계산된 MSE를 training MSE라고 한다. 그러나 우리가 정말 알고 싶은 것은 training set이 아니라 우리가 한 번도 본 적 없는 새로운 test set에서의 MSE일 것이다. 안타깝게도 training MSE가 낮다고 해서 test MSE도 항상 낮은 것은 아니다.

모델의 유연성이 높아질수록 trianing MSE는 지속적으로 감소하지만, test MSE는 대개 어느 정도까지는 감소하다가 다시 높아지는 U자형 곡선을 그린다. **과적합(Overfitting)**이란 training MSE는 낮지만 test MSE는 높은 상황을 일컫는다.

실제로 우리는 test data에 접근할 수 없기 때문에 교차 검증(cross-validation)과 같은 방법을 사용하여 가지고 있는 training data로부터 test MSE를 추정한다.



### 2.2 편향-분산 트레이드오프

$$x_0​$$에 대한 test MSE의 기댓값은 세 가지 요소의 합으로 표현 될 수 있다.


$$
E(y_0-\hat{f}(x_0))^2 = Var(\hat{f}(x_0))+[Bias(\hat{f}(x_0))]^2 + Var(\epsilon)
$$



위의 식에서 **분산(variance)**은 우변의 첫 번째 항에 해당하며, 다른 training data set을 사용하여 추정할 때 $$\hat{f}$$이 변화하는 양을 뜻한다. Training set이 조금씩 바뀌더라도 $$f$$에 대한 추정치가 지나치게 변동하지 않는 것이 바람직할 것이다.

한편, **편향(bias)**은 우변의 두 번째 항에 해당하며, 복잡한 실생활의 문제를 훨씬 간단한 모델로 근사함에 따라 발생하는 오차를 뜻한다. 

우리의 목표는 좌변의 test MSE의 기댓값을 최소화하는 것이기 때문에우변의 세 번째 항에 해당하는 irreducible error를 제외하고 분산과 편향을 동시에 감소시킬 수 있다면 좋을 것이다. 그러나 보통 모델이 유연할수록 편향은 감소하지만 분산은 증가하는 트레이드오프가 발생한다.



### 2.3 The Classification Setting

* Estimate $f$ on the basis of training observations $\left\{(x_1, y_1), . . . , (x_n, y_n) \right\}$, where $y_1, . . . , y_n​$ are <u>qualitative</u>.
* The most common approach for quantifying the accuracy of our estimate $\hat{f}​$ is the __training error rate__, the proportion of mistakes that are made if we apply our estimate $\hat{f}​$ to the training observations.

$$
\frac{1}{n} \sum_{i=1}^n I(y_i \ne \hat{y}_i)
$$

 * Here $\hat{y}_i$ is the predicted class label for the $i$th observation using $\hat{f}$. And $I(y_i \ne \hat{y}_i)$ is an indicator variable that equals 1 if $y_i \ne \hat{y}_i$ and zero if $y_i = \hat{y}_i$.
 * The __test error rate__ associated with a set of test observations of the form $(x_0, y_0)$ is given by

$$
Ave(I(y_i \ne \hat{y}_i))
$$

* where $\hat{y}_0​$ is the predicted class label that results from applying the classifier to the test observation with predictor $x_0​$.



#### The Bayes Classifier

* The test error rate is minimized, on average, by a very simple classifier that <u>assigns each observation to the most likely class, given its predictor values</u>.
* We should simply assign a test observation with predictor vector $x_0$ to the class $j$ for which $Pr(Y=j | X=x_0)$ is largest. 

* In a two-class problem where there are only two possible response values, say class 1 or class 2, the Bayes classifier corresponds to predicting class one if $Pr(Y = 1|X = x0) > 0.5​$, and class two otherwise.

![1552456226551](C:\Users\Jinny\AppData\Roaming\Typora\typora-user-images\1552456226551.png)

* A simulated data set in a two-dimensional space consisting of predictors $X_1​$ and $X_2​$.
  * The orange and blue circles correspond to <u>training observations</u> that belong to two different classes.
  * The orange shaded region reflects the set of points for which $Pr(Y = orange|X)​$ is greater than 50%, while the blue shaded region indicates the set of points for which the probability is below 50%.
  * The purple dashed line represents the points where the probability is exactly 50%. $\rightarrow$ __Bayes decision boundary__

* The Bayes classifier produces the lowest possible test error rate. $\rightarrow$ __Bayes error rate__
  * The error rate at $X = x_0$ will be $1−max_j Pr(Y = j|X = x_0)​$.
  * In general, the overall Bayes error rate is given by $1−E (max_j Pr(Y = j|X))$.

* The Bayes error rate is <u>analogous to the irreducible error</u>.



#### K-Nearest Neighbors

* For real data, we do not know the conditional distribution of $Y$ given $X$, and so computing the Bayes classifier is <u>impossible</u>.
* The Bayes classifier serves as an <u>unattainable gold standard</u> against which to compare other methods.
* Many approaches attempt to <u>estimate</u> the conditional distribution of $Y$ given $X​$, and then classify a given observation to the class with <u>highest estimated probability</u>.
* K-nearest neighbors (KNN) classifier: identifies the <u>K points</u> in the training data that are <u>closest to $x_0​$</u>, represented by $\mathcal{N}_0​$. It then estimates the conditional probability for class $j​$ as the fraction of points in $\mathcal{N}_0​$ whose response values equal $j​$:

$$
Pr(Y = j|X = x_0) = \frac{1}{K} \sum_{i∈\mathcal{N}_0} I(y_i = j)
$$

* Finally, KNN applies Bayes rule and classifies the test observation $x_0$ to the class <u>with the largest probability</u>.

![1552457835217](C:\Users\Jinny\AppData\Roaming\Typora\typora-user-images\1552457835217.png)

* 왼쪽 그림
  * An illustrative example of the KNN approach: make a prediction for the point labeled by the black cross.
  * Suppose that we choose $K = 3$.
  * KNN will first identify the <u>three observations that are closest to the cross</u>: two blue points and one orange point, resulting in <u>estimated probabilities</u> of 2/3 for the blue class and 1/3 for the orange class. Hence KNN will predict that the black cross belongs to the blue class.
* 오른쪽 그림
  * applied the KNN approach with $K = 3$ at all of the possible values for $X_1$ and $X_2​$, and have drawn in the corresponding KNN decision boundary.
* KNN can often produce classifiers that are <u>surprisingly close to the optimal Bayes classifier</u>.
* <u>The choice of $K$</u> has a drastic effect on the KNN classifier obtained.
  * When $K = 1$, the decision boundary is <u>overly flexible</u> and finds patterns in the data that don’t correspond to the Bayes decision boundary. $\rightarrow$ __low bias__ but very __high variance__
  * As K grows, the method becomes <u>less flexible</u> and produces a decision boundary that is close to linear. $\rightarrow$ __low-variance__ but __high bias__

![1552458170089](C:\Users\Jinny\AppData\Roaming\Typora\typora-user-images\1552458170089.png)

* <u>As $1/K$ increases</u>, the method becomes <u>more flexible</u>. As in the regression setting, the <u>training error rate consistently declines</u> as the flexibility increases. However, the <u>test error</u> exhibits a characteristic <u>U-shape</u>.

![1552458260371](C:\Users\Jinny\AppData\Roaming\Typora\typora-user-images\1552458260371.png)

* In both the regression and classification settings, <u>choosing the correct level of flexibility is critical</u> to the success of any statistical learning method.
