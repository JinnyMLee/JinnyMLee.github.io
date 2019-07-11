---
layout: post
title: ISL 2. Statistical Learning
subtitle: An Introduction to Statistical Learning
tags: [ISL]
use_math: true
---



## 2.1 통계 학습이란 무엇인가?

통계 학습은 **예측변인(predictor)** (혹은 입력변수, 독립변인, feature) $X$와 **반응변인(response)** (혹은 출력변수, 종속변인) $Y$로 이루어진다. 우리는 이 $X$와 $Y$ 사이에 어떤 관계가 있음을 가정하며 이것은 아래의 식처럼 표현될 수 있다.
$$
Y = f(X) + \epsilon
$$

* * Here $$f$$ is some fixed but unknown function of $$X_1$$, ..., $$X_p$$.

  * $$\epsilon$$ is a <u>random error term</u>, which is independent of $$X​$$ and has <u>mean zero</u>.
  * In this formulation, $$f$$ represents the <u>systematic information</u> that $$X$$ provides about $$Y​$$.

* Predict income using years of education

![1552388120038](C:\Users\이명진\AppData\Roaming\Typora\typora-user-images\1552388120038.png)

* * One must <u>estimate</u> $$f$$ <u>based on the observed points</u>.

  * The blue curve represents $$f$$ and the vertical lines represent the error terms $$\epsilon$$.

* In essence, __statistical learning__ <u>refers to a set of approaches for estimating $$f$$</u>.



### 2.1.1 Why Estimate f?



#### Prediction

* Since the <u>error term averages to zero</u>, we can predict $$Y$$ using

$$
\hat{Y} = \hat{f}(X)
$$

* where $$\hat{f}$$ represents our estimate for $$f$$, and $$\hat{Y}$$ represents the resulting prediction for $$Y$$.
* $$\hat{f}$$ is often treated as a <u>black box</u>, in the sense that one is <u>not typically concerned with the exact form of $$\hat{f}$$</u>, provided that it yields accurate predictions for $$Y$$.

* The accuracy of $$\hat{Y}$$ as a prediction for $$Y$$ depends on two quantities
  * __Reducible error__: Error introduced by the fact that $$\hat{f}$$ will not be a perfect estimate for $$f$$
    * estimation 과정에서 적절한 테크닉을 사용하면 reducible
  * __Irreducible error__: $$\epsilon$$의 variability. $$X$$를 통해 $$Y$$를 예측할 수 없는 부분 때문에 $$\epsilon$$ 존재.
    * The quantity $$\epsilon$$ may contain <u>unmeasured variables</u> that are useful in predicting $$Y​$$. It also may contain <u>unmeasurable variation</u>.
* Assume for a moment that both $\hat{f}$ and $$X$$ are fixed. Then, it is easy to show that

$$
E(Y-\hat{Y})^2 = E[f(X)+\epsilon-\hat{f}(X)]^2 \\
\qquad \qquad \quad \qquad =[f(X)-\hat{f}(X)]^2+Var(\epsilon)
$$

* * where $E(Y − \hat{Y} )^2$ represents the average, or __expected value__, of the squared difference between the predicted and actual value of $Y$

  * $[f(X)-\hat{f}(X)]^2$: reducible error
  * $Var(\epsilon)$: irreducible error

* The focus of this book is on techniques for estimating $f$ with the aim of <u>minimizing the reducible error</u>.

* <u>Irreducible error</u> will always provide an <u>upper bound</u> on the accuracy of our prediction for $Y​$. This bound is almost always <u>unknown</u> in practice.



#### Inference

* Understand <u>the relationship between $X$ and $Y$</u> , or more specifically, to understand <u>how $Y$ changes as a function of $X_1$, ..., $X_p$</u>.
* Now $\hat{f}$ cannot be treated as a black box, because <u>we need to know its exact form</u>.
* Questions
  * <u>Which predictors</u> are associated with the response?
  * <u>What is the relationship</u> between the response and each predictor?
  * Can the relationship between $Y$ and each predictor be adequately summarized using a <u>linear</u> equation, or is the relationship more complicated?



* Depending on whether our ultimate goal is prediction, inference, or a combination of the two, different methods for estimating $f​$ may be appropriate.



### 2.1.2 How Do We Estimate $f$?

* __Training data__: Observations we use to <u>train, or teach, our method how to estimate $f$</u>.
* Let $x_{ij}$ represent the value of the $j$th <u>predictor, or input</u>, for observation $i$, where $i$ = 1, 2, ..., $n$ and $j$ = 1, 2, ... , $p​$.
* Correspondingly, let $y_i$ represent the <u>response variable</u> for the $i​$th observation.
* Then our training data consist of $\left\{(x_1, y_1), (x_2, y_2), ... , (x_n, y_n) \right\}​$ where $x_i = (x_{i1}, x_{i2}, ... , x_{ip})^T​$.
* Goal: find a function $\hat{f}$ such that $Y ≈ \hat{f}(X)$ for any observation $(X, Y)$



#### Parametric Methods

1. make an <u>assumption about the functional form, or shape, of $f$</u>

   * Example: linear model

     $$f(X) = β_0 + β_1X_1 + β_2X_2 + . . . + β_pX_p$$ 

2. use the <u>training data to fit or train the model</u> & <u>estimate the parameters</u>

   * Example: (ordinary) least squares to estimate $β_0$, $β_1$, ... , $β_p​$

* Parametric methods <u>reduces the problem</u> of estimating $f$ down to one of <u>estimating a set of parameters</u>.

* _Disadvantage_: Model we choose will usually <u>not match</u> the true unknown form of $f$.

  $\Rightarrow $ can address this problem by <u>choosing flexible models</u> that can fit many different possible functional forms flexible for $f$.

* However, fitting a more flexible model requires <u>estimating a greater number of parameters</u>.

  $\Rightarrow ​$ __Overfitting problem__: A model follows the errors, or noise, too closely.



#### Non-parametric Methods

* do <u>not make explicit assumptions</u> about the functional form of $f​$.
* seek <u>an estimate of $f$ that gets as close to the data points as possible</u> without being too rough or wiggly.
* _Advantage_: potential to <u>accurately fit</u> a wider range of possible shapes for $f$.
* _Disadvantage_: <u>a very large number of observations is required</u> in order to obtain an accurate estimate for $f$.
* Example: Thin-plate spline



### 2.1.3 The Trade-Off Between Prediction Accuracy and Model Interpretability

![1552462590668](C:\Users\Jinny\AppData\Roaming\Typora\typora-user-images\1552462590668.png)

* If we are mainly interested in <u>inference</u>, then <u>restrictive models are much more interpretable</u>.
* Very <u>flexible</u> approaches can lead to such <u>complicated estimates</u> of $f$ that it is <u>difficult to understand</u> how any individual predictor is associated with the response.
* Flexible methods에서는 overfitting의 위험성이 있으므로 오히려 less flexible method를 사용하는 것이 더 정확한 predictions을 내릴 수도 있음.



### 2.1.4 Supervised Versus Unsupervised Learning

* __Supervised learning__: For each observation of the predictor measurement(s) $x_i$, $i = 1, . . . , n$ there is an <u>associated response measurement</u> $y_i​$.
  * Many classical statistical learning methods such as linear regression and logistic regression, as well as more modern approaches such as GAM, boosting, and support vector machines, operate in the supervised learning domain.
* __Unsupervised learning__: For every observation $i = 1, . . . , n$, we observe a vector of measurements $x_i$ but <u>no associated response</u> $y_i​$.
  * We can seek to understand the <u>relationships between the variables or between the observations</u>. $\Rightarrow​$ __cluster analysis__, or __clustering__
  * Visual inspection is simply not a viable way to identify clusters. For this reason, <u>automated clustering methods</u> are important.

* __Semi-supervised learning problem__: observation 중 일부는 predictor와 response가 둘 다 있고, 또 다른 일부는 predictor만 있는 경우



### 2.1.5 Regression Versus Classification Problems

* Variables can be characterized as either __quantitative__ or __qualitative__ (__categorical__).
* Problems with a <u>quantitative</u> response $\rightarrow$ __Regression problems__
* Problems with a <u>qualitative</u> response $\rightarrow$ __Classification problems__



## 2.2 Assessing Model Accuracy



### 2.2.1 Measuring the Quality of Fit

* How well its predictions <u>actually match</u> the observed data
* The most commonly-used measure is the __mean squared error (MSE)__.

$$
MSE = \frac{1}{n} \sum_{i=1}^n(y_i-\hat{f}(x_i))^2
$$

  * __Training MSE__: The MSE computed using the <u>training data</u> that was used to fit the model
  * 그렇지만 우리가 정말 알고 싶은 것은 __test MSE__. Test MSE가 가장 작은 모델을 선택해야 함.
      * We want to know whether $\hat{f}(x_0)$ is approximately equal to $y_0$, where $(x_0, y_0)$ is a <u>previously unseen test observation</u> not used to train the statistical learning method.

$$
Ave(y_0-\hat{f}(x_0))^2
$$

* No guarantee that the method with the lowest training MSE will also have the lowest test MSE.

![1552452993136](C:\Users\Jinny\AppData\Roaming\Typora\typora-user-images\1552452993136.png)

* 왼쪽 그림

  * The black curve: <u>true</u> $f$
  * The orange, blue and green curves: three possible <u>estimates</u> for $f​$ obtained using methods with increasing levels of <u>flexibility</u>. As the level of flexibility increases, the curves fit the observed spline
    data more closely.

* 오른쪽 그림

  * The grey curve: the average <u>training MSE</u> as a function of flexibility (__degrees of freedom__)
    * The training MSE <u>declines</u> monotonically as flexibility increases.

  * The red curve: the <u>test MSE</u>
    * The test MSE <u>initially declines</u> as the level of flexibility increases. However, at some point the test MSE levels off and then starts to <u>increase again</u>. (U-shaped)
    * <u>The blue curve</u> minimizes the test MSE.

  * The horizontal dashed line: $Var(\epsilon)$, the irreducible error, which corresponds to the <u>lowest achievable test MSE</u> among all possible methods.

  * As the flexibility of the statistical learning method increases, we observe a <u>monotone decrease in the training MSE</u> and a <u>U-shape in the test MSE</u>.

* When a given method yields a <u>small training MSE</u> but a <u>large test MSE</u>, we are said to be __overfitting__ the data.
* We almost always expect the <u>training MSE to be smaller than the test MSE</u>.
* Usually <u>no test data are available</u>. $\rightarrow$ __cross-validation__, which is a method for <u>estimating test MSE using the training data</u>

![1552462657084](C:\Users\Jinny\AppData\Roaming\Typora\typora-user-images\1552462657084.png)

![1552462683415](C:\Users\Jinny\AppData\Roaming\Typora\typora-user-images\1552462683415.png)



### 2.2.2 The Bias-Variance Trade-Off

* The <u>expected test MSE</u>, for a given value $x_0$, can always be decomposed into the sum of three fundamental quantities: the __variance__ of $\hat{f}(x_0)$, the squared __bias__ of $\hat{f}(x_0)$ and the variance of the error terms $\epsilon$.

$$
E(y_0-\hat{f}(x_0))^2 = Var(\hat{f}(x_0))+[Bias(\hat{f}(x_0))]^2 + Var(\epsilon)
$$

* $E(y_0-\hat{f}(x_0))^2$ refers the expected test MSE, the average test MSE that we would obtain if we repeatedly estimated $f$ using a large number of training sets, and tested each at $x_0$.
* In order to minimize the expected test error, we need to select a statistical learning method that simultaneously achieves __low variance__ and __low bias__.
* __Variance__ refers to <u>the amount by which $\hat{f}$ would change if we estimated it using a different training data set</u>. Ideally, the estimate for $f$ should not vary too much between training sets.
* __Bias__ refers to <u>the error that is introduced by approximating a real-life problem</u>, which may be extremely complicated, <u>by a much simpler model</u>.
* As a general rule, as we use more <u>flexible</u> methods, the <u>variance will increase</u> and the <u>bias will decrease</u>.

![1552454652027](C:\Users\Jinny\AppData\Roaming\Typora\typora-user-images\1552454652027.png)

 * The horizontal dashed line represents $Var(\epsilon)$, the irreducible error.



### 2.2.3 The Classification Setting

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
