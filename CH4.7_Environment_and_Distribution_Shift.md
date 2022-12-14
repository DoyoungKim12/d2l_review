# CH4.7_Environment_and_Distribution_Shift

<br/>

## 0. intro

가장 기저에 깔려있는 문제는

- 데이터가 어디서 오는지
- 모델에서의 최종 결과물 형태로 무엇을 할 것인지

이지만, 생각보다 우리는 이를 깊게 고려하지 않는다.

예들 들어, test set에서는 굉장히 좋은 성능을 낸 모델이라도 실제 배포 후 발생한 data shift에 성능을 내는 것을 실패하는 경우가 있다.

스니커즈를 신은 사람이 채납자일 가능성이 높다는 모델이 있다면, 모든 스니커즈를 신은 사람에게만 돈을 빌려 주지 않았다고 해 보자. 그리고 사람들은 이러한 경향을 알아차리고 모두 옥스퍼드를 신기 시작했다고 가정해 보자.

이런 케이스에서는, 다른 요인들에 대한 고려가 적은 상태로 옥스퍼드를 신은 사람에게 돈을 빌려주게 될 것이다. 

더 이상 모델이 제대로 작동하기 힘든 환경으로 변해버린 것이다.

<br/>

여기에서는, 이런 상황해 대해 피해를 줄이고 경향을 최대한 빨리 알아차리기 위한 방법들을 알아보고자 한다. 어떤 방법은 쉽고("올바른" 데이터에게 검증하는 것) 어떤 방법은 까다롭다(강화학습을 적용하는 등). 

<br/>

## 1. Types of Distribution Shift

데이터의 disttribution shift는 다양한 방법으로 일어날 수 있다. 어떻게 해야 모델 성능에 악영향에 피해를 최소화할 수 있을지부터 먼저 살펴보자.

우리의 train data는 $P_s(x, y)$ 같은 분포를 가진 데이터에서 샘플링하였다.

반면, test data label이 없는 $P_T(x, y)$라는 다른 분포이다.

$P_S$와 $P_T$의 분포가 어떻게 관련되어있는지 모르는 상태에서 robust한 모델을 만드는 것은 사실상 불가능할 것이다.

다행이도, 몇몇 가정 위에서 우리는 모델이 data shift를 감지하고 (운이 좋다면) 적용할 방법을 찾을 수 있다.

<br/>

### 1-1. Covarivate Shift

distribution shift 중에 가장 유명한 것은 바로 covariate shift이다. 

main concept은 input 데이터의 분포가 변할지라도, labeling function (예를 들어, $P(y|x)$와 같은 conditional dist.)은 변하지 않는다는 것이다.

이름이 covariate shift인 이유는 covariate (즉, features)의 분포 변화가 문제가 되기 때문이다.

예를 들어, 1번과 같은 개와 고양이의 사진을 주고 2번을 추론하라고 한다면 어떨까?

1번
<img width="459" alt="스크린샷 2022-10-24 오후 10 13 23" src="https://user-images.githubusercontent.com/62180861/197534100-507ebbd0-f4ba-4aad-afd9-c92f885d5ded.png">

2번
<img width="458" alt="스크린샷 2022-10-24 오후 10 13 30" src="https://user-images.githubusercontent.com/62180861/197534110-98766ca3-e32c-4c27-a9ad-6586efab24c8.png">


두 데이터셋 모두 개와 고양이는 맞지만(label이 동일), 특성이 다르다(feature 변화). 1번의 경우 사진 형태이고, 2번의 경우 애니메이션 형태로 1번으로 학습한 모델이 2번에서 잘 작동하기는 어려울 것이다.

<br/>

### 1-2. Label Shift

label shift의 경우 반대의 상황이다. 여기서 우리는 label의 $P(y)$가 변화하고 class-conditional dist. (즉, $P(x|y)$)는 동일한 상황을 생각해볼 수 있다.

물론, covariate shift와 label shift가 동시에 발생할 수도 있다.

예를 들어, label이 feature의 특성을 결정짓는 경우 label의 변화와 covariate의 변화가 동시에 발생하게 되는 것이다.

특이하게도, 이런 경우는 label shift assumption을 따른 방법을 다루기가 쉬워진다. 왜냐하면 label과 같이 생긴 objects를 포함시킬 수 있기 때문이다. 단순히 input의 특성만 가진 (hight-dimensional) objects와는 반대로 말이다.

<br/>

### 1-3. Concept Shift

concept shit는 label의 정의가 크게 달라지는 경우를 말한다. 예를 들어, 의학 용어의 기준이 바뀌는 사례를 들 수 있겠다.

또는 머신러닝으로 번역을 하는 경우를 생각해보면 $P(y|x)$의 분포가 언어에 따라 굉장히 차이난다는 것을 알 수 있다.

<br/>

## 2. Examples of Distribution Shift

공식과 알고리즘을 배우는 것으로 들어가기 앞서, covariate 또는 concept shift가 명확하지 않은 상황들에 대해 살펴보자.

<br/>

### 2-1. Medical Diagnostics

암을 판별하는 알고리즘을 디자인인하려고 한다고 해보자. 건강하고 아픈 사람들의 데이터를 모아 알고리즘을 학습시키게 될 것이다. 만약, 생각한대로 흘러간다면 우리는 잘 작동하는 암 판별 알고리즘을 결과물로 가지게 될 것이다..!

그러나 보통 이렇게 사용된 train data는 현실의 것과 괴리가 존재한다.

예를 들어, 피 검사 결과로 아픈 사람들을 구별하는 알고리즘을 만들기 위해 아픈 사람들과 건강한 사람들의 피를 모으는 작업을 상상해보자. 아픈 사람들의 경우 이미 내원을 하였고 검사를 받은 이력이 있기에 샘플을 모으기가 어렵지 않지만 건강한 사람의 경우 보다 까다롭다. 

이 알고리즘을 만들고자한 회사는 대학생들에게 피를 후원받고자 하였다.

이렇게 모은 데이터로 만든 알고리즘은 건강하거나 아픈 집단을 굉장히 잘 구분하였다, 그러나, 기본적으로 데이터에 편향이 존재하였기 때문에 (대상자들의 나이, 호르몬 영향도, 신체 지수, 식이 습관 등) 현실의 데이터와는 괴리가 있었다.

즉, sampling 과정 때문에 발생한 극심한 covariate shift에 직면하게 된 것이다.

<br/>

### 2-2. Self-Driving Cars

한 회사가 자율주행 자동차를 만들고 싶다고 해 보자. 이 과저에서 주요한 요소중 하나는 도로를 탐지하는 것이다. 실제 도로 데이터를 확보하기는 힘들기 때문에, 대신 그들은 게임 랜더링된 도로 데이터를 train data로 확보하였다.

이렇게 탄생한 알고리즘은 게임 렌더링된 test data에서는 문제없이 잘 작동하였다. 그러나, 현실 세계에서는 잘 작동할 수 있을리가 없었다.

<br/>

### 2-3. Nonstationary Distribution

distribution이 보다 천천히 변화(nonstationary distribution)하지만, 모델이 알맞게 업데이트 되지 않는 경우 조금 더 미묘한 상황이 발생한다.

예를 들어,
- 자동 마케팅 솔루션 알고리즘 -> 새로운 제품 출시 업데이트를 하지 않음
- 스팸 필터 -> 스팸 발신자가 새로운 형태의 스팸을 보냄
- 상품 추천 시스템 -> 겨울 기간에 만들어져 계속 산타 관련 상품을 추천함

<br/>

### 2-4. More Anecdotes

- face detector -> 벤치마크에서는 좋은 성능을 냈지만, test data에서는 그렇지 못함. test data에는 train set에 없었던 꽉 찬 얼굴 이미지가 많았기 때문이다.
- web search engine을 US에서 만들었는데 UK에 적용하고 싶음
- 각 1000개의 이미지가 있는 100개의 카테ㄹ고리에 대한 데이터 셋을 학습 -> 현실에는 label distribution이 non-uniform함

<br/>

## 3. Correction of DistributionShift

어떤 경우에는 shift가 발생하여도 모델이 잘 작동하지만 (운이 좋게도), 그렇지 않은 경우에는 shift를 다루기 위해 특정 방법들을 적용할 필요가 있다.

앞으로는 이 테크닉 들에 대해 좀 더 자세히 알아보도록 하겠다.

<br/>

### 3-1. Empirical Risk

우선, 모델을 학습할 때 정확히 어떤 과정들이 진행되는지 짚고 넘어가도록 하자.

training data에서 각 feature들과 거기에 연결된 label들을 iterate하게 학습하게 된다. 이렇게 돌아간 mini-batch에서는 모델의 파라미터 $f$를 업데이트 한다.

결과적으로 loss를 최소화하는 training은 다음과 같이 진행될 것이다.

<img width="250" alt="스크린샷 2022-10-25 오전 10 21 34" src="https://user-images.githubusercontent.com/62180861/197660017-b2ff4e29-61ee-4e08-a2e1-b4e0ad74a2c8.png">

여기에서의 $l$은 loss function으로 $f(x_i)$ 가 $y_i$를 얼마나 잘못 예측하는지를 측정한다.

통계학자들은 이를 *empirical risk* 라고 부른다. true distridution $p(x,y)$에서 온 전체 표본에 대한 loss의 평균이라고 볼 수 있겠다. 수식으로 표현하자면 아래와 같다.

<img width="394" alt="스크린샷 2022-10-25 오전 10 21 41" src="https://user-images.githubusercontent.com/62180861/197660021-6a8df083-c653-48cf-bbe2-8240545bf4bb.png">

<br/>

그러나, 보통의 경우 우리는 전체 표본을 확보할 수 없다.

따라서, 최대한 유사하게 추정하는 방향으로 진행해보고자 한다.

<br/>

### 3-2. Covariate Shift Correction

labed data $(x_i, y_i)$에 대하여 $P(y|x)$를 추정하고 싶다고 해 보자. 슬프게도, 관측치 $x_i$는 *target distribution* $p(x)$보다는 *source distribution* $q(x)$에서 나올 가능성이 크다. 다행히 기본적인 의존 가정 $p(y|x) = q(y|x)$는 변하지 않는다는 것이다.

만약, $q(x)$가 틀렸다면 아래와 같은 risk의 간단한 identity로 정정할 수 있다.

<img width="589" alt="스크린샷 2022-10-25 오전 10 21 48" src="https://user-images.githubusercontent.com/62180861/197660022-fa448f7b-79b2-4462-a322-31a0265c9adc.png">

즉, 각 data example을 실제 distribution에서 나올 확률로 가중치를 만들어주는 것이다. 

실제 분포 $q(x_i)$에서 sample된 데이터 $p(x_i)$가 나올 확률을 아래와 같이 $\beta_i$라고 정의한다면


<img width="150" alt="스크린샷 2022-10-25 오후 12 43 43" src="https://user-images.githubusercontent.com/62180861/197677396-9bcc5ffd-50d0-4301-877c-2681cc6c0321.png">

최종적인 *weighted empirical risk minimization*은  다음과 같이 표현될 수 있을 것이다.

<img width="269" alt="스크린샷 2022-10-25 오후 12 43 48" src="https://user-images.githubusercontent.com/62180861/197677401-85a3938f-e8f8-4a12-8d6e-cd0d5e2ceed1.png">

그런데 슬프게도, 보통은 $\beta_i$ 자체를 모르는 경우가 대부분이다. 따라서 *weighted empirical risk minimization*을 이용하기 위해서는 $\beta_i$를 먼저 추정해야한다.

이를 위해서는 $x \sim p(x)$가 필요하다. (오히려 $y \sim p(y)$는 필요치 않다.)

<br/>

이 상황에서 가장 좋은 성능을 내는 효과적인 방법은 **logistic regression**을 사용하는 것이다. (특히, softmax regression for binary classification) 이것이 estimated probability ratios를 구하기 위해 필요한 전부이다.

softmax regression을 할 때 배웠듯, 만약 $p(x)$에서 나온 데이터와 $q(x)$에서 나온 데이터를 구분할 수 없다면 이는 다시 말해 두 분포를 구분할 수 없다는 것이다. (즉, 유사한 분포) 반대로 어느 분포에서 나온 것인지 구분이 된다면 두 분포가 다르다고 볼 수 있다.

쉽게 설명하기 위해서 우리가 $p(x)$와 $q(x)$ 두 분포에서 나온 동일한 개수의 instance을 가지고 있다고 해 보자.

그리고 분포 $q(x)$에서 나온 데이터라면 -1, $p(x)$에서 나온 데이터라면 1인 label $z$를 추가한다.

그러면 아래와 같은 식을 도출할 수 있다.

<img width="520" alt="스크린샷 2022-10-26 오후 8 48 32" src="https://user-images.githubusercontent.com/62180861/198018515-5a0b18b6-0c62-49dd-ad2f-8647b65b8f05.png">

여기에 logisitic regression 방법을 적용한다면 아래와 같이 $P(z=1|x)$를 표현할 수 있다.

<img width="438" alt="스크린샷 2022-10-26 오후 8 50 37" src="https://user-images.githubusercontent.com/62180861/198018956-8e727f9f-41fa-485a-ae09-06080cd2c93b.png">

그러면 결과적으로 $q(x_i)$에서 $p(x_i)$가 나올 확률인 $\beta_i$는 다음과 같다.

<img width="436" alt="스크린샷 2022-10-26 오후 8 50 44" src="https://user-images.githubusercontent.com/62180861/198018960-1bc65a17-f964-48fe-95f7-1bd91043ab8c.png">

<br/>

그럼 이제 지금까지 말한 것들을 종합해 올바른 알고리즘을 써 보자. 우리 아래와 같은 train set과 unlabeled test set을 가지고 있다.

<img width="183" alt="스크린샷 2022-10-26 오후 9 17 57" src="https://user-images.githubusercontent.com/62180861/198024055-07cfdd63-be0f-436b-ae5d-9be97394efee.png"> <img width="107" alt="스크린샷 2022-10-26 오후 9 18 02" src="https://user-images.githubusercontent.com/62180861/198024059-af21aa0e-0774-46df-b56f-8c4000cef044.png">

$x_i \ for \ all \ 1 \leq i \leq n$는 모두 source dist.에서 나왔으며 $u_i \ for \ all \ 1 \leq i \leq m$는 모두 target dist. 에서 나왔다.

그렇다면,
1. binary-classification training set을 생성한다.
<img width="361" alt="스크린샷 2022-10-26 오후 9 23 14" src="https://user-images.githubusercontent.com/62180861/198025084-23c3eb78-4aac-4606-b0fe-1cae0ab5b16c.png">

2. logistic regression을 이용해 binary classifier를 훈련하고 function $h$를 얻는다.

3. $\beta_i \ = \ exp(h(x_i))$를 이용해 train data에 가중을 적용한다.

4. 이를 이용해 training을 진행한다.


한가지 명심해야할 점은, 위 알고리즘이 "target dist.이 train에서 발생할 확률이 0이 아님"이라는 가정 하에 이루어진다는 것이다. 만약 $p(x) \ > \ 0 \ but q(x) \ = \ 0$이라면, weight는 infinte해 질 것이기 때문이다.

<br/>

### 3-3. Label shift correction

우리가 $k$ categories에 대한 classification 문제를 다루고 있으며 $q$와 $p$ 분포가 각각 source dist. target dist 을 따른다고 가정해보자. 그리고 label shift가 발생하였고 (즉, $q(x|y) \neq p(x|y)$) class-conditional dist.은 동일하다고 (즉, $q(x|y) \ = \ p(x|y)$)해보자.

그렇다면, 우리는 risk를 다음과 같이 정의할 수 있다.

<img width="573" alt="스크린샷 2022-10-26 오후 9 46 00" src="https://user-images.githubusercontent.com/62180861/198029593-759e6eee-9870-4dfa-bd1b-dc3ef94c6805.png">

여기에서 weight는 label likelihood ratio를 따를 것이다.

<img width="112" alt="스크린샷 2022-10-26 오후 9 46 03" src="https://user-images.githubusercontent.com/62180861/198029602-e65449a7-dc7b-411f-a434-866fd815beb4.png">

label shift에서 좋은 점 하나는 만약 우리가 source dist.에 대해 좋은 모델을 가지고 있다면 ambient dimension(환경 공간?)을 다룰 필요 없이도 이 weight를 일관되게 추정할 수 있다는 것이다.

target label dist.을 추정하기 위해서 우선 좋은 성능의 classifier (train data로 학습된) 그리고 validation set에 대해 confusion matrix를 계산한다.

이 confusion matrix $C$ 는 심플한 $k$ X $k$ matrix로 각 column은 true label category이고 row는 모델 추론된 label category이다.

각 cell $c_{ij}$는 validation set에 대한 예측 인스턴스들이다. (true label이 $j$이고 model이 $i$로 예측한)

우리는 target data의 confusion matrix를 직접 계산할 수는 없다. 모든 현실 세계에 있는 데이터에 labeling을 할 수는 없기 때문에. 그래서 우리는 모델이 추론한 모든 test 데이터에 대해 각 라벨 i의 $\mu(\hat{y})\in R^{k}$를 계산한다.

만약 우리의 모델이 꽤나 accurate 하고 target data가 원래 라벨 카테고리에서만 나타나고 label shift 가정이 유지된다면 우리는 간단한 선형 모형을 통해 test label dist.를 추정할 수 있다.

<img width="127" alt="스크린샷 2022-10-27 오전 12 33 25" src="https://user-images.githubusercontent.com/62180861/198070181-0012a7dd-5b23-405d-a988-49af1d265984.png">

조금 풀어서 설명해보면 아래와 같다.

<img width="175" alt="스크린샷 2022-10-27 오전 10 59 57" src="https://user-images.githubusercontent.com/62180861/198173323-b48eb501-228b-435a-a8f0-e6d6f9bfd0e7.png">

target data에 대해 예측되는 라벨 i의 추정치 모집단은, 실제 라벨 j의 분포와 라벨이 j일 때 예측되는 i의 confusion matrix를 적용하면 구할 수 있을 것이다.

이 식을 다시 정리해보면 다음과 같다.

<img width="135" alt="스크린샷 2022-10-27 오전 11 11 42" src="https://user-images.githubusercontent.com/62180861/198174589-cd9ff484-10a3-4cef-9834-fc1bda749454.png">

결과적으로 우리는 target label dist.를 추정할 수 있는 것이다. source data label은 이미 가지고 있으므로 $q(y)$를 구하는 것을 어렵지 않으므로 $\beta_i$를 구하기 위해 $p(y_i) / q(y_i)$를 이용하면 된다. 결과적으로 weighted empirical risk minimization을 구할 수 있다.

<br/>

### 3-4. concept shift correction

concept shift는 다루기가 가장 어려운데, 가량 고양이와 강아지를 구분하는 문제에서 흰/검은 동물을 구분하는 문제가 넘어갔다고 가정해보자. 사실상 새로운 라벨에 대한 데이터를 다시 모아 모델을 학습시키는 것이 가장 좋은 방법일 것이다.

하지만 다행히 현실 세계에서는 concept shift가 점진적으로 일어난다. 예를 들어, 교통 카메라가 노후되어 화질이 점점 안좋아진다는 느낌으로 말이다. 

이런 경우에는 우리가 data change를 습득하기 위해 사용하였던 training network와 동일한 방법을 사용하면 된다. (즉, existing network weight를 new data를 이용해 조금 update)

<br/>

## 4. A Taxonomy of Learning Problems

지금까지는 dist. change를 다루는 법을 모았다면, 이제 다른 측면에서 나타나는 문제를 봐보도록 하자.

### 4-1. Batch learning

한번만 학습하고 계속 사용하는 것의 문제가 발생. (여기서의 batch는 1번 학습을 의미하는 것 하다.) 예를 들어, 고양이 인식 도어 프로세스를 생각해보면 처음 데이터로 학습을 한 이후에 사용을 할 때에는 다시 update하지 않고 계속 사용하게 될 것.

### 4-2. Online learning

주식 가격 예측 모델 같은 것을 생각해보면, 데이터로 먼저 예측을 하고 그 다음에 실제 라벨을 관찰할 수 있다.

그래서 online learning에서는 아래와 같은 사이클을 따라 계속 성능을 업데이트할 수 있다.

<img width="764" alt="스크린샷 2022-10-27 오전 1 08 14" src="https://user-images.githubusercontent.com/62180861/198078316-0f238acf-b2a6-4f3c-8c47-98c4459c2491.png">

### 4-3. Bandits

해볼 수 있는 행동의 범위가 제한적이라는 것도 문제 상황이 될 수 있다. 예를 들어, Multi-armed Bandit Problem을 생각해보면 이해가 쉬울 것이다.

### 4-4. Control

보통 어떤 액션을 취하게 되면, 환경에는 그 기록이 남게 되는 법이다. 예를 들어, coffee boiler controller는 이전에 boiler가 데워졌는지 여부에 따라 다른 온도를 측정하게 될 것이다.

또 다른 예로는, 뉴스 사이트는 개인이 이전에 어떤 뉴스를 봤는지에 따라 다른 뉴스를 노출시킬 것이다. 결국 나오는 결과는 less random 할 수 밖에 없다.

최근에는 문자 생성의 다양성을 높이고 이미지 생성 퀄리티를 높이기 위해 PID와 같은 control theory를 사용한다고 한다.

### 4-5. Reinforcement Learning

memory가 저장되는 환경이라면, 환경 자체가 변화하고 (우리를 이기려하는) 상황을 맞이할 수도 있을 것이다.

게임의 케이스를 생각해보거나, 자율 주행 차에서 운전자의 운전 스타일을 학습해가는 경우를 상상해보면 될 것 같다.

### 4-6.  Considering the Environment

환경의 변화가 발생할 수도 있다는 점도 생각해볼 문제이다.

예를 들어, 부동산 중개업자가 매매를 하려고 마음을 먹었을 때는 이미 그 매물이 없는 상태일 수도 있다. 

그리고 만약 우리그 그 환경의 변화가 어떤 방식으로 일어나는지 (점진적인지 급진적인지 등등) 알고 있다면 이를 모델에 반영해줄 수도 있을 것이다.

위에서 나왔던 concept shift에 대응하는 개념과도 유사하게 느껴진다.

<br/>

## 5. Fairness, Accountability, and Transparency in Machine Learning

모델을 생각할 때는 단순히 예측에서만 끝나는 것이 아니라 그 예측을 활용해 만들어질 결정까지도 고려를 해야한다.

또한 그 결과의 윤리적 문제도 고려를 해야한다.

적용의 측면에서 고민하다보면 단순히 accuracy가 모델의 정답 성능은 아니라는 것을 알게 될 것이다.

모델의 feedback loop를 고려해볼 수도 있다.예를 들어, 범죄가 많은 곳에 더 많은 경찰을 배치하기 위한 예측을 한다고 해 보자. 예측된 것에 경찰을 더 배치하면 더 많은 범죄가 발견될 수도 있고 이 피드백 이 다시금 그 지역에 더 많은 경찰을 배치하게 만들 수도 있는 것이다.

마지막으로, 우리가 푸는 문제가 실제로 옳은지에 대해 판단하는 과정도 있겠다. 예를 들어, 과연 뉴스 페이지에서 보여주는 뉴스가 해당 사람이 페이스북으로 좋아요를 찍은 관련 기사만 필터링이 되는 것이 맞을까?


<br/>

이런 여러 딜레마들은 ML 문제를 푸는 것에 있어 계속 나타나게 될 것이다. (생각할 가치가 있지 않을까..!)
