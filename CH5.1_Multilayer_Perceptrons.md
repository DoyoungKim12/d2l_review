# 5.1 Multilayer Perceptrons

- Section 4.1, 4.4, 4.5 에서 softmax regression에 대한 이론적 배경 및 코드 구현을 다루었습니다.
- Linear Model을 벗어나 deep neural network에 대한 이야기를 시작합니다.



## 5.1.1 Hidden Layers

- Deep neural network이 왜 필요한가? 
  $\rarr$ Linear model의 한계점 때문에 더 유연한 모델이 필요하다

- Linear Model 리뷰
  - example: Softmax Regression
    $\begin{split}\begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned}\end{split}$
  - 이 모델은 affine transformation으로 출력값을 산출하고 softmax를 적용합니다. 
    - affine transformation = linear transformation via weighted sum + translation via added bias
  - 만약 label과 input의 관계가 affine transformation으로 표현될 수 있는 linear 관계라면 Linear Model만 사용해도 충분할 것입니다.
  - 하지만 linearity 관계가 아닌 label-input 쌍이 많습니다.
  
  

### 5.1.1.1 Limitations of Linear Models

- Linearity에는 monotonicity (단조성)가 포함됩니다.
- monotonicity란 feature의 증가가 항상 
  output의 증가로 이어지거나 (weight가 양수일 때)
  output의 감소로 이어진다는 것을 의미합니다. (weight가 음수일 때)
- 실제 데이터/자연 현상에서 monotonicity 가정을 위반하는 경우는 너무나 흔합니다. 
  - y: 건강, x: 체온
  - Linear 모델로 이 관계를 표현하려면 $|37-x|$ 같은 feature engineering으로 
    feature와 output 사이의 monotonicity를 강제할 수 있습니다.
- 하지만 monotonicity가 만족이 된다고 linearity가 만족되는 것은 아닙니다.
  - y: 대출 상환 여부, x: income
  - 실제로 income이 높을수록 대출 상환 확률이 오를 것이고 
    x와 y는monotonic 관계를 가집니다
  - 하지만, x와 y는 선형적인 관계를 가지지 않습니다.
  - income \$0과 \$50,000를 가지는 사람의 대출 상환 여부 likelihood 차이와
    income \$1M와 $1.05M를 가지는 사람의 대출 상환 여부 likelihood 차이가 같지 않을 것입니다.
  - 로그 스케일링 등 전처리를 통해서 x와 y가 선형관계를 가지도록 유도할 수 있습니다.
- feature engineering으로 선형성을 유도할 수 있다면 문제가 없지 않는가?
  - 쉬운 feature engineering 해법을 찾을 수 없는 문제도 많다.
  - x: image pixel 값, y: is_dog
  - 이미지 픽셀 값을 입력으로 받는 선형 모델을 사용한다면:
    특정 위치의 픽셀 값 증가와 is_dog의 likelihood가 monotonic한 관계를 가진다거 가정하게 된다. 
  - 다르게 말하면 선형모델을 사용하면 각 픽셀의 밝기가 독립적으로 is_dog likelihood와 연관되어 있어야한다는 것이다.
  - 하지만 이미지 내 강아지 위치가 달라질 수 있고, 사진 반전의 케이스 등 이미지 픽셀 데이터는 선형 모델의 가정을 위반하는 경우가 많다.
  - Linear Model을 사용한다면 is_dog이 잘 예측되지 않을 수 밖에 없다.
  - 이미지 데이터에 feature engineering을 잘 수행하면 Linear Model과 사용할 수 있는 이미지 representation을 얻을 수 있을지도 모른다.
    하지만, 이 과정은 어렵다.
  - Deep neural network을 사용하면 우리는 hidden layer를 통해 이미지 representation과 linear 모델을 동시에 학습할 수 있다.
- 결론: deep neural network을 쓰면 복잡한 문제를 비교적 쉽게 해결할 수 있다.



### 5.1.1.2 Incorporating Hidden Layers

- Linear Model의 한계를 극복하기 위해서 1개 이상의 hidden layer를 사용할 수 있다.
- 가장 간단한 형태의 hidden layer는 fully connected layer를 stack하는 방식이다.
- 각 layer의 출력이 다음 layer에 입력되어 최종 출력값을 얻게된다.
- 출력값을 얻기 전 결과물($L-1$ layer의 출력)을 feature에 대한 representation으로 생각할 수 있고
- 마지막 layer를 linear predictor로 생각할 수 있다.
- 이런 모델 아키텍처 **multilayer perceptron** (MLP)라고 부른다.

![An MLP with a hidden layer of 5 hidden units. ](https://d2l.ai/_images/mlp.svg)

- 위 그림의 MLP는 4개의 input, 3개의 output을 가지고, 5개의 hidden unit을 가진다.



### 5.1.1.3 From Linear to Nonlinear

- linear model에 hidden layer를 stack한다고, 모델이 Linear 가정에서 자유로워지는건 아닙니다.



- MLP 모델의 입력이 각각 d개의 feature를 가지는 row n개로 구성된 minibatch matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$로 정의하고
- MLP 모델이 1개의 hidden layer를 가지고
- hidden layer가 h개의 hidden unit을 가진다면
- hidden layer의 출력값(hidden representation)은 $\mathbf{H} \in \mathbb{R}^{n \times h}$로 표기하고 n x h 크기를 가집니다
- hidden layer는 weight와 bias는 $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$와 $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$로 표기합니다
- output layer의 weight와 bias는 $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$와 $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$로 표기합니다.
- 최종 모델 출력값을 $\mathbf{O} \in \mathbb{R}^{n \times q}$로 표기합니다.



- 이런 모델을 다음과 같이 요약할 수 있습니다:
  $\begin{split}\begin{aligned}
      \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
      \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
  \end{aligned}\end{split}$
- 이 모델은 리니어 모델에 비해서 더 많은 parameter를 가지게 됩니다
- 하지만, parameter수가 증가했음에도 불구하고 이 모델은 기존 linear 모델 대비 장점이 없습니다.



- $H$ 는 $X$에 대한 affine function이고, 
  $O$도 $H$에 대한 affine function입니다.
- affine function의 affine function은 affine function입니다.

- 수식으로 설명을 하자면:

  - $O$ 계산식을 펼쳐보면 다음과 같습니다
    $\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.$ 

  - 보다 시피 hidden layer가 추가되어도 모델은 
    $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$, $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$를 가지는 linear model과 동일합니다.

  - hidden layer을 추가해도 얻을 수 있는 이점이 없습니다.

    

- 그러므로 MLP 아키텍처가 효과적으로 non-linear 관계를 표현할 수 있도록 하려면 
  nonlinear activation function $\sigma$ 를 각 hidden unit에 적용해줄 필요가 있습니다.

- 자주 사용하는 activation function으로는 ReLU $\sigma(x) = \mathrm{max}(0, x)$가 존재합니다.

  - ReLU는 $x$에 대해 elementwise 연산을 진행합니다.

- activation function $\sigma(\cdot)$의 출력값을 activation이라고 부릅니다.

- 보편적으로 activation function이 사용되면 MLP를 Linear Model이 아니게 됩니다.

  $\begin{split}\begin{aligned}
      \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
      \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
  \end{aligned}\end{split}$



- 더 표현력이 강한 MLP 모델을 얻기 위해서 우리는 activation function이 포함된 hidden layer를 계속 쌓아갈 수 있습니다.
- $\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$, $\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$ , ...



### 5.1.1.4 Universal Approximators

- 딥러닝 모델, MLP 모델의 표현력은 얼마나 큰가?에 대한 궁금증이 자연스럽게 발생합니다.
- 이 질문에 대한 답은 다음 논문들에서 찾아볼 수 있습니다.
  (1989) Approximation by superpositions of a sigmoidal function. *Mathematics of control, signals and systems*
  (1984) Interpolation of scattered data: distance matrices and conditionally positive definite functions
- 이 논문들은 1개의 hidden-layer가 존재하는 MLP 모델이
  모델 node 수가 충분하고, weight가 적절하다면 모든 function을 근사할 수 있다는 것을 보여줍니다.



## 5.1.2 Activation Functions

- Activation function은 한 neuron이 활성화될지 여부를 결정합니다.
- Activation function은 미분가능합니다.
- 대부분 Activation function은 non-linearity를 모델에 도입합니다.

- 자주 쓰이는 activation function 몇가지를 소개해드리겠습니다.



### 5.1.2.1 ReLU Function

- 가장 자주 쓰이는 activation function은 rectified linear unit (ReLU)입니다.

- ReLU가 보편화된 이유는 간단하면서 좋은 성능을 보이기 때문입니다.

- ReLU:
  $\operatorname{ReLU}(x) = \max(x, 0).$

- ReLU는 양수 입력값을 남겨두고, 음수 입력값을 0으로 치환합니다.

- ReLU 함수 그래프를 시각화해보면 다음과 같습니다
  ```python
  x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
  y = torch.relu(x)
  d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
  ```

  

![../_images/output_mlp_76f463_15_0.svg](https://d2l.ai/_images/output_mlp_76f463_15_0.svg)

- ReLU는 piecewise linear합니다.

- 입력값이 양수일 때 ReLU의 derivative는 1입니다

- 입력값이 음수일 때 ReLU의 derivative는 0입니다

- 입력값이 0일 때 ReLU는 미분 불가능합니다.

  - 이런 경우 우리는 예외 처리 형태로 derivative가 0이 되도록 설정해줍니다.
  - 이렇게 해도 문제가 없는 이유는 input이 실제로 0일 경우는 없기 때문입니다. (무슨말인지 모르겠습니다)
    - 원문: We can get away with this because the input may never actually be zero
      mathematicians would say that it’s nondifferentiable on a set of measure zero
  - 또한, 수학이 아닌 공학 영역에서 boundary condition에 대해서 너무 집착할 필요도 없습니다.

- ReLU의 도함수 그래프는 다음과 같습니다
  ```python
  y.backward(torch.ones_like(x), retain_graph=True)
  d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
  ```

  

  ![../_images/output_mlp_76f463_27_0.svg](https://d2l.ai/_images/output_mlp_76f463_27_0.svg)

- ReLU의 도함수 값은 0으로 사라지거나 입력값을 그대로 출력값으로 반환해줍니다.

- 이 특징은 모델 학습 과정에서 vanishing gradient 문제를 완화하는데 도움이 됩니다. (이후에 상세 설명)



- ReLU는 여러 파생 형태를 가지고 있습니다. 
- parameterized ReLU(pReLU)
  $\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$
  음수 입력값도 조금 통과시켜주는 형태의 activation function입니다.



 ### 5.1.2.2 Sigmoid Function

- Sigmoid 함수는  (-inf, inf) 입력을 (0, 1) 범위로 매핑하는 activation 함수입니다.
- 그렇기 때문에 squashing function으로도 불립니다.
- $\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$



- 초창기 neural network는 생물학적 neuron과 유사하게 fire / do not fire 형태로 작동했습니다.
- 이 때는 thresholding activation function이 많이 사용되었습니다.



- Gradient 기반 모델 학습이 각광받기 시작하면서 sigmoid 함수가 smooth, differentiable thresholding unit의 approximation의 개념으로 사용되었습니다.

- Sigmoid 함수는 지금도 output unit의 activation function으로 많이 사용되고 있습니다.

  - Binary classification에서 출력값을 확률로 해석하기 위해서 많이 사용됩니다.

- 하지만 지금은 hidden layer에서 ReLU에 의해 대체되고 있습니다.

  - 그 이유는 sigmoid 함수를 activation function으로 사용한다면 gradient vanishing 문제가 존재하기 때문입니다.
  - 입력값이 큰 양수값, 큰 음수값일 때 sigmoid 함수의 derivative는 0에 가까워집니다.
  - 이는 훈련 과정에서 벗어나기 힘든 plateau가 생길 가능성이 있다는 것을 의미합니다.

  

- 그럼에도 sigmoid 함수는 중요합니다. RNN에서 sigmoid unit이 중요한 역할을 하기도 합니다.



- sigmoid 함수를 시각화하면 다음과 같습니다.

  ```python
  y = torch.sigmoid(x)
  d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
  ```

- Sigmoid 함수의 derivative는 다음과 같습니다:
  $\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$

- Sigmoid 함수의 도함수를 시각화하면 다음과 같습니다:
  ![../_images/output_mlp_76f463_51_0.svg](https://d2l.ai/_images/output_mlp_76f463_51_0.svg)

  - derivative는 입력값이 0일 때 최대 0.25입니다.
  - 입력값이 0과 멀어지며 derivative는 0에 근사해집니다.



### 5.1.2.3 Tanh Function

- Sigmoid와 동일하게 tanh 함수도 입력값을 (-inf, inf) 범위에서 (-1, 1) 범위로 매핑합니다.

- $\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$

- tanh 함수를 시각화하면 다음과 같습니다.

- ```python
  y = torch.tanh(x)
  d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
  ```

- ![../_images/output_mlp_76f463_63_0.svg](https://d2l.ai/_images/output_mlp_76f463_63_0.svg)

- Sigmoid와 다르게 tanh 함수는 origin에 대해서 대칭이라는 특징을 가집니다.



- tanh 함수의 derivative는 다음과 같습니다:
  $\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$

- tanh 함수의 도함수를 시각화 해보면 다음과 같습니다
  ```python
  # Clear out previous gradients.
  x.grad.data.zero_()
  y.backward(torch.ones_like(x),retain_graph=True)
  d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
  ```

  ![../_images/output_mlp_76f463_75_0.svg](https://d2l.ai/_images/output_mlp_76f463_75_0.svg)

- tanh의 derivative 값은 x가 0일 때 최대 1의 값을 가질 수 있습니다.

- sigmoid 함수와 동일하게 x가 0에서 멀어지면 derivative 값은 0에 접근합니다.



## 5.1.3 Summary

- MLP 아키텍처에 nonlinearity를 추가하는 방법을 다루어보았다.
- 지금까지 배운 지식은 1990년대 기준 딥러닝 Practitioner의 지식과 비슷하다.
- 하지만 우리는 당시 Practitioner보다 더 좋은 딥러닝 프레임워크를 보유하고 있고
- ReLU같이 Optimization 과정을 더 간단하게 해주는 activation function을 가지고 있다.
- activation 함수에 대한 연구는 지금도 진행되고 있다.
  - swish activation function: $\sigma(x) = x \operatorname{sigmoid}(\beta x)$ (2017) 