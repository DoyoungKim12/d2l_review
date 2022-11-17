# 5.3. Forward Propagation, Backward Propagation, and Computational Graphs

- 지금까지 d2l에서 gradient descent을 이용한 모델 훈련을 진행했습니다. 

- 모델 훈련 진행 시 우리는 딥러닝 프레임워크에서 제공하는 backpropagation 기능을 이용했기 때문에

  forward pass 구현에만 집중 했었습니다. 

- 현재 사용되는 딥러닝 프레임 워크는 gradient를 자동으로 계산해주는 자동 미분 기능이 있습니다

- 이런 기능은 딥러닝 알고리즘 구현을 크게 간략화합니다. 

  - 자동 미분 기능이 등장하기 전 
    큰 모델에 작은 변화가 발생하면
    gradient 계산 방식 및 코드를 수동으로 업데이트 했습니다.
  - 그래서 당시 논문에는  weight 업데이트 룰 관련 내용이 수 페이지를 차지했습니다.



- 우리는 자동 미분 기능을 사용하지 않은 필요는 없지만 
  backpropagation에 대한 이해가 필요합니다.

- 이번 챕터에서 우리는 MLP 모델에서의 backpropagation에 대해 알아보겠습니다.



## 5.3.1 Forward Propagation

- Forward propagation (forward pass):
  - 모델의 intermediate variable (output 포함)의 연산 및 저장



- 다음과 같은 MLP 모델을 사용한다고 가정해보겠습니다.
  - hidden layer 1개
  - hidden layer bias term 없음
  - weight decay(l2 regularization)이 적용되었습니다.
- MLP 모델의 입력 vector 싸이즈 d  $\mathbf{x}\in \mathbb{R}^d$



- 이 모델의 forward pass를 수행하면서 얻게되는 intermediate variable은 3개가 존재합니다.

  - $\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$
    $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$: hidden layer weight

  - $\mathbf{h}= \phi (\mathbf{z}).$

    $\phi$: activation 함수

  - $\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$
    

- Loss 계산은 다음과 같습니다

  - $L = l(\mathbf{o}, y).$

- l2 regularization term은 다음과 같습니다.

  - $s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$
  - 여기서 사용하는 norm은 Frobenius norm으로 matrix를 vector로 펼친 후 l2 norm을 구한 것과 동일합니다.

- 모델의 최종 Loss는 다음과 같습니다.
  - $J = L + s.$
  - 이하 본문에서 $J$를 objective function이라고 지칭합니다



## 5.3.2 Computational Graph of Forward Propagation

- forward propagation의 computation graph를 그려보면 연산 operator와 variable 사이의 dependency를 파악하기 편리합니다.
- 아래 그림은 위에서 정의한 MLP 모델의 forward pass의 computation graph입니다.

![../_images/forward.svg](https://d2l.ai/_images/forward.svg)



- 네모 노드는 variable을 의미합니다
- 동그라미 노드는 operator를 의미합니다.





## 5.3.3 Backpropagation

- Backpropagation:
  - neural network 파라미터의 gradient를 계산하는 알고리즘입니다.
  - 이 알고리즘은 chain rule을 이용해서 network를 역순으로 횡단합니다. 
    output layer $\rarr$ input layer
  - 이 알고리즘은 모든 중간 결과값 (편미분 값)을 저장합니다.



-  $\mathsf{Y}=f(\mathsf{X})$, $ \mathsf{Z}=g(\mathsf{Y})$ 함수가 존재하고
  $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$가 임의 크기, 차원의 tensor라고 가정하다면

- 우리는 chain rule을 이용해서 $\frac{\partial \mathsf{Z}}{\partial \mathsf{X}}$를 구할 수 있습니다.
  $\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$

  prod 함수는 두 편미분 값을 "곱해주는" 연산을 의미합니다.

  - 편미분값이 vector일 경우 prod 함수는 단순한 matrix-matrix multiplication이지만
  - 편미분값이 더 높은 차원의 tensor의 경우 다른 종류의 "곱" 연산이 필요합니다.



- 우리가 앞서 사용한 MLP 모델의 parameter는 $\mathbf{W}^{(1)}$와 $\mathbf{W}^{(2)}$입니다
- 그러므로, backpropagation이 구해야할 값은 $\partial J/\partial \mathbf{W}^{(1)}$와 $\partial J/\partial \mathbf{W}^{(2)}$입니다



- backpropation 과정:
  ![../_images/forward.svg](https://d2l.ai/_images/forward.svg)
  
  - 역순으로 computation graph 종점에서 objective function $J$($J = L + s.$)값을 결정하는 $L, s$의 gradient 계산
  
    $\frac{\partial J}{\partial L} = 1$
    $\frac{\partial J}{\partial s} = 1.$
    $J$: objective function $L+s$
    $L$:  loss function
    $s$: regularization term
  
  
  
  - 모델 출력값 $o$의 gradient 계산  ($L = l(\mathbf{o}, y).$)
    $\frac{\partial J}{\partial \mathbf{o}}
    = \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
    = \frac{\partial L}{\partial \mathbf{o}}
    \in \mathbb{R}^q.$
  
  
  
  - $\mathbf{W}^{(1)}, \mathbf{W}^{(2)}$의 $s$에 대한 gradient 계산 ($s$($s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right)$))
    $\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
    \; \text{and} \;
    \frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$
  
    ![image](https://user-images.githubusercontent.com/46898478/202336767-c1eea42a-5988-430e-94f8-46f19eeb954c.png)
  
    
  
  - 이제 $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ ($\mathbf{W}^{(2)}$gradient)를 구할 수 있습니다.
    $\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$
  
    
  
  -  $\mathbf{W}^{(1)}$의 objective function $J$에 대한 gradient를 구하기 위해서는 backpropagation을 더 진행해야합니다.
  
  - hidden layer output $h$의 ($\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}$) gradient 계산
    $\frac{\partial J}{\partial \mathbf{h}}
    = \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
    = {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.$
  
    
  
  - $z$ (activation function 통과전 hidden layer 출력값)의 gradient 계산 ($\mathbf{h}= \phi (\mathbf{z}).$)
  
    activation function은 elementwise 연산이므로, $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$ 계산에서 $\text{prod}$ 함수는 elementwise multiplication을 수행해야합니다.
    $\frac{\partial J}{\partial \mathbf{z}}
    = \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
    = \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).$
  
  - 이제 $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ ($\mathbf{W}^{(2)}$의 gradient)를 구할 수 있습니다
    $\frac{\partial J}{\partial \mathbf{W}^{(1)}}
    = \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
    = \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.$



- 이로써 우리는 업데이트 해야할 parameter $\mathbf{W}^{(1)}, \mathbf{W}^{(2)}$의 gradient를 구했습니다.



## 5.3.4 Training Neural Networks

- Neural Network를 훈련 시킬 때, forward와 backward propagation은 상호 dependent합니다.
- 예:
  - Forward pass를 수행할 때
    - 딥러닝 프레임워크는 Computation Graph를 순방향으로 실행하고 중간 결과값을 모두 저장합니다
    - Backpropagation 단계에서 이 중간 결과값들이 gradient 계산에 사용됩니다.
  - 위 MLP 예시 예:
    - forward pass regularization term $s$ 연산에는$\mathbf{W}^{(1)}, \mathbf{W}^{(2)}$값이 필요합니다.
    - backward propagation에서 $\mathbf{W}^{(1)}, \mathbf{W}^{(2)}$의 gradient를 구할 때  forward pass의 중간 결과값 $h$가 필요합니다



- 그러므로 neural network를 훈련할 때
  - weight를 초기화하고
  - forward propagation과 backward propagation을 번갈아가면서 진행합니다.
  - backward propagation에서 중복 계산을 피하기 위해 forward propagation 중간 결과를 활용합니다
    - 중복 계산을 피하는 대가로 메모리를 training 단계에서 많이 사용하게됩니다.
    - 사용 메모리를 대략적으로 network layer 수와 batch size와 비례합니다.



## 5.3.5 Summary

- forward pass는 computation graph를 input $\rarr$  output layer 방향으로 순차적으로 실행하고 중간 결과값을 모두 저장합니다
- backward pass는 역순( output $\rarr$  input layer)으로 중간 결과값과 parameter의 gradient를 순차적으로 계산합니다.
- 훈련 시 forward pass backward pass는 상호 dependent합니다. 
- 훈련에 필요한 메모리는 test time보다 큽니다.