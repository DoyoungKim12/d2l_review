# 5.2. Implementation of Multilayer Perceptrons

- MLP 구현은 simple linear model 구현과 큰 차이가 없다
- 차이점은 우리는 이제 여러 layer를 쌓는다는 것이다

```python
import torch
from torch import nn
from d2l import torch as d2l
```



## 5.2.1 Implementation from Scratch

### 5.2.1.1 Initializing Model Parameters

- 사용 데이터: Fashion-MNIST
  - 10개 class의 $28 \times 28 = 784$ grayscale 이미지로 구성된 데이터셋.
- 이전과 동일하게 지금은 픽셀간의 공간적 연관성을 무시하겠다.
  - 즉, 단순히 784 input feature와 10개 ouput값을 가지는 모델을 훈련하는 문제로 정의합니다.
- 우리는  1개의 256개의 hidden unit을 가지는 hidden layer를 사용할 것이다.
  - layer 수와 layer의 hidden unit (width)수는 설정가능하다. (hyperparameter)
- 보편적으로 우리는 layer width를 $2^n$인 값으로 설정한다. (메모리 효율을 위함)



- 이번에도 우리는 우리의 parameter를 tensor 형태로 선언할 것이다.
- 각 layer마다 우리는 weight matrix와 bias vector를 선언할 필요가 있다.



```python
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```



### 5.2.1.2 Model

- ReLU activation 함수를 직접 선언해보겠다. 

```python
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```



- 다음으로는 모델 선언을 진행해보겠다.
  - 우리는 2차원 이미지를 1차원 벡터로 변경하에 MLP에 입력한다.

```python
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = X.reshape((-1, self.num_inputs))
    H = relu(torch.matmul(X, self.W1) + self.b1)
    return torch.matmul(H, self.W2) + self.b2
```



### 5.2.1.3 Training

- 모델 훈련은 softmax regression과 완전 동일하다.
- 모델, 데이터, trainer를 정의하고 fit 메서드를 이용해서 훈련을 진행한다

```python
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

![../_images/output_mlp-implementation_d1b2f2_51_0.svg](https://d2l.ai/_images/output_mlp-implementation_d1b2f2_51_0.svg)

## 5.2.2 Concise Implementation

- high-level API를 사용하면 MLP를 더 간결하게 구현할 수 있다.

### 5.2.2.1 Model

```python
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```

- softmax regression과 차이는 
  - 2개의 fully connected layer를 사용한다는 점
  - ReLU를 사용한다는 점입니다.



### 5.2.2.2 Training

- 훈련 코드는 softmax regression과 완전히 동일합니다.
- 이렇게 모듈화된 코드는 모델 아키텍처와 훈련 설정을 분리해서 고민할 수 있도록 합니다.

```python
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

![../_images/output_mlp-implementation_d1b2f2_75_0.svg](https://d2l.ai/_images/output_mlp-implementation_d1b2f2_75_0.svg)



### 5.2.3 Summary

- single에서 multiple layer를 사용하도록 전환하는 것은 실무에서 큰 어려움이 없다는 것을 확인했다
- 그럼에도 MLP를 직접 구현하는 것은 지저분하다.
  - 모델 파라미터를 직접 선언하고 관리해야하기 때문에 layer가 많아지면 코드가 길어질 것이다.
- 우리는 이제 1980년대 기준 SOTA 모델을 완전히 구현할 능력을 가졌다.