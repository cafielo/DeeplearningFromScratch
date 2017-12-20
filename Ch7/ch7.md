# Chap7 CNN


## 7.1 전체 구조 
- CNN = NN + convolution layer, pooling layer
- 인접한 계층의 모든 뉴런과 결합 > 완전 연결 (이렇게 완전 연결된 계층 > Affine 계층이라고함)
- Affine 계층을 이용하여 네트워크 구성
    - affine 레이어 뒤에 활성화 함수(relu, sigmoid, softmax)를 붙임
    - 구조: x > affine - relu > affine - relu > affine - softmax
￼ 
<img width="463" alt="screen shot 2017-12-17 at 1 57 13 pm" src="https://user-images.githubusercontent.com/5119286/34197347-6ba98d30-e5a9-11e7-8aaa-69879928b413.png">


### CNN의 계층을 이용하여 네트워크 구성 
- Affine 을  이용하여 네트워크 구성할때랑 다른점 
    - affine - relu vs.  conv - relu - (pooling)
    - 구조:  x > conv - relu > conv - relu > affine - relu > affine - softmax
<img width="537" alt="screen shot 2017-12-17 at 2 10 46 pm" src="https://user-images.githubusercontent.com/5119286/34197361-75ff1a16-e5a9-11e7-9574-d83e258fd6f6.png">
￼
<br><br>
## 7.2 합성곱 계층
### 7.2.1 Affine Layer의 문제점
- 데이터 형상이 무시가 된다 
    -  예: MNIST 
        - MNIST 이미지의 차원 (세로, 가로, 색상)으로 구성된 3차원  
        - MNIST 이미지 (28, 28, 1) 이렇게 3차원 데이터 임
        - 근데 28 * 28 * 1 = 784 784의 1차원으로 affine 계층에 만들어 넣어줌
    - 형상이 무시가 되는게 문제인 이유:
        - 공간적 정보의 누락: 인접한 픽셀간의 색이 비슷하다거나 
    - CNN layer 는 형상을 유지함

- CNN의 입출력 데이터를 feature map이라고 부름 

### 7.2.2 합성곱 연산  
- 합성곱 연산  
<img width="526" alt="screen shot 2017-12-17 at 2 20 23 pm" src="https://user-images.githubusercontent.com/5119286/34197570-72952a0e-e5aa-11e7-814b-14aaaab62417.png">


- 합성곱 연산을 이미지 처리에서 말하는 필터(커널) 연산에 해당함 
![convolution](https://user-images.githubusercontent.com/5119286/34197583-7e3ea3b2-e5aa-11e7-96fd-6b863d0a30b2.png)

- 필터에 편향까지 포함하면 아래와 같음 
<img width="548" alt="screen shot 2017-12-17 at 2 26 15 pm" src="https://user-images.githubusercontent.com/5119286/34197630-b5f6c8a2-e5aa-11e7-80b5-e1543efadea9.png">

￼
### 7.2.4 스트라이드 
- 필터를 얼마나 건너 띄게 할것인가?
![stride](https://user-images.githubusercontent.com/5119286/34197685-e6860960-e5aa-11e7-9356-3107434f1ed3.png)



- 필터 아웃풋 갯수 구하기   
￼![filteroutput](https://user-images.githubusercontent.com/5119286/34197805-6b128d5c-e5ab-11e7-9e04-bf59abbf3c44.png)

- 공식  
![formula](https://user-images.githubusercontent.com/5119286/34197815-7218d9bc-e5ab-11e7-9a0d-b3e44a47e42b.png)
￼

- 아웃풋은 정수가 되어야 함 

## 7.2.5 3차원 데이터의 합성곱

￼![screen shot 2017-12-17 at 3 24 48 pm](https://user-images.githubusercontent.com/5119286/34197901-beca4e08-e5ab-11e7-8845-e00f5d39c8bc.png)


![3dconv](https://user-images.githubusercontent.com/5119286/34197908-c5099b84-e5ab-11e7-85be-6c4c8992f4c0.png)


#### 중요한점 
- 채널수와 필터수 같아야함
- 필터간에는 크기가 같아야함


### 7.2.6 블록으로 생각하기 

![screen shot 2017-12-17 at 3 38 58 pm](https://user-images.githubusercontent.com/5119286/34197992-0bfed3c4-e5ac-11e7-9a6e-0ddbec494bb5.png)
- 이그림의 예에서 출력데이터는  한장의 특징 맵(feature map)입니다. (이말인 즉슨, 1채널만 나온다)
- 그럼 합성곱 연산을 다수의 채널을 보내고 싶으면?
![screen shot 2017-12-17 at 3 41 35 pm](https://user-images.githubusercontent.com/5119286/34198023-2548017a-e5ac-11e7-8050-99c65dc7004e.png)
 
- 이렇게 FN개의  필터를 이용해서 FN 채널의 데이터 출력 가능 
- 이게 CNN의 처리 흐름 
- bias 추가한 경우
![screen shot 2017-12-17 at 3 44 02 pm](https://user-images.githubusercontent.com/5119286/34198137-9857fa6c-e5ac-11e7-80b3-be101a8afb25.png)


### 7.2.7 배치 처리 
#### 합성곱을 배치 처리를 지원하고자 할때 
- 각계층에 차원 하나 늘려서 4차원으로 네트웍을 흐르도록 만들어줌 
![screen shot 2017-12-17 at 3 46 42 pm](https://user-images.githubusercontent.com/5119286/34198215-e5f95afe-e5ac-11e7-8de8-8950347651cf.png)


<br><br>
## 7.3 풀링 계층 

### 풀링? == 다운샘플링!
![screen shot 2017-12-19 at 1 27 25 am](https://user-images.githubusercontent.com/5119286/34198369-82a41538-e5ad-11e7-895a-97dcd8a0dc8e.png)


### 7.3.1 풀링 계층의 특징 
- 학습할 매개변수 없음
- 채널수가 변하지 않음 
￼![screen shot 2017-12-19 at 1 29 26 am](https://user-images.githubusercontent.com/5119286/34198406-9e7a5bfa-e5ad-11e7-8b39-c55df2720534.png)


- 입력의 변화에 영향을 적게 받는다(강건하다)
![screen shot 2017-12-19 at 1 29 36 am](https://user-images.githubusercontent.com/5119286/34198410-a2f67998-e5ad-11e7-8530-d28d15ad7518.png)

<br><br>
## 7.4 합성곱/풀링 계층 구현하기 

### 7.4.1 4차원 배열 
- CNN이 주로 쓰이는 이미지 환경으로 생각하면 4차원배열을 생각해야함 
- 차원: 데이터 갯수(1차), 채널개수(2차), 높이(3차), 너비(4차)
![screen shot 2017-12-20 at 5 50 23 pm](https://user-images.githubusercontent.com/5119286/34198583-4ae4b6ec-e5ae-11e7-8915-4775b50e6fe7.png)



### 7.4.2 im2col 로 데이터 전개하기 
- CNN에서 사용되는 이미지를 4차원 배열로 정리한후 for문을 이용해서 합성곱을 해도됨
- 그러나 성능적으로 느림
- 그래서 행렬로 계산하면 성능의 이점이 있음
- 4차원 데이터(데이터 갯수 + 채널수 + 높이 + 너비)를 im2col을 이용해서 행렬로 만듬 
![im2col](https://user-images.githubusercontent.com/5119286/34198616-6b5cff1a-e5ae-11e7-8e77-39c4351f577b.png)
![screen shot 2017-12-19 at 1 50 20 am](https://user-images.githubusercontent.com/5119286/34198733-c05bfc78-e5ae-11e7-9291-666565005a4c.png)

￼
### 7.4.3 CNN Layer  구현
- im2col 사용
![screen shot 2017-12-20 at 11 46 20 am](https://user-images.githubusercontent.com/5119286/34198838-165b0830-e5af-11e7-9b3d-34fbb539baae.png)
![screen shot 2017-12-20 at 11 46 26 am](https://user-images.githubusercontent.com/5119286/34198808-058e2c76-e5af-11e7-8919-18918d3bb09a.png)


- 코드 
```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        
        # 합성곱
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        return out


```
- 위의 코드에서 `reshape(FN, -1)` 은 다차원 배열의 원소수가 변환 후에도 똑같이 유지 되도록 적절히 묶어줌
  - 예 
    - (10, 3, 5, 5) 형상의 데이터를 `reshape(10, -1)`로 해주면 (10, 75) 인 형상으로 만들어줌
-  transpose 는 축 배열을 바꾸어줌
    - 바꾸기전: (데이터수, 높이, 너비, 채널) ===> (0, 1, 2, 3)
    - 바꾼후: (데이터수, 채널, 높이, 너비) ===> (0, 3, 1, 2) 


### 7.4.4 Pooling layer 구현
- 여기도 im2col 사용
![pooling1](https://user-images.githubusercontent.com/5119286/34199494-4d33b8be-e5b1-11e7-90f1-3535348a7b0a.png)
![pooling2](https://user-images.githubusercontent.com/5119286/34199499-51e1c43c-e5b1-11e7-9c3b-1c1a5d9de661.png)

- 코드 
```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 전개 (1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        
        # 최댓값 (2)
        out = np.max(col, axis=1)

        # 성형 (3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

```



## 7.5 CNN 구현하기 
- conv, pooling layer를 합성하여 CNN 조립
￼![screen shot 2017-12-20 at 3 01 59 pm](https://user-images.githubusercontent.com/5119286/34199859-738a284e-e5b2-11e7-8d28-095adaa79dd4.png)




### 위의 구조를 갖는 SimpleConvNet 구성
- 초기 파라미터
    - input_dim: 인풋데이터의 차원
    - conv_param:
        - filter_num
        - filter_size
        - stride
        - pad
        - hidden_size
        - output_size
        - weight_init_std
    
```python


class SimpleConvNet:
    """ConvNet
    conv - relu - pool - affine - relu - affine - softmax
    
    Parameters
    ----------
    input_size 
    hidden_size_list
    output_size 
    activation : 'relu' or 'sigmoid'
    weight_init_std : （e.g. 0.01）
    """
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 가중치, 편향 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # CNN  구성하는 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    
    # 오차역전파법으로 기울기 구하기 
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

  
```



## 7.6 CNN 시각화 하기 

### 7.6.1  1계층 가중치 시각화 
- 필터의 가중치를 학습전과 학습후 이미지로 확인 
<img width="568" alt="screen shot 2017-12-20 at 6 39 41 pm" src="https://user-images.githubusercontent.com/5119286/34200690-5277e7e2-e5b5-11e7-9586-a1fb32a99aa5.png">


- 필터의 특징에 따라 에지나, 블롭등 원시적인 정보들을 추출해 낼수 있음 
<img width="569" alt="screen shot 2017-12-20 at 6 43 36 pm" src="https://user-images.githubusercontent.com/5119286/34200834-e7a9d94c-e5b5-11e7-9815-8224df1dc1ed.png">


### 7.6.2 층 깊에 따른 추출 정보의 변화 
- 1계층 합성곱에서는 에지나, 블롭등의 저수준 정보가 추출
- 계층이 깊어질수록 추출되는 정보는 더 추상화됨 
![deeplearing](https://user-images.githubusercontent.com/5119286/34201763-e57cdfea-e5b8-11e7-98d7-940af2a0e2a4.png)


## 7.7 대표적인 CNN

### 7.7.1 LeNet
- 손글씨 인식 네트워크 (1998)
- Conv + Subsampling(Pooling 아님) 후 FC(Full connection)

### 7.7.2 AlexNet
- 2012년 발표
- 특징
    - Activation 함수로 Relu 사용
    - LRN(구소적 정규화) 사용
    - 드롭 아웃 사용


## 7.8 정리 
- CNN은 Affine Layer에 Conv + Pooling Layer 를 추가함
- im2col을 이용해서 Conv, Pooling Layer에서 연산시 성능 향상시킴
- CNN을 시각화 하면 계층이 깊어질수록 고급정보가 추출됨 
- 대표적 CNN은 LeNet, AlexNet이 있음

