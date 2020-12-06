# ChaNN

> QianChen 2020/11/19
>
> Learning note about building simple convolutional neural network based on numpy 
>
> reference[https://github.com/yizt/numpy_neural_network]



[TOC]



## Full connected layer

参考笔记《全连接层和卷积层的前向和反向传播推导》



## Flatten layer

压平层没有参数需要更新，所以它的反向传播只是误差的重新分配。

```python
### flatten layer
### (batchsize, channels, h, w) -> (batchsize, channels*h*w)
def flatten_forward(x):
    N = x.shape[0]
    return np.reshape(x,(N,-1))

def flatten_backward(dzNext, x):
    return np.reshape(dzNext, np.shape(x))
```

如上代码所示，它的前向传播是保持Batchsize维度不变，其他维度reshape在一起。

它的反向传播是误差的重新分配，即reshape的逆过程。



## Convolutional layer

卷积层的前向和反向传播是很复杂的，尤其是涉及到步长和补零的情况，具体的公式推导在[yizt的github](https://github.com/yizt/numpy_neural_network)上有非常详细的推导。这里为了更直观的了解卷积网络的前向和反向传播，参考[博客](https://www.brilliantcode.net/1748/convolutional-neural-networks-5-backpropagation-in-feature-maps-biases-of-cnns/), 采用图解的方式。

参考笔记《全连接层和卷积层的前向和反向传播推导》



## Pooling layer

和压平层一样，pooling层没有参数需要更新，所以它的反向传播只是误差的重新分配。

前向和反向算法参考代码即可。



## Activation function

#### ReLU


$$
ReLU(z)=\begin{cases}
z &  z>0 \\
0 & z<=0    \tag 1
\end{cases}
$$

a) 我们将激活函数也看做一层, 设第$l$层输出为$z^l$, 经过激活函数后的输出为$z^{l+1}$

b) 记损失函数L关于第$l$ 层的输出$z^l$ 的偏导为$\delta^l = \frac {\partial L} {\partial z^l}  $ 

​        则损失函数L关于关于第l层的偏导如下：
$$
\begin{align}
&\delta^l = \frac {\partial L} {\partial z^{l+1}}   \frac {\partial z^{l+1}} {\partial z^{l}}  \\
&=\delta^{l+1} \frac {\partial ReLU(z^l)} {\partial z^{l}} \\
&=\delta^{l+1} \begin{cases}
1    & z^l>0 \\
0    & z^l<=0   
\end{cases} \\
&= \begin{cases}
\delta^{l+1}    & z^l>0 \\
0    & z^l<=0    \tag 2
\end{cases}
\end{align}
$$



```python
### ReLU
def ReLU_forward(z):
    return np.maximum(0, z)

def ReLU_backward(dzNext, z):
    '''
    dzNext: gradient after activate function
    z: input of activation function
    '''
    dz = np.where(np.greater(z, 0), dzNext, 0)
    return dz
```



#### Leaky ReLU

 ReLU在取值小于零部分没有梯度，LeakyReLU在取值小于0部分给一个很小的梯度,一般$\alpha = 0.1$。

$$
LeakyReLU(z)=\begin{cases}
z &  z>0 \\
\alpha z & z<=0, \alpha=0.1    \tag 3
\end{cases}
$$

同Relu可知损失函数L关于关于第l层的偏导为:
$$
\begin{align}&\delta^l = \begin{cases}
\delta^{l+1}    & z^l>0 \\
\alpha\delta^{l+1}    & z^l<=0, \alpha=0.1    \tag 4
\end{cases}
\end{align}
$$



```python
### LeakyReLU
def LeakyReLU_forward(z, alpha=0.1):
    ## if z>0, return z. if z<0, return alpha*z.
    return np.where(np.greater(z, 0), z, alpha*z)

def LeakyReLU_backward(dzNext, z, alpha=0.1):
    return np.where(np.greater(z, 0), dzNext, 0.1*dzNext)

```



####PReLU

  参数化ReLU，形式同LeakyRelu,不过$\alpha$ 不是固定的常量而是根据数据学习到的。[论文地址](https://arxiv.org/pdf/1502.01852.pdf)

$$
PReLU(z)=\begin{cases}
z &  z>0 \\
\alpha z & z<=0, \alpha是与z相同维度的变量    \tag 5
\end{cases}
$$

a) 同LeakyRelu可知损失函数L关于关于第l层的偏导为:
$$
\begin{align}&\delta^l = \begin{cases}
\delta^{l+1}    & z^l>0 \\
\alpha\delta^{l+1}    & z^l<=0,\alpha是需要学习的参数    \tag 6
\end{cases}
\end{align}
$$


b) 损失函数L关于关于参数$\alpha$的偏导为:
$$
\begin{align}
&\frac {\partial L} {\partial \alpha} = \frac {\partial L} {\partial z^{l+1}}   \frac {\partial z^{l+1}} {\partial \alpha} \\
&=\delta^{l+1} \frac {\partial PReLU(z^l)} {\partial \alpha} \\
&=\delta^{l+1} \begin{cases}
0  & z^l >0 \\
z^l & z^l<=0
\end{cases} \\
&= \begin{cases}
0  & z^l >0 \\
\delta^{l+1}z^l & z^l<=0  \tag 7
\end{cases} 
\end{align}
$$

```python
### PReLU
def PReLU_forward(z, alpha):
    """
    z: input params
    alpha: learnable params
    """
    return np.where(np.greater(z, 0), z, alpha * z)

def PReLU_backwark(dzNext, z, alpha):
    dz = np.where(np.greater(z, 0), dzNext, alpha * dzNext)
    d_alpha = np.where(np.greater(z, 0), 0, z * dzNext)
    return d_alpha, dz
```

每次计算完后，alpha是需要更新的。



#### ELU

​           指数化ReLU，在取值小于0的部分使用指数，[论文地址](https://arxiv.org/pdf/1511.07289.pdf)


$$
ELU(z)=\begin{cases}
z &  z>0 \\
\alpha(\exp(z)-1) & z<=0, \alpha=0.1    \tag 8
\end{cases}
$$

同LeakyRelu可知损失函数L关于关于第l层的偏导为:
$$
\begin{align}&\delta^l = \begin{cases}
\delta^{l+1}    & z^l>0 \\
\alpha \delta^{l+1} \exp(z^l)    & z^l<=0    \tag 9
\end{cases}
\end{align}
$$

```python
### ELU
def ELU_forward(z, alpha=0.1):
    return np.where(np.greater(z,0), z, alpha*(np.exp(z)-1))

def ELU_backward(dzNext, z, alpha=0.1):
    return np.where(np.greater(z, 0), dzNext, alpha*dzNext*np.exp(z))
```



#### SELU

​           缩放指数型线性单元, 就是对ELU加上一个缩放因子$\lambda$。[论文地址](https://arxiv.org/pdf/1706.02515.pdf)


$$
SELU(z)=\lambda\begin{cases}
z &  z>0 \\
\alpha(\exp(z)-1) & z<=0    \tag {10}
\end{cases}
$$

​             其中$\lambda \approx 1.0507 , \alpha \approx  1.673$ (论文中有大段证明)



同ELU可知损失函数L关于关于第l层的偏导为:
$$
\begin{align}&\delta^l = \lambda \begin{cases}
\delta^{l+1}    & z^l>0 \\
\alpha \delta^{l+1} \exp(z^l)    & z^l<=0    \tag {11}
\end{cases}
\end{align}
$$

```python
### SELU
## in the paper, it proven that lambda = 1.0507, alpha = 1.673
def SELU_forward(z, lamda = 1.0507, alpha=1.673):
    return np.where(np.greater(z, 0), lamda*z, (lamda*alpha * np.exp(z)-1))

def SELU_backward(dzNext, z, lamda = 1.0507, alpha=1.673):
    return np.where(np.greater(z, 0), dzNext*lamda, lamda*alpha*dzNext*np.exp(z))
```



#### Softmax

$$
Sigmoid(z)=\frac 1 {1+\exp(-z)}    \tag {12}
$$

损失函数L关于关于第l层的偏导为:

$$
\begin{align} \delta^l &= \frac {\partial L} {\partial z^l} \\
&=\frac {\partial L} {\partial z^{l+1}} \frac {\partial z^{l+1}} {\partial z^l} \\
&=\delta^{l+1} (-1) (1+exp(-z))^{-2}exp(-z)(-1) \\
&=\delta^{l+1} \frac {exp(-z)} {(1+exp(-z))^2} \\
&=\delta^{l+1} Sigmoid(z) (1-Sigmoid(z))      \tag {13}
\end{align}
$$

```python
### sigmoid
def sigmoid_forward(z):
    return 1 / (1+np.exp(-z))

def sigmoid_backward(dzNext, z):
    return dzNext*sigmoid_forward(z)*(1-sigmoid_forward(z))
```



#### Tanh

$$
Tanh(z)=\frac{e^x-e^{-x}}{e^x+e^{-x}} \tag{14}
$$

损失函数L关于关于第l层的偏导为:
$$
\begin{align} \delta^l &= \frac {\partial L} {\partial z^l} \\
&=\frac {\partial L} {\partial z^{l+1}} \frac {\partial z^{l+1}} {\partial z^l} \\
&=\delta^{l+1} (1-(Tanh^2(z^l)) \\   \tag {15}
\end{align}
$$

```python
### Tanh
def Tanh_forward(z):
    return np.tanh(z)

def Tanh_backward(dzNext, z):
    return dzNext*(1-np.square(np.tanh(z)))
```



## Loss function

#### MSE

- 定义和求导

平方损失函数是一种L2范数损失函数。对于单个样本$(x,y*)$ ，定义如下：
$$
L(y,y*) = \frac 1 2(y-y^*)^2 \tag {1}
$$
​          其中$y$是神经网络最后一层的输出$y=z^n$ ,就是预测值
$$
\begin{align}
&\frac {\partial L} {\partial y_i} = \frac {\partial (\frac 1 2(y_i - y^*_i)^2)} {\partial y_i} \\ 
&=(y_i - y^*_i) * \frac {\partial (y_i - y^*_i)} {\partial y_i} \\
&=(y_i - y^*_i)    \ \ \ \ \ \ \ (2)
\end{align}
$$
​         更一般的表示为 $\frac {\partial L} {\partial y} = y - y^*$ ; 也就是$\delta^n=\frac {\partial L} {\partial y}=y-y^* = z^n-y^*      \tag {3}$

​         即使用均方误差情况下，损失函数L关于网络最后一层的导数就是预测值减实际值

```python
### MSE
def loss_MSE(z_predict ,z_true):
    loss = np.mean(np.sum(np.square(y_predict-y_true),axis=-1))
    dzLast = z_predict - z_true
    return loss, dzLast
```



#### Cross Entropy

- 定义和求导

​         交叉熵用于度量两个概率分布的差异;一般使用交叉熵损失前，会对网络输出做softmax变换进行概率归一化；所以我们这里介绍的交叉熵损失是带softmax变换的交叉熵。

​          softmax变换定义如下：
$$
a_i=e^{y_i}/\sum_k e^{y_k}  \tag {4}
$$
​          交叉熵损失定义如下：
$$
L(y,y^*) = - \frac{1}{n} \sum_i y^*_i \log a_i \tag {5}
$$


a) 我们先来求$a_i$ 关于$y_j$ 的偏导
$$
\begin{align*}
&\frac {\partial a_i} {\partial y_j} = \frac {\partial(e^{y_i}/\sum_k e^{y_k})} {\partial y_j} \\
&= \frac {\partial e^{y_i}} {\partial y_j} * \frac {1} {\sum_k e^{y_k}} +  e^{y_i} * \frac {-1} {(\sum_k e^{y_k})^2} * \frac {\partial (\sum_k e^{y_k})} {\partial y_j} \\
&= \frac {\partial e^{y_i}} {\partial y_j} * \frac {1} {\sum_k e^{y_k}} -   \frac {e^{y_i}} {(\sum_k e^{y_k})^2} * e^{y_j} \\
&=\begin{cases}
\frac {e^{y_j}} {\sum_k e^{y_k}} - \frac {(e^{y_j})^2} {(\sum_k e^{y_k})^2}  &  i=j \\
-\frac {e^{y_i}e^{y_j}} {(\sum_k e^{y_k})^2}  & i\neq\ j
\end{cases} \\
&=\begin{cases}
a_i(1-a_i)  &  i=j  \\
-a_ia_j  & i\neq\ j  \tag {6}
\end{cases} 
\end{align*}
$$


b) 然后我们来求L关于$y_j$ 的偏导
$$
\begin{align}
&\frac {\partial L} {\partial y_j} = - \sum_i\frac {\partial( y^*_i \log a_i )} {\partial a_i} * \frac {\partial a_i} {\partial y_j} \\
&=- \sum_i \frac {y^*_i} {a_i} * \frac {\partial a_i} {\partial y_j} \\
&= - \frac {y^*_j} {a_j} * a_j(1-a_j) + \sum_{i \neq\ j}  \frac {y^*_i} {a_i} * a_ia_j & //注意这里i是变量,j是固定的 \\
&=-y^*_j(1-a_j) + \sum_{i \neq\ j} y^*_ia_j \\
&= - y^*_j + \sum_iy^*_i a_j  & //所有真实标签的概率之和为1\\
&=a_j - y^*_j
\end{align}
$$
​     更一般的表示为 :

$\frac {\partial L} {\partial y} = a - y^*  \tag {7}$

​        所以使用带softmax变换的交叉熵损失函数，损失函数L关于网络最后一层的导数就是预测值经softmax变换后的值减去真实值。

- 最小化交叉熵损失函数相当于最大化似然概率。
- 如果网络中的激活函数是Sigmoid或者是softmax，那么适合用交叉熵函数来作为损失函数。
- 在变成中，计算softmax前需要先将输入值做一个整体的shift，这种shift不会影响输出，并且可以避免溢出错误。

```python
def loss_CrossEntropy(z_predict, z_true):
    ### in order to avoid overflow, y_predict should shift to a relatively small value and this thift will not change result of softmax 
    z_exp = np.exp(z_predict - np.max(z_predict, axis=-1, keepdims=True))
    
    ### calculate softmax before using cross entropy.
    z_probability = z_exp / np.sum(z_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(z_exp, axis=-1, keepdims=True))
    dzLast = z_probability - z_true
    return loss, dzLast
```



## Optimizers

无论是什么优化器，都可以用下面的公式来描述：
$$
w=w+\Delta w \tag1
$$
其中$w$是待更新的参数，$\Delta w$是参数的增量。

#### SGD

这里所说的SGD其实是mBGD。区别是，BGD是计算所有样本的梯度和，然后更新参数，每次下降都是梯度最小的方向下降，但是速度很慢。SGD是得到每个样本的梯度后都更新一次参数，所以每次更新不一定是梯度最小的方向，是震荡着向局部最优解的方向下降，速度很快。mBGD是两者的一个折中，每计算得到一个batch数据的梯度后，更新一次参数。

下面的SGD算法其实是加了动量的。

a)  权重参数$w$ , 权重梯度$\nabla_w$ 

b)  学习率 $\eta$ , 学习率衰减$decay$ (一般设置很小)

c)  动量大小$\gamma$ (一般设置为0.9) , t次迭代时累积的动量为$v_t$

​           则学习率的更新公式为:
$$
\eta_t = \eta /(1+t \cdot decay)  \tag 1
$$
​           累积动量和权重的更新公式如下：
$$
\begin{align}
&v_t=\gamma \cdot v_{t-1} + \eta_t \cdot \nabla_w   & 其中v_0=0    \tag 2 \\
&w = w - v_t  \tag 3
\end{align}
$$


这里加入动量的算法是有其物理背景的，借鉴了有摩擦的小球在曲面下降情况，具体参考[博客](https://zhuanlan.zhihu.com/p/81020717)



#### AdaGrad

​		AdaGrad根据自变量在每个维度的梯度值的大小来调整各个维度上的学习率，从而避免统一的学习率难以适应所有维度的问题。

​        Adagrad 的算法会使用一个小批量随机梯度按元素平方的累加变量每次迭代中，首先将梯度$\nabla_w$  按元素平方后(哈达码积)累加到变量 $s_t$
$$
s_t = s_{t-1} + \nabla_w^2  \tag4
$$
​        梯度的更新公式为:
$$
w = w - \frac {\eta_t} {\sqrt{s_t + \epsilon }} \cdot \nabla_w  \tag 5
$$
​        $\epsilon$ 是为了维持数值稳定性(避免除零)而添加的常数，例如 $10^{-6}$ ; $\eta_t$ 可以是常数，也可以像公式(1)一样为衰减学习率。

从公式中可以看出，随着算法不断迭代， ![[公式]](https://www.zhihu.com/equation?tex=r_i) 会越来越大，整体的学习率会越来越小。所以，一般来说AdaGrad算法一开始是激励收敛，到了后面就慢慢变成惩罚收敛，速度越来越慢。如果前期找不到一个合适的解，那么后期优化就比较困难。



#### RMSProp

​        当学习率在迭代早期降得较快且当前解依然不佳时，Adagrad 在迭代后期由于学习率过小，可能较难找到一个有用的解。为了应对这一问题，RMSProp 算法对 Adagrad 做了一点小小的修改。

​        不同于 Adagrad 里状态变量$s$是到目前时间步里所有梯度按元素平方和，RMSProp 将过去时间步里梯度按元素平方做指数加权移动平均。公式如下：
$$
s_t =\gamma \cdot s_{t-1} + (1-\gamma) \cdot \nabla_w^2  \ ;  \ \ \ \ 其中   0<\gamma <1\  \tag6
$$
​        权重更新公式仍然如AdaGrad
$$
w = w - \frac {\eta_t} {\sqrt{s_t + \epsilon }} \cdot \nabla_w  \tag 5
$$



#### Adam

Adam的名称来自Adaptive Momentum，可以看作是Momentum与RMSProp的一个结合体，该算法通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率，公式如下：
$$
s_t =\gamma_i \cdot s_{t-1} + (1-\gamma_i) \cdot \nabla_w^2  \   \ \ \ \ 其中   0<\gamma <1\  \tag7 \\
$$

$$
r_t = \beta_i r_{t-1}+(1-\beta_i)\nabla_w; \ \ \ \ 其中  0<\beta<1  \tag{8}\\
$$

$$
\gamma_{i+1} = \gamma_i * decay,\ \ (decay=0.9) \tag{9}\\
$$

$$
\beta_{i+1} = \beta_i * decay,\ \ (decay=0.99) \tag{10}\\
$$

$$
\Delta w = - \eta_t \frac{r_t}{\sqrt{\epsilon+s_t}} \tag{11}\\
$$

$$
w = w+ \Delta w \tag{12}
$$



![image-20201205215509107](/Users/apple/Library/Application Support/typora-user-images/image-20201205215509107.png)

