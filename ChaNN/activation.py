### chan 2020/11/24
### common activation function for build CNN based on numpy
### including: ReLU, LeakyReLU, PReLU, ELU, SELU, sigmoid, Tanh.

import numpy as np
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

### LeakyReLU
def LeakyReLU_forward(z, alpha=0.1):
    ## if z>0, return z. if z<0, return alpha*z.
    return np.where(np.greater(z, 0), z, alpha*z)

def LeakyReLU_backward(dzNext, z, alpha=0.1):
    return np.where(np.greater(z, 0), dzNext, 0.1*dzNext)

### PReLU [论文地址](https://arxiv.org/pdf/1502.01852.pdf)
def PReLU_forward(z, alpha=0.1):
    """
    z: input params
    alpha: learnable params
    """
    return np.where(np.greater(z, 0), z, alpha * z)

def PReLU_backward(dzNext, z, alpha=0.1):
    dz = np.where(np.greater(z, 0), dzNext, alpha * dzNext)
    d_alpha = np.where(np.greater(z, 0), 0, z * dzNext)
    return d_alpha, dz

### ELU [论文地址](https://arxiv.org/pdf/1511.07289.pdf)
def ELU_forward(z, alpha=0.1):
    return np.where(np.greater(z,0), z, alpha*(np.exp(z)-1))

def ELU_backward(dzNext, z, alpha=0.1):
    return np.where(np.greater(z, 0), dzNext, alpha*dzNext*np.exp(z))

### SELU [论文地址](https://arxiv.org/pdf/1706.02515.pdf)
## in the paper, it proven that lambda = 1.0507, alpha = 1.673
def SELU_forward(z, lamda = 1.0507, alpha=1.673):
    return np.where(np.greater(z, 0), lamda*z, (lamda*alpha * np.exp(z)-1))

def SELU_backward(dzNext, z, lamda = 1.0507, alpha=1.673):
    return np.where(np.greater(z, 0), dzNext*lamda, lamda*alpha*dzNext*np.exp(z))

### sigmoid
def sigmoid_forward(z):
    return 1 / (1+np.exp(-z))

def sigmoid_backward(dzNext, z):
    return dzNext*sigmoid_forward(z)*(1-sigmoid_forward(z))

### Tanh
def Tanh_forward(z):
    return np.tanh(z)

def Tanh_backward(dzNext, z):
    return dzNext*(1-np.square(np.tanh(z)))