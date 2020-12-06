### chan 2020/11/24
### common loss function for build CNN based on numpy
### including: MSE, Cross Entropy
import numpy as np

### MSE
def loss_MSE(z_predict ,z_true):
    loss = np.mean(np.sum(np.square(y_predict-y_true),axis=-1))
    dzLast = z_predict - z_true
    return loss, dzLast

### Cross Entropy
def loss_CrossEntropy(z_predict, z_true):
    ### in order to avoid overflow, y_predict should shift to a relatively small value and this thift will not change result of softmax 
    z_exp = np.exp(z_predict - np.max(z_predict, axis=-1, keepdims=True))
    
    ### calculate softmax before using cross entropy.
    z_probability = z_exp / np.sum(z_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(z_exp, axis=-1, keepdims=True))
    dzLast = z_probability - z_true
    return loss, dzLast
    