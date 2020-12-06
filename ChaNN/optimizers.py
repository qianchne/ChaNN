### chan 2020/11/25
### optimizers for building CNN based on numpy.
### including: 
### SGD, ...

import numpy as np

def _copy_weights_to_zeros(weights):
    result = {}
    result.keys()
    for key in weights.keys():
        result[key] = np.zeros_like(weights[key])
    return result

### SGD(without momentum)
class SGD(object):
    def __init__(self, lr=0.01, decay=1e-5):

        self.iterations = 0
        self.lr = self.init_lr = lr
        self.decay = decay

    def iterate(self, weights, gradients):

        ## update learning rate
        self.lr = self.init_lr / (1 + self.iterations * self.decay)
        for key in weights.keys():
            weights[key] -= self.lr * gradients[key]

        ## update iteration
        self.iterations += 1

### SGD(with momentum)
class SGD_momentum(object):
    def __init__(self, lr=0.01, momentum=0.9, decay=1e-5):
        '''
        lr: learning rate
        momentum: momentum factor
        decay: lr decay
        '''
        self.v = {}  # accumulation of momentum
        self.iterations = 0
        self.lr = self.init_lr = lr
        self.momentum = momentum
        self.decay = decay

    def iterate(self, weights, gradients):

        ## initial v by zeros
        if self.iterations == 0:
            self.v = _copy_weights_to_zeros(weights)
        ## update learning rate
        self.lr = self.init_lr / (1 + self.iterations * self.decay)
        for key in weights.keys():
            ## update momentum
            self.v[key] = self.momentum * self.v[key] + self.lr * gradients[key]
            weights[key] -= self.v[key]

        ## update iteration
        self.iterations += 1

### AdaGrad
class AdaGrad(object):

    def __init__(self, lr=0.01, epsilon=1e-6, decay=0):
        self.epsilon = epsilon
        self.iterations = 0
        self.s = {}
        self.lr = self.init_lr = lr
        self.decay = decay
    
    def iterate(self, weights, gradients):
        
        ## self.s saving accumulation of square of gradients
        if self.iterations == 0:
            self.s = _copy_weights_to_zeros(weights)
        
        self.lr = self.init_lr / (1 + self.iterations * self.decay)

        for key in weights.keys():
            ## update s
            #self.s[key] += np.square(weights[key]) ?????
            self.s[key] += np.square(gradients[key])
            ## update weights
            weights[key] -= (self.lr * gradients[key]) / np.sqrt(self.epsilon + self.s[key])


### RMSProp
class RMSProp(object):

    def __init__(self, lr=0.01, beta=0.9, epsilon=1e-6, decay=0):
        self.epsilon = epsilon
        self.iterations = 0
        self.s = {}
        self.lr = self.init_lr = lr
        self.decay = decay
        self.beta = beta
    
    def iterate(self, weights, gradients):
        
        ## self.s saving accumulation of square of gradients
        if self.iterations == 0:
            self.s = _copy_weights_to_zeros(weights)
        
        self.lr = self.init_lr / (1 + self.iterations * self.decay)

        for key in weights.keys():
            ## update s, give recent s larger weights
            #self.s[key] = self.s[key]*self.beta + (1-self.beta)*np.square(weights[key])
            self.s[key] = self.s[key]*self.beta + (1-self.beta)*np.square(gradients[key])
            ## update weights
            weights[key] -= (self.lr * gradients[key]) / np.sqrt(self.epsilon + self.s[key])

### Adam
class Adam(object):

    def __init__(self, lr=0.01, alpha=0.9, beta=0.99, epsilon=1e-6, decay=1e-6):
        self.epsilon = epsilon
        self.iterations = 0
        self.s = {}
        self.r = {}
        self.lr = self.init_lr = lr
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.alpha_init = 1
        self.beta_init = 1
    
    def iterate(self, weights, gradients):
        
        ## self.s saving accumulation of square of gradients
        if self.iterations == 0:
            self.s = _copy_weights_to_zeros(weights)
            self.r = _copy_weights_to_zeros(weights)
        
        self.lr = self.init_lr / (1 + self.iterations * self.decay)

        for key in weights.keys():
            ## update s, give recent s larger weights, like RMSProp
            self.s[key] = self.s[key]*self.beta_init + (1-self.beta_init)*np.square(gradients[key])
            ## update r, like momentun
            self.r[key] = self.r[key]*self.alpha_init + (1-self.alpha_init)*gradients[key]
            ## modify alpha and beta 
            self.alpha_init = self.alpha_init * self.alpha
            self.beta_init = self.beta_init * self.beta
            ## update weights
            weights[key] -= (self.lr * self.r[key]) / np.sqrt(self.epsilon + self.s[key])